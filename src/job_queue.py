"""
Job Queue Manager
Manages job queuing, execution, and concurrency control
"""

import logging
import threading
import queue
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3


@dataclass(order=True)
class QueuedJob:
    """Represents a job in the queue"""
    priority: int = field(compare=True)
    queued_at: float = field(compare=True)
    job_id: str = field(compare=False)
    config: Any = field(compare=False)
    callback: Callable = field(compare=False)

    def __init__(self, job_id: str, config: Any, callback: Callable, priority: JobPriority = JobPriority.NORMAL):
        # Priority is inverted for priority queue (lower number = higher priority)
        self.priority = -priority.value
        self.queued_at = time.time()
        self.job_id = job_id
        self.config = config
        self.callback = callback


class JobQueue:
    """
    Manages training job queue with configurable concurrency.

    Features:
    - Priority queue for job ordering
    - Configurable max concurrent jobs (default: 1)
    - Thread-safe job management
    - Automatic worker thread management
    - Job cancellation support
    - Queue position tracking
    """

    def __init__(self, max_concurrent_jobs: int = 1, job_manager=None):
        """
        Initialize job queue.

        Args:
            max_concurrent_jobs: Maximum number of jobs to run concurrently (default: 1)
            job_manager: JobManager instance for persistence
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_manager = job_manager

        # Priority queue for pending jobs
        self._queue = queue.PriorityQueue()

        # Track running jobs
        self._running_jobs: Dict[str, threading.Thread] = {}
        self._running_lock = threading.Lock()

        # Track all queued job IDs for fast lookup
        self._queued_jobs: Dict[str, QueuedJob] = {}
        self._queued_lock = threading.Lock()

        # Cancellation flags
        self._cancelled_jobs = set()
        self._cancel_lock = threading.Lock()

        # Worker threads
        self._workers: List[threading.Thread] = []
        self._shutdown = False

        # Start workers
        self._start_workers()

        logger.info(f"Job queue initialized with max_concurrent_jobs={max_concurrent_jobs}")

    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"JobQueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
            logger.debug(f"Started worker thread: {worker.name}")

    def _worker_loop(self):
        """Worker thread main loop"""
        logger.debug(f"Worker {threading.current_thread().name} started")

        while not self._shutdown:
            try:
                # Get next job from queue (with timeout to allow shutdown checks)
                try:
                    queued_job = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                job_id = queued_job.job_id

                # Claim the queued entry. remove() races us for it under the
                # same lock, so whoever pops it wins: a miss (or a cleared
                # callback) means the job was dropped from the queue while it
                # was waiting and must not run.
                with self._queued_lock:
                    claimed = self._queued_jobs.pop(job_id, None) is not None
                    callback = queued_job.callback
                    config = queued_job.config

                if not claimed or callback is None:
                    logger.info(f"Job {job_id} was removed from the queue before execution - skipping")
                    with self._cancel_lock:
                        self._cancelled_jobs.discard(job_id)
                    self._queue.task_done()
                    continue

                # A job whose record was deleted while it sat in the queue must
                # not execute: there is nothing left to report status to, and
                # its stored config is gone. This also covers deletions that
                # bypass the queue entirely (e.g. clearing all jobs).
                if self.job_manager is not None and self.job_manager.get_job(job_id) is None:
                    logger.info(f"Job {job_id} no longer exists (deleted while queued) - skipping execution")
                    with self._cancel_lock:
                        self._cancelled_jobs.discard(job_id)
                    self._queue.task_done()
                    continue

                # Check if job was cancelled while in queue
                with self._cancel_lock:
                    if job_id in self._cancelled_jobs:
                        logger.info(f"Job {job_id} was cancelled before execution")
                        self._cancelled_jobs.remove(job_id)
                        if self.job_manager:
                            self.job_manager.update_job(job_id, status="cancelled", progress=0.0)
                        self._queue.task_done()
                        continue

                # Mark job as running
                with self._running_lock:
                    self._running_jobs[job_id] = threading.current_thread()

                # Update job status to running (from queued)
                if self.job_manager:
                    self.job_manager.update_job(job_id, status="initializing", progress=0.0)

                logger.info(f"Worker {threading.current_thread().name} executing job {job_id}")

                # Execute the job callback
                try:
                    callback(job_id, config)
                except Exception as e:
                    logger.error(f"Job {job_id} execution failed: {e}", exc_info=True)
                    if self.job_manager:
                        self.job_manager.update_job(job_id, status="failed", error=str(e))
                finally:
                    # Remove from running jobs
                    with self._running_lock:
                        self._running_jobs.pop(job_id, None)

                    # Mark queue task as done
                    self._queue.task_done()

                    logger.info(f"Worker {threading.current_thread().name} completed job {job_id}")

            except Exception as e:
                logger.error(f"Worker {threading.current_thread().name} error: {e}", exc_info=True)

        logger.debug(f"Worker {threading.current_thread().name} shutting down")

    def submit(
        self,
        job_id: str,
        config: Any,
        callback: Callable,
        priority: JobPriority = JobPriority.NORMAL
    ) -> int:
        """
        Submit a job to the queue.

        Args:
            job_id: Unique job identifier
            config: Job configuration
            callback: Function to execute (signature: callback(job_id, config))
            priority: Job priority (LOW, NORMAL, HIGH)

        Returns:
            Queue position (0-indexed, 0 = next to run)
        """
        if self._shutdown:
            raise RuntimeError("Job queue is shutting down")

        # Create queued job
        queued_job = QueuedJob(job_id, config, callback, priority)

        # Register tracking and persist status BEFORE handing the job to the
        # queue. Once _queue.put() runs, an idle worker can pick the job up
        # immediately; if tracking/DB writes happened after (as they used to),
        # the worker's _queued_jobs.pop() could race the registration (leaking
        # a stale "queued" entry) and the DB status="queued" write could
        # overwrite the worker's "initializing".
        with self._queued_lock:
            self._queued_jobs[job_id] = queued_job

        try:
            if self.job_manager:
                self.job_manager.update_job(job_id, status="queued", progress=0.0)

            # Add to queue (job becomes visible to workers from this point)
            self._queue.put(queued_job)
        except BaseException:
            with self._queued_lock:
                self._queued_jobs.pop(job_id, None)
            raise

        # Calculate queue position
        position = self.get_queue_position(job_id)

        logger.info(f"Job {job_id} submitted to queue at position {position} with priority {priority.name}")

        return position

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a job (whether queued or running).

        For queued jobs: Removes from queue immediately
        For running jobs: Sets stop_requested flag (graceful stop)

        Args:
            job_id: Job identifier

        Returns:
            True if job was found and cancellation initiated
        """
        # Check if job is queued
        with self._queued_lock:
            if job_id in self._queued_jobs:
                # Mark as cancelled
                with self._cancel_lock:
                    self._cancelled_jobs.add(job_id)

                logger.info(f"Job {job_id} marked for cancellation (currently queued)")
                return True

        # Check if job is running
        with self._running_lock:
            if job_id in self._running_jobs:
                # For running jobs, use the existing stop mechanism
                if self.job_manager:
                    success = self.job_manager.request_stop(job_id)
                    if success:
                        logger.info(f"Stop requested for running job {job_id}")
                        return True
                return False

        logger.warning(f"Job {job_id} not found in queue or running jobs")
        return False

    def remove(self, job_id: str) -> bool:
        """
        Drop a queued job from the queue and release its config.

        Used when a job record is deleted: the queue must stop reporting the
        job as queued, must not hand it to a worker, and must not keep its
        config (which can hold API tokens) alive.

        Running jobs are untouched - use cancel() to stop those.

        Args:
            job_id: Job identifier

        Returns:
            True if a queued entry was removed
        """
        with self._queued_lock:
            queued_job = self._queued_jobs.pop(job_id, None)
            if queued_job is None:
                return False

            # The item itself stays in the PriorityQueue (it has no removal
            # API), so clear what the worker would act on: it sees the empty
            # callback under _queued_lock and skips the job, and the config
            # becomes collectable now rather than whenever the worker drains.
            queued_job.callback = None
            queued_job.config = None

        with self._cancel_lock:
            self._cancelled_jobs.discard(job_id)

        logger.info(f"Removed queued job {job_id} from the queue")
        return True

    def remove_all_queued(self) -> int:
        """
        Drop every queued job from the queue and release their configs.

        Running jobs are untouched.

        Returns:
            Number of queued entries removed
        """
        with self._queued_lock:
            queued_jobs = list(self._queued_jobs.values())
            self._queued_jobs.clear()
            for queued_job in queued_jobs:
                queued_job.callback = None
                queued_job.config = None

        with self._cancel_lock:
            for queued_job in queued_jobs:
                self._cancelled_jobs.discard(queued_job.job_id)

        if queued_jobs:
            logger.info(f"Removed {len(queued_jobs)} queued job(s) from the queue")

        return len(queued_jobs)

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get position of a job in the queue.

        Args:
            job_id: Job identifier

        Returns:
            Queue position (0 = next to run), or None if not in queue
        """
        with self._queued_lock:
            if job_id not in self._queued_jobs:
                return None

            # Get all queued jobs and sort by priority
            jobs = list(self._queued_jobs.values())
            jobs.sort()  # Uses dataclass __lt__ based on priority and time

            # Find position
            for i, job in enumerate(jobs):
                if job.job_id == job_id:
                    return i

            return None

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get queue status for a job.

        Returns:
            Dictionary with queue status information
        """
        # Check if queued
        with self._queued_lock:
            if job_id in self._queued_jobs:
                position = self.get_queue_position(job_id)
                queued_job = self._queued_jobs[job_id]
                return {
                    "state": "queued",
                    "position": position,
                    "queued_at": datetime.fromtimestamp(queued_job.queued_at).isoformat(),
                    "priority": JobPriority(-queued_job.priority).name
                }

        # Check if running
        with self._running_lock:
            if job_id in self._running_jobs:
                return {
                    "state": "running",
                    "position": None,
                    "worker": self._running_jobs[job_id].name
                }

        # Check if cancelled
        with self._cancel_lock:
            if job_id in self._cancelled_jobs:
                return {
                    "state": "cancelled",
                    "position": None
                }

        # Not in queue system
        return {
            "state": "unknown",
            "position": None
        }

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get overall queue statistics.

        Returns:
            Dictionary with queue stats
        """
        with self._queued_lock:
            queued_count = len(self._queued_jobs)

        with self._running_lock:
            running_count = len(self._running_jobs)

        return {
            "queued": queued_count,
            "running": running_count,
            "max_concurrent": self.max_concurrent_jobs,
            "workers": len(self._workers),
            "available_slots": max(0, self.max_concurrent_jobs - running_count)
        }

    def list_queued_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of all queued jobs in order.

        Returns:
            List of job info dictionaries
        """
        with self._queued_lock:
            jobs = list(self._queued_jobs.values())
            jobs.sort()  # Sort by priority and time

            return [
                {
                    "job_id": job.job_id,
                    "position": i,
                    "priority": JobPriority(-job.priority).name,
                    "queued_at": datetime.fromtimestamp(job.queued_at).isoformat()
                }
                for i, job in enumerate(jobs)
            ]

    def list_running_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of all running jobs.

        Returns:
            List of job info dictionaries
        """
        with self._running_lock:
            return [
                {
                    "job_id": job_id,
                    "worker": thread.name
                }
                for job_id, thread in self._running_jobs.items()
            ]

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all jobs to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if all jobs completed, False if the timeout expired first
        """
        if timeout is None:
            self._queue.join()
            return True

        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def shutdown(self, wait: bool = True):
        """
        Shutdown the job queue.

        Args:
            wait: If True, wait for running jobs to complete
        """
        logger.info("Shutting down job queue...")
        self._shutdown = True

        if wait:
            # Wait for queue to empty
            self._queue.join()

            # Wait for workers to finish
            for worker in self._workers:
                worker.join(timeout=5.0)

        logger.info("Job queue shut down complete")

    def __del__(self):
        """Cleanup on deletion"""
        if not self._shutdown:
            self.shutdown(wait=False)
