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

                # Remove from queued tracking
                with self._queued_lock:
                    self._queued_jobs.pop(job_id, None)

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
                    queued_job.callback(job_id, queued_job.config)
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

        # Add to queue
        self._queue.put(queued_job)

        # Track in queued jobs
        with self._queued_lock:
            self._queued_jobs[job_id] = queued_job

        # Update job status in database
        if self.job_manager:
            self.job_manager.update_job(job_id, status="queued", progress=0.0)

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

    def wait_for_completion(self, timeout: Optional[float] = None):
        """
        Wait for all jobs to complete.

        Args:
            timeout: Maximum time to wait in seconds
        """
        self._queue.join()

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
