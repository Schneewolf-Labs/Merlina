"""
Job queue: bounded-concurrency execution of training jobs.

One lock, one condition variable, two dictionaries. Every observable fact
about the queue (what is waiting, what is running, in which order) lives in
``_pending`` and ``_running`` and is only ever read or written under
``_cv``. There is no separate heap to keep in sync, so cancelling or removing
a job is a plain dictionary delete: nothing is left behind for a worker to
"skip" later, and nothing keeps the job's config (which can hold API tokens)
alive.

Lifecycle of a job in this queue::

    submit()  -> DB status "queued"      -> _pending
    worker    -> DB status "initializing" -> _running -> callback(job_id, config)
    cancel()  (pending) -> DB status "cancelled", gone from _pending
    cancel()  (running) -> job_manager.request_stop(); the job's own code
                           (the training loop / worker monitor) honours it
    remove()  (pending) -> gone from _pending, DB untouched (caller deleted
                           the row)

Ordering is by priority (HIGH first), then by submission order.

Persistence is optional: ``job_manager`` may be ``None`` (tests), in which
case the queue is purely in-memory. When it is set, the "initializing" write
doubles as the existence check — a job whose record was deleted while it
waited has no row to update, and ``update_job`` reporting ``False`` is what
stops it from running.
"""

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

JobCallback = Callable[[str, Any], None]


class JobPriority(Enum):
    """Job priority levels. Higher value runs first."""
    LOW = 1
    NORMAL = 2
    HIGH = 3


@dataclass
class QueuedJob:
    """A job waiting to run."""
    job_id: str
    config: Any
    callback: Optional[JobCallback]
    priority: JobPriority
    seq: int
    queued_at: float = field(default_factory=time.time)

    @property
    def sort_key(self):
        # Higher priority first, then FIFO within a priority level.
        return (-self.priority.value, self.seq)

    def release(self) -> None:
        """Drop references to the config and callback once the job can no
        longer run, so secrets in the config don't outlive the queue entry."""
        self.config = None
        self.callback = None


class JobQueue:
    """Priority queue with ``max_concurrent_jobs`` worker threads."""

    def __init__(self, max_concurrent_jobs: int = 1, job_manager=None, name: str = "jobs"):
        self.max_concurrent_jobs = max(1, int(max_concurrent_jobs))
        self.job_manager = job_manager
        self.name = name

        self._cv = threading.Condition()
        self._pending: Dict[str, QueuedJob] = {}
        self._running: Dict[str, threading.Thread] = {}
        self._seq = itertools.count()
        self._shutdown = False

        self._workers: List[threading.Thread] = [
            threading.Thread(
                target=self._worker_loop,
                name=f"JobQueueWorker-{name}-{i}",
                daemon=True,
            )
            for i in range(self.max_concurrent_jobs)
        ]
        for worker in self._workers:
            worker.start()

        logger.info(f"Job queue '{name}' started with max_concurrent_jobs={self.max_concurrent_jobs}")

    # ------------------------------------------------------------------ #
    # Worker side
    # ------------------------------------------------------------------ #

    def _worker_loop(self) -> None:
        me = threading.current_thread()
        while True:
            with self._cv:
                while not self._shutdown and not self._pending:
                    self._cv.wait()
                if self._shutdown:
                    return
                job = self._pop_next_locked()
                self._running[job.job_id] = me

            try:
                self._execute(job)
            finally:
                with self._cv:
                    self._running.pop(job.job_id, None)
                    self._cv.notify_all()  # wake wait_for_completion()

    def _pop_next_locked(self) -> QueuedJob:
        job_id = min(self._pending, key=lambda jid: self._pending[jid].sort_key)
        return self._pending.pop(job_id)

    def _execute(self, job: QueuedJob) -> None:
        job_id = job.job_id
        worker = threading.current_thread().name
        try:
            if self.job_manager is not None:
                # One write, two purposes: flip the status and prove the row
                # still exists. A job deleted while it waited must not run —
                # there is nothing left to report status to.
                if not self.job_manager.update_job(job_id, status="initializing", progress=0.0):
                    logger.info(f"Job {job_id} no longer exists (deleted while queued) - skipping")
                    return

            logger.info(f"{worker} executing job {job_id}")
            job.callback(job_id, job.config)
            logger.info(f"{worker} finished job {job_id}")
        except Exception as e:
            logger.error(f"Job {job_id} execution failed: {e}", exc_info=True)
            if self.job_manager is not None:
                try:
                    self.job_manager.update_job(job_id, status="failed", error=str(e))
                except Exception:
                    logger.exception(f"Could not mark job {job_id} as failed")
        finally:
            job.release()

    # ------------------------------------------------------------------ #
    # Submission / cancellation
    # ------------------------------------------------------------------ #

    def submit(
        self,
        job_id: str,
        config: Any,
        callback: JobCallback,
        priority: JobPriority = JobPriority.NORMAL,
    ) -> int:
        """
        Queue a job. Returns its 0-based position (0 = next to run).

        Raises ``ValueError`` if the job id is already queued or running and
        ``RuntimeError`` if the queue is shutting down. A failed DB write
        leaves no trace in the queue.
        """
        with self._cv:
            self._check_can_submit_locked(job_id)

        # The DB write happens outside the lock so API reads of queue state
        # never wait on SQLite. Nothing can observe the job until it is in
        # ``_pending``, so "queued" is guaranteed to land before any worker
        # can write "initializing".
        if self.job_manager is not None:
            self.job_manager.update_job(job_id, status="queued", progress=0.0)

        with self._cv:
            self._check_can_submit_locked(job_id)  # re-check: lock was dropped
            self._pending[job_id] = QueuedJob(
                job_id=job_id,
                config=config,
                callback=callback,
                priority=priority,
                seq=next(self._seq),
            )
            position = self._position_locked(job_id)
            self._cv.notify()

        logger.info(f"Job {job_id} queued at position {position} ({priority.name})")
        return position

    def _check_can_submit_locked(self, job_id: str) -> None:
        if self._shutdown:
            raise RuntimeError("Job queue is shutting down")
        if job_id in self._pending or job_id in self._running:
            raise ValueError(f"Job {job_id} is already queued or running")

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a job.

        Queued: removed immediately and recorded as ``cancelled``.
        Running: a graceful stop is requested through the job manager; the
        running code is responsible for honouring it.

        Returns True if the job was found in the queue.
        """
        with self._cv:
            job = self._pending.pop(job_id, None)
            running = job_id in self._running

        if job is not None:
            job.release()
            if self.job_manager is not None:
                self.job_manager.update_job(job_id, status="cancelled", progress=0.0)
            logger.info(f"Job {job_id} cancelled while queued")
            return True

        if running:
            if self.job_manager is None:
                return False
            if self.job_manager.request_stop(job_id):
                logger.info(f"Stop requested for running job {job_id}")
                return True
            return False

        return False

    def remove(self, job_id: str) -> bool:
        """
        Drop a queued job without touching its DB record (for use after the
        record itself was deleted). Running jobs are untouched.
        """
        with self._cv:
            job = self._pending.pop(job_id, None)
        if job is None:
            return False
        job.release()
        logger.info(f"Removed queued job {job_id}")
        return True

    def remove_all_queued(self) -> int:
        """Drop every queued job. Running jobs are untouched."""
        with self._cv:
            jobs = list(self._pending.values())
            self._pending.clear()
        for job in jobs:
            job.release()
        if jobs:
            logger.info(f"Removed {len(jobs)} queued job(s)")
        return len(jobs)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def _ordered_pending_locked(self) -> List[QueuedJob]:
        return sorted(self._pending.values(), key=lambda j: j.sort_key)

    def _position_locked(self, job_id: str) -> Optional[int]:
        for i, job in enumerate(self._ordered_pending_locked()):
            if job.job_id == job_id:
                return i
        return None

    def get_queue_position(self, job_id: str) -> Optional[int]:
        with self._cv:
            return self._position_locked(job_id)

    def get_status(self, job_id: str) -> Dict[str, Any]:
        with self._cv:
            job = self._pending.get(job_id)
            if job is not None:
                return {
                    "state": "queued",
                    "position": self._position_locked(job_id),
                    "queued_at": datetime.fromtimestamp(job.queued_at).isoformat(),
                    "priority": job.priority.name,
                }
            thread = self._running.get(job_id)
            if thread is not None:
                return {"state": "running", "position": None, "worker": thread.name}
        return {"state": "unknown", "position": None}

    def is_tracking(self, job_id: str) -> bool:
        """True if the job is queued or running here."""
        with self._cv:
            return job_id in self._pending or job_id in self._running

    def get_queue_stats(self) -> Dict[str, Any]:
        with self._cv:
            queued, running = len(self._pending), len(self._running)
        return {
            "queued": queued,
            "running": running,
            "max_concurrent": self.max_concurrent_jobs,
            "workers": len(self._workers),
            "available_slots": max(0, self.max_concurrent_jobs - running),
        }

    def list_queued_jobs(self) -> List[Dict[str, Any]]:
        with self._cv:
            return [
                {
                    "job_id": job.job_id,
                    "position": i,
                    "priority": job.priority.name,
                    "queued_at": datetime.fromtimestamp(job.queued_at).isoformat(),
                }
                for i, job in enumerate(self._ordered_pending_locked())
            ]

    def list_running_jobs(self) -> List[Dict[str, Any]]:
        with self._cv:
            return [
                {"job_id": job_id, "worker": thread.name}
                for job_id, thread in self._running.items()
            ]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Block until nothing is queued or running. Returns False on timeout."""
        with self._cv:
            return self._cv.wait_for(
                lambda: not self._pending and not self._running, timeout=timeout
            )

    def shutdown(self, wait: bool = True) -> None:
        """
        Stop accepting work. Workers finish their current job and exit;
        anything still queued stays queued in the DB for the restart
        reconciler (``src/worker_reattach.py``) to deal with.
        """
        with self._cv:
            if self._shutdown:
                return
            self._shutdown = True
            self._cv.notify_all()
        logger.info(f"Job queue '{self.name}' shutting down")
        if wait:
            for worker in self._workers:
                worker.join()
