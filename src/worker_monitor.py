"""
Monitor for a detached training worker process.

Training runs in a subprocess (``accelerate launch`` -> ``src/train_worker.py``)
that writes its own status and metrics to the job DB and appends JSONL
progress records to a file. The API side needs exactly one loop per worker:

* forward a stop request (SIGTERM to the worker's process group, escalating
  to SIGKILL after ``STOP_GRACE_SECONDS`` so a stop is always honoured),
* tail the progress file and relay records to WebSocket clients,
* when the worker exits, reconcile the DB: a job still in an active status
  gets marked failed (the worker died without saying why), the pid is
  cleared and the run directory removed.

:func:`monitor_worker` is that loop. It serves both the worker launched by
this process (``training_runner.run_training_distributed``) and a worker
re-adopted after an API restart (``worker_reattach``), so the two paths can't
drift apart. It never imports torch or the training stack.
"""

import json
import logging
import os
import shutil
import signal
import time
from typing import Any, Callable, Optional

from src.job_manager import TERMINAL_STATUSES
from src.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

STOP_GRACE_SECONDS = 180
POLL_INTERVAL_SECONDS = 1.0


def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists (regardless of ownership)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def signal_worker(pid: int, sig: int) -> None:
    """Signal the worker's whole process group (it is its own session leader)."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, OSError):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def forward_progress_record(record: dict, job_id: str, event_loop=None) -> None:
    """Relay one worker progress record to WebSocket clients.

    No DB writes here: the worker persists its own metrics and status. The
    progress file exists only so live clients keep getting updates.
    """
    ws = websocket_manager
    rec_type = record.get("type", "")

    if rec_type == "metrics":
        message = ws.status_message(
            job_id, "training", record.get("progress", 0.5),
            current_step=record.get("current_step"),
            total_steps=record.get("total_steps"),
            loss=record.get("loss"),
            learning_rate=record.get("learning_rate"),
            gpu_memory=record.get("gpu_memory"),
            extra_metrics=record.get("extra"),
        )
    elif rec_type == "eval":
        if "eval_loss" not in record and not record.get("extra"):
            return
        message = ws.status_message(
            job_id, "training", eval_loss=record.get("eval_loss"),
            current_step=record.get("current_step"),
            extra_metrics=record.get("extra"),
        )
    elif rec_type == "status":
        message = ws.status_message(
            job_id, record.get("status", "training"), message=record.get("message")
        )
    elif rec_type == "completed":
        message = {"type": "completed", "job_id": job_id,
                   "output_dir": record.get("output_dir", ""), "timestamp": time.time()}
    elif rec_type == "error":
        message = {"type": "error", "job_id": job_id,
                   "error": record.get("error", "Unknown error"), "timestamp": time.time()}
    else:
        return

    ws.publish(job_id, message, loop=event_loop)


class _ProgressTail:
    """Incrementally reads new JSONL lines from the worker's progress file."""

    def __init__(self, path: Optional[str], job_id: str, event_loop):
        self.path = path
        self.job_id = job_id
        self.event_loop = event_loop
        self.offset = 0

    def drain(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                f.seek(self.offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    forward_progress_record(record, self.job_id, self.event_loop)
                self.offset = f.tell()
        except Exception as e:
            logger.debug(f"Error reading progress file for {self.job_id}: {e}")


def monitor_worker(
    job_id: str,
    pid: int,
    progress_file: Optional[str],
    job_manager: Any,
    event_loop=None,
    *,
    is_alive: Optional[Callable[[], bool]] = None,
    returncode: Optional[Callable[[], Optional[int]]] = None,
    reattached: bool = False,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """
    Block until the worker exits, relaying progress and stop requests.

    Args:
        job_id: Job the worker is running.
        pid: Worker pid (the launcher; it leads its own process group).
        progress_file: JSONL file the worker appends to (may be None).
        job_manager: JobManager for stop-flag polling and final reconciliation.
        event_loop: Server event loop for WebSocket relays (falls back to
            the loop bound on ``websocket_manager``).
        is_alive: Liveness probe. Defaults to a pid check; a parent that
            owns the ``Popen`` should pass ``lambda: proc.poll() is None`` so
            the child is reaped.
        returncode: Exit-code getter used in the failure message, if known.
        reattached: Whether this monitor adopted the worker after a restart
            (only affects log/error wording).
        poll_interval: Seconds between iterations.
    """
    alive = is_alive or (lambda: pid_alive(pid))
    tail = _ProgressTail(progress_file, job_id, event_loop)
    stop_sent_at: Optional[float] = None

    logger.info(f"{'Re-attached to' if reattached else 'Monitoring'} training worker for job {job_id} (pid {pid})")

    while alive():
        # Nothing in here may kill the loop while the worker lives: a
        # transient DB error must not orphan a running job.
        try:
            job = job_manager.get_job(job_id)
            if job is not None and job.stop_requested:
                if stop_sent_at is None:
                    logger.info(f"Forwarding stop (SIGTERM) to worker for job {job_id}")
                    signal_worker(pid, signal.SIGTERM)
                    stop_sent_at = time.monotonic()
                elif time.monotonic() - stop_sent_at > STOP_GRACE_SECONDS:
                    logger.warning(
                        f"Job {job_id} did not exit {STOP_GRACE_SECONDS}s after SIGTERM - escalating to SIGKILL"
                    )
                    signal_worker(pid, signal.SIGKILL)
                    stop_sent_at = time.monotonic()  # don't spam SIGKILL every tick
        except Exception as e:
            logger.debug(f"Stop check failed for job {job_id}: {e}")

        tail.drain()
        time.sleep(poll_interval)

    # Worker is gone: pick up its final records, then reconcile the DB.
    tail.drain()
    _finalize(job_id, pid, progress_file, job_manager, event_loop,
              returncode() if returncode else None, reattached)


def _finalize(job_id, pid, progress_file, job_manager, event_loop, rc, reattached) -> None:
    try:
        job = job_manager.get_job(job_id)
        if job is not None and job.status not in TERMINAL_STATUSES:
            error = f"Training worker (pid {pid}) exited without reporting a final status"
            if reattached:
                error += " after the server re-attached to it"
            if rc is not None:
                error += f" (exit code {rc})"
            logger.error(f"Job {job_id}: {error}")
            # Single write: whoever sees the terminal status must also see the
            # cleared pid, or the job looks finished yet still advertises a
            # worker that is already gone.
            job_manager.update_job(job_id, status="failed", error=error, worker_pid=None)
            websocket_manager.publish(
                job_id, {"type": "error", "job_id": job_id, "error": error, "timestamp": time.time()},
                loop=event_loop,
            )
        else:
            job_manager.update_job(job_id, worker_pid=None)
    except Exception:
        logger.exception(f"Failed to finalize job {job_id} after its worker exited")

    # Run files (config.json with credentials, progress.jsonl) are no longer
    # needed once the worker is gone.
    if progress_file:
        shutil.rmtree(os.path.dirname(progress_file), ignore_errors=True)

    logger.info(f"Worker monitor for job {job_id} finished")
