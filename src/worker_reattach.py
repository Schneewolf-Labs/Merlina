"""
Re-attach to training workers that survived an API restart.

Training runs in a detached subprocess (``accelerate launch`` →
``src/train_worker.py``, started with ``start_new_session=True``), so an API
restart does not kill it — but it used to orphan it: the monitor thread died
with the old process, the job could no longer be stopped through the queue,
and its progress stopped flowing to WebSocket clients.

On startup the server calls :func:`reattach_orphaned_workers`, which scans the
job DB for jobs stuck in an active status and either:

* re-attaches a monitor to a worker whose recorded pid is still alive
  (tailing the durable progress file under ``data/worker_runs/<job_id>/``,
  forwarding stop requests, and finalizing the job when the worker exits), or
* marks the job failed when the worker is gone, so it stops presenting as
  "training" forever.

This module deliberately avoids importing the heavy training stack (torch,
grimoire, transformers) so the startup scan is cheap and works on servers
where the ML dependencies are mocked.
"""

import json
import logging
import os
import shutil
import signal
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Statuses that mean "this job should have a live worker or queue entry".
ACTIVE_STATUSES = {
    "started",
    "queued",
    "initializing",
    "loading_model",
    "loading_dataset",
    "training",
    "saving",
    "stopping",
    "running",
}

# Statuses a worker sets on its own way out; never overwrite these.
TERMINAL_STATUSES = {"completed", "failed", "stopped", "cancelled", "uploading"}

STOP_GRACE_SECONDS = 180

RESTART_LOST_ERROR = (
    "Server restarted while this job was active and its training worker is no "
    "longer running. The job did not finish. Use POST /jobs/{job_id}/retry to "
    "requeue it with the same configuration."
)


def _pid_alive(pid: int) -> bool:
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


def _is_our_worker(pid: int, job_id: str) -> bool:
    """Check the pid is alive AND still the worker we launched for job_id.

    Pids get reused; before signalling or trusting one recorded in the DB,
    verify its command line still mentions the job id (the launcher is
    invoked with ``--job-id <job_id>``). On platforms without /proc, fall
    back to plain liveness.
    """
    if not _pid_alive(pid):
        return False
    cmdline_path = f"/proc/{pid}/cmdline"
    try:
        with open(cmdline_path, "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        return job_id in cmdline
    except FileNotFoundError:
        # Either the process just exited, or there's no procfs. Distinguish:
        if os.path.isdir("/proc"):
            return False
        return True
    except OSError:
        # procfs exists but is unreadable — assume it's ours rather than
        # falsely declaring a live worker dead.
        return True


def _send_ws(coro, event_loop) -> None:
    """Best-effort WebSocket forward from this sync thread."""
    import asyncio

    try:
        if event_loop is None:
            coro.close()
            return
        asyncio.run_coroutine_threadsafe(coro, event_loop)
    except Exception as e:
        logger.debug(f"Could not send WebSocket update: {e}")


def _forward_record(record: dict, job_id: str, event_loop) -> None:
    """Forward one progress-file record to WebSocket clients.

    DB persistence is not needed here: the worker writes its own metrics and
    status to the DB directly; the progress file only feeds live clients.
    """
    from src.websocket_manager import websocket_manager

    rec_type = record.get("type", "")

    if rec_type == "metrics":
        _send_ws(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="training",
                progress=record.get("progress", 0.5),
                current_step=record.get("current_step"),
                total_steps=record.get("total_steps"),
                loss=record.get("loss"),
                learning_rate=record.get("learning_rate"),
                gpu_memory=record.get("gpu_memory"),
            ),
            event_loop,
        )
    elif rec_type == "eval":
        if "eval_loss" in record:
            _send_ws(
                websocket_manager.send_status_update(
                    job_id=job_id, status="training", eval_loss=record["eval_loss"]
                ),
                event_loop,
            )
    elif rec_type == "completed":
        _send_ws(
            websocket_manager.send_completion(
                job_id=job_id, output_dir=record.get("output_dir", "")
            ),
            event_loop,
        )
    elif rec_type == "error":
        _send_ws(
            websocket_manager.send_error(
                job_id=job_id, error=record.get("error", "Unknown error")
            ),
            event_loop,
        )
    elif rec_type == "status":
        _send_ws(
            websocket_manager.send_status_update(
                job_id=job_id,
                status=record.get("status", "training"),
                message=record.get("message"),
            ),
            event_loop,
        )


def _drain_progress(progress_file: Optional[str], last_pos: int, job_id: str, event_loop) -> int:
    """Read new JSONL records from the progress file; return the new offset."""
    if not progress_file:
        return last_pos
    try:
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _forward_record(json.loads(line), job_id, event_loop)
                    except json.JSONDecodeError:
                        pass
                last_pos = f.tell()
    except Exception as e:
        logger.debug(f"Error reading progress file for {job_id}: {e}")
    return last_pos


def _signal_worker(pid: int, job_id: str, sig: int) -> None:
    """Signal the worker's whole process group (it is its own session leader)."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, OSError):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _monitor_reattached_worker(
    job_id: str,
    pid: int,
    progress_file: Optional[str],
    job_manager: Any,
    event_loop=None,
) -> None:
    """Monitor a re-adopted worker: tail progress, forward stops, finalize."""
    logger.info(f"Re-attached to training worker for job {job_id} (pid {pid})")
    last_pos = 0
    stop_sigterm_at = None

    while _pid_alive(pid):
        try:
            job = job_manager.get_job(job_id)
            if job and job.stop_requested:
                if stop_sigterm_at is None:
                    logger.info(f"Forwarding stop (SIGTERM) to re-attached worker for job {job_id}")
                    _signal_worker(pid, job_id, signal.SIGTERM)
                    stop_sigterm_at = time.monotonic()
                elif time.monotonic() - stop_sigterm_at > STOP_GRACE_SECONDS:
                    logger.warning(
                        f"Job {job_id} did not exit {STOP_GRACE_SECONDS}s after "
                        f"SIGTERM — escalating to SIGKILL"
                    )
                    _signal_worker(pid, job_id, signal.SIGKILL)
        except Exception as e:
            logger.debug(f"Stop check failed for re-attached job {job_id}: {e}")

        last_pos = _drain_progress(progress_file, last_pos, job_id, event_loop)
        time.sleep(1.0)

    # Worker exited — pick up any final records, then reconcile status.
    _drain_progress(progress_file, last_pos, job_id, event_loop)

    try:
        job = job_manager.get_job(job_id)
        if job and job.status not in TERMINAL_STATUSES:
            error_msg = (
                f"Training worker (pid {pid}) exited without reporting a final "
                "status after the server re-attached to it."
            )
            logger.error(f"Job {job_id}: {error_msg}")
            # Single update: a watcher that sees the terminal status must also see the
            # cleared pid. Writing them separately leaves a window where the job looks
            # finished but still advertises a worker that is already gone.
            job_manager.update_job(
                job_id, status="failed", error=error_msg, worker_pid=None
            )
            from src.websocket_manager import websocket_manager

            _send_ws(websocket_manager.send_error(job_id=job_id, error=error_msg), event_loop)
        else:
            job_manager.update_job(job_id, worker_pid=None)
    except Exception:
        logger.exception(f"Failed to finalize re-attached job {job_id}")

    # Run files (config.json with credentials, progress.jsonl) are no longer
    # needed once the worker is gone.
    if progress_file:
        try:
            shutil.rmtree(os.path.dirname(progress_file), ignore_errors=True)
        except Exception:
            pass

    logger.info(f"Re-attached monitor for job {job_id} finished")


def reattach_orphaned_workers(job_manager: Any, event_loop=None) -> dict:
    """Reconcile active jobs from the DB with reality after a server start.

    Returns a dict with ``reattached`` and ``failed`` job-id lists.
    """
    reattached = []
    failed = []

    try:
        jobs = job_manager.list_jobs(limit=100000)
    except Exception:
        logger.exception("Could not list jobs for worker reattach scan")
        return {"reattached": reattached, "failed": failed}

    for job in jobs:
        if job.status not in ACTIVE_STATUSES:
            continue

        pid = getattr(job, "worker_pid", None)
        if pid and _is_our_worker(pid, job.job_id):
            thread = threading.Thread(
                target=_monitor_reattached_worker,
                args=(job.job_id, pid, getattr(job, "progress_file", None), job_manager, event_loop),
                name=f"Reattach-{job.job_id}",
                daemon=True,
            )
            thread.start()
            reattached.append(job.job_id)
        else:
            try:
                job_manager.update_job(
                    job.job_id,
                    status="failed",
                    error=RESTART_LOST_ERROR.replace("{job_id}", job.job_id),
                    worker_pid=None,
                )
                failed.append(job.job_id)
            except Exception:
                logger.exception(f"Could not mark orphaned job {job.job_id} as failed")

    if reattached:
        logger.info(f"Re-attached to {len(reattached)} running training worker(s): {reattached}")
    if failed:
        logger.info(
            f"Marked {len(failed)} job(s) failed after restart (no live worker): {failed}"
        )
    return {"reattached": reattached, "failed": failed}
