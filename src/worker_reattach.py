"""
Re-attach to training workers that survived an API restart.

Training runs in a detached subprocess (``accelerate launch`` ->
``src/train_worker.py``, started with ``start_new_session=True``), so an API
restart does not kill it. On startup the server calls
:func:`reattach_orphaned_workers`, which scans the job DB for jobs stuck in
an active status and either:

* re-attaches a :func:`src.worker_monitor.monitor_worker` to a worker whose
  recorded pid is still alive and still ours (progress keeps flowing to
  WebSocket clients, ``/stop`` keeps working, the job is finalized when the
  worker exits), or
* marks the job failed when the worker is gone, so it stops presenting as
  "training" forever.

Deliberately free of torch / training-stack imports so the startup scan is
cheap and works where the ML dependencies are mocked.
"""

import logging
import os
import threading
from typing import Any

from src.job_manager import ACTIVE_STATUSES, TERMINAL_STATUSES  # noqa: F401  (re-exported)
from src.worker_monitor import STOP_GRACE_SECONDS, monitor_worker, pid_alive  # noqa: F401

logger = logging.getLogger(__name__)

RESTART_LOST_ERROR = (
    "Server restarted while this job was active and its training worker is no "
    "longer running. The job did not finish. Use POST /jobs/{job_id}/retry to "
    "requeue it with the same configuration."
)

# Backwards-compatible alias.
_pid_alive = pid_alive


def _is_our_worker(pid: int, job_id: str) -> bool:
    """Check the pid is alive AND still the worker we launched for job_id.

    Pids get reused; before signalling or trusting one recorded in the DB,
    verify its command line still mentions the job id (the launcher is
    invoked with ``--job-id <job_id>``). On platforms without /proc, fall
    back to plain liveness.
    """
    if not pid_alive(pid):
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        return job_id in cmdline
    except FileNotFoundError:
        # Either the process just exited, or there's no procfs.
        return not os.path.isdir("/proc")
    except OSError:
        # procfs exists but is unreadable - assume it's ours rather than
        # falsely declaring a live worker dead.
        return True


def reattach_orphaned_workers(job_manager: Any, event_loop=None) -> dict:
    """Reconcile active jobs from the DB with reality after a server start.

    Returns a dict with ``reattached`` and ``failed`` job-id lists.
    """
    reattached, failed = [], []

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
            threading.Thread(
                target=monitor_worker,
                args=(job.job_id, pid, getattr(job, "progress_file", None), job_manager, event_loop),
                kwargs={"reattached": True},
                name=f"Reattach-{job.job_id}",
                daemon=True,
            ).start()
            reattached.append(job.job_id)
            continue

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
        logger.info(f"Marked {len(failed)} job(s) failed after restart (no live worker): {failed}")
    return {"reattached": reattached, "failed": failed}
