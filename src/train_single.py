#!/usr/bin/env python
"""
Standalone single-process training entry point.

Launched by ``run_training_subprocess`` (src/training_runner.py):

    python src/train_single.py --config-path /tmp/.../config.json \
        --job-id job_123 --db-path ./data/jobs.db \
        [--uploaded-datasets-dir /tmp/.../uploaded_datasets]

Running the job in its own process — instead of on a queue worker thread
inside the API server — means:

- model loading / tokenization GIL churn can't starve the API event loop;
- SIGTERM is always honored: during load phases it aborts immediately
  (there is no checkpoint worth saving yet), during training it triggers
  the same graceful stop-with-checkpoint as the in-process path;
- the parent monitor can SIGKILL a truly wedged job, so a stop request
  can never leave the queue worker hung;
- a hard crash (segfault, OOM kill) takes down this process, not the server.

Progress, metrics, and status flow through the shared SQLite DB; the
parent monitor relays them to WebSocket clients.
"""

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# Statuses during which a SIGTERM should abort immediately — training has
# not started yet, so there is no checkpoint worth saving and no step loop
# that would ever notice the stop_requested flag.
_ABORT_ON_SIGTERM_STATUSES = {"queued", "initializing", "loading_model", "loading_dataset"}


def _load_uploaded_datasets(uploads_dir):
    """Load serialized uploaded datasets back into the in-memory dict shape
    run_training_sync expects ({id: {content, filename, size}})."""
    uploaded = {}
    if not uploads_dir or not os.path.isdir(uploads_dir):
        return uploaded
    meta_path = os.path.join(uploads_dir, "meta.json")
    if not os.path.exists(meta_path):
        return uploaded
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for dataset_id, info in meta.items():
        file_path = os.path.join(uploads_dir, info["filename"])
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                content = f.read()
            uploaded[dataset_id] = {
                "content": content,
                "filename": info["filename"],
                "size": len(content),
            }
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Merlina single-process training worker")
    parser.add_argument("--config-path", required=True, help="Path to training config JSON")
    parser.add_argument("--job-id", required=True, help="Job identifier")
    parser.add_argument("--db-path", default="./data/jobs.db", help="Path to SQLite database")
    parser.add_argument("--uploaded-datasets-dir", default=None, help="Path to uploaded datasets temp dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from src.job_manager import JobManager
    job_manager = JobManager(db_path=args.db_path)

    def _sigterm_handler(signum, frame):
        # Always set the DB flag: the training callback checks it at every
        # step end and stops gracefully (checkpoint saved).
        try:
            job_manager.request_stop(args.job_id)
        except Exception:
            pass
        status = None
        try:
            job = job_manager.get_job(args.job_id)
            status = job.status if job else None
        except Exception:
            pass
        if status in _ABORT_ON_SIGTERM_STATUSES:
            raise KeyboardInterrupt(f"stop requested during '{status}'")

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        # Load + validate config. TrainingConfig lives in merlina.py;
        # importing it pulls in the FastAPI app module, which is harmless in
        # a subprocess (same pattern as src/train_worker.py) and keeps the
        # Pydantic model as the single source of truth.
        try:
            with open(args.config_path, "r") as f:
                config_dict = json.load(f)
            from merlina import TrainingConfig
            from pydantic import TypeAdapter
            config = TypeAdapter(TrainingConfig).validate_python(config_dict)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Failed to load training config: {e}", exc_info=True)
            try:
                job_manager.update_job(args.job_id, status="failed", error=f"Config load failed: {e}")
            except Exception:
                pass
            sys.exit(1)

        uploaded_datasets = _load_uploaded_datasets(args.uploaded_datasets_dir)

        from src.training_runner import run_training_sync

        # event_loop=None: WebSocket coroutines are closed unsent in this
        # process; the parent monitor relays DB updates to connected
        # clients instead.
        run_training_sync(args.job_id, config, job_manager, uploaded_datasets, event_loop=None)

    except KeyboardInterrupt:
        # Raised by the SIGTERM handler while still in a load phase — abort
        # cleanly, nothing to checkpoint.
        logger.info(f"Job {args.job_id} aborted by stop request before training started")
        try:
            job_manager.update_job(args.job_id, status="stopped")
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
