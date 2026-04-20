"""
HuggingFace Jobs training backend.

Submits a Merlina training run to HuggingFace Jobs instead of executing it
on the local server. Reuses ``src/train_worker.py`` as the entry point
inside the container; the config is shipped in a ``MERLINA_CONFIG_B64``
environment variable and progress is streamed back through the job's
stdout logs (each progress record prefixed with ``MERLINA_PROGRESS:``).

Requires ``huggingface_hub >= 0.34`` and a valid ``hf_token`` on the
training config (HF Pro subscription needed to launch jobs).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from typing import Any, Optional

from src.job_manager import JobManager
from src.training_runner import _handle_progress_record, send_websocket_update
from src.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

PROGRESS_PREFIX = "MERLINA_PROGRESS:"
POLL_INTERVAL_SECONDS = 5.0
TERMINAL_STATES = {"completed", "failed", "canceled", "cancelled", "error"}


def _import_hf_jobs():
    """Import the huggingface_hub Jobs API, raising a clear error if unavailable."""
    try:
        from huggingface_hub import (
            cancel_job,
            fetch_job_logs,
            inspect_job,
            run_job,
        )
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace Jobs requires huggingface_hub>=0.34. "
            "Upgrade with: pip install --upgrade huggingface_hub"
        ) from exc
    return run_job, inspect_job, fetch_job_logs, cancel_job


def _build_secrets(config: Any) -> dict:
    """Collect secrets to inject into the job (server-side encrypted)."""
    secrets = {}
    if getattr(config, "hf_token", None):
        secrets["HF_TOKEN"] = config.hf_token
    if getattr(config, "use_wandb", False) and getattr(config, "wandb_key", None):
        secrets["WANDB_API_KEY"] = config.wandb_key
    return secrets


def _encode_config(config: Any) -> str:
    """Serialize the Pydantic config to base64-encoded JSON for env transport."""
    raw = json.dumps(config.model_dump()).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _reject_unsupported_sources(config: Any) -> None:
    """HF Jobs can't read locally-uploaded dataset blobs; require Hub/URL sources."""
    dataset = getattr(config, "dataset", None)
    if dataset is None:
        return
    source = getattr(dataset, "source", None)
    source_type = getattr(source, "source_type", None) if source else None
    if source_type == "uploaded":
        raise ValueError(
            "HF Jobs backend cannot access uploaded dataset files. "
            "Push the dataset to the HuggingFace Hub or use a URL-based source."
        )


def _log_stream_worker(
    fetch_job_logs,
    hf_job_id: str,
    job_id: str,
    job_manager: JobManager,
    event_loop: Optional[asyncio.AbstractEventLoop],
    stop_event: threading.Event,
) -> None:
    """Consume streaming logs from the HF Job, forwarding progress records."""
    try:
        for raw_line in fetch_job_logs(job_id=hf_job_id):
            if stop_event.is_set():
                break
            line = (raw_line or "").rstrip()
            if not line:
                continue

            if PROGRESS_PREFIX in line:
                _, _, payload = line.partition(PROGRESS_PREFIX)
                payload = payload.strip()
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("Dropping malformed progress line: %s", payload[:200])
                    continue
                _handle_progress_record(record, job_id, job_manager, event_loop)
            else:
                logger.info("[hf-job %s] %s", hf_job_id, line)
    except Exception as exc:
        logger.warning("HF Job log stream ended: %s", exc)


def _poll_until_terminal(
    inspect_job,
    cancel_job,
    hf_job_id: str,
    job_id: str,
    job_manager: JobManager,
    stop_event: threading.Event,
) -> str:
    """Poll the job until it reaches a terminal state. Returns final status."""
    while True:
        # Forward stop requests to HF Jobs.
        job_record = job_manager.get_job(job_id)
        if job_record and job_record.stop_requested:
            logger.info("Forwarding stop request to HF Job %s", hf_job_id)
            try:
                cancel_job(job_id=hf_job_id)
            except Exception as exc:
                logger.warning("cancel_job failed: %s", exc)

        try:
            info = inspect_job(job_id=hf_job_id)
        except Exception as exc:
            logger.warning("inspect_job failed, will retry: %s", exc)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        status = (getattr(info, "status", None) or "").lower()
        if status in TERMINAL_STATES:
            stop_event.set()
            return status

        time.sleep(POLL_INTERVAL_SECONDS)


def run_training_hf_jobs(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    """
    Submit a Merlina training run to HuggingFace Jobs and relay its progress.

    Mirrors the signature of ``run_training_sync`` / ``run_training_distributed``
    so it can be swapped in by the training callback dispatcher.
    """
    run_job, inspect_job, fetch_job_logs, cancel_job = _import_hf_jobs()

    if not getattr(config, "hf_token", None):
        raise ValueError(
            "HF Jobs backend requires an hf_token (set in config or HF_TOKEN env)."
        )
    _reject_unsupported_sources(config)

    if uploaded_datasets:
        logger.warning(
            "Ignoring %d uploaded dataset(s) — HF Jobs backend only reads Hub/URL sources.",
            len(uploaded_datasets),
        )

    job_manager.update_job(job_id, status="initializing", progress=0.0)
    send_websocket_update(
        websocket_manager.send_status_update(
            job_id=job_id,
            status="initializing",
            progress=0.0,
        ),
        event_loop,
    )

    env = {
        "MERLINA_JOB_ID": job_id,
        "MERLINA_CONFIG_B64": _encode_config(config),
    }
    secrets = _build_secrets(config)

    command = ["/opt/merlina/scripts/hf_jobs_entrypoint.sh"]

    logger.info(
        "Submitting HF Job for %s: image=%s flavor=%s timeout=%s",
        job_id, config.hf_jobs_image, config.hf_jobs_flavor, config.hf_jobs_timeout,
    )

    try:
        info = run_job(
            image=config.hf_jobs_image,
            command=command,
            flavor=config.hf_jobs_flavor,
            env=env,
            secrets=secrets,
            timeout=config.hf_jobs_timeout,
            token=config.hf_token,
        )
    except Exception as exc:
        logger.error("HF Job submission failed: %s", exc, exc_info=True)
        job_manager.update_job(job_id, status="failed", error=f"HF Jobs submit failed: {exc}")
        send_websocket_update(
            websocket_manager.send_error(job_id=job_id, error=str(exc)),
            event_loop,
        )
        return

    hf_job_id = getattr(info, "id", None) or getattr(info, "job_id", None)
    hf_job_url = getattr(info, "url", None)
    logger.info("HF Job submitted: id=%s url=%s", hf_job_id, hf_job_url)

    # Store the HF job id in the error field for now (no dedicated column yet) so
    # it shows up in the UI and logs; a dedicated column can be added later.
    job_manager.update_job(
        job_id,
        status="running",
        progress=0.05,
    )
    send_websocket_update(
        websocket_manager.send_status_update(
            job_id=job_id,
            status="running",
            progress=0.05,
        ),
        event_loop,
    )

    stop_event = threading.Event()
    log_thread = threading.Thread(
        target=_log_stream_worker,
        args=(fetch_job_logs, hf_job_id, job_id, job_manager, event_loop, stop_event),
        daemon=True,
        name=f"hfjob-logs-{hf_job_id}",
    )
    log_thread.start()

    final_status = _poll_until_terminal(
        inspect_job, cancel_job, hf_job_id, job_id, job_manager, stop_event,
    )

    # Give the log thread a moment to drain remaining lines after the job ends.
    log_thread.join(timeout=15.0)

    current = job_manager.get_job(job_id)
    already_terminal = current and current.status in {"completed", "failed", "stopped"}

    if already_terminal:
        return

    if final_status == "completed":
        job_manager.update_job(job_id, status="completed", progress=1.0)
        send_websocket_update(
            websocket_manager.send_completion(
                job_id=job_id,
                output_dir=f"hf://{config.output_name}" if config.push_to_hub else "",
            ),
            event_loop,
        )
    elif final_status in {"canceled", "cancelled"}:
        job_manager.update_job(job_id, status="stopped")
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="stopped",
                progress=current.progress if current else 0.0,
            ),
            event_loop,
        )
    else:
        err = f"HF Job ended with status '{final_status}'"
        job_manager.update_job(job_id, status="failed", error=err)
        send_websocket_update(
            websocket_manager.send_error(job_id=job_id, error=err),
            event_loop,
        )
