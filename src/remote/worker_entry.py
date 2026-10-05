"""
Remote worker entry point — runs ON the provisioned instance.

Launched as the instance's start command (``python -m
src.remote.worker_entry``). Decodes its job from the environment, exposes
a tiny token-authenticated HTTP status API for the control plane to poll,
executes exactly one pipeline stage, pushes the stage's output artifacts
to the artifact store, and then idles until the control plane terminates
the instance.

Stages:
  train — runs the standard Merlina training pipeline (run_training_sync
          or DDP via run_training_distributed) against a local SQLite
          job DB, then pushes ``models/<output_name>`` as the "adapter"
          artifact.
  merge — pulls the "adapter" artifact, merges it into the base model on
          CPU (``merge_lora_to_directory``), and pushes/publishes the
          result. Needs RAM and disk, not GPU — this is the stage you run
          on a different, cheaper machine.

The module top level imports no ML libraries, so the status server and
payload handling are unit-testable anywhere; heavy imports happen inside
the stage functions on the instance.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from src.job_manager import JobManager
from src.remote.artifacts import store_from_description
from src.remote.worker_client import (
    STATE_DONE,
    STATE_FAILED,
    STATE_PUSHING,
    STATE_RUNNING,
    STATE_STARTING,
    TOKEN_HEADER,
    WORKER_PORT,
)

logger = logging.getLogger(__name__)

JOB_ENV_VAR = "MERLINA_REMOTE_JOB_B64"
TOKEN_ENV_VAR = "MERLINA_WORKER_TOKEN"
DEFAULT_DB_PATH = "./data/worker_jobs.db"


class WorkerState:
    """Thread-safe worker state shared with the status server."""

    def __init__(self, job_id: str, stage: str):
        self.job_id = job_id
        self.stage = stage
        self._lock = threading.Lock()
        self._state = STATE_STARTING
        self._error: Optional[str] = None
        self._artifacts: List[str] = []

    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def fail(self, error: str) -> None:
        with self._lock:
            self._state = STATE_FAILED
            self._error = error

    def add_artifact(self, name: str) -> None:
        with self._lock:
            self._artifacts.append(name)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "error": self._error,
                "artifacts": list(self._artifacts),
            }


def build_status_payload(state: WorkerState, job_manager: Optional[JobManager],
                         since_step: int = 0) -> Dict[str, Any]:
    """Compose the /status response the orchestrator's client expects."""
    payload = state.snapshot()
    payload["job"] = {}
    payload["new_metrics"] = []
    if job_manager is not None:
        job = job_manager.get_job(state.job_id)
        if job is not None:
            payload["job"] = {
                "status": job.status,
                "progress": job.progress,
                "current_step": job.current_step,
                "total_steps": job.total_steps,
                "loss": job.loss,
                "eval_loss": job.eval_loss,
                "learning_rate": job.learning_rate,
                "error": job.error,
            }
        payload["new_metrics"] = [
            m for m in job_manager.get_metrics(state.job_id)
            if (m.get("step") or 0) > since_step
        ]
    return payload


def make_handler(state: WorkerState, job_manager: Optional[JobManager], token: str):
    """Build the request handler class bound to this worker's state."""

    class StatusHandler(BaseHTTPRequestHandler):
        # Provider HTTP proxies make these URLs reachable by anyone who can
        # guess an instance id, so every endpoint requires the run token.

        def _authorized(self) -> bool:
            return bool(token) and self.headers.get(TOKEN_HEADER) == token

        def _respond(self, code: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._authorized():
                self._respond(401, {"error": "unauthorized"})
                return
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._respond(200, {"ok": True, "job_id": state.job_id,
                                    "stage": state.stage,
                                    "state": state.snapshot()["state"]})
            elif parsed.path == "/status":
                qs = parse_qs(parsed.query)
                try:
                    since_step = int(qs.get("since_step", ["0"])[0])
                except ValueError:
                    since_step = 0
                self._respond(200, build_status_payload(state, job_manager, since_step))
            else:
                self._respond(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                self._respond(401, {"error": "unauthorized"})
                return
            if urlparse(self.path).path == "/stop":
                if job_manager is not None:
                    job_manager.request_stop(state.job_id)
                logger.info("Stop requested by control plane")
                self._respond(200, {"ok": True})
            else:
                self._respond(404, {"error": "not found"})

        def log_message(self, fmt, *args):  # quiet the default stderr spam
            logger.debug("status server: " + fmt, *args)

    return StatusHandler


def start_status_server(state: WorkerState, job_manager: Optional[JobManager],
                        token: str, port: int = WORKER_PORT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(state, job_manager, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="merlina-worker-status")
    thread.start()
    logger.info("Worker status server listening on :%d", port)
    return server


def decode_job_payload(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = env if env is not None else os.environ
    raw = env.get(JOB_ENV_VAR)
    if not raw:
        raise RuntimeError(f"{JOB_ENV_VAR} is not set — this process must be "
                           "launched by the Merlina remote orchestrator.")
    return json.loads(base64.b64decode(raw))


# ---------------------------------------------------------------------------
# Stage implementations (heavy imports stay inside)
# ---------------------------------------------------------------------------

def _load_training_config(config_dict: Dict[str, Any]):
    """Validate the config dict against the canonical Pydantic model.

    Same pattern as src/train_worker.py: importing merlina triggers app
    module init, which is harmless in a worker process and keeps
    TrainingConfig as the single source of truth.
    """
    from pydantic import TypeAdapter
    from merlina import TrainingConfig
    return TypeAdapter(TrainingConfig).validate_python(config_dict)


def _resolve_store(payload: Dict[str, Any]):
    return store_from_description(
        payload["artifact_store"],
        token=payload["config"].get("hf_token") or os.environ.get("HF_TOKEN"),
    )


def run_train_stage(payload: Dict[str, Any], job_manager: JobManager,
                    state: WorkerState) -> None:
    job_id = payload["job_id"]
    config = _load_training_config(payload["config"])
    job_manager.create_job(job_id, payload["config"])
    state.set_state(STATE_RUNNING)

    from src.training_runner import (
        _get_distributed_gpu_count,
        run_training_distributed,
        run_training_sync,
    )

    num_gpus = _get_distributed_gpu_count(config)
    if config.multi_gpu_strategy == "ddp" and num_gpus > 1:
        run_training_distributed(job_id, config, job_manager, {}, None)
    else:
        run_training_sync(job_id, config, job_manager, {}, None)

    job = job_manager.get_job(job_id)
    if job is None or job.status == "failed":
        raise RuntimeError(job.error if job else "training produced no job record")

    output_dir = Path("./models") / config.output_name
    if not output_dir.is_dir():
        raise RuntimeError(f"Training finished but no output at {output_dir}")

    state.set_state(STATE_PUSHING)
    store = _resolve_store(payload)
    for artifact in payload.get("artifacts_out") or ["adapter"]:
        store.push_dir(output_dir, artifact)
        state.add_artifact(artifact)


def run_merge_stage(payload: Dict[str, Any], job_manager: JobManager,
                    state: WorkerState) -> None:
    job_id = payload["job_id"]
    config_dict = payload["config"]
    job_manager.create_job(job_id, config_dict)
    state.set_state(STATE_RUNNING)

    store = _resolve_store(payload)
    work_dir = Path("./merge_work")
    adapter_dir = work_dir / "adapter"
    merged_dir = work_dir / "merged"

    artifacts_in = payload.get("artifacts_in") or ["adapter"]
    store.pull_dir(artifacts_in[0], adapter_dir)

    from src.gguf_exporter import merge_lora_to_directory  # heavy: lazy
    merge_lora_to_directory(
        config_dict["base_model"],
        adapter_dir,
        merged_dir,
        model_type=config_dict.get("model_type", "auto"),
    )

    state.set_state(STATE_PUSHING)
    if config_dict.get("push_to_hub"):
        # Publish straight to the user's target repo — re-shipping a merged
        # model through the artifact store would double the upload.
        from huggingface_hub import HfApi
        token = config_dict.get("hf_token") or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        repo_id = config_dict["output_name"]
        api.create_repo(repo_id, private=config_dict.get("hf_hub_private", True),
                        exist_ok=True)
        api.upload_folder(folder_path=str(merged_dir), repo_id=repo_id,
                          commit_message="Upload from Merlina remote run")
        state.add_artifact(f"hub:{repo_id}")
    else:
        for artifact in payload.get("artifacts_out") or ["merged"]:
            store.push_dir(merged_dir, artifact)
            state.add_artifact(artifact)

    job_manager.update_job(job_id, status="completed", progress=1.0,
                           output_dir=str(merged_dir))


STAGE_RUNNERS = {
    "train": run_train_stage,
    "merge": run_merge_stage,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    payload = decode_job_payload()
    token = os.environ.get(TOKEN_ENV_VAR, "")
    if not token:
        raise RuntimeError(f"{TOKEN_ENV_VAR} is not set — refusing to serve "
                           "an unauthenticated status endpoint.")

    job_id = payload["job_id"]
    stage = payload["stage"]
    runner = STAGE_RUNNERS.get(stage)
    if runner is None:
        raise RuntimeError(f"Unknown stage {stage!r}")

    db_path = os.environ.get("MERLINA_WORKER_DB", DEFAULT_DB_PATH)
    job_manager = JobManager(db_path=db_path)
    state = WorkerState(job_id, stage)
    start_status_server(state, job_manager, token)

    logger.info("Remote worker starting stage %r for job %s", stage, job_id)
    try:
        runner(payload, job_manager, state)
        state.set_state(STATE_DONE)
        logger.info("Stage %r complete", stage)
    except Exception as e:
        logger.error("Stage %r failed: %s", stage, e, exc_info=True)
        state.fail(str(e))

    # Stay alive so the control plane can read the terminal state and any
    # error; it terminates the instance when it's done with us.
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
