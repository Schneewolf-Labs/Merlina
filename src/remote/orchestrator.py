"""
Remote run orchestrator — the control-plane side of remote mode.

``run_training_remote`` is the remote sibling of ``run_training_sync``:
same signature shape, same JobManager/WebSocket lifecycle, but instead of
training in-process it builds a stage plan, provisions instances through
a ComputeProvider, polls the on-instance worker for progress, hands
artifacts between stages via an ArtifactStore, and always tears the
instances down.

Stages are deliberately independent: the train stage's only output is an
artifact in the store, so the merge stage can run locally, on another
instance, or be skipped — and a crash between stages loses nothing that
was already pushed.

No torch/transformers imports here; the heavy lifting happens on the
workers (or in lazily-imported local merge code).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from src.job_manager import JobManager
from src.websocket_manager import websocket_manager

from .artifacts import ArtifactStore, HFHubArtifactStore
from .plan import RemotePlanError, build_remote_plan
from .providers import ComputeProvider, ProviderError, get_provider
from .spec import InstanceSpec, RemotePlan, StagePlan
from .worker_client import (
    STATE_FAILED,
    WORKER_PORT,
    HttpWorkerClient,
    WorkerClient,
)

logger = logging.getLogger(__name__)

JOB_ENV_VAR = "MERLINA_REMOTE_JOB_B64"
TOKEN_ENV_VAR = "MERLINA_WORKER_TOKEN"

DEFAULT_WORKER_IMAGE = "ghcr.io/schneewolf-labs/merlina:latest"
DEFAULT_POLL_INTERVAL_S = 10.0
DEFAULT_BOOT_TIMEOUT_MIN = 30.0
DEFAULT_STOP_GRACE_MIN = 15.0

# Merge instances need RAM/disk, not GPU horsepower. Until CPU-instance
# sizing is wired up we rent the cheapest viable GPU box with a big disk.
MERGE_STAGE_FALLBACK_GPU = "NVIDIA RTX A6000"


class RemoteStageError(RuntimeError):
    """A remote stage failed on the worker."""


class RemoteStopped(Exception):
    """The user requested a stop; not an error."""


class RemoteBudgetExceeded(RuntimeError):
    """Runtime cap hit — instances were terminated to stop the spend."""


def run_training_remote(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    *,
    provider: Optional[ComputeProvider] = None,
    store: Optional[ArtifactStore] = None,
    worker_client_factory: Optional[Callable[[str, str], WorkerClient]] = None,
    poll_interval_s: Optional[float] = None,
    boot_timeout_min: Optional[float] = None,
) -> None:
    """
    Execute a remote training run end to end. Mirrors the local runners'
    contract: updates the JobManager/WebSocket throughout and never raises
    (failures land in the job record).

    The keyword-only arguments exist for dependency injection in tests;
    production callers rely on settings-based defaults.
    """
    from config import settings  # late import: avoids cycles at module load

    poll_interval_s = poll_interval_s or getattr(settings, "remote_poll_interval_seconds", DEFAULT_POLL_INTERVAL_S)
    boot_timeout_min = boot_timeout_min or getattr(settings, "remote_boot_timeout_minutes", DEFAULT_BOOT_TIMEOUT_MIN)
    remote_cfg = config.remote

    try:
        _update(job_manager, event_loop, job_id, status="provisioning", progress=0.0,
                message="Planning remote run")

        if uploaded_datasets and _uses_uploaded_dataset(config):
            raise RemotePlanError(
                "Uploaded datasets can't travel to remote workers yet — push the "
                "dataset to the HuggingFace Hub and use a 'huggingface' source."
            )

        if provider is None:
            provider = get_provider(
                remote_cfg.provider,
                api_key=getattr(settings, "runpod_api_key", None),
            )

        offers = None
        try:
            offers = provider.list_gpu_offers()
        except ProviderError as e:
            logger.warning("Could not fetch live GPU offers (%s); using static catalog", e)

        plan = build_remote_plan(config, offers=offers, job_id=job_id)
        if store is None:
            store = _default_store(config, job_id)
        job_manager.update_job(job_id, metrics={
            "remote_plan": plan.to_dict(),
            "artifact_store": store.describe(),
        })
        for w in plan.warnings:
            logger.warning("Remote plan warning for %s: %s", job_id, w)

        client_factory = worker_client_factory or (
            lambda url, token: HttpWorkerClient(url, token)
        )

        # ---- Stage 1: train on a provisioned GPU instance ----
        train_stage = plan.stage("train")
        _run_remote_stage(
            stage=train_stage,
            spec=_train_instance_spec(job_id, config, plan, store, settings),
            job_id=job_id,
            config=config,
            job_manager=job_manager,
            event_loop=event_loop,
            provider=provider,
            client_factory=client_factory,
            poll_interval_s=poll_interval_s,
            boot_timeout_min=boot_timeout_min,
            mirror_job_fields=True,
        )

        # ---- Pull the adapter home ----
        _update(job_manager, event_loop, job_id, status="downloading", progress=0.92,
                message="Pulling trained adapter from artifact store")
        output_dir = Path("./models") / config.output_name
        store.pull_dir("adapter", output_dir)
        job_manager.update_job(job_id, output_dir=str(output_dir))

        # ---- Stage 2: merge (local / remote / skipped per plan) ----
        merge_stage = plan.stage("merge")
        if merge_stage is not None:
            _update(job_manager, event_loop, job_id, status="merging", progress=0.94,
                    message=f"Merging adapter ({merge_stage.target})")
            if merge_stage.target == "local":
                _run_local_merge(config, output_dir)
            elif merge_stage.target == "remote":
                _run_remote_stage(
                    stage=merge_stage,
                    spec=_merge_instance_spec(job_id, config, plan, store, settings),
                    job_id=job_id,
                    config=config,
                    job_manager=job_manager,
                    event_loop=event_loop,
                    provider=provider,
                    client_factory=client_factory,
                    poll_interval_s=poll_interval_s,
                    boot_timeout_min=boot_timeout_min,
                    mirror_job_fields=False,
                )
        elif config.push_to_hub:
            # Adapter-only publish: the artifact we pulled is small.
            _update(job_manager, event_loop, job_id, status="uploading", progress=0.97,
                    message="Publishing adapter to the HuggingFace Hub")
            _publish_dir_to_hub(
                output_dir,
                repo_id=config.output_name,
                token=config.hf_token,
                private=getattr(config, "hf_hub_private", True),
            )

        _update(job_manager, event_loop, job_id, status="completed", progress=1.0)
        _send_ws(event_loop, websocket_manager.send_completion(
            job_id=job_id, output_dir=str(output_dir)))
        logger.info("Remote training job %s completed", job_id)

    except RemoteStopped:
        logger.info("Remote training job %s stopped by user", job_id)
        job_manager.update_job(job_id, status="stopped")
        job = job_manager.get_job(job_id)
        _send_ws(event_loop, websocket_manager.send_status_update(
            job_id=job_id, status="stopped",
            progress=job.progress if job else 0.0))
    except Exception as e:
        logger.error("Remote training job %s failed: %s", job_id, e, exc_info=True)
        job_manager.update_job(job_id, status="failed", error=str(e))
        _send_ws(event_loop, websocket_manager.send_error(job_id=job_id, error=str(e)))


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------

def _run_remote_stage(
    *,
    stage: StagePlan,
    spec: InstanceSpec,
    job_id: str,
    config: Any,
    job_manager: JobManager,
    event_loop,
    provider: ComputeProvider,
    client_factory: Callable[[str, str], WorkerClient],
    poll_interval_s: float,
    boot_timeout_min: float,
    mirror_job_fields: bool,
) -> None:
    """
    Provision an instance, drive one worker stage to completion on it, and
    terminate the instance no matter what. Raises on failure/stop/budget.
    """
    remote_cfg = config.remote
    token = spec.env.get(TOKEN_ENV_VAR, "")
    deadline = time.monotonic() + remote_cfg.max_runtime_hours * 3600

    instance = provider.provision(spec)
    logger.info("Stage %r for job %s on %s instance %s",
                stage.name, job_id, provider.name, instance.instance_id)
    failed = False
    try:
        url = provider.proxy_url(instance, WORKER_PORT)
        if not url:
            raise RemoteStageError(
                f"Provider {provider.name} returned no proxy URL for instance "
                f"{instance.instance_id} — cannot reach the worker."
            )
        client = client_factory(url, token)

        _wait_for_worker(
            client, instance, provider, job_id, job_manager,
            boot_timeout_s=boot_timeout_min * 60,
            poll_interval_s=poll_interval_s,
            deadline=deadline,
        )

        _poll_stage_to_completion(
            client=client,
            stage=stage,
            job_id=job_id,
            job_manager=job_manager,
            event_loop=event_loop,
            poll_interval_s=poll_interval_s,
            deadline=deadline,
            mirror_job_fields=mirror_job_fields,
        )
    except BaseException:
        failed = True
        raise
    finally:
        if failed and remote_cfg.keep_instance_on_failure:
            logger.warning(
                "Keeping instance %s alive for debugging (keep_instance_on_failure) — "
                "REMEMBER TO TERMINATE IT, it is still billing.",
                instance.instance_id,
            )
        else:
            try:
                provider.terminate(instance.instance_id)
            except Exception as e:
                logger.error(
                    "FAILED TO TERMINATE instance %s — terminate it manually in the "
                    "%s console to stop billing: %s",
                    instance.instance_id, provider.name, e,
                )


def _wait_for_worker(client, instance, provider, job_id, job_manager,
                     *, boot_timeout_s, poll_interval_s, deadline) -> None:
    """Wait for the worker HTTP server to come up on a fresh instance."""
    boot_deadline = time.monotonic() + boot_timeout_s
    while not client.health():
        now = time.monotonic()
        if now > boot_deadline:
            raise RemoteStageError(
                f"Worker on instance {instance.instance_id} did not become healthy "
                f"within {boot_timeout_s / 60:.0f} minutes."
            )
        if now > deadline:
            raise RemoteBudgetExceeded("max_runtime_hours reached while waiting for boot.")
        job = job_manager.get_job(job_id)
        if job and job.stop_requested:
            raise RemoteStopped()
        current = provider.get_instance(instance.instance_id)
        if current.status in ("exited", "terminated"):
            raise RemoteStageError(
                f"Instance {instance.instance_id} {current.status} before the worker "
                "came up (image pull failure or capacity reclaim?)."
            )
        time.sleep(poll_interval_s)


def _poll_stage_to_completion(*, client, stage, job_id, job_manager, event_loop,
                              poll_interval_s, deadline, mirror_job_fields) -> None:
    """Poll worker status until a terminal state, mirroring progress home."""
    from config import settings
    stop_grace_s = getattr(settings, "remote_stop_grace_minutes", DEFAULT_STOP_GRACE_MIN) * 60

    since_step = 0
    stop_sent_at: Optional[float] = None

    while True:
        status = client.status(since_step=since_step)

        if mirror_job_fields:
            since_step = _mirror_status(status, job_id, job_manager, event_loop, since_step)

        if status.state == STATE_FAILED:
            raise RemoteStageError(
                f"Stage {stage.name!r} failed on the worker: {status.error or 'unknown error'}"
            )
        if status.is_terminal:
            return

        now = time.monotonic()
        if now > deadline:
            raise RemoteBudgetExceeded(
                "max_runtime_hours reached — terminating the instance. Raise "
                "remote.max_runtime_hours if the run legitimately needs longer."
            )

        job = job_manager.get_job(job_id)
        if job and job.stop_requested:
            if stop_sent_at is None:
                logger.info("Forwarding stop request to remote worker for job %s", job_id)
                client.request_stop()
                stop_sent_at = now
            elif now - stop_sent_at > stop_grace_s:
                raise RemoteStopped()

        time.sleep(poll_interval_s)


def _mirror_status(status, job_id, job_manager, event_loop, since_step) -> int:
    """Mirror worker job fields/metrics into the local job record + WS."""
    job_fields = status.job or {}
    update = {k: job_fields[k] for k in
              ("status", "progress", "current_step", "total_steps",
               "loss", "eval_loss", "learning_rate")
              if job_fields.get(k) is not None}
    # Terminal worker-side statuses must not clobber the control-plane
    # lifecycle (we still have stages to run after the worker finishes).
    if update.get("status") in ("completed", "failed", "stopped"):
        update.pop("status")
    if update:
        job_manager.update_job(job_id, **update)
        _send_ws(event_loop, websocket_manager.send_status_update(
            job_id=job_id,
            status=update.get("status", job_fields.get("status", "training")),
            progress=job_fields.get("progress"),
            current_step=job_fields.get("current_step"),
            total_steps=job_fields.get("total_steps"),
            loss=job_fields.get("loss"),
            eval_loss=job_fields.get("eval_loss"),
            learning_rate=job_fields.get("learning_rate"),
        ))
    for metric in status.new_metrics:
        step = metric.get("step")
        if step is None:
            continue
        job_manager.add_metric(
            job_id,
            step=step,
            loss=metric.get("loss"),
            eval_loss=metric.get("eval_loss"),
            learning_rate=metric.get("learning_rate"),
            gpu_memory_used=metric.get("gpu_memory_used"),
        )
        since_step = max(since_step, step)
    return since_step


# ---------------------------------------------------------------------------
# Instance specs & worker payloads
# ---------------------------------------------------------------------------

def _stage_payload(job_id: str, stage: StagePlan, config: Any, store: ArtifactStore,
                   multi_gpu_strategy: Optional[str] = None) -> dict:
    """The job description a worker decodes from its environment."""
    config_dict = config.model_dump()
    # Workers must never re-dispatch to remote, publish from the train
    # stage, or export GGUFs on rented metal (publishing is a control-plane
    # or merge-stage concern).
    config_dict["remote"] = None
    if stage.name == "train":
        config_dict["push_to_hub"] = False
        config_dict["export_gguf"] = False
        if multi_gpu_strategy:
            config_dict["multi_gpu_strategy"] = multi_gpu_strategy
    return {
        "job_id": job_id,
        "stage": stage.name,
        "config": config_dict,
        "artifact_store": store.describe(),
        "artifacts_in": stage.artifacts_in,
        "artifacts_out": stage.artifacts_out,
    }


def _worker_env(payload: dict, token: str, config: Any) -> dict:
    env = {
        JOB_ENV_VAR: base64.b64encode(json.dumps(payload).encode()).decode(),
        TOKEN_ENV_VAR: token,
    }
    if config.hf_token:
        env["HF_TOKEN"] = config.hf_token
    wandb_key = getattr(config, "wandb_key", None)
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
    return env


def _train_instance_spec(job_id: str, config: Any, plan: RemotePlan,
                         store: ArtifactStore, settings) -> InstanceSpec:
    stage = plan.stage("train")
    sizing = stage.sizing
    token = secrets.token_urlsafe(32)
    payload = _stage_payload(job_id, stage, config, store,
                             multi_gpu_strategy=sizing.multi_gpu_strategy)
    return InstanceSpec(
        name=f"merlina-{job_id}-train",
        image=config.remote.worker_image
              or getattr(settings, "remote_worker_image", DEFAULT_WORKER_IMAGE),
        gpu_type_id=sizing.gpu_type_id,
        gpu_count=sizing.gpu_count,
        container_disk_gb=config.remote.container_disk_gb
                          or max(int(sizing.disk_required_gb), 50),
        volume_gb=config.remote.volume_gb or 0,
        cloud_type=sizing.cloud_type,
        env=_worker_env(payload, token, config),
        http_ports=[WORKER_PORT],
        docker_start_cmd=["python", "-m", "src.remote.worker_entry"],
    )


def _merge_instance_spec(job_id: str, config: Any, plan: RemotePlan,
                         store: ArtifactStore, settings) -> InstanceSpec:
    from .sizing import estimate_disk_gb
    stage = plan.stage("merge")
    token = secrets.token_urlsafe(32)
    payload = _stage_payload(job_id, stage, config, store)
    disk_gb = int(estimate_disk_gb(plan.model_specs, merge_stage=True)) \
        if plan.model_specs else 200
    return InstanceSpec(
        name=f"merlina-{job_id}-merge",
        image=config.remote.worker_image
              or getattr(settings, "remote_worker_image", DEFAULT_WORKER_IMAGE),
        gpu_type_id=MERGE_STAGE_FALLBACK_GPU,
        gpu_count=1,
        container_disk_gb=max(disk_gb, 100),
        cloud_type=config.remote.cloud_type,
        env=_worker_env(payload, token, config),
        http_ports=[WORKER_PORT],
        docker_start_cmd=["python", "-m", "src.remote.worker_entry"],
    )


# ---------------------------------------------------------------------------
# Local merge / publish / store helpers
# ---------------------------------------------------------------------------

def _run_local_merge(config: Any, adapter_dir: Path) -> Path:
    """Merge the pulled adapter into the base on the control plane."""
    from src.gguf_exporter import merge_lora_to_directory  # heavy: lazy
    merged_dir = adapter_dir.parent / f"{config.output_name}-merged"
    merge_lora_to_directory(
        config.base_model,
        adapter_dir,
        merged_dir,
        model_type=getattr(config, "model_type", "auto"),
    )
    if config.push_to_hub:
        _publish_dir_to_hub(
            merged_dir,
            repo_id=config.output_name,
            token=config.hf_token,
            private=getattr(config, "hf_hub_private", True),
        )
    return merged_dir


def _publish_dir_to_hub(local_dir: Path, *, repo_id: str,
                        token: Optional[str], private: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                      commit_message="Upload from Merlina remote run")
    logger.info("Published %s to https://huggingface.co/%s", local_dir, repo_id)


def _default_store(config: Any, job_id: str) -> ArtifactStore:
    """Private HF repo per run: <user>/merlina-run-<job_id>."""
    repo_id = config.remote.artifact_repo
    if not repo_id:
        if not config.hf_token:
            raise RemotePlanError(
                "Remote runs need an artifact store. Provide an HF token (the "
                "default store is a private HF repo) or set remote.artifact_repo."
            )
        from huggingface_hub import HfApi
        username = HfApi(token=config.hf_token).whoami()["name"]
        repo_id = f"{username}/merlina-run-{job_id}"
    return HFHubArtifactStore(repo_id, token=config.hf_token, private=True)


def _uses_uploaded_dataset(config: Any) -> bool:
    dataset = getattr(config, "dataset", None)
    if dataset is None:
        return False
    sources = [dataset.source] + list(getattr(dataset, "additional_sources", []) or [])
    return any(getattr(s, "source_type", "") == "upload" for s in sources)


# ---------------------------------------------------------------------------
# WebSocket / job plumbing
# ---------------------------------------------------------------------------

def _send_ws(event_loop, coro) -> None:
    """Schedule a WebSocket coroutine from this sync thread (best effort)."""
    try:
        if event_loop is None:
            coro.close()
            return
        asyncio.run_coroutine_threadsafe(coro, event_loop)
    except Exception as e:
        logger.debug("Could not send WebSocket update: %s", e)


def _update(job_manager: JobManager, event_loop, job_id: str, *,
            status: str, progress: Optional[float] = None,
            message: Optional[str] = None) -> None:
    fields = {"status": status}
    if progress is not None:
        fields["progress"] = progress
    job_manager.update_job(job_id, **fields)
    _send_ws(event_loop, websocket_manager.send_status_update(
        job_id=job_id, status=status, progress=progress, message=message))
