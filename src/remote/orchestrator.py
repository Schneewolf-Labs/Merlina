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
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

from src.job_manager import JobManager
from src.websocket_manager import websocket_manager

from .artifacts import ArtifactStore, artifact_worker_env, store_for_job
from .lifecycle import RemoteLifecycle, RemoteAlreadyClaimed, RemoteLeaseLost, terminate_confirmed
from .plan import RemotePlanError, build_remote_plan
from .providers import ComputeProvider, ProviderError, ProvisionRejected, get_provider
from .spec import InstanceSpec, RemotePlan, StagePlan
from .worker_client import (
    SERVE_PORT,
    STATE_FAILED,
    STATE_SERVING,
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
DEFAULT_STATUS_ERROR_TOLERANCE_MIN = 10.0

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
    lifecycle: Optional[RemoteLifecycle] = None,
    execution_guard: Any = None,
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
    lifecycle = lifecycle or RemoteLifecycle(job_manager.db_path)
    try:
        owner = lifecycle.claim(job_id, remote_cfg.max_runtime_hours)
    except RemoteAlreadyClaimed:
        logger.warning("Ignoring duplicate remote dispatch for %s", job_id)
        return
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop, args=(lifecycle, job_id, owner, heartbeat_stop), daemon=True)
    heartbeat_thread.start()
    final_state, final_error = "failed", None

    try:
        _update(job_manager, event_loop, job_id, status="provisioning", progress=0.0,
                message="Planning remote run")

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
        from .inputs import stage_remote_inputs
        prepared_config = stage_remote_inputs(config.model_dump(), uploaded_datasets, store)
        job_manager.update_job(job_id, metrics={
            "remote_plan": plan.to_dict(),
            "artifact_store": store.describe(),
            "remote_inputs": prepared_config.get("dataset"),
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
            spec=_train_instance_spec(job_id, config, plan, store, settings,
                                      prepared_config=prepared_config),
            job_id=job_id,
            config=config,
            job_manager=job_manager,
            event_loop=event_loop,
            provider=provider,
            client_factory=client_factory,
            poll_interval_s=poll_interval_s,
            boot_timeout_min=boot_timeout_min,
            mirror_job_fields=True,
            lifecycle=lifecycle, owner=owner, execution_guard=execution_guard,
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
                # The merged model is already on local disk; re-uploading it
                # to the run store would only add cost and failure modes.
                _run_local_merge(config, output_dir)
            elif merge_stage.target == "remote":
                _run_remote_stage(
                    stage=merge_stage,
                    spec=_merge_instance_spec(job_id, config, plan, store, settings,
                                              prepared_config=prepared_config),
                    job_id=job_id,
                    config=config,
                    job_manager=job_manager,
                    event_loop=event_loop,
                    provider=provider,
                    client_factory=client_factory,
                    poll_interval_s=poll_interval_s,
                    boot_timeout_min=boot_timeout_min,
                    mirror_job_fields=False,
                    lifecycle=lifecycle, owner=owner, execution_guard=execution_guard,
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

        _check_run(lifecycle, job_id, owner, job_manager)
        final_state = "completed"
        _update(job_manager, event_loop, job_id, status="completed", progress=1.0)
        _send_ws(event_loop, websocket_manager.send_completion(
            job_id=job_id, output_dir=str(output_dir)))
        logger.info("Remote training job %s completed", job_id)

    except RemoteStopped:
        final_state = "stopped"
        logger.info("Remote training job %s stopped by user", job_id)
        job_manager.update_job(job_id, status="stopped")
        job = job_manager.get_job(job_id)
        _send_ws(event_loop, websocket_manager.send_status_update(
            job_id=job_id, status="stopped",
            progress=job.progress if job else 0.0))
    except Exception as e:
        final_error = str(e)
        logger.error("Remote training job %s failed: %s", job_id, e, exc_info=True)
        job_manager.update_job(job_id, status="failed", error=str(e))
        _send_ws(event_loop, websocket_manager.send_error(job_id=job_id, error=str(e)))
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)
        lifecycle.finish(job_id, owner, final_state, final_error)
        resources = lifecycle.resources(job_id)
        job = job_manager.get_job(job_id)
        job_manager.update_job(job_id, metrics={**(job.metrics or {}), "remote_resources": resources} if job else {})
        if execution_guard is not None:
            # An ambiguous create can still appear in provider inventory later.
            attempts = [a for a in lifecycle.attempts() if a["job_id"] == job_id]
            if all(a["state"] not in ("provisioning", "provisioning_unknown") for a in attempts):
                try:
                    execution_guard.settle(job_id, resources)
                except Exception:
                    logger.exception("Settlement pending for remote job %s", job_id)


def _heartbeat_loop(lifecycle, job_id, owner, stop, *, interval=None, max_backoff=None):
    """Renew the lease until stopped or definitively lost.

    Transient failures (e.g. sqlite "database is locked" during a long pull or
    Hub push) are retried with backoff; giving up would let the lease lapse and
    the reconciler would terminate a healthy job.
    """
    interval = interval if interval is not None else min(20.0, lifecycle.lease_seconds / 3)
    max_backoff = max_backoff if max_backoff is not None else interval
    delay, backoff = interval, min(1.0, max_backoff)
    while not stop.wait(delay):
        try:
            lifecycle.heartbeat(job_id, owner)
        except RemoteLeaseLost:
            logger.error("Remote execution lease lost for %s; heartbeat stopped", job_id)
            return
        except Exception as exc:
            logger.warning("Remote execution heartbeat failed for %s (%s); retrying in %.1fs",
                           job_id, exc, backoff)
            delay, backoff = backoff, min(backoff * 2, max_backoff)
        else:
            delay, backoff = interval, min(1.0, max_backoff)


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
    lifecycle: RemoteLifecycle,
    owner: str,
    execution_guard: Any = None,
    keep_on_failure: Any = None,
    drive: Optional[Callable[..., List[str]]] = None,
) -> None:
    """
    Provision an instance, drive one worker stage on it, and terminate the
    instance no matter what. Raises on failure/stop/budget.

    By default the stage is driven to a terminal worker state
    (:func:`_poll_stage_to_completion`). A ``drive`` callable replaces that
    for stages whose useful phase is not terminal — the serve stage holds
    in ``serving`` until stopped. It is called as ``drive(client=...,
    instance=..., tick=..., deadline=...)`` once the worker is healthy and
    returns the stage's artifact names; returning normally means success,
    and the instance is terminated either way.

    ``keep_on_failure`` overrides ``config.remote.keep_instance_on_failure``;
    it may be a predicate on the exception (retain only for real failures).
    """
    if keep_on_failure is None:
        keep_on_failure = config.remote.keep_instance_on_failure
    token = spec.env.get(TOKEN_ENV_VAR, "")
    _check_run(lifecycle, job_id, owner, job_manager)
    if execution_guard is not None:
        execution_guard.before_provision(job_id, stage.name, spec)
    label = lifecycle.intent(job_id, owner, stage.name, provider.name)
    spec.name = label
    deadline = time.monotonic() + max(0, lifecycle.run(job_id)["deadline"] - time.time())
    instance = None
    failed = False
    error: Optional[BaseException] = None

    def tick():
        _check_run(lifecycle, job_id, owner, job_manager, check_stop=False)
        if execution_guard is not None:
            execution_guard.usage(job_id, lifecycle.resources(job_id))

    try:
        instance = provider.provision(spec)
        lifecycle.record_instance(job_id, stage.name, label, instance)
        lifecycle.stage_state(job_id, stage.name, "running")
        tick()
        url = provider.proxy_url(instance, WORKER_PORT)
        if not url:
            raise RemoteStageError(f"Provider {provider.name} returned no worker proxy URL")
        client = client_factory(url, token)
        _wait_for_worker(
            client, instance, provider, job_id, job_manager,
            boot_timeout_s=boot_timeout_min * 60, poll_interval_s=poll_interval_s,
            deadline=deadline, tick=tick,
        )
        if drive is not None:
            artifacts = drive(client=client, instance=instance, tick=tick, deadline=deadline)
        else:
            artifacts = _poll_stage_to_completion(
                client=client, stage=stage, job_id=job_id, job_manager=job_manager,
                event_loop=event_loop, poll_interval_s=poll_interval_s,
                deadline=deadline, mirror_job_fields=mirror_job_fields, tick=tick, lifecycle=lifecycle,
            )
        lifecycle.stage_state(job_id, stage.name, "completed", artifacts=artifacts)
    except BaseException as exc:
        failed, error = True, exc
        # Only a definitive provider refusal proves nothing was created;
        # anything else before an instance id leaves the create ambiguous.
        definite = instance is not None or isinstance(exc, ProvisionRejected)
        lifecycle.stage_state(job_id, stage.name,
                              "failed" if definite else "provisioning_unknown",
                              error=type(exc).__name__)
        raise
    finally:
        if instance is not None:
            keep = keep_on_failure(error) if callable(keep_on_failure) else keep_on_failure
            if failed and keep and execution_guard is None:
                lifecycle.cleanup_result(provider.name, instance.instance_id, retained=True,
                                         error="Retained for debugging until total job deadline")
            else:
                terminate_confirmed(lifecycle, provider, instance.instance_id)


def _check_run(lifecycle, job_id, owner, job_manager, *, check_stop=True):
    run = lifecycle.run(job_id)
    if time.time() >= run["deadline"]:
        raise RemoteBudgetExceeded("max_runtime_hours reached across the entire remote job")
    lifecycle.heartbeat(job_id, owner)
    job = job_manager.get_job(job_id)
    if job is None or (check_stop and job.stop_requested):
        raise RemoteStopped()


def _wait_for_worker(client, instance, provider, job_id, job_manager,
                     *, boot_timeout_s, poll_interval_s, deadline, tick=lambda: None) -> None:
    """Wait for the worker HTTP server to come up on a fresh instance."""
    boot_deadline = time.monotonic() + boot_timeout_s
    tick()
    while not client.health():
        tick()
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
                              poll_interval_s, deadline, mirror_job_fields, tick=lambda: None, lifecycle=None):
    """Poll worker status until a terminal state, mirroring progress home."""
    from config import settings
    stop_grace_s = getattr(settings, "remote_stop_grace_minutes", DEFAULT_STOP_GRACE_MIN) * 60
    error_tolerance_s = getattr(settings, "remote_status_error_tolerance_minutes",
                                DEFAULT_STATUS_ERROR_TOLERANCE_MIN) * 60

    since_step = 0
    stop_sent_at: Optional[float] = None
    unreachable_since: Optional[float] = None

    while True:
        tick()
        try:
            status = client.status(since_step=since_step)
        except Exception as e:
            # Provider proxies return 404/5xx while routing settles or blips;
            # one failed poll must not abandon a healthy, billing worker.
            if _is_auth_error(e):
                raise RemoteStageError(f"Worker rejected the control plane's token: {e}") from e
            now = time.monotonic()
            if unreachable_since is None:
                unreachable_since = now
                logger.warning("Worker status for job %s unreachable (%s); retrying", job_id, e)
            elif now - unreachable_since > error_tolerance_s:
                raise RemoteStageError(
                    f"Worker status unreachable for {error_tolerance_s / 60:.0f} minutes: {e}"
                ) from e
            if now > deadline:
                raise RemoteBudgetExceeded("max_runtime_hours reached while the worker was unreachable.")
            job = job_manager.get_job(job_id)
            if job and job.stop_requested:
                raise RemoteStopped()
            time.sleep(poll_interval_s)
            continue
        if unreachable_since is not None:
            logger.info("Worker status for job %s reachable again", job_id)
            unreachable_since = None

        if mirror_job_fields:
            since_step = _mirror_status(status, job_id, job_manager, event_loop, since_step)

        if status.state == STATE_FAILED:
            raise RemoteStageError(
                f"Stage {stage.name!r} failed on the worker: {status.error or 'unknown error'}"
            )
        if status.is_terminal:
            job = job_manager.get_job(job_id)
            if (job and job.stop_requested) or status.job.get("status") == "stopped":
                raise RemoteStopped()
            return status.artifacts

        now = time.monotonic()
        if now > deadline:
            raise RemoteBudgetExceeded(
                "max_runtime_hours reached — terminating the instance. Raise "
                "remote.max_runtime_hours if the run legitimately needs longer."
            )

        job = job_manager.get_job(job_id)
        if job and job.stop_requested:
            stop_deadline = lifecycle.request_cancel(job_id, stop_grace_s) if lifecycle else None
            if stop_deadline is not None and time.time() >= stop_deadline:
                raise RemoteStopped()
            if stop_sent_at is None:
                logger.info("Forwarding stop request to remote worker for job %s", job_id)
                client.request_stop()
                stop_sent_at = now
            elif now - stop_sent_at > stop_grace_s:
                raise RemoteStopped()

        time.sleep(poll_interval_s)


def _is_auth_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) in (401, 403)


def _mirror_status(status, job_id, job_manager, event_loop, since_step) -> int:
    """Mirror worker job fields/metrics into the local job record + WS."""
    job_fields = status.job or {}
    checkpoint = job_fields.get("checkpoint_publication")
    if isinstance(checkpoint, dict):
        current = job_manager.get_job(job_id)
        metrics = dict(current.metrics or {}) if current else {}
        metrics["checkpoint_publication"] = checkpoint
        job_manager.update_job(job_id, metrics=metrics)
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
            extra=metric.get("extra") or None,
        )
        since_step = max(since_step, step)
    return since_step


# ---------------------------------------------------------------------------
# Instance specs & worker payloads
# ---------------------------------------------------------------------------

def _stage_payload(job_id: str, stage: StagePlan, config: Any, store: ArtifactStore,
                   multi_gpu_strategy: Optional[str] = None,
                   prepared_config: Optional[dict] = None) -> dict:
    """The job description a worker decodes from its environment."""
    import copy
    config_dict = copy.deepcopy(prepared_config) if prepared_config is not None else config.model_dump()
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
    env.update(artifact_worker_env(payload.get("artifact_store", {})))
    return env


def _train_instance_spec(job_id: str, config: Any, plan: RemotePlan,
                         store: ArtifactStore, settings,
                         prepared_config: Optional[dict] = None) -> InstanceSpec:
    stage = plan.stage("train")
    sizing = stage.sizing
    token = secrets.token_urlsafe(32)
    payload = _stage_payload(job_id, stage, config, store,
                             multi_gpu_strategy=sizing.multi_gpu_strategy,
                             prepared_config=prepared_config)
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
        env={**_worker_env(payload, token, config),
             **(store.worker_environment() if callable(getattr(store, "worker_environment", None)) else {})},
        http_ports=[WORKER_PORT],
        docker_start_cmd=["python", "-m", "src.remote.worker_entry"],
    )


def _merge_instance_spec(job_id: str, config: Any, plan: RemotePlan,
                         store: ArtifactStore, settings,
                         prepared_config: Optional[dict] = None) -> InstanceSpec:
    from .sizing import estimate_disk_gb
    stage = plan.stage("merge")
    token = secrets.token_urlsafe(32)
    payload = _stage_payload(job_id, stage, config, store, prepared_config=prepared_config)
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
        env={**_worker_env(payload, token, config),
             **(store.worker_environment() if callable(getattr(store, "worker_environment", None)) else {})},
        http_ports=[WORKER_PORT],
        docker_start_cmd=["python", "-m", "src.remote.worker_entry"],
    )


# ---------------------------------------------------------------------------
# Remote inference sessions (the "serve" stage)
# ---------------------------------------------------------------------------

# Session outcomes that are a cap doing its job, not an error.
_DEADLINE_REASONS = ("max_runtime_hours", "max_total_cost_usd")


def run_inference_remote(
    session_id: str,
    request: Any,
    sessions: Any,
    *,
    plan: Any = None,
    job_manager: Optional[JobManager] = None,
    provider: Optional[ComputeProvider] = None,
    worker_client_factory: Optional[Callable[[str, str], WorkerClient]] = None,
    poll_interval_s: Optional[float] = None,
    boot_timeout_min: Optional[float] = None,
    lifecycle: Optional[RemoteLifecycle] = None,
    execution_guard: Any = None,
) -> None:
    """
    Run one remote inference session end to end: rent an instance, serve the
    model with vLLM, mark the session ``ready`` with its OpenAI base URL, hold
    until a stop / deadline / total-cost cap / idle timeout / worker failure,
    and always terminate the instance. Never raises; the outcome lands in the
    session row (``sessions`` is an :class:`InferenceSessionStore`).

    Reuses the training stage machinery: durable claim + lease, intent label
    per stage ("serve") before provisioning, ``terminate_confirmed`` in a
    ``finally`` — so the reconciler cleans up an orphaned serve pod exactly
    like an orphaned training pod.
    """
    from config import settings
    from .inference import build_serve_plan
    from .sessions import forget_api_key, remember_api_key

    poll_interval_s = poll_interval_s or getattr(settings, "remote_poll_interval_seconds", DEFAULT_POLL_INTERVAL_S)
    boot_timeout_min = boot_timeout_min or getattr(settings, "remote_boot_timeout_minutes", DEFAULT_BOOT_TIMEOUT_MIN)
    lifecycle = lifecycle or RemoteLifecycle(sessions.db_path)
    owner: Optional[str] = None
    heartbeat_stop = threading.Event()
    heartbeat_thread: Optional[threading.Thread] = None
    final_status, stop_reason, error = "failed", "error", None

    try:
        if provider is None:
            provider = get_provider(request.provider, api_key=getattr(settings, "runpod_api_key", None))
        if plan is None:
            offers = None
            try:
                offers = provider.list_gpu_offers()
            except ProviderError as e:
                logger.warning("Could not fetch live GPU offers (%s); using static catalog", e)
            plan = build_serve_plan(request, offers=offers, job_manager=job_manager,
                                    lifecycle=lifecycle, settings=settings)
        try:
            owner = lifecycle.claim(session_id, plan.effective_max_hours)
        except RemoteAlreadyClaimed:
            logger.warning("Ignoring duplicate dispatch for inference session %s", session_id)
            return
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop, args=(lifecycle, session_id, owner, heartbeat_stop), daemon=True)
        heartbeat_thread.start()
        sessions.update(session_id, status="provisioning", plan=plan.to_dict(),
                        served_model_name=plan.served_model_name,
                        deadline_at=lifecycle.run(session_id)["deadline"])

        api_key = secrets.token_urlsafe(32)
        remember_api_key(session_id, api_key)
        spec = _serve_instance_spec(session_id, request, plan, api_key, job_manager=job_manager)
        client_factory = worker_client_factory or (lambda url, token: HttpWorkerClient(url, token))

        outcome: dict = {}

        def drive(*, client, instance, tick, deadline):
            outcome["reason"] = _drive_serve_session(
                session_id=session_id, sessions=sessions, client=client, instance=instance,
                provider=provider, plan=plan, lifecycle=lifecycle, owner=owner, tick=tick,
                poll_interval_s=poll_interval_s, settings=settings)
            return []

        _run_remote_stage(
            stage=StagePlan(name="serve", target="remote", sizing=plan.sizing),
            spec=spec, job_id=session_id, config=None, job_manager=sessions,
            event_loop=None, provider=provider, client_factory=client_factory,
            poll_interval_s=poll_interval_s, boot_timeout_min=boot_timeout_min,
            mirror_job_fields=False, lifecycle=lifecycle, owner=owner,
            execution_guard=execution_guard,
            # Debug retention is for failures only: stops and caps always end the bill.
            keep_on_failure=lambda exc: bool(request.keep_instance_on_failure) and not isinstance(
                exc, (RemoteStopped, RemoteBudgetExceeded)),
            drive=drive,
        )
        stop_reason = outcome.get("reason", "user")
        final_status = "expired" if stop_reason in _DEADLINE_REASONS else "stopped"
    except RemoteStopped:
        final_status, stop_reason = "stopped", "user"
    except (RemoteBudgetExceeded, RemoteLeaseLost) as e:
        run = lifecycle.run(session_id) if owner else None
        if run and time.time() >= run["deadline"]:
            final_status = "expired"
            current = sessions.get(session_id)
            stop_reason = ((current.metrics or {}).get("deadline_reason") if current else None) \
                or getattr(plan, "deadline_reason", None) or "max_runtime_hours"
        else:
            error = str(e)
    except Exception as e:
        error = str(e)
        logger.error("Inference session %s failed: %s", session_id, e)
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        forget_api_key(session_id)
        if owner is not None:
            lifecycle.finish(session_id, owner, "failed" if final_status == "failed" else "stopped", error)
            resources = lifecycle.resources(session_id)
            current = sessions.get(session_id)
            metrics = dict(current.metrics or {}) if current else {}
            metrics.update(remote_resources=resources, est_cost_usd=_estimate_spend(resources))
            sessions.update(session_id, status=final_status, stop_reason=stop_reason,
                            error=error, ended_at=time.time(), metrics=metrics)
            if execution_guard is not None:
                attempts = [a for a in lifecycle.attempts() if a["job_id"] == session_id]
                if all(a["state"] not in ("provisioning", "provisioning_unknown") for a in attempts):
                    try:
                        execution_guard.settle(session_id, resources)
                    except Exception:
                        logger.exception("Settlement pending for inference session %s", session_id)
        elif error is not None:
            # Failed before claiming (bad plan, no provider key): nothing was rented.
            sessions.update(session_id, status="failed", stop_reason="error", error=error,
                            ended_at=time.time())
        logger.info("Inference session %s ended: %s (%s)", session_id, final_status, stop_reason)


def _serve_instance_spec(session_id: str, request: Any, plan: Any, api_key: str,
                         *, job_manager: Optional[JobManager] = None) -> InstanceSpec:
    """Instance for a serve stage: worker port + the API-key-guarded serve port."""
    from .inference import resolve_hf_token, serve_spec
    from .serve_worker import API_KEY_ENV_VAR

    job_config = None
    source_job = getattr(request.model, "job_id", None)
    if source_job and job_manager is not None:
        job = job_manager.get_job(source_job)
        job_config = job.config if job else None
    payload = {
        "job_id": session_id,
        "stage": "serve",
        "serve": serve_spec(plan),
        # Tokens never ride in the payload; HF_TOKEN travels as its own variable.
        "config": {"base_model": plan.model_id, "output_name": plan.served_model_name},
        "artifact_store": plan.artifact_store or {},
        "artifacts_in": [plan.artifact] if plan.artifact else [],
        "artifacts_out": [],
    }
    env = {
        JOB_ENV_VAR: base64.b64encode(json.dumps(payload).encode()).decode(),
        TOKEN_ENV_VAR: secrets.token_urlsafe(32),
        API_KEY_ENV_VAR: api_key,
    }
    hf_token = resolve_hf_token(request, job_config)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if plan.artifact_store:
        env.update(artifact_worker_env(plan.artifact_store))
    return InstanceSpec(
        name=f"merlina-{session_id}-serve",
        image=plan.worker_image,
        gpu_type_id=plan.sizing.gpu_type_id,
        gpu_count=plan.sizing.gpu_count,
        container_disk_gb=getattr(request, "container_disk_gb", None)
                          or max(int(plan.sizing.disk_required_gb), 50),
        cloud_type=plan.sizing.cloud_type,
        env=env,
        http_ports=[WORKER_PORT, SERVE_PORT],
        docker_start_cmd=["python3", "-m", "src.remote.worker_entry"],
    )


class _TolerantStatusPoller:
    """``client.status()`` that rides out provider-proxy blips (returns None)
    for up to ``tolerance_s`` before declaring the worker lost."""

    def __init__(self, client: WorkerClient, tolerance_s: float, label: str):
        self.client, self.tolerance_s, self.label = client, tolerance_s, label
        self._since: Optional[float] = None

    def poll(self):
        try:
            status = self.client.status()
        except Exception as e:
            if _is_auth_error(e):
                raise RemoteStageError(f"Worker rejected the control plane's token: {e}") from e
            now = time.monotonic()
            if self._since is None:
                self._since = now
                logger.warning("Worker status for %s unreachable (%s); retrying", self.label, e)
            elif now - self._since > self.tolerance_s:
                raise RemoteStageError(
                    f"Worker status unreachable for {self.tolerance_s / 60:.0f} minutes: {e}") from e
            return None
        if self._since is not None:
            logger.info("Worker status for %s reachable again", self.label)
            self._since = None
        return status


def _drive_serve_session(*, session_id, sessions, client, instance, provider, plan,
                         lifecycle, owner, tick, poll_interval_s, settings) -> str:
    """Wait for ``serving``, publish the endpoint, hold. Returns the stop reason;
    raises :class:`RemoteStageError` when the worker or vLLM fails."""
    deadline_reason = plan.deadline_reason
    # The plan priced the session from offers; the provider's actual rate
    # may differ. Re-derive the total-cost deadline durably (never later).
    if plan.max_total_cost_usd and instance.cost_per_hr:
        run = lifecycle.run(session_id)
        cost_deadline = run["started_at"] + plan.max_total_cost_usd / instance.cost_per_hr * 3600
        if cost_deadline < run["deadline"]:
            deadline_reason = "max_total_cost_usd"
        sessions.update(session_id, deadline_at=lifecycle.tighten_deadline(session_id, owner, cost_deadline))

    def stop_requested() -> bool:
        current = sessions.get(session_id)
        return current is None or current.stop_requested

    def record(**fields):
        current = sessions.get(session_id)
        sessions.update(session_id, metrics={**(current.metrics if current else {}), **fields})

    sessions.update(session_id, status="starting")
    record(deadline_reason=deadline_reason, instance={
        "instance_id": instance.instance_id, "provider": instance.provider,
        "cost_per_hr": instance.cost_per_hr or plan.sizing.est_cost_per_hr,
        "gpu_type_id": plan.sizing.gpu_type_id, "gpu_count": plan.sizing.gpu_count})

    def failure(status) -> RemoteStageError:
        tail = status.serve.get("log_tail")
        if tail:
            record(log_tail=tail)
        return RemoteStageError(f"Serve stage failed on the worker: {status.error or 'unknown error'}")

    tolerance_s = getattr(settings, "remote_status_error_tolerance_minutes",
                          DEFAULT_STATUS_ERROR_TOLERANCE_MIN) * 60
    poller = _TolerantStatusPoller(client, tolerance_s, f"inference session {session_id}")

    # ---- Phase 1: model download + vLLM startup ----
    while True:
        tick()
        if stop_requested():
            raise RemoteStopped()
        status = poller.poll()
        if status is not None:
            if status.state == STATE_FAILED:
                raise failure(status)
            if status.state == STATE_SERVING:
                break
            if status.is_terminal:
                raise RemoteStageError("Worker finished before it started serving")
        time.sleep(poll_interval_s)

    base_url = (provider.proxy_url(instance, SERVE_PORT) or "").rstrip("/")
    if not base_url:
        raise RemoteStageError(f"Provider {provider.name} returned no serve proxy URL")
    base_url += status.serve.get("openai_base_path") or "/v1"
    served = status.serve.get("served_model_name") or plan.served_model_name
    sessions.update(session_id, status="ready", base_url=base_url, served_model_name=served,
                    ready_at=time.time())
    logger.info("Inference session %s ready at %s (model %r)", session_id, base_url, served)

    # ---- Phase 2: hold until something ends the session ----
    idle_limit_s = plan.idle_timeout_minutes * 60 if plan.idle_timeout_minutes else None
    last_stats = None
    while True:
        try:
            tick()
        except RemoteBudgetExceeded:
            return deadline_reason
        if stop_requested():
            return "user"
        status = poller.poll()
        if status is not None:
            if status.state == STATE_FAILED:
                raise failure(status)
            if status.is_terminal:
                return "worker_stopped"
            stats = {k: status.serve.get(k) for k in
                     ("requests", "inflight", "idle_seconds", "rejected_requests")}
            if stats != last_stats:
                record(serve=stats)
                last_stats = stats
            if idle_limit_s and (status.serve.get("idle_seconds") or 0) >= idle_limit_s:
                logger.info("Inference session %s idle for %.0f min; stopping",
                            session_id, idle_limit_s / 60)
                return "idle_timeout"
        time.sleep(poll_interval_s)


def _estimate_spend(resources: List[dict], now: Optional[float] = None) -> Optional[float]:
    """Dollars billed so far across a run's resources (None when unpriced)."""
    now = now or time.time()
    total, priced = 0.0, False
    for r in resources:
        if r.get("cost_per_hr") is None:
            continue
        priced = True
        end = r.get("terminated_at") or now
        total += max(0.0, end - r["started_at"]) / 3600 * r["cost_per_hr"]
    return round(total, 4) if priced else None


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
    return store_for_job(config, job_id)


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
