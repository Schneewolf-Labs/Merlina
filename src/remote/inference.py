"""
Remote inference planning — what a serve session would rent, and for how long.

``build_serve_plan`` is the serve-side sibling of
:func:`src.remote.plan.build_remote_plan`. It backs both the dry-run
endpoint (``POST /remote/inference/plan``) and the real session
(``run_inference_remote``), so the GPU, price, and deadline the owner
approves are exactly what runs.

A session serves a HuggingFace repo, a model pushed to object storage
(``merlina models push``, served with ``model.store_model``), or the output
of a finished *remote* training job pulled from that job's artifact store:
the "merged" weights, or the "adapter" on top of its base model (vLLM
LoRA). The latter closes the train → serve → eval loop without anything
touching the control plane's disk.

No torch/transformers/vLLM imports here.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .plan import RemotePlanError
from .serve_worker import (
    DEFAULT_STARTUP_TIMEOUT_MIN,
    build_vllm_command,
    check_vllm_args,
    lora_rank_for,
)
from .sizing import (
    FALLBACK_GPU_CATALOG,
    estimate_disk_gb,
    fetch_model_specs,
    pick_instance,
    store_model_specs,
    serve_vram_breakdown,
    tensor_parallel_counts,
)
from .spec import GpuOffer, ModelSpecs, ServePlan

logger = logging.getLogger(__name__)

DEFAULT_SERVE_IMAGE = "ghcr.io/schneewolf-labs/merlina-serve:latest"
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_NUM_SEQS = 8
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
MAX_SERVE_RUNTIME_HOURS = 24.0
# A session whose total-cost cap leaves less than this would likely spend
# its whole budget downloading and loading weights.
MIN_USEFUL_SESSION_HOURS = 0.25
SERVE_DISK_FLOOR_GB = 50

from src.model_refs import _NAME_RE as _STORE_MODEL_RE  # noqa: E402  (names may nest: hf/<org>/<repo>)
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


# ---------------------------------------------------------------------------
# Request models (shared by the API and the orchestrator)
# ---------------------------------------------------------------------------

class InferenceModelSource(BaseModel):
    """What to serve: a HuggingFace repo, a model in object storage, or a
    Merlina training job's output."""
    hf_repo: Optional[str] = Field(
        None, description="HuggingFace model id to serve, e.g. 'Qwen/Qwen3-8B'")
    revision: Optional[str] = Field(None, description="Branch, tag or commit of hf_repo")
    store_model: Optional[str] = Field(
        None, description="A model pushed with `merlina models push` to the artifact "
                          "store (models/<name>), e.g. 'C0' — no Hub repo needed")
    job_id: Optional[str] = Field(
        None, description="A completed *remote* Merlina training job whose artifact to serve")
    artifact: Literal["adapter", "merged"] = Field(
        "adapter",
        description="With job_id: 'adapter' serves the base model + LoRA adapter (vLLM "
                    "--enable-lora); 'merged' serves merged weights (needs a remote merge stage)")

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if sum(map(bool, (self.hf_repo, self.store_model, self.job_id))) != 1:
            raise ValueError("Set exactly one of model.hf_repo, model.store_model or model.job_id")
        if self.store_model and not _STORE_MODEL_RE.match(self.store_model):
            raise ValueError("model.store_model must be a name like 'C0' or 'hf/org/repo' "
                             "(letters, digits, '.', '_', '-', up to three '/')")
        if self.hf_repo and not _HF_REPO_RE.match(self.hf_repo):
            raise ValueError("model.hf_repo must be a HuggingFace repo id like 'org/name' "
                             "(local paths don't travel to rented instances)")
        if self.revision and not self.hf_repo:
            raise ValueError("model.revision applies to model.hf_repo only")
        return self


class RemoteInferenceRequest(BaseModel):
    """
    Start (or dry-run plan) a remote inference session: rent a GPU instance,
    serve the model with vLLM behind an OpenAI-compatible API, and tear the
    instance down when stopped, idle, or out of time/budget.
    """
    model: InferenceModelSource
    served_model_name: Optional[str] = Field(
        None, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
        description="Name clients pass as 'model' (default: the repo / output name)")
    hf_token: Optional[str] = Field(
        None, description="HF token for gated/private repos and HF artifact stores "
                          "(default: the training job's token, then server HF_TOKEN). Never stored.")
    provider: Literal["runpod"] = Field("runpod", description="Compute provider")
    gpu_type_id: Optional[str] = Field(None, description="Explicit GPU type (default: cheapest fit)")
    gpu_count: Optional[int] = Field(None, ge=1, le=8, description="GPUs on the instance")
    cloud_type: Literal["secure", "community"] = Field("secure")
    container_disk_gb: Optional[int] = Field(None, ge=20, description="Disk override (default: sized)")
    max_cost_per_hr: Optional[float] = Field(
        None, gt=0, description="Reject instance picks above this hourly cost")
    max_runtime_hours: float = Field(
        2.0, gt=0, le=MAX_SERVE_RUNTIME_HOURS,
        description="Hard cap — the instance is terminated when this elapses (max 24)")
    max_total_cost_usd: Optional[float] = Field(
        None, gt=0,
        description="Spend cap — the session ends at start + max_total_cost_usd / $/hr "
                    "if that is earlier than max_runtime_hours")
    idle_timeout_minutes: Optional[float] = Field(
        None, ge=1, le=24 * 60,
        description="Stop after this long with no requests (counted from when serving starts)")
    max_model_len: Optional[int] = Field(
        None, ge=256, le=1_048_576,
        description=f"Context length (default: min(model max, {DEFAULT_MAX_MODEL_LEN}))")
    max_num_seqs: int = Field(
        DEFAULT_MAX_NUM_SEQS, ge=1, le=1024,
        description="Concurrent sequences; the plan reserves KV cache for all of them at full length")
    tensor_parallel_size: Optional[int] = Field(
        None, ge=1, le=8, description="vLLM tensor parallelism (default: the instance's GPU count)")
    gpu_memory_utilization: float = Field(DEFAULT_GPU_MEMORY_UTILIZATION, gt=0.3, le=0.98)
    vllm_args: List[str] = Field(
        default_factory=list,
        description="Extra `vllm serve` flags, e.g. ['--enable-prefix-caching']. "
                    "Host/port/api-key/served-name are managed by Merlina.")
    worker_image: Optional[str] = Field(
        None, description="Serve image override (default: settings.remote_serve_image)")
    keep_instance_on_failure: bool = Field(
        False, description="Keep the instance after a FAILURE for debugging (bills until "
                           "the session deadline). Stops, idle and caps always terminate.")

    @model_validator(mode="after")
    def _check_vllm_args(self):
        problems = check_vllm_args(self.vllm_args)
        if problems:
            raise ValueError("; ".join(problems))
        return self


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def resolve_hf_token(request: Any, job_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Request token, then the training job's token, then the server's."""
    token = getattr(request, "hf_token", None) or (job_config or {}).get("hf_token")
    if token:
        return token
    try:
        from config import resolve_hf_token as _server_token
        return _server_token(None)
    except Exception:
        return None


def _job_artifacts(lifecycle, job_id: str) -> List[str]:
    pushed: List[str] = []
    for attempt in lifecycle.attempts() if lifecycle is not None else []:
        if attempt["job_id"] != job_id or attempt["state"] != "completed":
            continue
        try:
            pushed.extend(json.loads(attempt.get("artifacts") or "[]"))
        except (TypeError, ValueError):
            pass
        if attempt["stage_name"] == "train" and "adapter" not in pushed:
            pushed.append("adapter")      # a completed train stage always pushes it
    return pushed


def _store_source(name: str, request: RemoteInferenceRequest, store) -> Dict[str, Any]:
    from .artifacts import ArtifactStoreError, model_store
    try:
        store = store if store is not None else model_store()
    except ArtifactStoreError as e:
        raise RemotePlanError(str(e)) from e
    if getattr(store, "tier", None) == "internal":
        raise RemotePlanError(
            "The artifact store is the internal (LAN) tier, which a rented instance can't "
            "reach. Push the model to the external tier and serve it from there.")
    try:
        exists = store.exists(name)
    except Exception as e:
        raise RemotePlanError(f"Could not read models/{name} from the artifact store: {e}") from e
    if not exists:
        raise RemotePlanError(
            f"No model {name!r} in the store (models/{name}); push it with "
            f"`merlina models push <dir> {name}`")
    return {
        "source_kind": "artifact", "model_id": f"store:{name}", "revision": None,
        "artifact": name, "artifact_store": store.describe(), "lora": False,
        "max_lora_rank": None, "served_model_name": request.served_model_name or name,
        "base_served_model_name": None, "job_config": None, "warnings": [], "store": store,
    }


def resolve_model_source(request: RemoteInferenceRequest, *, job_manager=None,
                         lifecycle=None, model_store=None) -> Dict[str, Any]:
    """Turn the request's model source into what the worker should load."""
    src = request.model
    if src.store_model:
        return _store_source(src.store_model, request, model_store)
    if src.hf_repo:
        return {
            "source_kind": "hf", "model_id": src.hf_repo, "revision": src.revision,
            "artifact": None, "artifact_store": None, "lora": False, "max_lora_rank": None,
            "served_model_name": request.served_model_name or src.hf_repo.rsplit("/", 1)[-1],
            "base_served_model_name": None, "job_config": None, "warnings": [],
        }

    job = job_manager.get_job(src.job_id) if job_manager is not None else None
    if job is None:
        raise RemotePlanError(f"Training job {src.job_id!r} not found")
    cfg = job.config or {}
    if not (cfg.get("remote") or {}).get("enabled"):
        raise RemotePlanError(
            f"Job {src.job_id} trained locally, so its output is on this machine rather than "
            "in an artifact store a rented instance can reach. Push it to the HuggingFace Hub "
            "and serve it with model.hf_repo instead.")
    if job.status != "completed":
        raise RemotePlanError(f"Job {src.job_id} is {job.status!r}; only completed jobs can be served")
    store = (job.metrics or {}).get("artifact_store")
    if not store:
        raise RemotePlanError(f"Job {src.job_id} has no recorded artifact store")
    pushed = _job_artifacts(lifecycle, src.job_id)
    if src.artifact not in pushed:
        hub = [a[4:] for a in pushed if a.startswith("hub:")]
        hint = (f" Its merged model was published to the Hub — serve it with "
                f"model.hf_repo='{hub[0]}'." if hub else
                " The merged model only reaches the store when the job used "
                "remote.merge_strategy='remote' without push_to_hub; serve the adapter instead.")
        raise RemotePlanError(f"Job {src.job_id} has no {src.artifact!r} artifact in its store.{hint}")

    warnings: List[str] = []
    base_model = cfg.get("base_model")
    lora = src.artifact == "adapter" and bool(cfg.get("use_lora", True))
    max_lora_rank = None
    if lora:
        try:
            max_lora_rank = lora_rank_for(cfg.get("lora_r"))
        except ValueError as e:
            raise RemotePlanError(str(e)) from e
        if cfg.get("modules_to_save"):
            warnings.append(
                f"The adapter fully retrains {cfg['modules_to_save']} (modules_to_save), which "
                "vLLM LoRA serving does not apply — outputs would differ from the trained model. "
                "Serve the 'merged' artifact for this job instead.")
    output_name = str(cfg.get("output_name") or src.job_id).rsplit("/", 1)[-1]
    served = request.served_model_name or output_name
    return {
        "source_kind": "artifact", "model_id": base_model, "revision": None,
        "artifact": src.artifact, "artifact_store": store, "lora": lora,
        "max_lora_rank": max_lora_rank, "served_model_name": served,
        "base_served_model_name": base_model if lora else None,
        "job_config": cfg, "warnings": warnings,
    }


def build_serve_plan(
    request: RemoteInferenceRequest,
    *,
    offers: Optional[List[GpuOffer]] = None,
    model_specs: Optional[ModelSpecs] = None,
    job_manager=None,
    lifecycle=None,
    settings=None,
    model_store=None,
) -> ServePlan:
    """Size, price, and time-box a serve session without provisioning anything."""
    if settings is None:
        from config import settings as _settings
        settings = _settings
    source = resolve_model_source(request, job_manager=job_manager, lifecycle=lifecycle,
                                  model_store=model_store)
    warnings: List[str] = list(source["warnings"])

    if model_specs is None and source.get("store") is not None:
        try:
            model_specs = store_model_specs(source["store"], source["artifact"])
        except Exception as e:
            raise RemotePlanError(
                f"Could not read config.json for {source['model_id']} (needed for sizing): "
                f"{e}") from e
    elif model_specs is None:
        try:
            model_specs = fetch_model_specs(
                source["model_id"], hf_token=resolve_hf_token(request, source["job_config"]),
                revision=source["revision"])
        except Exception as e:
            raise RemotePlanError(
                f"Could not read config.json for '{source['model_id']}' from the Hub "
                f"(needed for sizing): {e}") from e
    if not model_specs.weight_bytes_total:
        warnings.append("No safetensors index found — weights sized as bf16 from the estimated "
                        "parameter count; set gpu_type_id explicitly if that is wrong.")

    model_max = model_specs.max_position_embeddings
    max_model_len = request.max_model_len or min(model_max or DEFAULT_MAX_MODEL_LEN,
                                                 DEFAULT_MAX_MODEL_LEN)
    if request.max_model_len and model_max and request.max_model_len > model_max:
        warnings.append(f"max_model_len {request.max_model_len} exceeds the model's "
                        f"max_position_embeddings ({model_max}); vLLM may refuse to start.")

    breakdown = serve_vram_breakdown(model_specs, max_model_len, request.max_num_seqs,
                                     lora=source["lora"])
    disk_gb = max(estimate_disk_gb(model_specs), SERVE_DISK_FLOOR_GB)

    counts = tensor_parallel_counts(model_specs)
    if request.tensor_parallel_size:
        if request.tensor_parallel_size not in counts:
            warnings.append(f"tensor_parallel_size {request.tensor_parallel_size} does not divide "
                            f"the model's {model_specs.num_attention_heads} attention heads.")
        counts = [request.tensor_parallel_size]
    try:
        decision = pick_instance(
            breakdown["total_gb"], disk_gb, offers or FALLBACK_GPU_CATALOG,
            cloud_type=request.cloud_type, max_cost_per_hr=request.max_cost_per_hr,
            gpu_type_id=request.gpu_type_id, gpu_count=request.gpu_count,
            gpu_counts=[request.gpu_count] if request.gpu_count else counts,
        )
    except ValueError as e:
        raise RemotePlanError(str(e)) from e
    tp = request.tensor_parallel_size or decision.gpu_count
    if decision.gpu_count % tp:
        raise RemotePlanError(f"tensor_parallel_size {tp} does not fit {decision.gpu_count} GPUs")
    decision.parallelism = "single_gpu" if tp == 1 else "tensor_parallel"
    if not offers:
        warnings.append("Sized against the static GPU catalog (no live provider offers) — "
                        "prices and availability are approximate.")

    cost = decision.est_cost_per_hr
    effective_hours, reason = request.max_runtime_hours, "max_runtime_hours"
    if request.max_total_cost_usd is not None:
        if not cost:
            raise RemotePlanError(
                "max_total_cost_usd needs a known hourly price, but the selected GPU has none; "
                "pick a priced gpu_type_id or drop the total-cost cap.")
        cost_hours = request.max_total_cost_usd / cost
        if cost_hours < effective_hours:
            effective_hours, reason = cost_hours, "max_total_cost_usd"
    if effective_hours < MIN_USEFUL_SESSION_HOURS:
        warnings.append(f"The caps allow only ~{effective_hours * 60:.0f} minutes, most of which "
                        "may go to downloading and loading the weights.")

    serve = {
        "source_kind": source["source_kind"], "model_id": source["model_id"],
        "revision": source["revision"], "lora": source["lora"],
        "served_model_name": source["served_model_name"],
        "base_served_model_name": source["base_served_model_name"],
        "max_lora_rank": source["max_lora_rank"], "tensor_parallel_size": tp,
        "max_model_len": max_model_len, "max_num_seqs": request.max_num_seqs,
        "gpu_memory_utilization": request.gpu_memory_utilization,
        "vllm_args": list(request.vllm_args),
    }
    model_ref = (source["model_id"] if source["source_kind"] == "hf" or source["lora"]
                 else f"<{source['model_id']}>" if source.get("store") is not None
                 else f"<artifact:{source['artifact']}>")
    command = build_vllm_command(serve, model_ref=model_ref,
                                 lora_path=f"<artifact:{source['artifact']}>")

    return ServePlan(
        model_source=request.model.model_dump(exclude_none=True),
        model_id=source["model_id"],
        revision=source["revision"],
        served_model_name=source["served_model_name"],
        base_served_model_name=source["base_served_model_name"],
        lora=source["lora"],
        max_lora_rank=source["max_lora_rank"],
        artifact=source["artifact"],
        artifact_store=source["artifact_store"],
        model_specs=model_specs,
        sizing=decision,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        max_num_seqs=request.max_num_seqs,
        gpu_memory_utilization=request.gpu_memory_utilization,
        vram_breakdown=breakdown,
        max_runtime_hours=request.max_runtime_hours,
        max_total_cost_usd=request.max_total_cost_usd,
        idle_timeout_minutes=request.idle_timeout_minutes,
        effective_max_hours=effective_hours,
        deadline_reason=reason,
        est_max_total_cost_usd=round(cost * effective_hours, 2) if cost else None,
        worker_image=(request.worker_image
                      or getattr(settings, "remote_serve_image", None) or DEFAULT_SERVE_IMAGE),
        vllm_args=list(request.vllm_args),
        vllm_command=command,
        warnings=warnings,
    )


def serve_spec(plan: ServePlan) -> Dict[str, Any]:
    """The ``serve`` block of the worker payload."""
    return {
        "source_kind": "artifact" if plan.artifact else "hf",
        "model_id": plan.model_id,
        "revision": plan.revision,
        "artifact": plan.artifact,
        "lora": plan.lora,
        "served_model_name": plan.served_model_name,
        "base_served_model_name": plan.base_served_model_name,
        "max_lora_rank": plan.max_lora_rank,
        "tensor_parallel_size": plan.tensor_parallel_size,
        "max_model_len": plan.max_model_len,
        "max_num_seqs": plan.max_num_seqs,
        "gpu_memory_utilization": plan.gpu_memory_utilization,
        "vllm_args": list(plan.vllm_args),
        "startup_timeout_minutes": DEFAULT_STARTUP_TIMEOUT_MIN,
    }

