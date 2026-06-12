"""
Build the piecemeal stage plan for a remote run.

A plan answers three questions before any money is spent:
  1. What does this model actually need (MoE-aware sizing)?
  2. Which instance shape should the train stage rent?
  3. Where should the merge happen — locally, on a separate big-disk
     instance, or nowhere (adapter-only, the sane default for 1T-class
     models whose merged weights would be terabytes)?

The same function backs the ``POST /remote/plan`` preview endpoint and
the orchestrator's actual execution, so what you preview is what runs.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from .sizing import (
    FALLBACK_GPU_CATALOG,
    estimate_disk_gb,
    estimate_train_vram_gb,
    fetch_model_specs,
    pick_instance,
)
from .spec import GpuOffer, ModelSpecs, RemotePlan, StagePlan

logger = logging.getLogger(__name__)

PREFERENCE_MODES = {"orpo", "dpo", "simpo", "cpo", "ipo", "kto"}

# Above this native-weight footprint, "auto" merge strategy skips merging:
# the merged model would need to be re-downloaded/re-uploaded whole, and
# adapter-only artifacts are the practical format for very large bases.
AUTO_MERGE_MAX_WEIGHTS_GB = 80.0


class RemotePlanError(ValueError):
    """The requested remote run cannot be planned."""


def check_remote_compatible(config: Any) -> List[str]:
    """
    Return blocking problems that make a config unrunnable remotely.

    Remote workers can only reach datasets that exist somewhere durable —
    HuggingFace Hub datasets work; local files and in-memory uploads on
    the control plane don't travel (yet).
    """
    problems: List[str] = []
    dataset = getattr(config, "dataset", None)
    sources = []
    if dataset is not None:
        sources.append(dataset.source)
        sources.extend(getattr(dataset, "additional_sources", []) or [])
    for src in sources:
        if getattr(src, "source_type", "huggingface") != "huggingface":
            problems.append(
                f"Dataset source type '{src.source_type}' is not available on remote "
                "workers — push the dataset to the HuggingFace Hub and use a "
                "'huggingface' source for remote runs."
            )
    mode = (getattr(config, "training_mode", "") or "").lower()
    if mode.startswith("vlm_") or mode.startswith("diffusion_"):
        problems.append(
            f"training_mode '{mode}' is not supported on remote workers yet "
            "(text-LLM modes only)."
        )
    return problems


def build_remote_plan(
    config: Any,
    *,
    offers: Optional[List[GpuOffer]] = None,
    model_specs: Optional[ModelSpecs] = None,
    job_id: Optional[str] = None,
) -> RemotePlan:
    """
    Produce a :class:`RemotePlan` from a TrainingConfig with ``remote`` set.

    ``offers`` should be live provider offers when available; the static
    fallback catalog is used otherwise (preview without an API key).
    Raises :class:`RemotePlanError` when the run can't be planned.
    """
    remote = config.remote
    warnings: List[str] = []

    problems = check_remote_compatible(config)
    if problems:
        raise RemotePlanError(" ".join(problems))

    if model_specs is None:
        try:
            model_specs = fetch_model_specs(config.base_model, hf_token=config.hf_token)
        except Exception as e:
            raise RemotePlanError(
                f"Could not read config.json for '{config.base_model}' from the Hub "
                f"(needed for sizing): {e}"
            ) from e

    if model_specs.total_params_b is None:
        warnings.append(
            "Model config is missing standard size fields — sizing falls back to "
            "conservative defaults; consider setting remote.gpu_type_id explicitly."
        )

    vram_gb = estimate_train_vram_gb(
        model_specs,
        use_4bit=config.use_4bit,
        lora_r=getattr(config, "lora_r", 64) or 64,
        batch_size=getattr(config, "batch_size", 1) or 1,
        max_length=getattr(config, "max_length", 2048) or 2048,
        gradient_checkpointing=True,
        preference_mode=(config.training_mode or "").lower() in PREFERENCE_MODES,
    )
    disk_gb = estimate_disk_gb(model_specs)

    decision = pick_instance(
        vram_gb,
        disk_gb,
        offers or FALLBACK_GPU_CATALOG,
        cloud_type=remote.cloud_type,
        max_cost_per_hr=remote.max_cost_per_hr,
        gpu_type_id=remote.gpu_type_id,
        gpu_count=remote.gpu_count,
    )
    if model_specs.is_moe:
        decision.rationale.append(
            f"MoE model: ~{model_specs.total_params_b:.0f}B total / "
            f"~{model_specs.active_params_b:.0f}B active parameters."
        )
    if not (offers):
        warnings.append(
            "Sized against the static GPU catalog (no live provider offers) — "
            "prices and availability are approximate."
        )

    stages = [StagePlan(
        name="train",
        target="remote",
        sizing=decision,
        artifacts_out=["adapter"],
    )]

    merge_target, merge_notes = _resolve_merge_target(remote.merge_strategy, model_specs)
    if merge_target != "skip":
        stages.append(StagePlan(
            name="merge",
            target=merge_target,
            artifacts_in=["adapter"],
            artifacts_out=["merged"],
            notes=merge_notes,
        ))
    else:
        warnings.extend(merge_notes)

    return RemotePlan(
        job_id=job_id,
        model_specs=model_specs,
        stages=stages,
        warnings=warnings,
    )


def _resolve_merge_target(strategy: str, specs: ModelSpecs):
    """Decide where (or whether) the LoRA merge happens."""
    weights_gb = None
    if specs.weight_bytes_total:
        weights_gb = specs.weight_bytes_total / 1024 ** 3
    elif specs.total_params_b:
        weights_gb = specs.total_params_b * 2.0

    if strategy == "skip":
        return "skip", ["Merge skipped by request — the run produces an adapter only."]
    if strategy == "local":
        notes = ["Merge runs on the control plane after the adapter is pulled."]
        if weights_gb and weights_gb > AUTO_MERGE_MAX_WEIGHTS_GB:
            notes.append(
                f"Warning: base weights are ~{weights_gb:.0f} GB — the local merge "
                "needs that much RAM/disk (CPU bf16 reload)."
            )
        return "local", notes
    if strategy == "remote":
        return "remote", [
            "Merge runs on a separate provisioned instance (no GPU needed — "
            "sized for RAM and disk)."
        ]

    # "auto"
    if weights_gb is not None and weights_gb > AUTO_MERGE_MAX_WEIGHTS_GB:
        return "skip", [
            f"Auto merge strategy: base weights ~{weights_gb:.0f} GB exceed the "
            f"{AUTO_MERGE_MAX_WEIGHTS_GB:.0f} GB auto-merge limit, so the run is "
            "adapter-only. Set remote.merge_strategy='remote' to merge on a "
            "big-disk instance anyway."
        ]
    return "local", ["Auto merge strategy: base is small enough to merge locally."]
