"""
MoE-aware model sizing and instance selection for remote runs.

Unlike the local preflight estimator (which guesses hidden size from the
parameter count in the model's *name*), this module reads the model's
actual ``config.json`` — so a 1T-parameter MoE like Kimi-K2 with 7168
hidden size and 32B active parameters sizes correctly instead of
absurdly. Only ``huggingface_hub`` is needed; never torch/transformers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .spec import GpuOffer, ModelSpecs, SizingDecision

logger = logging.getLogger(__name__)

# Bytes per parameter for the loaded base weights.
BYTES_PER_PARAM_4BIT = 0.55     # NF4 + double-quant + bnb bookkeeping
BYTES_PER_PARAM_BF16 = 2.0

# AdamW on LoRA params: bf16 grads (2) + fp32 master copy (4) + two fp32
# moments (8) on top of the bf16 weight itself.
OPTIMIZER_BYTES_PER_TRAINABLE = 16.0

CUDA_CONTEXT_GB_PER_GPU = 2.0   # CUDA context + framework overhead per GPU
VRAM_HEADROOM_FACTOR = 1.2      # fragmentation, temporary buffers
USABLE_VRAM_FRACTION = 0.94     # never plan to the last byte of a GPU


# Static fallback so /remote/plan works without a provider API key.
# Prices are rough on-demand secure-cloud figures; live offers from the
# provider always take precedence when available.
FALLBACK_GPU_CATALOG: List[GpuOffer] = [
    GpuOffer("NVIDIA GeForce RTX 4090", "RTX 4090", 24, 0.69, 0.44, 8),
    GpuOffer("NVIDIA RTX A6000", "RTX A6000", 48, 0.79, 0.49, 8),
    GpuOffer("NVIDIA L40S", "L40S", 48, 1.10, 0.79, 8),
    GpuOffer("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.79, 1.19, 8),
    GpuOffer("NVIDIA H100 80GB HBM3", "H100 80GB", 80, 2.99, 1.99, 8),
    GpuOffer("NVIDIA H200", "H200 141GB", 141, 3.99, 3.45, 8),
    GpuOffer("NVIDIA B200", "B200 192GB", 192, 5.99, 4.99, 8),
]


def parse_model_specs(model_id: str, config: Dict[str, Any],
                      weight_bytes_total: Optional[int] = None) -> ModelSpecs:
    """
    Build :class:`ModelSpecs` from a parsed ``config.json`` dict.

    Handles both dense decoders and DeepSeek-V3-style MoE configs (which
    Kimi-K2 uses): ``n_routed_experts`` / ``num_experts`` routed experts,
    ``num_experts_per_tok`` active, optional shared experts, and a number
    of leading dense layers (``first_k_dense_replace``).
    """
    # Some configs nest the decoder under text_config (VLMs)
    text_cfg = config.get("text_config", config)

    def _get(*names, default=None):
        for n in names:
            if text_cfg.get(n) is not None:
                return text_cfg[n]
        return default

    hidden = _get("hidden_size", "n_embd", "d_model")
    layers = _get("num_hidden_layers", "n_layer", "num_layers")
    heads = _get("num_attention_heads", "n_head")
    kv_heads = _get("num_key_value_heads", default=heads)
    head_dim = _get("head_dim")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads

    specs = ModelSpecs(
        model_id=model_id,
        hidden_size=hidden,
        num_layers=layers,
        vocab_size=_get("vocab_size"),
        intermediate_size=_get("intermediate_size", "n_inner"),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        tie_word_embeddings=bool(_get("tie_word_embeddings", default=False)),
        architectures=list(config.get("architectures") or []),
        num_routed_experts=_get("n_routed_experts", "num_experts", "num_local_experts"),
        num_experts_per_tok=_get("num_experts_per_tok", "num_experts_per_token"),
        num_shared_experts=_get("n_shared_experts"),
        moe_intermediate_size=_get("moe_intermediate_size"),
        first_k_dense_layers=_get("first_k_dense_replace", default=0) or 0,
        weight_bytes_total=weight_bytes_total,
    )

    specs.total_params_b = _estimate_params_b(specs, active_only=False)
    specs.active_params_b = _estimate_params_b(specs, active_only=True)
    return specs


def _estimate_params_b(specs: ModelSpecs, active_only: bool) -> Optional[float]:
    """Estimate decoder parameter count (in billions) from config fields."""
    h, L, v = specs.hidden_size, specs.num_layers, specs.vocab_size
    if not (h and L and v):
        return None

    embed = v * h * (1 if specs.tie_word_embeddings else 2)

    kv_dim = (specs.num_key_value_heads or specs.num_attention_heads or 0) * (specs.head_dim or 0)
    q_dim = (specs.num_attention_heads or 0) * (specs.head_dim or 0)
    if q_dim and kv_dim:
        attn = h * q_dim + 2 * h * kv_dim + q_dim * h   # q, k, v, o
    else:
        attn = 4 * h * h

    inter = specs.intermediate_size or 4 * h
    dense_mlp = 3 * h * inter   # gated MLP: gate, up, down

    if specs.is_moe:
        moe_inter = specs.moe_intermediate_size or inter
        expert = 3 * h * moe_inter
        n_routed = specs.num_experts_per_tok if active_only else specs.num_routed_experts
        n_routed = n_routed or 0
        shared = (specs.num_shared_experts or 0) * expert
        router = h * (specs.num_routed_experts or 0)
        moe_mlp = n_routed * expert + shared + router

        dense_layers = min(specs.first_k_dense_layers, L)
        moe_layers = L - dense_layers
        mlp_total = dense_layers * dense_mlp + moe_layers * moe_mlp
    else:
        mlp_total = L * dense_mlp

    total = embed + L * attn + mlp_total
    return total / 1e9


def fetch_model_specs(model_id: str, hf_token: Optional[str] = None) -> ModelSpecs:
    """
    Fetch ``config.json`` (and the safetensors index, when present) from
    the HuggingFace Hub and parse it into :class:`ModelSpecs`.
    """
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_id, "config.json", token=hf_token)
    with open(config_path) as f:
        config = json.load(f)

    weight_bytes = None
    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json", token=hf_token)
        with open(index_path) as f:
            index = json.load(f)
        weight_bytes = (index.get("metadata") or {}).get("total_size")
    except Exception:
        pass  # single-file or non-safetensors repos have no index

    return parse_model_specs(model_id, config, weight_bytes)


def estimate_train_vram_gb(
    specs: ModelSpecs,
    *,
    use_4bit: bool = True,
    lora_r: int = 64,
    batch_size: int = 1,
    max_length: int = 2048,
    gradient_checkpointing: bool = True,
    preference_mode: bool = False,
) -> float:
    """
    Estimate total VRAM (GB) for a LoRA/QLoRA run, across all GPUs.

    Uses real architecture numbers: base weights from the (exact when
    available) weight footprint, optimizer state from the actual LoRA
    parameter count, activations from the true hidden size.
    """
    gib = 1024 ** 3

    if specs.weight_bytes_total and not use_4bit:
        weights_gb = specs.weight_bytes_total / gib
    else:
        params_b = specs.total_params_b or 1.0
        bpp = BYTES_PER_PARAM_4BIT if use_4bit else BYTES_PER_PARAM_BF16
        weights_gb = params_b * bpp

    # LoRA on attention projections: 2 low-rank matrices per projection,
    # 4 projections per layer.
    h, L = specs.hidden_size or 4096, specs.num_layers or 32
    lora_params = 8 * lora_r * h * L
    optimizer_gb = lora_params * OPTIMIZER_BYTES_PER_TRAINABLE / gib

    # Activations: per-layer residual streams in bf16; preference modes
    # (ORPO/DPO/...) forward chosen + rejected, doubling the batch.
    eff_batch = batch_size * (2 if preference_mode else 1)
    act_per_layer = eff_batch * max_length * h * 2  # bytes, bf16
    layers_resident = max(L * 0.15, 4) if gradient_checkpointing else L
    activations_gb = act_per_layer * layers_resident * 4 / gib  # ~4 tensors/layer

    total = (weights_gb + optimizer_gb + activations_gb) * VRAM_HEADROOM_FACTOR
    return round(total, 1)


def estimate_disk_gb(specs: ModelSpecs, *, merge_stage: bool = False) -> float:
    """
    Disk needed on the instance: base weights download + checkpoints.
    A merge stage holds base + merged copies simultaneously.
    """
    gib = 1024 ** 3
    if specs.weight_bytes_total:
        weights_gb = specs.weight_bytes_total / gib
    else:
        weights_gb = (specs.total_params_b or 1.0) * BYTES_PER_PARAM_BF16

    base_overhead = 30.0  # container image, pip cache, datasets, logs
    if merge_stage:
        return round(weights_gb * 2.2 + base_overhead, 0)
    # Training keeps the HF download + adapter checkpoints (small)
    return round(weights_gb * 1.1 + 10.0 + base_overhead, 0)


def pick_instance(
    vram_required_gb: float,
    disk_required_gb: float,
    offers: List[GpuOffer],
    *,
    cloud_type: str = "secure",
    max_cost_per_hr: Optional[float] = None,
    gpu_type_id: Optional[str] = None,
    gpu_count: Optional[int] = None,
) -> SizingDecision:
    """
    Choose the cheapest feasible (gpu_type, count) for a training stage.

    Feasibility is model-parallel aware: one GPU big enough wins outright;
    otherwise layers shard across N GPUs (single process, device_map=auto)
    and the *sum* of usable VRAM must cover the requirement, minus
    per-GPU runtime overhead. Explicit ``gpu_type_id``/``gpu_count``
    overrides skip the search but still get a feasibility check.

    Raises ``ValueError`` when nothing fits (or the explicit pick can't).
    """
    rationale = [f"Need ~{vram_required_gb:.0f} GB VRAM and ~{disk_required_gb:.0f} GB disk."]

    def usable(offer: GpuOffer, count: int) -> float:
        return (offer.vram_gb * USABLE_VRAM_FRACTION - CUDA_CONTEXT_GB_PER_GPU) * count

    if gpu_type_id:
        offer = next((o for o in offers if o.gpu_type_id == gpu_type_id), None)
        if offer is None:
            offer = GpuOffer(gpu_type_id, gpu_type_id, vram_gb=0)
        count = gpu_count or 1
        if offer.vram_gb and usable(offer, count) < vram_required_gb:
            raise ValueError(
                f"Explicit pick {count}× {gpu_type_id} has ~{usable(offer, count):.0f} GB "
                f"usable VRAM but the run needs ~{vram_required_gb:.0f} GB."
            )
        price = offer.price_per_hr(cloud_type)
        rationale.append(f"Using explicit instance selection: {count}× {gpu_type_id}.")
        return SizingDecision(
            gpu_type_id=gpu_type_id,
            gpu_count=count,
            cloud_type=cloud_type,
            vram_required_gb=vram_required_gb,
            disk_required_gb=disk_required_gb,
            est_cost_per_hr=price * count if price else None,
            parallelism="single_gpu" if count == 1 else "model_parallel",
            rationale=rationale,
        )

    best: Optional[SizingDecision] = None
    for offer in offers:
        if not offer.available:
            continue
        price = offer.price_per_hr(cloud_type)
        for count in range(1, max(offer.max_gpu_count, 1) + 1):
            if usable(offer, count) < vram_required_gb:
                continue
            cost = price * count if price else None
            if max_cost_per_hr is not None and cost is not None and cost > max_cost_per_hr:
                break  # more GPUs only cost more
            candidate = SizingDecision(
                gpu_type_id=offer.gpu_type_id,
                gpu_count=count,
                cloud_type=cloud_type,
                vram_required_gb=vram_required_gb,
                disk_required_gb=disk_required_gb,
                est_cost_per_hr=cost,
                parallelism="single_gpu" if count == 1 else "model_parallel",
                rationale=list(rationale),
            )
            if _is_better(candidate, best):
                best = candidate
            break  # smallest feasible count for this type found

    if best is None:
        budget = f" under ${max_cost_per_hr:.2f}/hr" if max_cost_per_hr is not None else ""
        raise ValueError(
            f"No available instance type fits ~{vram_required_gb:.0f} GB VRAM{budget}. "
            "Consider 4-bit quantization, a shorter max_length, or raising the budget."
        )

    best.rationale.append(
        f"Picked {best.gpu_count}× {best.gpu_type_id} ({best.parallelism}), "
        + (f"~${best.est_cost_per_hr:.2f}/hr." if best.est_cost_per_hr else "price unknown.")
    )
    return best


def _is_better(candidate: SizingDecision, best: Optional[SizingDecision]) -> bool:
    if best is None:
        return True
    c_cost = candidate.est_cost_per_hr
    b_cost = best.est_cost_per_hr
    if c_cost is not None and b_cost is not None and c_cost != b_cost:
        return c_cost < b_cost
    if (c_cost is None) != (b_cost is None):
        return c_cost is not None  # prefer known prices
    return candidate.gpu_count < best.gpu_count
