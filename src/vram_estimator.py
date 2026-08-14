"""
Architecture-aware VRAM estimation for training jobs.

The previous estimator guessed model size from the model *name* and used a
linear formula with a hidden-size approximation (``hidden ≈ B × 512``) that
was wildly wrong outside the 7B range, ignored vocabulary size entirely, and
knew nothing about training mode, optimizer choice, or attention
implementation. This module replaces it with an estimate built from the
model's actual architecture:

1. **Model profile** — read the real ``config.json`` (local directory or the
   HuggingFace Hub, honoring the local cache and offline mode) to get hidden
   size, layer count, GQA head counts, intermediate size, vocab size, and
   embedding tying. Exact parameter counts come from safetensors metadata
   when available; otherwise they are derived from the architecture dims.
   Only when no config can be found does it fall back to the old
   name-pattern heuristic, with representative dims for that size class.

2. **Component model** — VRAM is estimated per component and returned as a
   breakdown so callers can tell users *what* is eating memory:
   weights (quantization-aware — bitsandbytes leaves embeddings and
   ``lm_head`` in bf16), reference model (DPO/KTO without LoRA), trainable
   parameters (LoRA adapters sized from the actual target modules, plus any
   ``modules_to_save``), gradients, optimizer states (8-bit vs 32-bit vs
   SGD...), layer activations (flash vs eager attention, gradient
   checkpointing, and the 2× forward batch of paired preference modes), and
   logits/loss memory (∝ vocab size — the dominant term for modern
   large-vocab models, and the reason "the same 7B" OOMs on Llama-3 but not
   Mistral).

Every number here is an estimate; real usage varies with allocator
fragmentation and library versions. The goal is to be within ~20% and,
more importantly, to *rank* configs correctly so pre-flight suggestions
("enable gradient checkpointing", "this is logits memory — reduce
max_length") point at the actual bottleneck.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GB = 1024 ** 3

# Training modes that concatenate chosen+rejected into one forward pass,
# doubling the effective forward batch (TRL ORPO/DPO/CPO... trainers).
PAIRED_PREFERENCE_MODES = {"orpo", "dpo", "simpo", "cpo", "ipo"}

# Modes that keep a frozen reference model. With LoRA the reference is
# obtained by disabling adapters (no extra copy); without LoRA TRL loads a
# second full model.
REFERENCE_MODEL_MODES = {"dpo", "kto"}

# Optimizer state bytes per trainable parameter.
OPTIMIZER_STATE_BYTES = {
    "paged_adamw_8bit": 2.0,
    "adamw_8bit": 2.0,
    "paged_adamw_32bit": 8.0,
    "adamw_torch": 8.0,
    "adamw_hf": 8.0,
    "adafactor": 0.5,   # factored second moments are sub-linear
    "sgd": 0.0,
    "muon": 4.0,        # momentum buffer
}
DEFAULT_OPTIMIZER_STATE_BYTES = 8.0

# NF4 weight + double-quantized absmax constants (bytes per parameter).
NF4_BYTES_PER_PARAM = 0.53

# Merlina's default LoRA target set (mirrors TrainingConfig.target_modules).
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]

# Fixed CUDA context / cublas workspace overhead (GB) and fragmentation
# headroom applied to the subtotal.
CUDA_CONTEXT_GB = 0.9
FRAGMENTATION_FACTOR = 0.08

# Representative architectures per size class, used only when no config.json
# can be found and the size had to be guessed from the model name.
# (hidden, layers, heads, kv_heads, intermediate, vocab)
_ARCH_PRESETS = [
    (0.5, dict(hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
               num_key_value_heads=2, intermediate_size=4864, vocab_size=151_936)),
    (1.0, dict(hidden_size=2048, num_hidden_layers=16, num_attention_heads=32,
               num_key_value_heads=8, intermediate_size=8192, vocab_size=128_256)),
    (3.0, dict(hidden_size=3072, num_hidden_layers=28, num_attention_heads=24,
               num_key_value_heads=8, intermediate_size=8192, vocab_size=128_256)),
    (7.0, dict(hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
               num_key_value_heads=8, intermediate_size=14336, vocab_size=32_768)),
    (8.0, dict(hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
               num_key_value_heads=8, intermediate_size=14336, vocab_size=128_256)),
    (12.0, dict(hidden_size=5120, num_hidden_layers=40, num_attention_heads=32,
                num_key_value_heads=8, intermediate_size=14336, vocab_size=131_072)),
    (14.0, dict(hidden_size=5120, num_hidden_layers=48, num_attention_heads=40,
                num_key_value_heads=8, intermediate_size=13824, vocab_size=152_064)),
    (24.0, dict(hidden_size=5120, num_hidden_layers=40, num_attention_heads=32,
                num_key_value_heads=8, intermediate_size=32768, vocab_size=131_072)),
    (32.0, dict(hidden_size=5120, num_hidden_layers=64, num_attention_heads=40,
                num_key_value_heads=8, intermediate_size=27648, vocab_size=152_064)),
    (70.0, dict(hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
                num_key_value_heads=8, intermediate_size=28672, vocab_size=128_256)),
]

_profile_cache: Dict[str, "ModelProfile"] = {}
_profile_cache_lock = threading.Lock()


@dataclass
class ModelProfile:
    """Architecture facts about a model, from the most reliable source found."""
    model_id: str
    num_params: Optional[float] = None          # total parameter count
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    vocab_size: Optional[int] = None
    tie_word_embeddings: bool = False
    # Hybrid architectures (Qwen3.5/3.6/3.8 and friends) interleave linear-attention layers with
    # full-attention ones. Those layers have a fused training path that is off unless optional
    # kernels are installed, and the difference is 7x per step — worth surfacing, so the profile
    # carries enough to detect it.
    linear_attention_layers: int = 0
    source: str = "unknown"                     # local_config | hub_config | name_heuristic
    params_exact: bool = False                  # True when read from safetensors metadata

    @property
    def has_dims(self) -> bool:
        return all(v is not None for v in (
            self.hidden_size, self.num_hidden_layers,
            self.intermediate_size, self.vocab_size,
        ))

    @property
    def kv_dim(self) -> Optional[int]:
        if self.num_key_value_heads and self.head_dim:
            return self.num_key_value_heads * self.head_dim
        return self.hidden_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "num_params_billions": round(self.num_params / 1e9, 3) if self.num_params else None,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "source": self.source,
            "params_exact": self.params_exact,
        }


@dataclass
class VRAMEstimate:
    """Per-component VRAM estimate. All values in GB."""
    total_gb: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    confidence: str = "low"                     # high | medium | low
    notes: List[str] = field(default_factory=list)
    profile: Optional[ModelProfile] = None

    def dominant_component(self) -> Optional[str]:
        """Largest single component (excluding fixed overhead)."""
        candidates = {k: v for k, v in self.breakdown.items()
                      if k not in ("cuda_context", "fragmentation_buffer")}
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_gb": round(self.total_gb, 2),
            "breakdown_gb": {k: round(v, 2) for k, v in self.breakdown.items()},
            "confidence": self.confidence,
            "notes": self.notes,
            "model_profile": self.profile.to_dict() if self.profile else None,
        }


def get_model_size_from_name(model_name: str) -> Optional[float]:
    """Extract model size in billions from a model name (e.g. '...-8B-...')."""
    model_lower = model_name.lower()
    patterns = [
        r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)b',        # 8x7b MoE → total (must match first)
        r'(\d+(?:\.\d+)?)\s*b(?:illion)?(?:\b|$)',  # 7b, 7B, 7billion
        r'(\d+(?:\.\d+)?)-b(?:\b|$)',               # 7-b
        r'(\d+(?:\.\d+)?)_b(?:\b|$)',               # 7_b
    ]
    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2 and groups[1] is not None:
                return float(groups[0]) * float(groups[1])
            return float(groups[0])
    return None


def _extract_text_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the text-decoder sub-config for VLMs, or the config itself."""
    for key in ("text_config", "llm_config", "language_config"):
        sub = cfg.get(key)
        if isinstance(sub, dict) and "hidden_size" in sub:
            return sub
    return cfg


def _profile_from_config_dict(model_id: str, cfg: Dict[str, Any], source: str) -> ModelProfile:
    text_cfg = _extract_text_config(cfg)
    hidden = text_cfg.get("hidden_size")
    heads = text_cfg.get("num_attention_heads")
    head_dim = text_cfg.get("head_dim")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    profile = ModelProfile(
        model_id=model_id,
        hidden_size=hidden,
        num_hidden_layers=text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers"),
        num_attention_heads=heads,
        num_key_value_heads=text_cfg.get("num_key_value_heads") or heads,
        head_dim=head_dim,
        intermediate_size=text_cfg.get("intermediate_size"),
        vocab_size=text_cfg.get("vocab_size"),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings", False))),
        linear_attention_layers=sum(
            1 for t in (text_cfg.get("layer_types") or []) if t == "linear_attention"
        ),
        source=source,
    )
    if profile.has_dims:
        profile.num_params = _derive_param_count(profile)
    return profile


def _derive_param_count(p: ModelProfile) -> float:
    """Compute total parameters from architecture dims (dense decoder)."""
    h = p.hidden_size
    v = p.vocab_size
    i = p.intermediate_size
    layers = p.num_hidden_layers
    q_dim = (p.num_attention_heads or 0) * (p.head_dim or 0) or h
    kv = p.kv_dim or h

    embed = v * h
    lm_head = 0 if p.tie_word_embeddings else v * h
    attn = h * q_dim + 2 * h * kv + q_dim * h
    mlp = 3 * h * i          # gated (SwiGLU) MLP: gate + up + down
    norms = 2 * h
    return float(embed + lm_head + layers * (attn + mlp + norms) + h)


def _params_from_safetensors_index(model_dir: Path) -> Optional[float]:
    """Parameter count from a local safetensors index (total bytes / dtype width)."""
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        return None
    try:
        with open(index_file) as f:
            index = json.load(f)
        total_bytes = index.get("metadata", {}).get("total_size")
        if total_bytes:
            # Checkpoints are almost universally bf16/fp16 (2 bytes/param).
            return total_bytes / 2.0
    except (OSError, ValueError, KeyError) as e:
        logger.debug(f"Could not read safetensors index: {e}")
    return None


def _profile_from_local_path(model_path: str) -> Optional[ModelProfile]:
    model_dir = Path(model_path)
    config_file = model_dir / "config.json"
    if not config_file.exists():
        return None
    try:
        with open(config_file) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug(f"Could not parse {config_file}: {e}")
        return None

    profile = _profile_from_config_dict(model_path, cfg, source="local_config")
    exact = _params_from_safetensors_index(model_dir)
    if exact:
        profile.num_params = exact
        profile.params_exact = True
    return profile


def _profile_from_hub(model_id: str, hf_token: Optional[str]) -> Optional[ModelProfile]:
    """Fetch config.json (cache-aware) and safetensors param counts from the Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        config_path = hf_hub_download(
            model_id, "config.json", token=hf_token or None,
            etag_timeout=10.0,
        )
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception as e:
        logger.debug(f"Could not fetch config.json for {model_id}: {e}")
        return None

    profile = _profile_from_config_dict(model_id, cfg, source="hub_config")

    # Exact parameter count from safetensors metadata (best-effort).
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, token=hf_token or None, timeout=10.0)
        st = getattr(info, "safetensors", None)
        total = getattr(st, "total", None) if st else None
        if total:
            profile.num_params = float(total)
            profile.params_exact = True
    except Exception as e:
        logger.debug(f"Could not fetch safetensors metadata for {model_id}: {e}")

    return profile


def _profile_from_name(model_id: str) -> Optional[ModelProfile]:
    size_b = get_model_size_from_name(model_id)
    if size_b is None:
        return None
    # Closest representative architecture for this size class.
    preset_size, dims = min(_ARCH_PRESETS, key=lambda x: abs(x[0] - size_b))
    heads = dims["num_attention_heads"]
    profile = ModelProfile(
        model_id=model_id,
        num_params=size_b * 1e9,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=dims["num_key_value_heads"],
        head_dim=dims["hidden_size"] // heads,
        intermediate_size=dims["intermediate_size"],
        vocab_size=dims["vocab_size"],
        source="name_heuristic",
    )
    return profile


def get_model_profile(model_id: str, hf_token: Optional[str] = None,
                      use_cache: bool = True) -> Optional[ModelProfile]:
    """
    Build a ModelProfile from the most reliable available source:
    local config.json → Hub config.json (honors HF cache / offline mode) →
    name-pattern heuristic. Returns None when nothing can be determined.
    """
    if use_cache:
        with _profile_cache_lock:
            cached = _profile_cache.get(model_id)
        if cached is not None:
            return cached

    from .preflight_checks import is_local_model_path

    profile: Optional[ModelProfile] = None
    if is_local_model_path(model_id):
        profile = _profile_from_local_path(model_id)
    else:
        profile = _profile_from_hub(model_id, hf_token)

    if profile is None or not profile.has_dims:
        fallback = _profile_from_name(model_id)
        if fallback is not None:
            # Keep an exact param count if we managed to get one.
            if profile is not None and profile.num_params and profile.params_exact:
                fallback.num_params = profile.num_params
                fallback.params_exact = True
            profile = fallback

    if profile is not None and profile.num_params:
        with _profile_cache_lock:
            _profile_cache[model_id] = profile
        return profile
    return None


def clear_profile_cache() -> None:
    with _profile_cache_lock:
        _profile_cache.clear()


def _lora_param_count(p: ModelProfile, lora_r: int, target_modules: List[str]) -> float:
    """Adapter parameters: r × (in + out) per targeted projection per layer."""
    h = p.hidden_size
    i = p.intermediate_size
    q_dim = (p.num_attention_heads or 0) * (p.head_dim or 0) or h
    kv = p.kv_dim or h
    module_dims = {
        "q_proj": (h, q_dim),
        "k_proj": (h, kv),
        "v_proj": (h, kv),
        "o_proj": (q_dim, h),
        "gate_proj": (h, i),
        "up_proj": (h, i),
        "down_proj": (i, h),
    }
    per_layer = 0
    for module in target_modules or []:
        dims = module_dims.get(module)
        if dims:
            per_layer += lora_r * (dims[0] + dims[1])
    return float(per_layer * (p.num_hidden_layers or 0))


def _modules_to_save_param_count(p: ModelProfile, modules_to_save: List[str]) -> float:
    """Full-copy trainable modules (embed_tokens / lm_head)."""
    count = 0.0
    for module in modules_to_save or []:
        if module in ("embed_tokens", "lm_head"):
            count += float((p.vocab_size or 0) * (p.hidden_size or 0))
    return count


def estimate_vram(
    profile: ModelProfile,
    batch_size: int = 1,
    max_length: int = 2048,
    training_mode: str = "orpo",
    use_4bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 64,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
    gradient_checkpointing: bool = False,
    optimizer_type: str = "paged_adamw_8bit",
    attn_implementation: str = "auto",
    use_liger: bool = False,
) -> VRAMEstimate:
    """
    Estimate peak training VRAM for a model profile + training settings.

    Returns a per-component breakdown (GB) so callers can point at the
    dominant consumer instead of just printing a single scary number.
    """
    notes: List[str] = []
    mode = (training_mode or "orpo").lower()
    params = profile.num_params or 0.0
    h = profile.hidden_size or 4096
    layers = profile.num_hidden_layers or 32
    i_size = profile.intermediate_size or int(h * 3.5)
    vocab = profile.vocab_size or 128_256
    kv = profile.kv_dim or h
    heads = profile.num_attention_heads or 32

    embed_params = float(vocab * h)
    lm_head_params = 0.0 if profile.tie_word_embeddings else float(vocab * h)

    # Paired preference modes concatenate chosen+rejected → 2× forward batch.
    forward_batch = batch_size * (2 if mode in PAIRED_PREFERENCE_MODES else 1)
    if mode in PAIRED_PREFERENCE_MODES:
        notes.append(
            f"{mode.upper()} runs chosen+rejected together, doubling the "
            f"effective forward batch to {forward_batch}."
        )

    # ---- Model weights -------------------------------------------------
    if use_4bit:
        # bitsandbytes quantizes Linear layers only; embeddings stay bf16 and
        # lm_head is skipped by default.
        quantized = max(params - embed_params - lm_head_params, 0.0)
        weights_bytes = quantized * NF4_BYTES_PER_PARAM + (embed_params + lm_head_params) * 2.0
    else:
        weights_bytes = params * 2.0

    # ---- Reference model (DPO/KTO without LoRA) ------------------------
    ref_bytes = 0.0
    if mode in REFERENCE_MODEL_MODES and not use_lora:
        ref_bytes = weights_bytes
        notes.append(
            f"{mode.upper()} without LoRA loads a second frozen reference "
            "model; with LoRA the reference is free (adapters disabled)."
        )

    # ---- Trainable parameters, gradients, optimizer states -------------
    if use_lora:
        trainable = _lora_param_count(
            profile, lora_r, target_modules or DEFAULT_LORA_TARGET_MODULES
        )
        trainable += _modules_to_save_param_count(profile, modules_to_save or [])
        # PEFT keeps adapter weights in fp32: weights + grads at 4 bytes each.
        trainable_bytes = trainable * 4.0
        grad_bytes = trainable * 4.0
        if modules_to_save:
            notes.append(
                "modules_to_save trains full embedding/lm_head copies — far "
                "more trainable parameters than the LoRA adapters themselves."
            )
    else:
        trainable = params
        trainable_bytes = 0.0     # already counted under weights
        grad_bytes = trainable * 2.0
        notes.append("Full fine-tuning: gradients and optimizer states cover every parameter.")

    opt_bytes_per_param = OPTIMIZER_STATE_BYTES.get(
        (optimizer_type or "").lower(), DEFAULT_OPTIMIZER_STATE_BYTES
    )
    optimizer_bytes = trainable * opt_bytes_per_param

    # ---- Layer activations --------------------------------------------
    # Stored-for-backward elements per token per layer (bf16 = 2 bytes):
    # LN outputs, q/k/v projections, attention output, MLP gate/up/product.
    per_token_layer_bytes = 2.0 * (5 * h + 2 * kv + 3 * i_size)
    activation_bytes = forward_batch * max_length * per_token_layer_bytes * layers

    eager_attn = (attn_implementation or "auto").lower() == "eager"
    if eager_attn:
        # Materialized attention matrices: scores + softmax per head.
        attn_matrix_bytes = 2.0 * 2 * forward_batch * heads * (max_length ** 2) * layers
        activation_bytes += attn_matrix_bytes
        notes.append(
            "Eager attention materializes full attention matrices; "
            "flash_attention_2 or sdpa would remove this term."
        )

    if gradient_checkpointing:
        # Only block inputs are kept; one layer is re-materialized at a time.
        stored = 2.0 * forward_batch * max_length * h * layers
        one_layer = forward_batch * max_length * per_token_layer_bytes
        if eager_attn:
            one_layer += 2.0 * 2 * forward_batch * heads * (max_length ** 2)
        activation_bytes = stored + one_layer

    # ---- Logits / loss -------------------------------------------------
    # bf16 logits + fp32 upcast + log-softmax: the dominant term for
    # large-vocab models. Liger's fused chunked cross-entropy avoids
    # materializing the full logits tensor.
    logits_bytes = forward_batch * max_length * (2.0 * h + vocab * (2.0 + 4.0 + 4.0))
    if use_liger:
        logits_bytes *= 0.15

    # ---- Total ---------------------------------------------------------
    breakdown_bytes = {
        "model_weights": weights_bytes,
        "reference_model": ref_bytes,
        "trainable_params": trainable_bytes,
        "gradients": grad_bytes,
        "optimizer_states": optimizer_bytes,
        "activations": activation_bytes,
        "logits_and_loss": logits_bytes,
    }
    subtotal = sum(breakdown_bytes.values())
    breakdown_gb = {k: v / GB for k, v in breakdown_bytes.items() if v > 0}
    breakdown_gb["cuda_context"] = CUDA_CONTEXT_GB
    breakdown_gb["fragmentation_buffer"] = (subtotal / GB) * FRAGMENTATION_FACTOR

    total_gb = sum(breakdown_gb.values())

    if profile.source in ("local_config", "hub_config"):
        confidence = "high" if profile.params_exact else "medium"
    else:
        confidence = "low"
        notes.append(
            "Model architecture guessed from the model name — the real "
            "config.json could not be read."
        )

    return VRAMEstimate(
        total_gb=total_gb,
        breakdown=breakdown_gb,
        confidence=confidence,
        notes=notes,
        profile=profile,
    )


def suggest_reductions(estimate: VRAMEstimate, config: Any) -> List[str]:
    """
    Ordered, targeted suggestions for an over-budget estimate, based on
    which component actually dominates.
    """
    suggestions: List[str] = []
    b = estimate.breakdown
    dominant = estimate.dominant_component()

    if dominant == "activations" and not getattr(config, "gradient_checkpointing", False):
        suggestions.append(
            f"enable gradient_checkpointing (activations are the largest "
            f"consumer at ~{b.get('activations', 0):.1f}GB)"
        )
    if dominant == "logits_and_loss" and not getattr(config, "use_liger", False):
        suggestions.append(
            f"enable use_liger — fused cross-entropy avoids materializing the "
            f"full logits tensor (~{b.get('logits_and_loss', 0):.1f}GB for this "
            "vocab size)"
        )
    if not getattr(config, "use_4bit", True):
        suggestions.append("enable 4-bit quantization")
    if getattr(config, "batch_size", 1) > 1:
        suggestions.append(f"reduce batch_size (currently {config.batch_size})")
    if getattr(config, "max_length", 2048) > 1024:
        suggestions.append(f"reduce max_length (currently {config.max_length})")
    if not getattr(config, "gradient_checkpointing", False) and "enable gradient_checkpointing" not in " ".join(suggestions):
        suggestions.append("enable gradient_checkpointing")
    opt = (getattr(config, "optimizer_type", "") or "").lower()
    if b.get("optimizer_states", 0) > 1.0 and "8bit" not in opt:
        suggestions.append("switch to paged_adamw_8bit optimizer")
    if b.get("reference_model", 0) > 0:
        suggestions.append("enable LoRA so no separate reference model is loaded")
    if not suggestions:
        suggestions.append("use a smaller model")
    return suggestions
