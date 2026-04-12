"""
Shared utility functions for Merlina training system.

This module consolidates common utilities used across multiple modules
to avoid code duplication and ensure consistency.
"""

import json
import os
import re
import torch
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


def get_num_gpus() -> int:
    """
    Get the number of GPUs available for training.

    Returns:
        Number of available GPUs, minimum of 1 for calculations
        that require at least one device.
    """
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def calculate_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: Optional[int] = None
) -> int:
    """
    Calculate the effective batch size accounting for gradient accumulation and multiple GPUs.

    The effective batch size determines the actual number of samples used for
    each parameter update, which is important for training dynamics.

    Formula: effective_batch_size = per_device_batch_size × gradient_accumulation_steps × num_gpus

    Args:
        batch_size: Per-device batch size (samples per forward pass per GPU)
        gradient_accumulation_steps: Number of steps to accumulate gradients before update
        num_gpus: Number of GPUs (defaults to auto-detection via get_num_gpus())

    Returns:
        The effective batch size for training

    Examples:
        >>> calculate_effective_batch_size(4, 8, 1)  # 4 * 8 * 1 = 32
        32
        >>> calculate_effective_batch_size(2, 4, 2)  # 2 * 4 * 2 = 16
        16
    """
    if num_gpus is None:
        num_gpus = get_num_gpus()
    return batch_size * gradient_accumulation_steps * num_gpus


def get_gpu_memory_gb(device_id: int = 0) -> Optional[float]:
    """
    Get total GPU memory in gigabytes for a specific device.

    Args:
        device_id: CUDA device index (default: 0)

    Returns:
        Total GPU memory in GB, or None if CUDA is not available
    """
    if not torch.cuda.is_available():
        return None

    try:
        props = torch.cuda.get_device_properties(device_id)
        return props.total_memory / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Failed to get GPU memory for device {device_id}: {e}")
        return None


def get_current_gpu_memory_usage_gb(device_id: int = 0) -> Optional[float]:
    """
    Get current GPU memory usage in gigabytes.

    Args:
        device_id: CUDA device index (default: 0)

    Returns:
        Current GPU memory usage in GB, or None if CUDA is not available
    """
    if not torch.cuda.is_available():
        return None

    try:
        return torch.cuda.max_memory_allocated(device_id) / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Failed to get GPU memory usage for device {device_id}: {e}")
        return None


def get_torch_dtype():
    """
    Determine the optimal torch dtype based on GPU capability.

    Returns bfloat16 for Ampere+ GPUs (compute capability >= 8),
    otherwise returns float16.

    Returns:
        torch.dtype: Either torch.bfloat16 or torch.float16
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def supports_flash_attention() -> bool:
    """
    Check if the current GPU supports Flash Attention 2.

    Flash Attention 2 requires compute capability >= 8 (Ampere or newer).

    Returns:
        True if Flash Attention 2 is supported
    """
    if not torch.cuda.is_available():
        return False

    compute_cap = torch.cuda.get_device_capability()[0]
    return compute_cap >= 8


# ---------------------------------------------------------------------------
# VLM state-dict key validation / fixing
# ---------------------------------------------------------------------------

_NESTED_LM_RE = re.compile(r"(\.language_model){2,}")


def _fix_key(key: str) -> str:
    """Collapse repeated .language_model nesting and fix misplaced visual prefix."""
    key = _NESTED_LM_RE.sub(".language_model", key)
    if ".language_model.visual." in key:
        key = key.replace(".language_model.visual.", ".visual.", 1)
    return key


def fix_vlm_state_dict_on_disk(
    model_dir: str,
    base_model_name: str | None = None,
    *,
    is_vlm: bool = True,
) -> bool:
    """Validate saved safetensors for VLM key-naming bugs and fix in-place.

    After a PEFT merge-and-unload on VLM architectures (e.g. Qwen3.5-VL),
    save_pretrained can produce:
      - Triple-nested language_model keys
      - Visual keys under model.language_model.visual instead of model.visual
      - Missing multimodal weights (visual encoder, MTP head)

    This function detects those problems and fixes the safetensors files on
    disk so they match the base model's key layout.

    Args:
        model_dir:  Path to the saved model directory (contains safetensors).
        base_model_name:  HF repo id or local path of the base model.
                          Used to graft missing visual/MTP weights.
        is_vlm:  If False, skips all checks (fast no-op for text-only models).

    Returns:
        True if any fixes were applied, False if the model was already correct.
    """
    from safetensors.torch import load_file, save_file

    if not is_vlm:
        return False

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_shard = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        all_keys = set(index["weight_map"].keys())
        sharded = True
    elif os.path.exists(single_shard):
        tensors = load_file(single_shard)
        all_keys = set(tensors.keys())
        sharded = False
    else:
        return False

    # --- Detect problems ---
    has_nesting = any(_NESTED_LM_RE.search(k) for k in all_keys)
    has_misplaced_visual = any(".language_model.visual." in k for k in all_keys)
    needs_key_fix = has_nesting or has_misplaced_visual

    # Check for missing keys by comparing against base model index
    missing_keys: dict[str, str] = {}  # key -> base shard file
    if base_model_name:
        try:
            from huggingface_hub import hf_hub_download
            base_index_path = hf_hub_download(
                base_model_name,
                filename="model.safetensors.index.json",
                local_dir=os.path.join(model_dir, "_base_index"),
            )
            with open(base_index_path) as f:
                base_index = json.load(f)
            # Build expected keys after fixing
            fixed_keys = {_fix_key(k) for k in all_keys} if needs_key_fix else all_keys
            for base_key, base_shard in base_index["weight_map"].items():
                if base_key not in fixed_keys:
                    missing_keys[base_key] = base_shard
        except Exception as e:
            logger.warning(f"Could not check base model for missing keys: {e}")

    if not needs_key_fix and not missing_keys:
        return False

    logger.info(f"VLM state dict issues detected — fixing in-place")
    if needs_key_fix:
        logger.info(f"  Key nesting: {has_nesting}, misplaced visual: {has_misplaced_visual}")
    if missing_keys:
        logger.info(f"  Missing keys from base model: {len(missing_keys)}")

    # --- Fix keys in safetensors ---
    if sharded:
        _fix_sharded(model_dir, index, needs_key_fix, missing_keys, base_model_name)
    else:
        _fix_single(model_dir, tensors, needs_key_fix, missing_keys, base_model_name)

    # Clean up temp base index
    import shutil
    base_idx_dir = os.path.join(model_dir, "_base_index")
    if os.path.isdir(base_idx_dir):
        shutil.rmtree(base_idx_dir, ignore_errors=True)

    return True


def _fix_sharded(model_dir, index, needs_key_fix, missing_keys, base_model_name):
    """Fix key names across sharded safetensors and graft missing keys."""
    from safetensors.torch import load_file, save_file

    # Fix existing shards
    if needs_key_fix:
        new_weight_map = {}
        shard_files = set(index["weight_map"].values())
        for shard_name in shard_files:
            shard_path = os.path.join(model_dir, shard_name)
            tensors = load_file(shard_path)
            fixed = OrderedDict()
            for key, tensor in tensors.items():
                fixed[_fix_key(key)] = tensor
            save_file(fixed, shard_path)
            for new_key in fixed:
                new_weight_map[new_key] = shard_name
            del tensors, fixed
        index["weight_map"] = new_weight_map

    # Graft missing keys from base model
    if missing_keys and base_model_name:
        grafted = _graft_missing_keys(model_dir, missing_keys, base_model_name)
        # Append grafted keys to the last shard
        if grafted:
            shard_names = sorted(set(index["weight_map"].values()))
            last_shard_name = shard_names[-1]
            last_shard_path = os.path.join(model_dir, last_shard_name)
            existing = load_file(last_shard_path)
            merged = OrderedDict(sorted({**existing, **grafted}.items()))
            save_file(merged, last_shard_path)
            for key in grafted:
                index["weight_map"][key] = last_shard_name
            # Update total size
            extra_size = sum(t.nelement() * t.element_size() for t in grafted.values())
            index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + extra_size
            del existing, merged

    # Write updated index
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _fix_single(model_dir, tensors, needs_key_fix, missing_keys, base_model_name):
    """Fix key names in a single safetensors file and graft missing keys."""
    from safetensors.torch import save_file

    fixed = OrderedDict()
    if needs_key_fix:
        for key, tensor in tensors.items():
            fixed[_fix_key(key)] = tensor
    else:
        fixed = OrderedDict(tensors)

    if missing_keys and base_model_name:
        grafted = _graft_missing_keys(model_dir, missing_keys, base_model_name)
        fixed.update(grafted)

    save_file(OrderedDict(sorted(fixed.items())), os.path.join(model_dir, "model.safetensors"))


def _graft_missing_keys(model_dir, missing_keys, base_model_name):
    """Download and return missing tensors from the base model."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    grafted = {}
    # Group missing keys by their source shard
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in missing_keys.items():
        shard_to_keys.setdefault(shard, []).append(key)

    cache_dir = os.path.join(model_dir, "_base_shards")
    for shard_name, keys in shard_to_keys.items():
        try:
            shard_path = hf_hub_download(
                base_model_name,
                filename=shard_name,
                local_dir=cache_dir,
            )
            shard_tensors = load_file(shard_path)
            for key in keys:
                if key in shard_tensors:
                    grafted[key] = shard_tensors[key]
                    logger.info(f"  Grafted {key} from base model")
                else:
                    logger.warning(f"  Key {key} not found in base shard {shard_name}")
            del shard_tensors
        except Exception as e:
            logger.warning(f"  Failed to download base shard {shard_name}: {e}")

    # Clean up downloaded shards
    import shutil
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    return grafted
