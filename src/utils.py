"""
Shared utility functions for Merlina training system.

This module consolidates common utilities used across multiple modules
to avoid code duplication and ensure consistency.
"""

import torch
import logging
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
