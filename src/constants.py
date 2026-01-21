"""
Constants and configuration values for Merlina training system.

This module centralizes hard-coded values that were previously scattered
throughout the codebase, making them easier to maintain and configure.
"""

from typing import Dict, List


# =============================================================================
# VRAM Estimates (in GB with 4-bit quantization)
# =============================================================================

# Estimated VRAM requirements for different model sizes (with 4-bit quantization)
# These are conservative estimates to prevent OOM errors
VRAM_ESTIMATES_4BIT: Dict[str, int] = {
    "3b": 6,
    "7b": 10,
    "8b": 10,
    "13b": 16,
    "14b": 18,
    "34b": 30,
    "70b": 60,
}

# Multiplier for full precision vs 4-bit quantization
FULL_PRECISION_VRAM_MULTIPLIER = 3.5


# =============================================================================
# Disk Space Estimates (in GB)
# =============================================================================

# Estimated disk space requirements for model checkpoints and outputs
DISK_SPACE_ESTIMATES: Dict[str, int] = {
    "3b": 10,
    "7b": 20,
    "8b": 20,
    "13b": 30,
    "14b": 30,
    "34b": 70,
    "70b": 140,
}

# Default disk space estimate when model size cannot be determined
DEFAULT_DISK_SPACE_ESTIMATE = 25


# =============================================================================
# Gated Models (require HuggingFace token)
# =============================================================================

# List of model prefixes that are gated and require authentication
GATED_MODEL_PREFIXES: List[str] = [
    "meta-llama/Llama-2",
    "meta-llama/Meta-Llama-3",
    "meta-llama/Llama-3",
    "mistralai/Mixtral",
    "mistralai/Mistral",
    "google/gemma",
]


# =============================================================================
# Job Queue Configuration
# =============================================================================

# Default maximum concurrent jobs (set to 1 for single GPU to prevent OOM)
DEFAULT_MAX_CONCURRENT_JOBS = 1

# Queue worker polling interval (seconds)
QUEUE_POLL_INTERVAL = 1.0

# Maximum queue size (0 = unlimited)
MAX_QUEUE_SIZE = 0


# =============================================================================
# Database Configuration
# =============================================================================

# SQLite connection timeout (seconds)
DATABASE_TIMEOUT = 30.0

# Busy timeout for SQLite (milliseconds)
DATABASE_BUSY_TIMEOUT_MS = 30000


# =============================================================================
# Training Defaults
# =============================================================================

# Default dataset format when not specified
DEFAULT_FORMAT_TYPE = "tokenizer"

# Default test split size
DEFAULT_TEST_SIZE = 0.01

# Maximum number of processes for dataset mapping
MAX_DATASET_MAP_PROCESSES = 4

# Default random seed for reproducibility
DEFAULT_SEED = 42


# =============================================================================
# Training Parameter Validation Thresholds
# =============================================================================

# LoRA rank warnings
MAX_RECOMMENDED_LORA_RANK = 256

# Effective batch size warnings
MIN_EFFECTIVE_BATCH_SIZE = 4
MAX_EFFECTIVE_BATCH_SIZE = 128

# Learning rate warnings
MAX_RECOMMENDED_LEARNING_RATE = 1e-4


# =============================================================================
# GPU / CUDA Configuration
# =============================================================================

# Minimum compute capability for Flash Attention 2
FLASH_ATTENTION_MIN_COMPUTE_CAP = 8


# =============================================================================
# WebSocket Configuration
# =============================================================================

# Timeout for WebSocket message sending (seconds)
WEBSOCKET_SEND_TIMEOUT = 5.0


# =============================================================================
# File Upload Configuration
# =============================================================================

# Supported file formats for dataset uploads
SUPPORTED_UPLOAD_FORMATS: Dict[str, str] = {
    '.json': 'json',
    '.jsonl': 'json',
    '.csv': 'csv',
    '.parquet': 'parquet',
    '.pq': 'parquet'
}


# =============================================================================
# HuggingFace Configuration
# =============================================================================

# Default repository visibility for uploads
DEFAULT_HF_PRIVATE = True


# =============================================================================
# Weights & Biases Configuration
# =============================================================================

# Default W&B project name
DEFAULT_WANDB_PROJECT = "merlina-training"


# =============================================================================
# Helper Functions
# =============================================================================

def get_vram_estimate(model_name: str, use_4bit: bool = True) -> int:
    """
    Get estimated VRAM requirement for a model.

    Args:
        model_name: Model name or path (case-insensitive matching)
        use_4bit: Whether 4-bit quantization is enabled

    Returns:
        Estimated VRAM in GB, or None if cannot be determined
    """
    model_lower = model_name.lower()

    for size_key, vram in VRAM_ESTIMATES_4BIT.items():
        if size_key in model_lower:
            if use_4bit:
                return vram
            return int(vram * FULL_PRECISION_VRAM_MULTIPLIER)

    return None


def get_disk_space_estimate(model_name: str) -> int:
    """
    Get estimated disk space requirement for model checkpoints.

    Args:
        model_name: Model name or path (case-insensitive matching)

    Returns:
        Estimated disk space in GB
    """
    model_lower = model_name.lower()

    for size_key, space in DISK_SPACE_ESTIMATES.items():
        if size_key in model_lower:
            return space

    return DEFAULT_DISK_SPACE_ESTIMATE


def is_gated_model(model_name: str) -> bool:
    """
    Check if a model is gated and requires authentication.

    Args:
        model_name: Model name or HuggingFace repo ID

    Returns:
        True if the model is gated
    """
    return any(prefix in model_name for prefix in GATED_MODEL_PREFIXES)
