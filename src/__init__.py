"""
Merlina Source Modules
Core functionality for magical LLM training
"""

from .job_manager import JobManager, JobRecord

try:
    from .websocket_manager import websocket_manager, WebSocketManager
except ImportError:
    # websocket_manager requires fastapi which may not be available in
    # subprocess workers (e.g. train_worker.py launched via accelerate).
    websocket_manager = None
    WebSocketManager = None

# GPU-dependent modules load only when their public exports are requested.
# This keeps remote services and the hosted API importable on CPU-only hosts.
from importlib import import_module

_LAZY_EXPORTS = {
    'PreflightValidator': 'preflight_checks', 'validate_config': 'preflight_checks',
    'ValidationError': 'preflight_checks',
    'resolve_llama_cpp': 'llama_cpp_resolver', 'LlamaCppResolution': 'llama_cpp_resolver',
    'get_num_gpus': 'utils', 'calculate_effective_batch_size': 'utils',
    'get_gpu_memory_gb': 'utils', 'get_current_gpu_memory_usage_gb': 'utils',
    'get_torch_dtype': 'utils', 'supports_flash_attention': 'utils',
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f'.{module}', __name__), name)
    globals()[name] = value
    return value


from .constants import (
    VRAM_ESTIMATES_4BIT,
    DISK_SPACE_ESTIMATES,
    GATED_MODEL_PREFIXES,
    DEFAULT_MAX_CONCURRENT_JOBS,
    DEFAULT_WANDB_PROJECT,
    get_vram_estimate,
    get_disk_space_estimate,
    is_gated_model,
)

# Custom exceptions for better error handling
from .exceptions import (
    MerlinaError,
    DatasetError,
    DatasetNotFoundError,
    DatasetLoadError,
    DatasetValidationError,
    ModelError,
    ModelNotFoundError,
    ModelAccessError,
    TrainingError,
    TrainingConfigError,
    JobError,
    JobNotFoundError,
)

# Note: training_runner imports transformers/peft which may have environment issues
# Import it explicitly when needed rather than at package level
# from .training_runner import run_training_sync, WebSocketCallback

__all__ = [
    # Job management
    'JobManager',
    'JobRecord',
    # WebSocket
    'websocket_manager',
    'WebSocketManager',
    # Validation
    'PreflightValidator',
    'validate_config',
    'ValidationError',
    # llama.cpp integration
    'resolve_llama_cpp',
    'LlamaCppResolution',
    # Utilities
    'get_num_gpus',
    'calculate_effective_batch_size',
    'get_gpu_memory_gb',
    'get_current_gpu_memory_usage_gb',
    'get_torch_dtype',
    'supports_flash_attention',
    # Constants
    'VRAM_ESTIMATES_4BIT',
    'DISK_SPACE_ESTIMATES',
    'GATED_MODEL_PREFIXES',
    'DEFAULT_MAX_CONCURRENT_JOBS',
    'DEFAULT_WANDB_PROJECT',
    'get_vram_estimate',
    'get_disk_space_estimate',
    'is_gated_model',
    # Exceptions
    'MerlinaError',
    'DatasetError',
    'DatasetNotFoundError',
    'DatasetLoadError',
    'DatasetValidationError',
    'ModelError',
    'ModelNotFoundError',
    'ModelAccessError',
    'TrainingError',
    'TrainingConfigError',
    'JobError',
    'JobNotFoundError',
]

from version import __version__
