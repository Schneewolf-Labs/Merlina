"""
Merlina Source Modules
Core functionality for magical LLM training
"""

from .job_manager import JobManager, JobRecord
from .websocket_manager import websocket_manager, WebSocketManager
from .preflight_checks import PreflightValidator, validate_config, ValidationError

# Utility functions and constants
from .utils import (
    get_num_gpus,
    calculate_effective_batch_size,
    get_gpu_memory_gb,
    get_current_gpu_memory_usage_gb,
    get_torch_dtype,
    supports_flash_attention,
)

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

__version__ = '1.1.0'
