"""
Merlina Source Modules
Core functionality for magical LLM training
"""

from .job_manager import JobManager, JobRecord
from .websocket_manager import websocket_manager, WebSocketManager
from .preflight_checks import PreflightValidator, validate_config, ValidationError

# Note: training_runner imports transformers/peft which may have environment issues
# Import it explicitly when needed rather than at package level
# from .training_runner import run_training_sync, WebSocketCallback

__all__ = [
    'JobManager',
    'JobRecord',
    'websocket_manager',
    'WebSocketManager',
    'PreflightValidator',
    'validate_config',
    'ValidationError',
    # 'run_training_sync',
    # 'WebSocketCallback',
]

__version__ = '1.1.0'
