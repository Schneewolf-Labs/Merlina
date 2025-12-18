"""
Merlina Dataset Module
Modular dataset loading, formatting, and validation for ORPO training
"""

from .base import DatasetLoader, DatasetFormatter, DatasetPipeline
from .loaders import HuggingFaceLoader, LocalFileLoader, UploadedDatasetLoader
from .formatters import (
    ChatMLFormatter,
    Llama3Formatter,
    MistralFormatter,
    Qwen3Formatter,
    CustomFormatter,
    TokenizerFormatter,
    get_formatter
)
from .validators import validate_dataset_schema, DatasetValidationError
from .factory import (
    create_loader,
    create_loader_from_config,
    create_pipeline_from_config,
    LoaderCreationError
)

__all__ = [
    'DatasetLoader',
    'DatasetFormatter',
    'DatasetPipeline',
    'HuggingFaceLoader',
    'LocalFileLoader',
    'UploadedDatasetLoader',
    'ChatMLFormatter',
    'Llama3Formatter',
    'MistralFormatter',
    'Qwen3Formatter',
    'CustomFormatter',
    'TokenizerFormatter',
    'get_formatter',
    'validate_dataset_schema',
    'DatasetValidationError',
    'create_loader',
    'create_loader_from_config',
    'create_pipeline_from_config',
    'LoaderCreationError',
]
