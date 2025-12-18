"""
Factory functions for creating dataset loaders and pipelines.

This module centralizes the creation of loaders to eliminate code duplication
across merlina.py and training_runner.py.
"""

from typing import Dict, Optional, Tuple, Any
import logging

from .base import DatasetLoader, DatasetPipeline
from .loaders import HuggingFaceLoader, LocalFileLoader, UploadedDatasetLoader
from .formatters import get_formatter

logger = logging.getLogger(__name__)


class LoaderCreationError(Exception):
    """Raised when loader creation fails."""
    pass


def create_loader(
    source_type: str,
    repo_id: Optional[str] = None,
    split: str = "train",
    file_path: Optional[str] = None,
    file_format: Optional[str] = None,
    dataset_id: Optional[str] = None,
    uploaded_datasets: Optional[Dict[str, Tuple[bytes, str]]] = None,
    hf_token: Optional[str] = None
) -> DatasetLoader:
    """
    Create a dataset loader based on source type.

    Args:
        source_type: Type of source ("huggingface", "local_file", or "upload")
        repo_id: HuggingFace repository ID (required for huggingface source)
        split: Dataset split to load (default: "train")
        file_path: Path to local file (required for local_file source)
        file_format: File format (json, csv, parquet)
        dataset_id: ID of uploaded dataset (required for upload source)
        uploaded_datasets: Dict mapping dataset_id to (content, filename) tuples
        hf_token: HuggingFace API token for private datasets

    Returns:
        Configured DatasetLoader instance

    Raises:
        LoaderCreationError: If source_type is invalid or required params missing
    """
    if source_type == "huggingface":
        if not repo_id:
            raise LoaderCreationError("repo_id is required for huggingface source")
        return HuggingFaceLoader(
            repo_id=repo_id,
            split=split,
            token=hf_token
        )

    elif source_type == "local_file":
        if not file_path:
            raise LoaderCreationError("file_path is required for local_file source")
        return LocalFileLoader(
            file_path=file_path,
            file_format=file_format
        )

    elif source_type == "upload":
        if not dataset_id:
            raise LoaderCreationError("dataset_id is required for upload source")
        if uploaded_datasets is None or dataset_id not in uploaded_datasets:
            raise LoaderCreationError(f"Uploaded dataset '{dataset_id}' not found")

        file_content, filename = uploaded_datasets[dataset_id]
        return UploadedDatasetLoader(
            file_content=file_content,
            filename=filename,
            file_format=file_format
        )

    else:
        raise LoaderCreationError(f"Invalid source_type: {source_type}")


def create_loader_from_config(
    source_config: Any,
    uploaded_datasets: Optional[Dict[str, Tuple[bytes, str]]] = None,
    hf_token: Optional[str] = None
) -> DatasetLoader:
    """
    Create a dataset loader from a source configuration object.

    This is a convenience wrapper around create_loader() that accepts
    a Pydantic model or dict-like config object.

    Args:
        source_config: Configuration object with source_type and related fields
        uploaded_datasets: Dict mapping dataset_id to (content, filename) tuples
        hf_token: HuggingFace API token for private datasets

    Returns:
        Configured DatasetLoader instance

    Raises:
        LoaderCreationError: If configuration is invalid
    """
    # Handle both Pydantic models and dicts
    if hasattr(source_config, 'model_dump'):
        config_dict = source_config.model_dump()
    elif hasattr(source_config, 'dict'):
        config_dict = source_config.dict()
    elif isinstance(source_config, dict):
        config_dict = source_config
    else:
        # Assume it's an object with attributes
        config_dict = {
            'source_type': getattr(source_config, 'source_type', None),
            'repo_id': getattr(source_config, 'repo_id', None),
            'split': getattr(source_config, 'split', 'train'),
            'file_path': getattr(source_config, 'file_path', None),
            'file_format': getattr(source_config, 'file_format', None),
            'dataset_id': getattr(source_config, 'dataset_id', None),
        }

    return create_loader(
        source_type=config_dict.get('source_type'),
        repo_id=config_dict.get('repo_id'),
        split=config_dict.get('split', 'train'),
        file_path=config_dict.get('file_path'),
        file_format=config_dict.get('file_format'),
        dataset_id=config_dict.get('dataset_id'),
        uploaded_datasets=uploaded_datasets,
        hf_token=hf_token
    )


def create_pipeline_from_config(
    dataset_config: Any,
    uploaded_datasets: Optional[Dict[str, Tuple[bytes, str]]] = None,
    hf_token: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    seed: int = 42,
    shuffle: bool = True
) -> DatasetPipeline:
    """
    Create a complete dataset pipeline from a dataset configuration.

    Args:
        dataset_config: Dataset configuration with source, format, and other settings
        uploaded_datasets: Dict mapping dataset_id to (content, filename) tuples
        hf_token: HuggingFace API token for private datasets
        tokenizer: Tokenizer instance (required for 'tokenizer' format type)
        seed: Random seed for shuffling/splitting
        shuffle: Whether to shuffle the dataset

    Returns:
        Configured DatasetPipeline instance

    Raises:
        LoaderCreationError: If configuration is invalid
    """
    # Create loader
    loader = create_loader_from_config(
        source_config=dataset_config.source,
        uploaded_datasets=uploaded_datasets,
        hf_token=hf_token
    )

    # Get format configuration
    format_config = dataset_config.format
    format_type = getattr(format_config, 'format_type', 'chatml')
    custom_templates = getattr(format_config, 'custom_templates', None)
    enable_thinking = getattr(format_config, 'enable_thinking', False)

    # Create formatter
    formatter = get_formatter(
        format_type=format_type,
        custom_templates=custom_templates,
        tokenizer=tokenizer if format_type == 'tokenizer' else None,
        enable_thinking=enable_thinking
    )

    # Get pipeline settings
    column_mapping = getattr(dataset_config, 'column_mapping', None)
    test_size = getattr(dataset_config, 'test_size', 0.1)
    max_samples = getattr(dataset_config, 'max_samples', None)
    training_mode = getattr(dataset_config, 'training_mode', 'orpo')

    # Create pipeline
    return DatasetPipeline(
        loader=loader,
        formatter=formatter,
        column_mapping=column_mapping,
        test_size=test_size,
        max_samples=max_samples,
        seed=seed,
        shuffle=shuffle,
        training_mode=training_mode
    )
