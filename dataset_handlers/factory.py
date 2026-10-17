"""
Factory functions for creating dataset loaders and pipelines.

This module centralizes the creation of loaders to eliminate code duplication
across merlina.py and training_runner.py.
"""

from typing import Dict, Optional, Tuple, Any
import logging

from .base import DatasetLoader, DatasetPipeline
from .loaders import (
    HuggingFaceLoader,
    StreamingHuggingFaceLoader,
    InternalStoreLoader,
    MirroredHuggingFaceLoader,
    LocalFileLoader,
    UploadedDatasetLoader,
)
from .internal_store import InternalStore, InternalStoreConfig
from .formatters import get_formatter

logger = logging.getLogger(__name__)

# Which tier each store name reads its settings from.
_STORE_PREFIX = {"internal": "internal", "external": "external"}

# Per-backend cache. ``False`` means "built and found unconfigured" (distinct
# from a missing key = "not built yet"), so an unconfigured backend is not
# re-probed on every load. Keyed by store name ("internal" / "external").
_STORE_CACHE: "dict[str, InternalStore | bool]" = {}


def _reset_store_cache() -> None:
    """Clear the per-backend store cache (used by tests)."""
    _STORE_CACHE.clear()


def get_store(which: str) -> Optional[InternalStore]:
    """
    Build the named store backend ("internal" = MinIO, "external" = R2) from app
    settings, or return None when that backend is not configured. Cached. The
    import of ``config`` is lazy so dataset_handlers stays importable without the
    application config present (e.g. in isolated unit tests).
    """
    which = (which or "internal").lower()
    if which not in _STORE_CACHE:
        prefix = _STORE_PREFIX.get(which, "s3")
        try:
            from config import settings  # lazy: app config, not a package dep
            cfg = InternalStoreConfig.from_settings(settings, prefix=prefix)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"'{which}' dataset store not available: {e}")
            cfg = None
        _STORE_CACHE[which] = InternalStore(cfg) if cfg else False
    return _STORE_CACHE[which] or None


def active_store_name() -> str:
    """The store this process resolves swl:// against (``DATASET_TIER``)."""
    try:
        from config import settings  # lazy: app config, not a package dep
        return str(getattr(settings, "dataset_tier", None)
                   or getattr(settings, "dataset_store", None) or "internal").lower()
    except Exception:  # pragma: no cover - defensive
        return "internal"


def default_internal_store() -> Optional[InternalStore]:
    """
    The store for the active backend (``DATASET_TIER``: "internal" by default,
    "external" on a rented GPU box), or None when it isn't configured.
    """
    return get_store(active_store_name())


def default_prefer_internal() -> bool:
    """Whether huggingface sources should be served through the internal-store
    mirror by default (``DATASET_MIRROR`` in app settings). False if unset."""
    try:
        from config import settings  # lazy: app config, not a package dep
        value = getattr(settings, "dataset_mirror", None)
        if value is None:
            value = getattr(settings, "prefer_internal", None)
        return bool(value)
    except Exception:  # pragma: no cover - defensive
        return False


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
    hf_token: Optional[str] = None,
    streaming: bool = False,
    streaming_batch_size: int = 10000,
    config_name: Optional[str] = None,
    uri: Optional[str] = None,
    store: Optional[InternalStore] = None,
    max_samples: Optional[int] = None,
    prefer_internal: bool = False,
) -> DatasetLoader:
    """
    Create a dataset loader based on source type.

    Args:
        source_type: Type of source ("huggingface", "internal", "local_file", or "upload")
        repo_id: HuggingFace repository ID (required for huggingface source)
        split: Dataset split to load (default: "train")
        file_path: Path to local file (required for local_file source)
        file_format: File format (json, csv, parquet)
        dataset_id: ID of uploaded dataset (required for upload source)
        uploaded_datasets: Dict mapping dataset_id to (content, filename) tuples
        hf_token: HuggingFace API token for private datasets
        streaming: Use streaming mode for HuggingFace datasets (for large datasets)
        streaming_batch_size: Batch size for streaming materialization (default: 10000)
        config_name: HuggingFace dataset configuration / subset name (optional)
        uri: swl:// address (required for internal source)
        store: Configured InternalStore (required for internal source)
        max_samples: Optional row cap (used by the internal source)

    Returns:
        Configured DatasetLoader instance

    Raises:
        LoaderCreationError: If source_type is invalid or required params missing
    """
    if source_type == "internal":
        if not uri:
            raise LoaderCreationError("uri is required for internal source")
        if store is None:
            raise LoaderCreationError(
                "The internal dataset store is not configured "
                "(set STORAGE_INTERNAL_ENDPOINT / _BUCKET / _ACCESS_KEY / _SECRET_KEY)."
            )
        return InternalStoreLoader(
            uri=uri,
            store=store,
            split=split,
            max_samples=max_samples,
            streaming=streaming,
            batch_size=streaming_batch_size,
        )

    if source_type == "huggingface":
        if not repo_id:
            raise LoaderCreationError("repo_id is required for huggingface source")
        # Mirror-on-first-use: serve HF through the internal store when preferred
        # and a store is available. Streaming sources are left direct — a mirror
        # materializes the split, which a streaming loader is specifically avoiding.
        if prefer_internal and store is not None and not streaming:
            return MirroredHuggingFaceLoader(
                repo_id=repo_id,
                store=store,
                split=split,
                config_name=config_name,
                token=hf_token,
                max_samples=max_samples,
            )
        if prefer_internal and store is not None and streaming:
            # Stream from an existing mirror; never force a full snapshot just
            # to stream a slice of it (that's what streaming avoids).
            from .mirror import mirror_name
            from .internal_store import LATEST
            name = mirror_name(repo_id, config_name, split)
            try:
                mirrored = store.exists(name, LATEST)
            except Exception:
                mirrored = False
            if mirrored:
                return InternalStoreLoader(
                    uri=f"swl://{name}", store=store, split="train",
                    max_samples=max_samples, streaming=True, batch_size=streaming_batch_size,
                )
        if streaming:
            return StreamingHuggingFaceLoader(
                repo_id=repo_id,
                split=split,
                token=hf_token,
                batch_size=streaming_batch_size,
                config_name=config_name,
            )
        return HuggingFaceLoader(
            repo_id=repo_id,
            split=split,
            token=hf_token,
            config_name=config_name,
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

        # Handle both old tuple format (content, filename) and new dict format
        dataset_entry = uploaded_datasets[dataset_id]
        if isinstance(dataset_entry, dict):
            file_content = dataset_entry["content"]
            filename = dataset_entry["filename"]
        else:
            # Legacy tuple format for backwards compatibility
            file_content, filename = dataset_entry

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
    hf_token: Optional[str] = None,
    store: Optional[InternalStore] = None,
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
            'streaming': getattr(source_config, 'streaming', False),
            'streaming_batch_size': getattr(source_config, 'streaming_batch_size', 10000),
            'config_name': getattr(source_config, 'config_name', None),
            'uri': getattr(source_config, 'uri', None),
            'max_samples': getattr(source_config, 'max_samples', None),
        }

    # An swl:// repo_id is a convenient shorthand for an internal source, so a
    # config can point at the store without restating source_type.
    source_type = config_dict.get('source_type')
    uri = config_dict.get('uri')
    repo_id = config_dict.get('repo_id')
    if not source_type and isinstance(repo_id, str) and repo_id.startswith('swl://'):
        source_type = 'internal'
    if source_type == 'internal' and not uri and isinstance(repo_id, str):
        uri = repo_id

    # Fall back to the store configured in app settings so callers that don't
    # thread one through still resolve internal sources.
    if source_type == 'internal' and store is None:
        store = default_internal_store()

    # For huggingface sources, honor the mirror-on-first-use preference: when
    # enabled and a store is available, serve (and cache) through the store.
    prefer_internal = bool(config_dict.get('prefer_internal')) or default_prefer_internal()
    if source_type == 'huggingface' and prefer_internal and store is None:
        store = default_internal_store()

    return create_loader(
        source_type=source_type,
        repo_id=repo_id,
        split=config_dict.get('split', 'train'),
        file_path=config_dict.get('file_path'),
        file_format=config_dict.get('file_format'),
        dataset_id=config_dict.get('dataset_id'),
        uploaded_datasets=uploaded_datasets,
        hf_token=hf_token,
        streaming=config_dict.get('streaming', False),
        streaming_batch_size=config_dict.get('streaming_batch_size', 10000),
        config_name=config_dict.get('config_name'),
        uri=uri,
        store=store,
        max_samples=config_dict.get('max_samples'),
        prefer_internal=prefer_internal,
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
        training_mode=training_mode,
        # Forwarded regardless of format_type: chatml is the default, and a tool-calling dataset
        # still needs the model's template to render calls faithfully.
        tokenizer=tokenizer,
    )


def load_swl_dataset(uri: str, max_samples=None):
    """Load an ``swl://name[@rev]`` dataset from the active store as a regular
    Dataset. For runners that load datasets themselves (VLM, diffusion)
    rather than through :func:`create_loader_from_config`."""
    store = default_internal_store()
    if store is None:
        raise LoaderCreationError(
            f"{uri}: the internal dataset store is not configured "
            "(set STORAGE_INTERNAL_ENDPOINT / _BUCKET / _ACCESS_KEY / _SECRET_KEY).")
    return InternalStoreLoader(uri=uri, store=store, split="train", max_samples=max_samples).load()
