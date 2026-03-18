"""
Base classes for dataset handling
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Callable, List
from datasets import Dataset, concatenate_datasets
import logging
from .messages_converter import has_messages_format, convert_messages_dataset

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """Abstract base class for dataset loading strategies"""

    @abstractmethod
    def load(self) -> Dataset:
        """Load dataset from source"""
        pass

    @abstractmethod
    def get_source_info(self) -> dict:
        """Get information about the dataset source"""
        pass


class DatasetFormatter(ABC):
    """Abstract base class for dataset formatting strategies"""

    @abstractmethod
    def format(self, row: dict) -> dict:
        """
        Transform a dataset row to the expected format.

        Args:
            row: Dictionary with keys like 'system', 'prompt', 'chosen', 'rejected'

        Returns:
            Dictionary with keys 'prompt', 'chosen', 'rejected' in final format
        """
        pass

    @abstractmethod
    def get_format_info(self) -> dict:
        """Get information about the format type"""
        pass


class DatasetPipeline:
    """
    Orchestrates dataset loading, validation, formatting, and splitting.
    This is the main interface for preparing datasets for training.
    """

    def __init__(
        self,
        loader: DatasetLoader,
        formatter: DatasetFormatter,
        column_mapping: Optional[dict] = None,
        test_size: float = 0.01,
        max_samples: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = True,
        training_mode: str = "orpo",
        convert_messages_format: bool = True,
        additional_loaders: Optional[List[DatasetLoader]] = None,
        additional_column_mappings: Optional[List[Optional[dict]]] = None,
        additional_convert_messages: Optional[List[bool]] = None
    ):
        """
        Initialize dataset pipeline.

        Args:
            loader: DatasetLoader instance to load the primary dataset
            formatter: DatasetFormatter instance to format rows
            column_mapping: Map dataset columns to expected names
                           e.g., {'input': 'prompt', 'output_good': 'chosen'}
            test_size: Fraction of data to use for evaluation
            max_samples: Optional limit on number of samples (for testing)
            seed: Random seed for train/test split
            shuffle: Whether to shuffle the dataset before splitting
            training_mode: Training mode ('sft' or 'orpo'). For SFT, rejected is optional.
            convert_messages_format: Whether to automatically detect and convert messages format
            additional_loaders: Optional list of additional DatasetLoader instances to concatenate
            additional_column_mappings: Optional list of column mappings for additional datasets
            additional_convert_messages: Optional list of convert_messages_format flags for additional datasets
        """
        self.loader = loader
        self.formatter = formatter
        self.column_mapping = column_mapping or {}
        self.test_size = test_size
        self.max_samples = max_samples
        self.seed = seed
        self.shuffle = shuffle
        self.training_mode = training_mode
        self.convert_messages_format = convert_messages_format
        self.additional_loaders = additional_loaders or []
        self.additional_column_mappings = additional_column_mappings or []
        self.additional_convert_messages = additional_convert_messages or []

    def _load_single_dataset(self, loader: DatasetLoader, column_mapping: Optional[dict],
                              convert_messages: bool) -> Dataset:
        """Load a single dataset, apply messages conversion and column mapping."""
        dataset = loader.load()
        logger.info(f"Loaded {len(dataset)} samples from {loader.get_source_info()}")

        # Convert messages format if detected and enabled
        if convert_messages and has_messages_format(dataset):
            logger.info("Detected messages format, converting to standard format...")
            dataset = convert_messages_dataset(dataset)

        # Apply column mapping if provided
        if column_mapping:
            logger.info(f"Applying column mapping: {column_mapping}")
            dataset = self._apply_column_mapping(dataset, column_mapping)

        return dataset

    def _load_and_concat(self) -> Dataset:
        """Load primary and additional datasets, normalize columns, and concatenate."""
        # Load primary dataset
        dataset = self._load_single_dataset(
            self.loader, self.column_mapping, self.convert_messages_format
        )

        if not self.additional_loaders:
            return dataset

        # Load additional datasets
        all_datasets = [dataset]
        for i, loader in enumerate(self.additional_loaders):
            col_mapping = self.additional_column_mappings[i] if i < len(self.additional_column_mappings) else None
            convert_msgs = self.additional_convert_messages[i] if i < len(self.additional_convert_messages) else True

            additional_ds = self._load_single_dataset(loader, col_mapping, convert_msgs)
            all_datasets.append(additional_ds)

        # Normalize columns across all datasets before concatenation
        # Find the union of all columns
        all_columns = set()
        for ds in all_datasets:
            all_columns.update(ds.column_names)

        # Add missing columns with empty strings to each dataset
        normalized = []
        for ds in all_datasets:
            missing = all_columns - set(ds.column_names)
            if missing:
                logger.info(f"Adding missing columns {missing} to dataset for concatenation")
                ds = ds.map(lambda row: {col: "" for col in missing}, desc="Normalizing columns")
            normalized.append(ds)

        # Concatenate all datasets
        combined = concatenate_datasets(normalized)
        logger.info(f"Concatenated {len(all_datasets)} datasets: {len(combined)} total samples")
        return combined

    def prepare(self) -> tuple[Dataset, Dataset]:
        """
        Load, validate, format, and split dataset.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info("Loading dataset...")
        dataset = self._load_and_concat()

        # Validate schema
        logger.info("Validating dataset schema...")
        self._validate_schema(dataset)

        # Limit samples if requested
        if self.max_samples and len(dataset) > self.max_samples:
            logger.info(f"Limiting dataset to {self.max_samples} samples")
            dataset = dataset.select(range(self.max_samples))

        # Format dataset
        logger.info("Formatting dataset...")
        dataset = dataset.map(
            self.formatter.format,
            num_proc=min(os.cpu_count() or 1, 4),  # Limit to 4 processes
            desc="Formatting dataset"
        )

        # Split dataset
        logger.info(f"Splitting dataset (test_size={self.test_size}, shuffle={self.shuffle})...")
        split = dataset.train_test_split(test_size=self.test_size, seed=self.seed, shuffle=self.shuffle)

        train_dataset = split["train"]
        eval_dataset = split["test"]

        logger.info(f"Prepared {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")

        return train_dataset, eval_dataset

    def preview(self, num_samples: int = 5) -> list[dict]:
        """
        Load and preview raw dataset samples without formatting.

        Args:
            num_samples: Number of samples to preview

        Returns:
            List of raw dataset rows
        """
        dataset = self._load_and_concat()

        # Get first N samples
        preview_dataset = dataset.select(range(min(num_samples, len(dataset))))

        return [dict(row) for row in preview_dataset]

    def preview_formatted(self, num_samples: int = 5) -> list[dict]:
        """
        Load and preview formatted dataset samples.

        Args:
            num_samples: Number of samples to preview

        Returns:
            List of formatted dataset rows
        """
        dataset = self._load_and_concat()

        # Validate schema
        self._validate_schema(dataset)

        # Get first N samples and format
        preview_dataset = dataset.select(range(min(num_samples, len(dataset))))
        formatted = preview_dataset.map(self.formatter.format)

        return [dict(row) for row in formatted]

    def _apply_column_mapping(self, dataset: Dataset, column_mapping: Optional[dict] = None) -> Dataset:
        """Apply column name mapping to dataset"""
        mapping = column_mapping if column_mapping is not None else self.column_mapping
        # Rename columns according to mapping
        for old_name, new_name in mapping.items():
            if old_name in dataset.column_names and old_name != new_name:
                dataset = dataset.rename_column(old_name, new_name)

        return dataset

    def _validate_schema(self, dataset: Dataset):
        """Validate that dataset has required columns based on training mode"""
        # For SFT and KTO modes, rejected is not required
        if self.training_mode in ('sft', 'kto'):
            required_columns = {'prompt', 'chosen'}
        else:
            required_columns = {'prompt', 'chosen', 'rejected'}
        optional_columns = {'system', 'reasoning', 'rejected'}

        available_columns = set(dataset.column_names)
        missing_required = required_columns - available_columns

        if missing_required:
            raise ValueError(
                f"Dataset missing required columns: {missing_required}. "
                f"Available columns: {available_columns}. "
                f"Use column_mapping to map your dataset columns to the expected format."
            )

        logger.info(f"Dataset schema validated (mode={self.training_mode}). Columns: {available_columns}")
