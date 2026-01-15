"""
Base classes for dataset handling
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Callable
from datasets import Dataset
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
        convert_messages_format: bool = True
    ):
        """
        Initialize dataset pipeline.

        Args:
            loader: DatasetLoader instance to load the dataset
            formatter: DatasetFormatter instance to format rows
            column_mapping: Map dataset columns to expected names
                           e.g., {'input': 'prompt', 'output_good': 'chosen'}
            test_size: Fraction of data to use for evaluation
            max_samples: Optional limit on number of samples (for testing)
            seed: Random seed for train/test split
            shuffle: Whether to shuffle the dataset before splitting
            training_mode: Training mode ('sft' or 'orpo'). For SFT, rejected is optional.
            convert_messages_format: Whether to automatically detect and convert messages format
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

    def prepare(self) -> tuple[Dataset, Dataset]:
        """
        Load, validate, format, and split dataset.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info("Loading dataset...")
        dataset = self.loader.load()

        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # Convert messages format if detected and enabled
        if self.convert_messages_format and has_messages_format(dataset):
            logger.info("Detected messages format, converting to standard format...")
            dataset = convert_messages_dataset(dataset)

        # Apply column mapping if provided
        if self.column_mapping:
            logger.info(f"Applying column mapping: {self.column_mapping}")
            dataset = self._apply_column_mapping(dataset)

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
        dataset = self.loader.load()

        # Convert messages format if detected and enabled
        if self.convert_messages_format and has_messages_format(dataset):
            logger.info("Detected messages format, converting to standard format...")
            dataset = convert_messages_dataset(dataset)

        # Apply column mapping if provided
        if self.column_mapping:
            dataset = self._apply_column_mapping(dataset)

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
        dataset = self.loader.load()

        # Convert messages format if detected and enabled
        if self.convert_messages_format and has_messages_format(dataset):
            logger.info("Detected messages format, converting to standard format...")
            dataset = convert_messages_dataset(dataset)

        # Apply column mapping if provided
        if self.column_mapping:
            dataset = self._apply_column_mapping(dataset)

        # Validate schema
        self._validate_schema(dataset)

        # Get first N samples and format
        preview_dataset = dataset.select(range(min(num_samples, len(dataset))))
        formatted = preview_dataset.map(self.formatter.format)

        return [dict(row) for row in formatted]

    def _apply_column_mapping(self, dataset: Dataset) -> Dataset:
        """Apply column name mapping to dataset"""
        # Rename columns according to mapping
        for old_name, new_name in self.column_mapping.items():
            if old_name in dataset.column_names and old_name != new_name:
                dataset = dataset.rename_column(old_name, new_name)

        return dataset

    def _validate_schema(self, dataset: Dataset):
        """Validate that dataset has required columns based on training mode"""
        # For SFT mode, rejected is not required
        if self.training_mode == 'sft':
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
