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


class MultiDatasetPipeline:
    """
    Orchestrates loading, formatting, and concatenation of multiple datasets.
    Each dataset can have its own source, formatter, and column mapping.
    """

    def __init__(
        self,
        pipelines: list[DatasetPipeline],
        test_size: float = 0.01,
        max_samples: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = True,
        training_mode: str = "orpo"
    ):
        """
        Initialize multi-dataset pipeline.

        Args:
            pipelines: List of DatasetPipeline instances, one per dataset
            test_size: Fraction of combined data to use for evaluation
            max_samples: Optional limit on total number of samples
            seed: Random seed for train/test split
            shuffle: Whether to shuffle the combined dataset before splitting
            training_mode: Training mode ('sft' or 'orpo')
        """
        self.pipelines = pipelines
        self.test_size = test_size
        self.max_samples = max_samples
        self.seed = seed
        self.shuffle = shuffle
        self.training_mode = training_mode

    def prepare(self) -> tuple[Dataset, Dataset]:
        """
        Load, format, and concatenate all datasets, then split.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        from datasets import concatenate_datasets

        all_formatted_datasets = []

        for i, pipeline in enumerate(self.pipelines):
            logger.info(f"Processing dataset {i + 1}/{len(self.pipelines)}...")

            # Load dataset
            dataset = pipeline.loader.load()
            logger.info(f"Dataset {i + 1} loaded with {len(dataset)} samples")

            # Convert messages format if detected and enabled
            if pipeline.convert_messages_format and has_messages_format(dataset):
                logger.info(f"Dataset {i + 1}: Detected messages format, converting...")
                dataset = convert_messages_dataset(dataset)

            # Apply column mapping if provided
            if pipeline.column_mapping:
                logger.info(f"Dataset {i + 1}: Applying column mapping: {pipeline.column_mapping}")
                dataset = pipeline._apply_column_mapping(dataset)

            # Validate schema
            logger.info(f"Dataset {i + 1}: Validating schema...")
            pipeline._validate_schema(dataset)

            # Limit samples if requested for this specific dataset
            if pipeline.max_samples and len(dataset) > pipeline.max_samples:
                logger.info(f"Dataset {i + 1}: Limiting to {pipeline.max_samples} samples")
                dataset = dataset.select(range(pipeline.max_samples))

            # Format dataset
            logger.info(f"Dataset {i + 1}: Formatting dataset...")
            dataset = dataset.map(
                pipeline.formatter.format,
                num_proc=min(os.cpu_count() or 1, 4),
                desc=f"Formatting dataset {i + 1}"
            )

            all_formatted_datasets.append(dataset)
            logger.info(f"Dataset {i + 1} prepared with {len(dataset)} samples")

        # Concatenate all datasets
        logger.info(f"Concatenating {len(all_formatted_datasets)} datasets...")
        combined_dataset = concatenate_datasets(all_formatted_datasets)
        logger.info(f"Combined dataset has {len(combined_dataset)} total samples")

        # Apply global max_samples limit
        if self.max_samples and len(combined_dataset) > self.max_samples:
            logger.info(f"Limiting combined dataset to {self.max_samples} samples")
            combined_dataset = combined_dataset.select(range(self.max_samples))

        # Split dataset
        logger.info(f"Splitting dataset (test_size={self.test_size}, shuffle={self.shuffle})...")
        split = combined_dataset.train_test_split(
            test_size=self.test_size,
            seed=self.seed,
            shuffle=self.shuffle
        )

        train_dataset = split["train"]
        eval_dataset = split["test"]

        logger.info(f"Prepared {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")

        return train_dataset, eval_dataset

    def preview(self, dataset_index: int = 0, num_samples: int = 5) -> list[dict]:
        """
        Preview raw samples from a specific dataset.

        Args:
            dataset_index: Index of the dataset to preview (0-based)
            num_samples: Number of samples to preview

        Returns:
            List of raw dataset rows
        """
        if dataset_index >= len(self.pipelines):
            raise ValueError(f"Dataset index {dataset_index} out of range (have {len(self.pipelines)} datasets)")

        return self.pipelines[dataset_index].preview(num_samples)

    def preview_formatted(self, dataset_index: int = 0, num_samples: int = 5) -> list[dict]:
        """
        Preview formatted samples from a specific dataset.

        Args:
            dataset_index: Index of the dataset to preview (0-based)
            num_samples: Number of samples to preview

        Returns:
            List of formatted dataset rows
        """
        if dataset_index >= len(self.pipelines):
            raise ValueError(f"Dataset index {dataset_index} out of range (have {len(self.pipelines)} datasets)")

        return self.pipelines[dataset_index].preview_formatted(num_samples)

    def get_dataset_info(self) -> list[dict]:
        """Get information about each dataset in the pipeline"""
        info = []
        for i, pipeline in enumerate(self.pipelines):
            info.append({
                "index": i,
                "source": pipeline.loader.get_source_info(),
                "format": pipeline.formatter.get_format_info(),
                "column_mapping": pipeline.column_mapping,
                "max_samples": pipeline.max_samples
            })
        return info
