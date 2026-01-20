"""
Dataset loader implementations for different sources.

This module provides concrete implementations of DatasetLoader for:
- HuggingFace Hub datasets
- Local files (JSON, JSONL, CSV, Parquet)
- Uploaded file content (from web UI)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Dict
import logging

from datasets import load_dataset, Dataset
from .base import DatasetLoader

logger = logging.getLogger(__name__)


class HuggingFaceLoader(DatasetLoader):
    """
    Load dataset from HuggingFace Hub.

    This loader supports both public and private datasets. For private datasets,
    provide an API token with appropriate permissions.

    Example:
        >>> loader = HuggingFaceLoader("schneewolflabs/Athanor-DPO")
        >>> dataset = loader.load()
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        token: Optional[str] = None
    ) -> None:
        """
        Initialize HuggingFace dataset loader.

        Args:
            repo_id: HuggingFace repository ID (e.g., "schneewolflabs/Athanor-DPO")
            split: Dataset split to load (default: "train")
            token: Optional HuggingFace API token for private/gated datasets
        """
        self.repo_id = repo_id
        self.split = split
        self.token = token

    def load(self) -> Dataset:
        """
        Load dataset from HuggingFace Hub.

        Returns:
            The loaded Dataset

        Raises:
            ValueError: If the dataset cannot be loaded
        """
        logger.info(f"Loading dataset from HuggingFace: {self.repo_id} (split: {self.split})")

        try:
            dataset = load_dataset(
                self.repo_id,
                split=self.split,
                token=self.token
            )
            logger.info(f"Successfully loaded {len(dataset)} samples from {self.repo_id}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            raise ValueError(f"Failed to load dataset '{self.repo_id}': {str(e)}")

    def get_source_info(self) -> dict:
        """
        Get information about the dataset source.

        Returns:
            Dictionary containing source type, repo ID, and split
        """
        return {
            "source_type": "huggingface",
            "repo_id": self.repo_id,
            "split": self.split
        }


class LocalFileLoader(DatasetLoader):
    """
    Load dataset from local file.

    Supports JSON, JSONL, CSV, and Parquet formats. The format is automatically
    inferred from the file extension, or can be explicitly specified.

    Example:
        >>> loader = LocalFileLoader("./data/training.jsonl")
        >>> dataset = loader.load()
    """

    SUPPORTED_FORMATS: Dict[str, str] = {
        '.json': 'json',
        '.jsonl': 'json',
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.pq': 'parquet'
    }

    def __init__(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None
    ) -> None:
        """
        Initialize local file loader.

        Args:
            file_path: Path to the dataset file (absolute or relative)
            file_format: Optional file format override ('json', 'csv', 'parquet').
                        If not provided, will be inferred from file extension.

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the format cannot be determined
        """
        self.file_path = Path(file_path)
        self.file_format = file_format

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        # Infer format from extension if not provided
        if not self.file_format:
            ext = self.file_path.suffix.lower()
            self.file_format = self.SUPPORTED_FORMATS.get(ext)

            if not self.file_format:
                raise ValueError(
                    f"Could not infer file format from extension '{ext}'. "
                    f"Supported formats: {list(self.SUPPORTED_FORMATS.values())}"
                )

    def load(self) -> Dataset:
        """Load dataset from local file"""
        logger.info(f"Loading dataset from local file: {self.file_path} (format: {self.file_format})")

        try:
            if self.file_format == 'json':
                # Check if it's JSONL or regular JSON
                with open(self.file_path, 'r') as f:
                    first_line = f.readline().strip()

                # If first line is valid JSON object, it's JSONL
                try:
                    json.loads(first_line)
                    is_jsonl = True
                except json.JSONDecodeError:
                    is_jsonl = False

                if is_jsonl:
                    dataset = load_dataset('json', data_files=str(self.file_path), split='train')
                else:
                    # Regular JSON array
                    with open(self.file_path, 'r') as f:
                        data = json.load(f)
                    dataset = Dataset.from_list(data if isinstance(data, list) else [data])

            elif self.file_format == 'csv':
                dataset = load_dataset('csv', data_files=str(self.file_path), split='train')

            elif self.file_format == 'parquet':
                dataset = load_dataset('parquet', data_files=str(self.file_path), split='train')

            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")

            logger.info(f"Successfully loaded {len(dataset)} samples from {self.file_path}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset from file: {e}")
            raise ValueError(f"Failed to load dataset from '{self.file_path}': {str(e)}")

    def get_source_info(self) -> dict:
        """Get information about the dataset source"""
        return {
            "source_type": "local_file",
            "file_path": str(self.file_path),
            "file_format": self.file_format
        }


class UploadedDatasetLoader(DatasetLoader):
    """
    Load dataset from uploaded file content.

    This loader handles file content uploaded through the web UI by writing
    it to a temporary file, loading it with LocalFileLoader, and cleaning
    up the temporary file afterward.
    """

    def __init__(
        self,
        file_content: bytes,
        filename: str,
        file_format: Optional[str] = None
    ) -> None:
        """
        Initialize uploaded dataset loader.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename (used to infer format if not specified)
            file_format: Optional file format override ('json', 'csv', 'parquet')

        Raises:
            ValueError: If format cannot be inferred from filename
        """
        self.file_content = file_content
        self.filename = filename
        self.file_format = file_format
        self._temp_path: Optional[str] = None

        # Infer format from filename if not provided
        if not self.file_format:
            ext = Path(filename).suffix.lower()
            self.file_format = LocalFileLoader.SUPPORTED_FORMATS.get(ext)

            if not self.file_format:
                raise ValueError(
                    f"Could not infer file format from filename '{filename}'. "
                    f"Supported formats: {list(LocalFileLoader.SUPPORTED_FORMATS.values())}"
                )

    def _cleanup_temp_file(self) -> None:
        """Clean up the temporary file if it exists."""
        if self._temp_path:
            try:
                temp_file = Path(self._temp_path)
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {self._temp_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file '{self._temp_path}': {e}")
            finally:
                self._temp_path = None

    def load(self) -> Dataset:
        """
        Load dataset from uploaded content.

        Creates a temporary file, writes the content to it, loads the dataset
        using LocalFileLoader, and ensures cleanup of the temporary file.

        Returns:
            The loaded Dataset

        Raises:
            ValueError: If the dataset cannot be loaded
        """
        logger.info(f"Loading dataset from upload: {self.filename} (format: {self.file_format})")

        # Ensure any previous temp file is cleaned up
        self._cleanup_temp_file()

        try:
            # Write content to temporary file with appropriate suffix
            suffix = Path(self.filename).suffix or f'.{self.file_format}'

            # Create temp file with delete=False so we control cleanup
            fd, self._temp_path = tempfile.mkstemp(suffix=suffix)
            try:
                os.write(fd, self.file_content)
            finally:
                os.close(fd)

            # Use LocalFileLoader to load from temp file
            loader = LocalFileLoader(self._temp_path, self.file_format)
            dataset = loader.load()

            logger.info(f"Successfully loaded {len(dataset)} samples from upload")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load uploaded dataset: {e}")
            raise ValueError(f"Failed to load uploaded dataset '{self.filename}': {str(e)}")

        finally:
            # Always clean up the temp file
            self._cleanup_temp_file()

    def get_source_info(self) -> dict:
        """
        Get information about the dataset source.

        Returns:
            Dictionary containing source type, filename, and format
        """
        return {
            "source_type": "upload",
            "filename": self.filename,
            "file_format": self.file_format
        }
