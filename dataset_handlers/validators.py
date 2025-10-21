"""
Dataset validation utilities
"""

import logging
from typing import Optional, Set
from datasets import Dataset

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails"""
    pass


def validate_dataset_schema(
    dataset: Dataset,
    required_columns: Optional[Set[str]] = None,
    optional_columns: Optional[Set[str]] = None
) -> bool:
    """
    Validate that dataset has required columns.

    Args:
        dataset: Dataset to validate
        required_columns: Set of column names that must be present
        optional_columns: Set of column names that may be present

    Returns:
        True if validation passes

    Raises:
        DatasetValidationError: If validation fails
    """
    if required_columns is None:
        required_columns = {'prompt', 'chosen', 'rejected'}

    if optional_columns is None:
        optional_columns = {'system'}

    available_columns = set(dataset.column_names)
    missing_required = required_columns - available_columns

    if missing_required:
        raise DatasetValidationError(
            f"Dataset missing required columns: {missing_required}. "
            f"Available columns: {available_columns}. "
            f"Required columns: {required_columns}"
        )

    logger.info(f"Dataset schema validated. Columns: {available_columns}")
    return True


def validate_dataset_samples(
    dataset: Dataset,
    max_prompt_length: Optional[int] = None,
    max_chosen_length: Optional[int] = None,
    max_rejected_length: Optional[int] = None
) -> dict:
    """
    Validate dataset samples and return statistics.

    Args:
        dataset: Dataset to validate
        max_prompt_length: Optional maximum prompt length
        max_chosen_length: Optional maximum chosen response length
        max_rejected_length: Optional maximum rejected response length

    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_samples': len(dataset),
        'empty_prompts': 0,
        'empty_chosen': 0,
        'empty_rejected': 0,
        'long_prompts': 0,
        'long_chosen': 0,
        'long_rejected': 0,
        'issues': []
    }

    for idx, row in enumerate(dataset):
        prompt = str(row.get('prompt', ''))
        chosen = str(row.get('chosen', ''))
        rejected = str(row.get('rejected', ''))

        # Check for empty fields
        if not prompt.strip():
            stats['empty_prompts'] += 1
            stats['issues'].append(f"Row {idx}: Empty prompt")

        if not chosen.strip():
            stats['empty_chosen'] += 1
            stats['issues'].append(f"Row {idx}: Empty chosen response")

        if not rejected.strip():
            stats['empty_rejected'] += 1
            stats['issues'].append(f"Row {idx}: Empty rejected response")

        # Check for length limits
        if max_prompt_length and len(prompt) > max_prompt_length:
            stats['long_prompts'] += 1

        if max_chosen_length and len(chosen) > max_chosen_length:
            stats['long_chosen'] += 1

        if max_rejected_length and len(rejected) > max_rejected_length:
            stats['long_rejected'] += 1

    return stats
