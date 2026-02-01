"""
Utilities for deduplicating datasets.

Provides functions to detect and remove duplicate samples from datasets
based on configurable fields (prompt, chosen, or both).

Deduplication strategies:
- "prompt": Deduplicate based on prompt field only
- "chosen": Deduplicate based on chosen field only
- "prompt_chosen": Deduplicate based on combined prompt+chosen (default)
- "exact": Deduplicate based on all fields (exact row match)
"""

import hashlib
import logging
from typing import Optional, Literal
from datasets import Dataset

logger = logging.getLogger(__name__)

DedupeStrategy = Literal["prompt", "chosen", "prompt_chosen", "exact"]


def _compute_hash(text: str) -> str:
    """Compute a hash for a given text string."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _get_row_key(row: dict, strategy: DedupeStrategy) -> str:
    """
    Compute a deduplication key for a row based on strategy.

    Args:
        row: Dataset row dictionary
        strategy: Which fields to use for deduplication

    Returns:
        Hash string representing the row's dedupe key
    """
    if strategy == "prompt":
        key_text = str(row.get('prompt', ''))
    elif strategy == "chosen":
        key_text = str(row.get('chosen', ''))
    elif strategy == "prompt_chosen":
        key_text = f"{row.get('prompt', '')}|||{row.get('chosen', '')}"
    elif strategy == "exact":
        # Sort keys for consistent ordering
        key_text = '|||'.join(f"{k}:{v}" for k, v in sorted(row.items()))
    else:
        raise ValueError(f"Unknown deduplication strategy: {strategy}")

    return _compute_hash(key_text)


def has_duplicates(
    dataset: Dataset,
    strategy: DedupeStrategy = "prompt_chosen"
) -> bool:
    """
    Check if dataset contains duplicate samples.

    Args:
        dataset: Dataset to check
        strategy: Which fields to use for duplicate detection

    Returns:
        True if duplicates exist, False otherwise
    """
    if len(dataset) == 0:
        return False

    seen_hashes = set()

    for row in dataset:
        row_hash = _get_row_key(dict(row), strategy)
        if row_hash in seen_hashes:
            return True
        seen_hashes.add(row_hash)

    return False


def count_duplicates(
    dataset: Dataset,
    strategy: DedupeStrategy = "prompt_chosen"
) -> int:
    """
    Count the number of duplicate samples in a dataset.

    Args:
        dataset: Dataset to check
        strategy: Which fields to use for duplicate detection

    Returns:
        Number of duplicate samples (total rows - unique rows)
    """
    if len(dataset) == 0:
        return 0

    seen_hashes = set()

    for row in dataset:
        row_hash = _get_row_key(dict(row), strategy)
        seen_hashes.add(row_hash)

    return len(dataset) - len(seen_hashes)


def deduplicate_dataset(
    dataset: Dataset,
    strategy: DedupeStrategy = "prompt_chosen",
    keep: Literal["first", "last"] = "first"
) -> Dataset:
    """
    Remove duplicate samples from a dataset.

    Args:
        dataset: Dataset to deduplicate
        strategy: Which fields to use for deduplication
            - "prompt": Deduplicate based on prompt field only
            - "chosen": Deduplicate based on chosen field only
            - "prompt_chosen": Deduplicate based on combined prompt+chosen (default)
            - "exact": Deduplicate based on all fields
        keep: Which duplicate to keep
            - "first": Keep the first occurrence (default)
            - "last": Keep the last occurrence

    Returns:
        Dataset with duplicates removed
    """
    if len(dataset) == 0:
        logger.info("Dataset is empty, nothing to deduplicate")
        return dataset

    original_count = len(dataset)

    # Build index of unique rows
    seen_hashes = {}  # hash -> index

    for idx, row in enumerate(dataset):
        row_hash = _get_row_key(dict(row), strategy)

        if keep == "first":
            # Only store first occurrence
            if row_hash not in seen_hashes:
                seen_hashes[row_hash] = idx
        else:  # keep == "last"
            # Always overwrite to keep last occurrence
            seen_hashes[row_hash] = idx

    # Get indices to keep (sorted to maintain order)
    indices_to_keep = sorted(seen_hashes.values())

    # Select unique rows
    deduplicated = dataset.select(indices_to_keep)

    removed_count = original_count - len(deduplicated)

    if removed_count > 0:
        logger.info(
            f"Removed {removed_count} duplicate samples "
            f"({original_count} -> {len(deduplicated)}) "
            f"using strategy='{strategy}', keep='{keep}'"
        )
    else:
        logger.info(f"No duplicates found (strategy='{strategy}')")

    return deduplicated
