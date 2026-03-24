"""
Tests for dataset deduplication utilities.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from dataset_handlers.deduplication import (
    has_duplicates,
    count_duplicates,
    deduplicate_dataset
)


def test_has_duplicates_true():
    """Test detection of duplicates when they exist"""
    print("Testing duplicate detection (with duplicates)...")

    dataset = Dataset.from_list([
        {"prompt": "Hello", "chosen": "Hi there!"},
        {"prompt": "Hello", "chosen": "Hi there!"},  # Duplicate
        {"prompt": "Goodbye", "chosen": "Bye!"}
    ])

    assert has_duplicates(dataset, strategy="prompt_chosen"), "Should detect duplicates"
    print("✓ Duplicate detection works (with duplicates)")


def test_has_duplicates_false():
    """Test detection when no duplicates exist"""
    print("Testing duplicate detection (no duplicates)...")

    dataset = Dataset.from_list([
        {"prompt": "Hello", "chosen": "Hi there!"},
        {"prompt": "Goodbye", "chosen": "Bye!"},
        {"prompt": "How are you?", "chosen": "I'm fine!"}
    ])

    assert not has_duplicates(dataset, strategy="prompt_chosen"), "Should not detect duplicates"
    print("✓ Duplicate detection works (no duplicates)")


def test_count_duplicates():
    """Test counting of duplicate samples"""
    print("Testing duplicate counting...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "1"},
        {"prompt": "A", "chosen": "1"},  # Duplicate 1
        {"prompt": "A", "chosen": "1"},  # Duplicate 2
        {"prompt": "B", "chosen": "2"},
        {"prompt": "B", "chosen": "2"},  # Duplicate 3
    ])

    count = count_duplicates(dataset, strategy="prompt_chosen")
    assert count == 3, f"Expected 3 duplicates, got {count}"
    print("✓ Duplicate counting works")


def test_deduplicate_prompt_chosen():
    """Test deduplication based on prompt+chosen"""
    print("Testing deduplication (prompt_chosen strategy)...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "1", "rejected": "x"},
        {"prompt": "A", "chosen": "1", "rejected": "y"},  # Duplicate (different rejected)
        {"prompt": "B", "chosen": "2", "rejected": "z"},
    ])

    result = deduplicate_dataset(dataset, strategy="prompt_chosen")

    assert len(result) == 2, f"Expected 2 samples, got {len(result)}"
    assert result[0]["prompt"] == "A"
    assert result[1]["prompt"] == "B"
    print("✓ Deduplication (prompt_chosen) works")


def test_deduplicate_prompt_only():
    """Test deduplication based on prompt only"""
    print("Testing deduplication (prompt strategy)...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "1"},
        {"prompt": "A", "chosen": "2"},  # Same prompt, different chosen
        {"prompt": "B", "chosen": "3"},
    ])

    result = deduplicate_dataset(dataset, strategy="prompt")

    assert len(result) == 2, f"Expected 2 samples, got {len(result)}"
    print("✓ Deduplication (prompt) works")


def test_deduplicate_chosen_only():
    """Test deduplication based on chosen only"""
    print("Testing deduplication (chosen strategy)...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "same"},
        {"prompt": "B", "chosen": "same"},  # Same chosen, different prompt
        {"prompt": "C", "chosen": "different"},
    ])

    result = deduplicate_dataset(dataset, strategy="chosen")

    assert len(result) == 2, f"Expected 2 samples, got {len(result)}"
    print("✓ Deduplication (chosen) works")


def test_deduplicate_exact():
    """Test deduplication based on exact match"""
    print("Testing deduplication (exact strategy)...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "1", "rejected": "x"},
        {"prompt": "A", "chosen": "1", "rejected": "x"},  # Exact duplicate
        {"prompt": "A", "chosen": "1", "rejected": "y"},  # Different rejected, not a duplicate
    ])

    result = deduplicate_dataset(dataset, strategy="exact")

    assert len(result) == 2, f"Expected 2 samples, got {len(result)}"
    print("✓ Deduplication (exact) works")


def test_deduplicate_keep_first():
    """Test keeping first occurrence"""
    print("Testing keep='first'...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "first"},
        {"prompt": "A", "chosen": "second"},  # Duplicate prompt
        {"prompt": "A", "chosen": "third"},   # Duplicate prompt
    ])

    result = deduplicate_dataset(dataset, strategy="prompt", keep="first")

    assert len(result) == 1
    assert result[0]["chosen"] == "first", "Should keep first occurrence"
    print("✓ keep='first' works")


def test_deduplicate_keep_last():
    """Test keeping last occurrence"""
    print("Testing keep='last'...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "first"},
        {"prompt": "A", "chosen": "second"},  # Duplicate prompt
        {"prompt": "A", "chosen": "third"},   # Duplicate prompt
    ])

    result = deduplicate_dataset(dataset, strategy="prompt", keep="last")

    assert len(result) == 1
    assert result[0]["chosen"] == "third", "Should keep last occurrence"
    print("✓ keep='last' works")


def test_empty_dataset():
    """Test with empty dataset"""
    print("Testing with empty dataset...")

    dataset = Dataset.from_list([])

    assert not has_duplicates(dataset)
    assert count_duplicates(dataset) == 0
    result = deduplicate_dataset(dataset)
    assert len(result) == 0
    print("✓ Empty dataset handling works")


def test_no_duplicates():
    """Test when there are no duplicates to remove"""
    print("Testing with no duplicates...")

    dataset = Dataset.from_list([
        {"prompt": "A", "chosen": "1"},
        {"prompt": "B", "chosen": "2"},
        {"prompt": "C", "chosen": "3"},
    ])

    result = deduplicate_dataset(dataset, strategy="prompt_chosen")

    assert len(result) == 3, "Should keep all samples when no duplicates"
    print("✓ No duplicates handling works")


def test_integration_with_pipeline():
    """Test integration with DatasetPipeline"""
    print("Testing integration with DatasetPipeline...")

    from dataset_handlers.base import DatasetPipeline
    from dataset_handlers.loaders import LocalFileLoader
    from dataset_handlers.formatters import ChatMLFormatter
    import json
    import tempfile

    # Create test dataset with duplicates
    test_data = [
        {"prompt": "Hello", "chosen": "Hi there!"},
        {"prompt": "Hello", "chosen": "Hi there!"},  # Duplicate
        {"prompt": "What's 2+2?", "chosen": "It's 4"},
        {"prompt": "What's 2+2?", "chosen": "It's 4"},  # Duplicate
        {"prompt": "Goodbye", "chosen": "Bye!"}
    ]

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        # Create pipeline with deduplication enabled
        loader = LocalFileLoader(temp_path)
        formatter = ChatMLFormatter()
        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            training_mode='sft',
            deduplicate=True,
            dedupe_strategy='prompt_chosen',
            test_size=0.01
        )

        # Prepare should deduplicate
        train_dataset, eval_dataset = pipeline.prepare()

        total_samples = len(train_dataset) + len(eval_dataset)
        assert total_samples == 3, f"Expected 3 unique samples, got {total_samples}"

        print("✓ Pipeline integration works")

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    print("Running deduplication tests...\n")

    test_has_duplicates_true()
    test_has_duplicates_false()
    test_count_duplicates()
    test_deduplicate_prompt_chosen()
    test_deduplicate_prompt_only()
    test_deduplicate_chosen_only()
    test_deduplicate_exact()
    test_deduplicate_keep_first()
    test_deduplicate_keep_last()
    test_empty_dataset()
    test_no_duplicates()
    test_integration_with_pipeline()

    print("\n✓ All tests passed!")
