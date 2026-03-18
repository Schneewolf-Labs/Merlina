#!/usr/bin/env python3
"""
Test multi-dataset concatenation feature.
Tests that multiple datasets can be loaded, column-mapped, and concatenated.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset_handlers import (
    DatasetPipeline,
    LocalFileLoader,
    get_formatter
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_single_dataset_still_works():
    """Verify backwards compatibility - single dataset without additional loaders."""
    print("1. Testing single dataset (backwards compatibility)...")

    loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset.json'))
    formatter = get_formatter("chatml")

    pipeline = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        test_size=0.25
    )

    preview = pipeline.preview(num_samples=10)
    assert len(preview) == 4, f"Expected 4 samples, got {len(preview)}"
    assert 'prompt' in preview[0], "Missing 'prompt' column"
    print(f"   OK: {len(preview)} samples loaded")


def test_concat_same_schema():
    """Test concatenating two datasets with the same column schema."""
    print("2. Testing concat with same schema...")

    primary_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset.json'))
    additional_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset_additional.json'))
    formatter = get_formatter("chatml")

    pipeline = DatasetPipeline(
        loader=primary_loader,
        formatter=formatter,
        test_size=0.25,
        additional_loaders=[additional_loader]
    )

    preview = pipeline.preview(num_samples=20)
    # 4 from primary + 3 from additional = 7
    assert len(preview) == 7, f"Expected 7 samples, got {len(preview)}"
    print(f"   OK: {len(preview)} samples after concatenation (4 + 3)")


def test_concat_different_columns_with_mapping():
    """Test concatenating datasets with different columns using column mapping."""
    print("3. Testing concat with different columns + mapping...")

    primary_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset.json'))
    additional_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset_different_columns.json'))
    formatter = get_formatter("chatml")

    # The additional dataset has 'question', 'answer', 'bad_answer' columns
    # Map them to standard names
    additional_col_mapping = {
        'question': 'prompt',
        'answer': 'chosen',
        'bad_answer': 'rejected'
    }

    pipeline = DatasetPipeline(
        loader=primary_loader,
        formatter=formatter,
        test_size=0.25,
        additional_loaders=[additional_loader],
        additional_column_mappings=[additional_col_mapping]
    )

    preview = pipeline.preview(num_samples=20)
    # 4 from primary + 2 from additional = 6
    assert len(preview) == 6, f"Expected 6 samples, got {len(preview)}"

    # Verify all samples have standard columns
    for sample in preview:
        assert 'prompt' in sample, f"Missing 'prompt' in sample: {sample.keys()}"
        assert 'chosen' in sample, f"Missing 'chosen' in sample: {sample.keys()}"

    print(f"   OK: {len(preview)} samples with column mapping applied")


def test_concat_with_formatting():
    """Test that formatted preview works with concatenated datasets."""
    print("4. Testing formatted preview with concat...")

    primary_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset.json'))
    additional_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset_additional.json'))
    formatter = get_formatter("chatml")

    pipeline = DatasetPipeline(
        loader=primary_loader,
        formatter=formatter,
        test_size=0.25,
        additional_loaders=[additional_loader]
    )

    formatted = pipeline.preview_formatted(num_samples=5)
    assert len(formatted) == 5, f"Expected 5 formatted samples, got {len(formatted)}"

    for sample in formatted:
        assert 'prompt' in sample
        assert 'chosen' in sample
        # ChatML format should include special tokens
        assert '<|im_start|>' in sample['prompt'], "Formatted prompt missing ChatML tokens"

    print(f"   OK: {len(formatted)} formatted samples")


def test_concat_prepare():
    """Test full prepare() with concatenated datasets for training."""
    print("5. Testing prepare() with concat...")

    primary_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset.json'))
    additional_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset_additional.json'))
    formatter = get_formatter("chatml")

    pipeline = DatasetPipeline(
        loader=primary_loader,
        formatter=formatter,
        test_size=0.25,
        shuffle=True,
        additional_loaders=[additional_loader]
    )

    train_ds, eval_ds = pipeline.prepare()
    total = len(train_ds) + len(eval_ds)
    assert total == 7, f"Expected 7 total samples, got {total}"
    print(f"   OK: train={len(train_ds)}, eval={len(eval_ds)}, total={total}")


def test_concat_three_datasets():
    """Test concatenating three datasets."""
    print("6. Testing three-way concat...")

    loader1 = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset.json'))
    loader2 = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset_additional.json'))
    loader3 = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset_different_columns.json'))
    formatter = get_formatter("chatml")

    col_mapping_3 = {
        'question': 'prompt',
        'answer': 'chosen',
        'bad_answer': 'rejected'
    }

    pipeline = DatasetPipeline(
        loader=loader1,
        formatter=formatter,
        test_size=0.25,
        additional_loaders=[loader2, loader3],
        additional_column_mappings=[None, col_mapping_3]
    )

    preview = pipeline.preview(num_samples=20)
    # 4 + 3 + 2 = 9
    assert len(preview) == 9, f"Expected 9 samples, got {len(preview)}"
    print(f"   OK: {len(preview)} samples from 3 datasets (4 + 3 + 2)")


def test_concat_with_max_samples():
    """Test that max_samples limits the total combined dataset."""
    print("7. Testing max_samples with concat...")

    primary_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset.json'))
    additional_loader = LocalFileLoader(os.path.join(FIXTURES_DIR, 'test_dataset_additional.json'))
    formatter = get_formatter("chatml")

    pipeline = DatasetPipeline(
        loader=primary_loader,
        formatter=formatter,
        test_size=0.25,
        max_samples=5,
        additional_loaders=[additional_loader]
    )

    train_ds, eval_ds = pipeline.prepare()
    total = len(train_ds) + len(eval_ds)
    assert total == 5, f"Expected 5 total samples (limited), got {total}"
    print(f"   OK: {total} samples after max_samples limit")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Multi-Dataset Concatenation")
    print("=" * 60)

    tests = [
        test_single_dataset_still_works,
        test_concat_same_schema,
        test_concat_different_columns_with_mapping,
        test_concat_with_formatting,
        test_concat_prepare,
        test_concat_three_datasets,
        test_concat_with_max_samples,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
