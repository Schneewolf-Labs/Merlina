"""
Tests for StreamingHuggingFaceLoader and streaming dataset support.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from dataset_handlers.loaders import StreamingHuggingFaceLoader, HuggingFaceLoader
from dataset_handlers.factory import create_loader, create_loader_from_config
from dataset_handlers.base import DatasetPipeline


def _make_iterable(rows):
    """Create a simple iterable that mimics an IterableDataset."""
    for row in rows:
        yield row


def test_streaming_loader_basic():
    """Test that StreamingHuggingFaceLoader materializes rows into a Dataset."""
    print("Testing streaming loader basic materialization...")

    sample_rows = [
        {"prompt": f"Question {i}", "chosen": f"Answer {i}", "rejected": f"Bad {i}"}
        for i in range(25)
    ]

    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = _make_iterable(sample_rows)

        loader = StreamingHuggingFaceLoader(
            repo_id="test/dataset",
            split="train",
            batch_size=10,
        )
        dataset = loader.load()

    assert isinstance(dataset, Dataset), "Should return a Dataset object"
    assert len(dataset) == 25, f"Expected 25 rows, got {len(dataset)}"
    assert dataset[0]["prompt"] == "Question 0"
    assert dataset[24]["chosen"] == "Answer 24"
    print(f"  Loaded {len(dataset)} rows successfully")
    print("  Streaming loader basic materialization: PASSED")


def test_streaming_loader_max_samples():
    """Test that max_samples limits the number of rows streamed."""
    print("Testing streaming loader max_samples...")

    sample_rows = [
        {"prompt": f"Q{i}", "chosen": f"A{i}", "rejected": f"B{i}"}
        for i in range(1000)
    ]

    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = _make_iterable(sample_rows)

        loader = StreamingHuggingFaceLoader(
            repo_id="test/dataset",
            split="train",
            max_samples=50,
            batch_size=20,
        )
        dataset = loader.load()

    assert len(dataset) == 50, f"Expected 50 rows, got {len(dataset)}"
    print(f"  Stopped at {len(dataset)} rows (max_samples=50)")
    print("  Streaming loader max_samples: PASSED")


def test_streaming_loader_calls_load_dataset_with_streaming():
    """Test that the loader passes streaming=True to load_dataset."""
    print("Testing streaming=True passed to load_dataset...")

    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = _make_iterable([{"prompt": "Q", "chosen": "A", "rejected": "B"}])

        loader = StreamingHuggingFaceLoader(
            repo_id="org/repo",
            split="test",
            token="hf_token_123",
        )
        loader.load()

        mock_load.assert_called_once_with(
            "org/repo",
            split="test",
            token="hf_token_123",
            streaming=True,
        )

    print("  streaming=True correctly passed to load_dataset")
    print("  Streaming flag test: PASSED")


def test_streaming_loader_empty_dataset():
    """Test that an empty streamed dataset raises ValueError."""
    print("Testing streaming loader with empty dataset...")

    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = _make_iterable([])

        loader = StreamingHuggingFaceLoader(repo_id="test/empty", split="train")

        try:
            loader.load()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "no rows" in str(e).lower()
            print(f"  Correctly raised ValueError: {e}")

    print("  Empty dataset test: PASSED")


def test_streaming_loader_batch_boundaries():
    """Test that batching works correctly across boundaries."""
    print("Testing streaming loader batch boundaries...")

    # 33 rows with batch_size=10 -> 3 full batches + 1 partial batch
    sample_rows = [{"col": f"val_{i}"} for i in range(33)]

    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = _make_iterable(sample_rows)

        loader = StreamingHuggingFaceLoader(
            repo_id="test/dataset",
            split="train",
            batch_size=10,
        )
        dataset = loader.load()

    assert len(dataset) == 33, f"Expected 33 rows, got {len(dataset)}"
    assert dataset[32]["col"] == "val_32"
    print(f"  33 rows across 4 batches (3x10 + 1x3): correct")
    print("  Batch boundaries test: PASSED")


def test_streaming_loader_source_info():
    """Test get_source_info returns streaming metadata."""
    print("Testing streaming loader source info...")

    loader = StreamingHuggingFaceLoader(
        repo_id="org/model",
        split="train",
        max_samples=5000,
        batch_size=2000,
    )
    info = loader.get_source_info()

    assert info["source_type"] == "huggingface"
    assert info["streaming"] is True
    assert info["max_samples"] == 5000
    assert info["batch_size"] == 2000
    print(f"  Source info: {info}")
    print("  Source info test: PASSED")


def test_factory_creates_streaming_loader():
    """Test that create_loader returns StreamingHuggingFaceLoader when streaming=True."""
    print("Testing factory streaming loader creation...")

    loader = create_loader(
        source_type="huggingface",
        repo_id="test/repo",
        streaming=True,
        streaming_batch_size=5000,
    )

    assert isinstance(loader, StreamingHuggingFaceLoader)
    assert loader.batch_size == 5000
    print("  Factory created StreamingHuggingFaceLoader correctly")
    print("  Factory test: PASSED")


def test_factory_creates_normal_loader_by_default():
    """Test that create_loader returns HuggingFaceLoader when streaming=False."""
    print("Testing factory default (non-streaming) loader creation...")

    loader = create_loader(
        source_type="huggingface",
        repo_id="test/repo",
    )

    assert isinstance(loader, HuggingFaceLoader)
    print("  Factory created HuggingFaceLoader by default")
    print("  Default loader test: PASSED")


def test_factory_from_config_streaming():
    """Test that create_loader_from_config passes streaming from config dict."""
    print("Testing factory from config with streaming...")

    config = {
        "source_type": "huggingface",
        "repo_id": "org/dataset",
        "split": "train",
        "streaming": True,
        "streaming_batch_size": 8000,
    }

    loader = create_loader_from_config(source_config=config)

    assert isinstance(loader, StreamingHuggingFaceLoader)
    assert loader.batch_size == 8000
    print("  Config dict streaming correctly propagated")
    print("  Factory from config test: PASSED")


def test_factory_from_pydantic_config_streaming():
    """Test that create_loader_from_config works with Pydantic-like objects."""
    print("Testing factory from Pydantic config with streaming...")

    class MockSource:
        def model_dump(self):
            return {
                "source_type": "huggingface",
                "repo_id": "org/dataset",
                "split": "train",
                "streaming": True,
                "streaming_batch_size": 15000,
            }

    loader = create_loader_from_config(source_config=MockSource())

    assert isinstance(loader, StreamingHuggingFaceLoader)
    assert loader.batch_size == 15000
    print("  Pydantic config streaming correctly propagated")
    print("  Pydantic factory test: PASSED")


def test_pipeline_sets_max_samples_on_streaming_loader():
    """Test that DatasetPipeline passes max_samples hint to streaming loader."""
    print("Testing pipeline max_samples hint to streaming loader...")

    from dataset_handlers.formatters import get_formatter

    sample_rows = [
        {"prompt": f"Q{i}", "chosen": f"A{i}", "rejected": f"R{i}"}
        for i in range(100)
    ]

    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = _make_iterable(sample_rows)

        loader = StreamingHuggingFaceLoader(
            repo_id="test/dataset",
            split="train",
            batch_size=50,
        )
        formatter = get_formatter("chatml")

        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            max_samples=30,
            test_size=0.1,
        )

        train_ds, eval_ds = pipeline.prepare()

    # Pipeline should have set max_samples on the loader
    total = len(train_ds) + len(eval_ds)
    assert total == 30, f"Expected 30 total samples, got {total}"
    print(f"  Pipeline produced {len(train_ds)} train + {len(eval_ds)} eval = {total} total")
    print("  Pipeline max_samples hint test: PASSED")


def test_streaming_ignores_non_hf_sources():
    """Test that streaming flag is ignored for non-HuggingFace sources."""
    print("Testing streaming ignored for local_file source...")

    from dataset_handlers.loaders import LocalFileLoader

    loader = create_loader(
        source_type="local_file",
        file_path=str(Path(__file__).parent / "fixtures" / "test_dataset.json"),
        streaming=True,  # Should be ignored
    )

    assert isinstance(loader, LocalFileLoader)
    print("  Streaming flag correctly ignored for local_file")
    print("  Non-HF streaming test: PASSED")


if __name__ == "__main__":
    tests = [
        test_streaming_loader_basic,
        test_streaming_loader_max_samples,
        test_streaming_loader_calls_load_dataset_with_streaming,
        test_streaming_loader_empty_dataset,
        test_streaming_loader_batch_boundaries,
        test_streaming_loader_source_info,
        test_factory_creates_streaming_loader,
        test_factory_creates_normal_loader_by_default,
        test_factory_from_config_streaming,
        test_factory_from_pydantic_config_streaming,
        test_pipeline_sets_max_samples_on_streaming_loader,
        test_streaming_ignores_non_hf_sources,
    ]

    print("=" * 60)
    print("Testing Streaming Dataset Loader")
    print("=" * 60)

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n--- {test.__name__} ---")
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
