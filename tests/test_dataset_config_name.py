"""
Tests for HuggingFace dataset configuration / subset selection (`config_name`).

Covers that the `name=` argument is threaded to `datasets.load_dataset` through
the loaders and the factory, and is omitted entirely when no config is set so
default-config datasets keep loading exactly as before.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_handlers.loaders import HuggingFaceLoader, StreamingHuggingFaceLoader
from dataset_handlers.factory import create_loader, create_loader_from_config


def _make_iterable(rows):
    for row in rows:
        yield row


def test_loader_omits_name_when_no_config():
    """Default behavior: no `name` kwarg passed to load_dataset."""
    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = MagicMock(__len__=lambda self: 3)
        HuggingFaceLoader(repo_id="org/repo", split="train", token="t").load()
        _, kwargs = mock_load.call_args
        assert "name" not in kwargs, "should not pass name= when config unset"
    print("  default load omits name=")


def test_loader_passes_config_name():
    """When config_name is set, it is passed as load_dataset(name=...)."""
    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = MagicMock(__len__=lambda self: 3)
        HuggingFaceLoader(
            repo_id="org/repo", split="train", config_name="high_quality"
        ).load()
        mock_load.assert_called_once_with(
            "org/repo", split="train", token=None, name="high_quality"
        )
    print("  config_name forwarded as name=")


def test_streaming_loader_passes_config_name():
    """Streaming loader forwards config_name alongside streaming=True."""
    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = _make_iterable(
            [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        )
        StreamingHuggingFaceLoader(
            repo_id="org/repo", split="test", token="tok",
            config_name="scored",
        ).load()
        mock_load.assert_called_once_with(
            "org/repo", split="test", token="tok", streaming=True, name="scored"
        )
    print("  streaming loader forwards config_name")


def test_factory_threads_config_name():
    """create_loader passes config_name into the loader (both modes)."""
    loader = create_loader("huggingface", repo_id="org/repo", config_name="hq")
    assert isinstance(loader, HuggingFaceLoader)
    assert loader.config_name == "hq"
    assert loader.get_source_info()["config_name"] == "hq"

    stream = create_loader("huggingface", repo_id="org/repo",
                           config_name="hq", streaming=True)
    assert isinstance(stream, StreamingHuggingFaceLoader)
    assert stream.config_name == "hq"
    print("  factory threads config_name into both loaders")


def test_factory_from_config_reads_config_name():
    """create_loader_from_config reads config_name from a config object."""
    cfg = MagicMock(spec=[])
    cfg.source_type = "huggingface"
    cfg.repo_id = "org/repo"
    cfg.split = "train"
    cfg.config_name = "high_quality"
    cfg.streaming = False
    loader = create_loader_from_config(cfg)
    assert loader.config_name == "high_quality"
    print("  create_loader_from_config reads config_name")


def test_dataset_source_model_field():
    """The DatasetSource Pydantic model accepts and defaults config_name."""
    try:
        from merlina import DatasetSource
    except Exception as e:  # heavy deps (torch/fastapi) absent outside pytest
        print(f"  SKIP (merlina import needs full deps): {e}")
        return
    src = DatasetSource(source_type="huggingface", repo_id="org/repo")
    assert src.config_name is None
    src2 = DatasetSource(source_type="huggingface", repo_id="org/repo",
                         config_name="scored")
    assert src2.config_name == "scored"
    print("  DatasetSource.config_name field present")


if __name__ == "__main__":
    tests = [
        test_loader_omits_name_when_no_config,
        test_loader_passes_config_name,
        test_streaming_loader_passes_config_name,
        test_factory_threads_config_name,
        test_factory_from_config_reads_config_name,
        test_dataset_source_model_field,
    ]
    passed = failed = 0
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
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
