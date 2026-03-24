"""
Tests for multi-dataset concatenation support.

Verifies that DatasetPipeline can load and concatenate multiple datasets,
and that the API accepts additional_sources in the config.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from merlina import TrainingConfig, DatasetConfig, DatasetSource


# ---------------------------------------------------------------------------
# 1. Pydantic model accepts additional_sources
# ---------------------------------------------------------------------------

class TestMultiDatasetConfig:
    """Ensure DatasetConfig handles additional_sources correctly."""

    def test_additional_sources_default_empty(self):
        config = DatasetConfig()
        assert config.additional_sources == []

    def test_additional_sources_accepts_list(self):
        config = DatasetConfig(
            additional_sources=[
                DatasetSource(source_type="huggingface", repo_id="org/dataset-2"),
                DatasetSource(source_type="huggingface", repo_id="org/dataset-3"),
            ]
        )
        assert len(config.additional_sources) == 2
        assert config.additional_sources[0].repo_id == "org/dataset-2"
        assert config.additional_sources[1].repo_id == "org/dataset-3"

    def test_source_has_column_mapping(self):
        source = DatasetSource(
            source_type="huggingface",
            repo_id="org/dataset",
            column_mapping={"input": "prompt", "output": "chosen"}
        )
        assert source.column_mapping == {"input": "prompt", "output": "chosen"}

    def test_source_column_mapping_default_none(self):
        source = DatasetSource(source_type="huggingface", repo_id="org/dataset")
        assert source.column_mapping is None

    def test_training_config_with_additional_sources(self):
        """Full TrainingConfig with multiple datasets should serialize properly."""
        config = TrainingConfig(
            output_name="test",
            dataset=DatasetConfig(
                source=DatasetSource(source_type="huggingface", repo_id="org/main"),
                additional_sources=[
                    DatasetSource(
                        source_type="huggingface",
                        repo_id="org/extra",
                        column_mapping={"text": "prompt", "response": "chosen"}
                    ),
                ],
            )
        )
        assert config.dataset.source.repo_id == "org/main"
        assert len(config.dataset.additional_sources) == 1
        assert config.dataset.additional_sources[0].column_mapping["text"] == "prompt"

    def test_additional_sources_in_model_fields(self):
        assert "additional_sources" in DatasetConfig.model_fields

    def test_backward_compat_no_additional_sources(self):
        """Configs without additional_sources should still work."""
        config = DatasetConfig(
            source=DatasetSource(source_type="huggingface", repo_id="org/dataset"),
        )
        assert config.additional_sources == []


# ---------------------------------------------------------------------------
# 2. DatasetPipeline concatenation
# ---------------------------------------------------------------------------

class TestPipelineConcatenation:
    """Test that DatasetPipeline loads and concatenates multiple sources."""

    def _make_mock_dataset(self, rows):
        """Create a mock HF Dataset from a list of dicts."""
        from datasets import Dataset
        return Dataset.from_list(rows)

    def _make_mock_loader(self, rows):
        """Create a mock DatasetLoader that returns given rows."""
        loader = MagicMock()
        loader.load.return_value = self._make_mock_dataset(rows)
        loader.get_source_info.return_value = {"source_type": "mock"}
        return loader

    def _make_identity_formatter(self):
        """Formatter that passes rows through unchanged."""
        formatter = MagicMock()
        formatter.format = lambda row: row
        return formatter

    def test_single_loader_no_concat(self):
        """Without additional_loaders, pipeline works as before."""
        from dataset_handlers.base import DatasetPipeline

        loader = self._make_mock_loader([
            {"prompt": "q1", "chosen": "a1", "rejected": "r1"},
            {"prompt": "q2", "chosen": "a2", "rejected": "r2"},
        ])
        pipeline = DatasetPipeline(
            loader=loader,
            formatter=self._make_identity_formatter(),
            test_size=0.5,
            training_mode="orpo",
        )
        train, eval_ds = pipeline.prepare()
        assert len(train) + len(eval_ds) == 2

    def test_two_loaders_concatenated(self):
        """Two loaders should produce concatenated results."""
        from dataset_handlers.base import DatasetPipeline

        primary = self._make_mock_loader([
            {"prompt": "q1", "chosen": "a1", "rejected": "r1"},
        ])
        extra = self._make_mock_loader([
            {"prompt": "q2", "chosen": "a2", "rejected": "r2"},
            {"prompt": "q3", "chosen": "a3", "rejected": "r3"},
        ])
        pipeline = DatasetPipeline(
            loader=primary,
            formatter=self._make_identity_formatter(),
            test_size=0.01,
            training_mode="orpo",
            additional_loaders=[(extra, None, True)],
        )
        train, eval_ds = pipeline.prepare()
        total = len(train) + len(eval_ds)
        assert total == 3, f"Expected 3 samples from 1+2 datasets, got {total}"

    def test_three_loaders_concatenated(self):
        """Three loaders should all be concatenated."""
        from dataset_handlers.base import DatasetPipeline

        loaders = [
            self._make_mock_loader([{"prompt": f"q{i}", "chosen": f"a{i}", "rejected": f"r{i}"}])
            for i in range(3)
        ]
        pipeline = DatasetPipeline(
            loader=loaders[0],
            formatter=self._make_identity_formatter(),
            test_size=0.01,
            training_mode="orpo",
            additional_loaders=[
                (loaders[1], None, True),
                (loaders[2], None, True),
            ],
        )
        train, eval_ds = pipeline.prepare()
        assert len(train) + len(eval_ds) == 3

    def test_per_source_column_mapping(self):
        """Each source's column_mapping is applied independently."""
        from dataset_handlers.base import DatasetPipeline

        # Primary has standard columns
        primary = self._make_mock_loader([
            {"prompt": "q1", "chosen": "a1", "rejected": "r1"},
        ])
        # Extra has different column names
        extra = self._make_mock_loader([
            {"input": "q2", "output": "a2", "bad": "r2"},
        ])
        pipeline = DatasetPipeline(
            loader=primary,
            formatter=self._make_identity_formatter(),
            test_size=0.01,
            training_mode="orpo",
            additional_loaders=[
                (extra, {"input": "prompt", "output": "chosen", "bad": "rejected"}, True),
            ],
        )
        train, eval_ds = pipeline.prepare()
        total = len(train) + len(eval_ds)
        assert total == 2

        # Verify columns are normalized
        all_data = list(train) + list(eval_ds)
        for row in all_data:
            assert "prompt" in row
            assert "chosen" in row
            assert "rejected" in row

    def test_sft_mode_concat(self):
        """Concatenation works in SFT mode (no rejected required)."""
        from dataset_handlers.base import DatasetPipeline

        primary = self._make_mock_loader([
            {"prompt": "q1", "chosen": "a1"},
        ])
        extra = self._make_mock_loader([
            {"prompt": "q2", "chosen": "a2"},
        ])
        pipeline = DatasetPipeline(
            loader=primary,
            formatter=self._make_identity_formatter(),
            test_size=0.5,
            training_mode="sft",
            additional_loaders=[(extra, None, True)],
        )
        train, eval_ds = pipeline.prepare()
        assert len(train) + len(eval_ds) == 2
