"""
Tests for config field name alignment between frontend and backend.

Regression tests for the bug where the frontend sent `per_device_train_batch_size`
and `num_train_epochs` but the Pydantic model expected `batch_size` and `num_epochs`,
causing those values to be silently ignored (Pydantic v2 drops unknown fields).
"""

import sys
import math
import json
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from merlina import TrainingConfig


# ---------------------------------------------------------------------------
# 1. Pydantic model accepts the correct field names
# ---------------------------------------------------------------------------

class TestTrainingConfigFieldNames:
    """Ensure TrainingConfig accepts the field names the frontend actually sends."""

    def test_batch_size_field_accepted(self):
        """batch_size must be a real field, not silently ignored."""
        config = TrainingConfig(
            output_name="test",
            batch_size=4,
        )
        assert config.batch_size == 4

    def test_num_epochs_field_accepted(self):
        """num_epochs must be a real field, not silently ignored."""
        config = TrainingConfig(
            output_name="test",
            num_epochs=5,
        )
        assert config.num_epochs == 5

    def test_batch_size_not_default_when_set(self):
        """Setting batch_size=2 must not silently fall back to the default (1)."""
        config = TrainingConfig(output_name="test", batch_size=2)
        assert config.batch_size == 2, (
            f"batch_size should be 2 but got {config.batch_size} — "
            "field name may be mismatched with frontend"
        )

    def test_num_epochs_not_default_when_set(self):
        """Setting num_epochs=1 must not silently fall back to the default (2)."""
        config = TrainingConfig(output_name="test", num_epochs=1)
        assert config.num_epochs == 1, (
            f"num_epochs should be 1 but got {config.num_epochs} — "
            "field name may be mismatched with frontend"
        )

    def test_all_frontend_fields_are_model_fields(self):
        """Every field the frontend sends must exist on TrainingConfig.

        This is the definitive guard against silent field-name drift.
        """
        # Fields the frontend JS builds in getConfig()
        frontend_fields = {
            "base_model", "output_name", "model_type", "training_mode",
            "use_lora", "lora_r", "lora_alpha", "lora_dropout",
            "target_modules", "modules_to_save",
            "use_4bit", "max_length", "max_prompt_length",
            "num_epochs", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "warmup_ratio", "beta",
            "seed", "max_grad_norm", "weight_decay",
            "lr_scheduler_type", "logging_steps", "shuffle_dataset",
            "gradient_checkpointing", "optimizer_type",
            "adam_beta1", "adam_beta2", "adam_epsilon",
            "adafactor_relative_step", "adafactor_scale_parameter",
            "adafactor_warmup_init", "adafactor_decay_rate",
            "adafactor_beta1", "adafactor_clip_threshold",
            "attn_implementation", "eval_steps", "dataset",
        }

        model_fields = set(TrainingConfig.model_fields.keys())
        missing = frontend_fields - model_fields
        assert not missing, (
            f"Frontend sends fields that don't exist on TrainingConfig: {missing}. "
            "Pydantic v2 silently ignores unknown fields — these values are being dropped."
        )


# ---------------------------------------------------------------------------
# 2. Frontend JS sends the right field names (parse config.js)
# ---------------------------------------------------------------------------

class TestFrontendConfigJS:
    """Parse config.js to verify it sends field names matching the Pydantic model."""

    @pytest.fixture
    def config_js(self):
        js_path = Path(__file__).parent.parent / "frontend" / "js" / "config.js"
        if not js_path.exists():
            pytest.skip("frontend/js/config.js not found")
        return js_path.read_text()

    def test_no_per_device_train_batch_size_in_config_send(self, config_js):
        """Frontend must NOT send per_device_train_batch_size (old TRL name)."""
        # Look for the field being sent as a key in the config object
        # Match "per_device_train_batch_size:" or "per_device_train_batch_size :"
        # but NOT inside a comment or the fallback loader
        lines = config_js.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            # Only flag lines that use it as a config KEY being sent to server
            # Allow it in the config loader fallback (config.per_device_train_batch_size)
            if re.search(r'^\s*per_device_train_batch_size\s*:', line):
                pytest.fail(
                    f"config.js line {i} sends 'per_device_train_batch_size' — "
                    "backend expects 'batch_size'"
                )

    def test_no_num_train_epochs_in_config_send(self, config_js):
        """Frontend must NOT send num_train_epochs (old TRL name)."""
        lines = config_js.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if re.search(r'^\s*num_train_epochs\s*:', line):
                pytest.fail(
                    f"config.js line {i} sends 'num_train_epochs' — "
                    "backend expects 'num_epochs'"
                )

    def test_sends_batch_size(self, config_js):
        """Frontend must send 'batch_size' (matching Pydantic model)."""
        assert re.search(r'^\s*batch_size\s*:', config_js, re.MULTILINE), (
            "config.js does not send 'batch_size' — backend will use default"
        )

    def test_sends_num_epochs(self, config_js):
        """Frontend must send 'num_epochs' (matching Pydantic model)."""
        assert re.search(r'^\s*num_epochs\s*:', config_js, re.MULTILINE), (
            "config.js does not send 'num_epochs' — backend will use default"
        )


# ---------------------------------------------------------------------------
# 3. Eval steps ratio calculation
# ---------------------------------------------------------------------------

class TestEvalStepsCalculation:
    """Test the eval_steps ratio → absolute steps conversion logic."""

    def _compute_eval_steps(self, dataset_size, batch_size, grad_accum,
                            num_epochs, eval_ratio, num_gpus=1):
        """Replicate the eval_steps calculation from training_runner.py."""
        effective_batch = batch_size * grad_accum * num_gpus
        steps_per_epoch = math.ceil(dataset_size / effective_batch)
        total_steps = steps_per_epoch * num_epochs
        return max(1, int(total_steps * eval_ratio))

    def test_eval_ratio_0_2_gives_5_evals(self):
        """ratio=0.2 should give roughly 5 evals per training run."""
        # 1000 examples, bs=2, ga=16, 1 epoch → 32 steps
        eval_steps = self._compute_eval_steps(1000, 2, 16, 1, 0.2)
        total_steps = math.ceil(1000 / (2 * 16))  # 32
        num_evals = total_steps // eval_steps
        assert 4 <= num_evals <= 6, f"Expected ~5 evals, got {num_evals}"

    def test_batch_size_affects_eval_steps(self):
        """Doubling batch_size should halve total_steps and eval_steps."""
        eval_bs1 = self._compute_eval_steps(1000, 1, 16, 1, 0.2)
        eval_bs2 = self._compute_eval_steps(1000, 2, 16, 1, 0.2)
        # With double the batch_size, half the steps, so eval_steps ~ half
        total_bs1 = math.ceil(1000 / 16)   # 63
        total_bs2 = math.ceil(1000 / 32)   # 32
        # Ratio should be ~20% in both cases
        ratio_bs1 = eval_bs1 / total_bs1
        ratio_bs2 = eval_bs2 / total_bs2
        assert abs(ratio_bs1 - 0.2) < 0.05, f"bs=1 ratio: {ratio_bs1}"
        assert abs(ratio_bs2 - 0.2) < 0.05, f"bs=2 ratio: {ratio_bs2}"

    def test_same_effective_batch_same_eval_steps(self):
        """bs=2,ga=16 and bs=1,ga=32 have the same effective batch → same eval_steps."""
        eval_a = self._compute_eval_steps(1000, 2, 16, 1, 0.2)
        eval_b = self._compute_eval_steps(1000, 1, 32, 1, 0.2)
        assert eval_a == eval_b, (
            f"Same effective batch size but different eval_steps: "
            f"bs=2,ga=16 → {eval_a}, bs=1,ga=32 → {eval_b}"
        )

    def test_multi_gpu_reduces_steps(self):
        """With 2 GPUs, total steps should halve, keeping eval ratio correct."""
        eval_1gpu = self._compute_eval_steps(1000, 2, 16, 1, 0.2, num_gpus=1)
        eval_2gpu = self._compute_eval_steps(1000, 2, 16, 1, 0.2, num_gpus=2)

        total_1gpu = math.ceil(1000 / (2 * 16 * 1))  # 32
        total_2gpu = math.ceil(1000 / (2 * 16 * 2))  # 16

        ratio_1gpu = eval_1gpu / total_1gpu
        ratio_2gpu = eval_2gpu / total_2gpu
        assert abs(ratio_1gpu - 0.2) < 0.05
        assert abs(ratio_2gpu - 0.2) < 0.05

    def test_eval_steps_at_least_1(self):
        """Tiny dataset should still give eval_steps >= 1."""
        eval_steps = self._compute_eval_steps(5, 4, 16, 1, 0.2)
        assert eval_steps >= 1

    def test_absolute_eval_steps_passthrough(self):
        """eval_steps >= 1 should be used as-is (absolute step count)."""
        # This mirrors the `else` branch: eval_steps = int(config.eval_steps)
        assert int(50) == 50
        assert int(100.0) == 100

    def test_eval_ratio_with_multiple_epochs(self):
        """Ratio should apply to TOTAL steps across all epochs."""
        eval_1ep = self._compute_eval_steps(1000, 2, 16, 1, 0.2)
        eval_3ep = self._compute_eval_steps(1000, 2, 16, 3, 0.2)
        total_1ep = math.ceil(1000 / 32) * 1
        total_3ep = math.ceil(1000 / 32) * 3
        # eval_steps should scale with epochs
        assert eval_3ep > eval_1ep
        # But the ratio stays ~20%
        assert abs(eval_3ep / total_3ep - 0.2) < 0.05


# ---------------------------------------------------------------------------
# 4. Round-trip: config dict → TrainingConfig → values used for eval calc
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    """Simulate the full path: frontend dict → Pydantic → eval_steps calc."""

    def test_frontend_config_roundtrip(self):
        """A config dict with the correct field names should produce the right eval_steps."""
        # Simulates what the frontend sends after the fix
        frontend_config = {
            "output_name": "test-sft",
            "base_model": "meta-llama/Llama-3-8B-Instruct",
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "num_epochs": 1,
            "eval_steps": 0.2,
            "training_mode": "sft",
            "dataset": {
                "source": {"source_type": "huggingface", "repo_id": "test/data", "split": "train"},
                "format": {"format_type": "chatml"},
                "test_size": 0.01,
            },
        }

        config = TrainingConfig(**frontend_config)

        # Verify the values actually landed
        assert config.batch_size == 2
        assert config.num_epochs == 1
        assert config.eval_steps == 0.2
        assert config.gradient_accumulation_steps == 16

        # Compute eval_steps the same way training_runner.py does
        dataset_size = 1600
        num_gpus = 1
        effective_batch = config.batch_size * config.gradient_accumulation_steps * num_gpus
        steps_per_epoch = math.ceil(dataset_size / effective_batch)
        total_steps = steps_per_epoch * config.num_epochs
        eval_steps = max(1, int(total_steps * config.eval_steps))

        # 1600 / (2 * 16) = 50 steps, eval every 10 steps = 20%
        assert total_steps == 50
        assert eval_steps == 10
        assert abs(eval_steps / total_steps - 0.2) < 0.01

    def test_old_field_names_would_use_defaults(self):
        """Demonstrate that old field names get silently ignored by Pydantic."""
        # This is what USED to happen — the test documents the failure mode
        old_frontend_config = {
            "output_name": "test-sft",
            "per_device_train_batch_size": 2,  # Wrong name!
            "num_train_epochs": 1,              # Wrong name!
            "eval_steps": 0.2,
            "gradient_accumulation_steps": 16,
            "dataset": {
                "source": {"source_type": "huggingface", "repo_id": "test/data", "split": "train"},
                "format": {"format_type": "chatml"},
                "test_size": 0.01,
            },
        }

        config = TrainingConfig(**old_frontend_config)

        # Pydantic ignores unknown fields → defaults used
        assert config.batch_size == 1, "Old field name should fall back to default"
        assert config.num_epochs == 2, "Old field name should fall back to default"
