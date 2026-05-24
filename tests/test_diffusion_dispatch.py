"""Lightweight tests for the diffusion backend dispatch + Pydantic schema.

These tests don't load any model weights — they exercise the dispatch
glue, the Pydantic TrainingConfig surface, and the diffusion mode
registry. Heavy end-to-end tests (real GPU + real model) live elsewhere
and are skipped under pytest.
"""
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------
# _resolve_sibling_runner: text/vlm/diffusion routing
# ---------------------------------------------------------------------

def _cfg(**fields):
    """Build a config-like object with only the dispatch-relevant fields."""
    return SimpleNamespace(model_type=fields.get("model_type", "auto"),
                           training_mode=fields.get("training_mode", "orpo"))


class TestResolveSiblingRunner:
    def test_text_mode_returns_none(self):
        from src.training_runner import _resolve_sibling_runner
        assert _resolve_sibling_runner(_cfg(training_mode="orpo")) is None
        assert _resolve_sibling_runner(_cfg(training_mode="sft")) is None
        assert _resolve_sibling_runner(_cfg(training_mode="dpo")) is None

    def test_vlm_prefix_dispatches_to_vlm_runner(self):
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(training_mode="vlm_stage1"))
        assert fn is not None
        assert fn.__name__ == "run_vlm_training_sync"

    def test_diffusion_prefix_dispatches_to_diffusion_runner(self):
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(training_mode="diffusion_qwen_image"))
        assert fn is not None
        assert fn.__name__ == "run_diffusion_training_sync"

    def test_explicit_model_type_vlm_wins(self):
        """model_type='vlm' dispatches to VLM even with a non-vlm_ training_mode."""
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(model_type="vlm", training_mode="sft"))
        assert fn is not None
        assert fn.__name__ == "run_vlm_training_sync"

    def test_explicit_model_type_diffusion_wins(self):
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(model_type="diffusion", training_mode="sft"))
        assert fn is not None
        assert fn.__name__ == "run_diffusion_training_sync"

    def test_auto_model_type_falls_back_to_training_mode(self):
        from src.training_runner import _resolve_sibling_runner
        assert _resolve_sibling_runner(_cfg(model_type="auto", training_mode="sft")) is None
        assert _resolve_sibling_runner(
            _cfg(model_type="auto", training_mode="vlm_stage1")
        ).__name__ == "run_vlm_training_sync"

    def test_missing_attrs_doesnt_crash(self):
        """Bare object without model_type / training_mode falls through to text."""
        from src.training_runner import _resolve_sibling_runner
        empty = SimpleNamespace()
        assert _resolve_sibling_runner(empty) is None


# ---------------------------------------------------------------------
# diffusion runner registry
# ---------------------------------------------------------------------

class TestDiffusionModeRegistry:
    def test_known_modes(self):
        from src.training_runner_diffusion import _DIFFUSION_MODES
        assert "diffusion_qwen_image" in _DIFFUSION_MODES
        assert "diffusion_qwen_edit" in _DIFFUSION_MODES
        assert "diffusion_sdxl" in _DIFFUSION_MODES

    def test_resolve_unknown_mode_raises(self):
        from src.training_runner_diffusion import _resolve_adapter_and_loss
        with pytest.raises(ValueError, match="Unknown diffusion training_mode"):
            _resolve_adapter_and_loss("diffusion_not_real")

    def test_mode_tuple_shape(self):
        """Each registry entry is (adapter_name, loss_name, default_targets)."""
        from src.training_runner_diffusion import _DIFFUSION_MODES
        for mode, spec in _DIFFUSION_MODES.items():
            assert len(spec) == 3, f"{mode}: expected 3-tuple, got {spec}"
            adapter_name, loss_name, targets = spec
            assert isinstance(adapter_name, str)
            assert isinstance(loss_name, str)
            assert isinstance(targets, list) and len(targets) > 0


# ---------------------------------------------------------------------
# Pydantic TrainingConfig — diffusion fields accepted
# ---------------------------------------------------------------------

class TestTrainingConfigDiffusionFields:
    def test_diffusion_fields_optional(self):
        """A text-mode request should not need any diffusion field set."""
        from merlina import TrainingConfig
        cfg = TrainingConfig(output_name="text-run")
        assert cfg.model_name is None
        assert cfg.image_resolution is None
        assert cfg.lora_rank is None
        assert cfg.dataset_jsonl_path is None

    def test_diffusion_fields_round_trip(self):
        from merlina import TrainingConfig
        cfg = TrainingConfig(
            output_name="diffusion-run",
            model_type="diffusion",
            training_mode="diffusion_qwen_image",
            model_name="Qwen/Qwen-Image",
            image_resolution=1024,
            lora_rank=32,
            lora_target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            dataset_jsonl_path="/tmp/train.jsonl",
        )
        assert cfg.model_name == "Qwen/Qwen-Image"
        assert cfg.image_resolution == 1024
        assert cfg.lora_rank == 32
        assert cfg.lora_target_modules == ["to_k", "to_q", "to_v", "to_out.0"]

    def test_image_resolution_bounds(self):
        from merlina import TrainingConfig
        with pytest.raises(Exception):  # Pydantic ValidationError
            TrainingConfig(output_name="x", image_resolution=100)  # below min 256
        with pytest.raises(Exception):
            TrainingConfig(output_name="x", image_resolution=4096)  # above max 2048

    def test_lora_rank_bounds(self):
        from merlina import TrainingConfig
        with pytest.raises(Exception):
            TrainingConfig(output_name="x", lora_rank=2)  # below min 4
