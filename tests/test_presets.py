"""
Tests for training presets (recommended hyperparameters per method).

Validates that presets exist for all training modes, contain required fields,
and have sane values backed by paper recommendations.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.presets import get_preset, get_all_presets, TRAINING_PRESETS


# ---------------------------------------------------------------------------
# All supported training modes
# ---------------------------------------------------------------------------

ALL_MODES = ["sft", "orpo", "dpo", "simpo", "cpo", "ipo", "kto"]
PREFERENCE_MODES = ["orpo", "dpo", "simpo", "cpo", "ipo", "kto"]

# Fields every preset must have
REQUIRED_SETTINGS = {
    "learning_rate", "num_epochs", "batch_size",
    "gradient_accumulation_steps", "warmup_ratio", "weight_decay",
    "lora_dropout", "max_grad_norm", "lr_scheduler_type",
}


# ---------------------------------------------------------------------------
# 1. Preset existence and structure
# ---------------------------------------------------------------------------

class TestPresetStructure:
    """Every training mode has a well-formed preset."""

    def test_all_modes_have_presets(self):
        for mode in ALL_MODES:
            preset = get_preset(mode)
            assert preset is not None, f"Missing preset for '{mode}'"

    def test_unknown_mode_returns_none(self):
        assert get_preset("nonexistent") is None

    def test_get_all_presets_returns_all(self):
        all_presets = get_all_presets()
        for mode in ALL_MODES:
            assert mode in all_presets, f"'{mode}' missing from get_all_presets()"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_preset_has_label(self, mode):
        preset = get_preset(mode)
        assert "label" in preset
        assert isinstance(preset["label"], str)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_preset_has_notes(self, mode):
        preset = get_preset(mode)
        assert "notes" in preset
        assert len(preset["notes"]) > 10, "Notes should explain the recommendations"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_preset_has_required_settings(self, mode):
        settings = get_preset(mode)["settings"]
        missing = REQUIRED_SETTINGS - set(settings.keys())
        assert not missing, f"Preset '{mode}' missing settings: {missing}"


# ---------------------------------------------------------------------------
# 2. Value sanity checks (paper-backed)
# ---------------------------------------------------------------------------

class TestPresetValues:
    """Preset values match paper recommendations."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_learning_rate_positive(self, mode):
        lr = get_preset(mode)["settings"]["learning_rate"]
        assert 1e-8 < lr < 1e-2, f"{mode} LR {lr} out of range"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_epochs_reasonable(self, mode):
        epochs = get_preset(mode)["settings"]["num_epochs"]
        assert 1 <= epochs <= 5, f"{mode} epochs={epochs} out of range"

    @pytest.mark.parametrize("mode", PREFERENCE_MODES)
    def test_preference_modes_have_beta(self, mode):
        settings = get_preset(mode)["settings"]
        assert "beta" in settings, f"Preference mode '{mode}' must have beta"

    @pytest.mark.parametrize("mode", PREFERENCE_MODES)
    def test_preference_modes_zero_lora_dropout(self, mode):
        """Paper consensus: disable dropout for preference optimization."""
        dropout = get_preset(mode)["settings"]["lora_dropout"]
        assert dropout == 0.0, f"{mode} should have lora_dropout=0 for preference training"

    @pytest.mark.parametrize("mode", PREFERENCE_MODES)
    def test_preference_modes_one_epoch(self, mode):
        """Paper consensus: preference methods overfit fast, 1 epoch is optimal."""
        epochs = get_preset(mode)["settings"]["num_epochs"]
        assert epochs == 1, f"{mode} should default to 1 epoch (overfitting risk)"

    def test_sft_higher_learning_rate(self):
        """SFT LoRA learning rate should be ~10x higher than preference methods."""
        sft_lr = get_preset("sft")["settings"]["learning_rate"]
        dpo_lr = get_preset("dpo")["settings"]["learning_rate"]
        assert sft_lr > dpo_lr * 5, (
            f"SFT LR ({sft_lr}) should be much higher than DPO LR ({dpo_lr})"
        )

    def test_simpo_beta_is_high(self):
        """SimPO paper uses beta 2.0-10.0, NOT the DPO-like 0.1."""
        beta = get_preset("simpo")["settings"]["beta"]
        assert beta >= 2.0, f"SimPO beta should be >= 2.0 (paper), got {beta}"

    def test_simpo_has_gamma(self):
        gamma = get_preset("simpo")["settings"]["gamma"]
        assert gamma > 0, "SimPO should have positive gamma (reward margin)"

    def test_ipo_beta_is_small(self):
        """IPO beta=0.01 was optimal across models in HF benchmarks."""
        beta = get_preset("ipo")["settings"]["beta"]
        assert beta <= 0.05, f"IPO beta should be small (0.01 recommended), got {beta}"

    def test_kto_batch_size_at_least_4(self):
        """KTO needs per-step batch_size >= 4 for stable KL estimation."""
        batch_size = get_preset("kto")["settings"]["batch_size"]
        assert batch_size >= 4, f"KTO batch_size must be >= 4 (KL estimation), got {batch_size}"

    def test_dpo_has_label_smoothing(self):
        settings = get_preset("dpo")["settings"]
        assert "label_smoothing" in settings

    def test_cpo_has_label_smoothing(self):
        settings = get_preset("cpo")["settings"]
        assert "label_smoothing" in settings


# ---------------------------------------------------------------------------
# 3. API endpoint (via TestClient)
# ---------------------------------------------------------------------------

# Mock heavy dependencies for FastAPI import
class FakeTensor:
    pass

class FakeModule:
    pass

mock_torch = MagicMock()
mock_torch.__spec__ = MagicMock()
mock_torch.Tensor = FakeTensor
mock_torch.nn = MagicMock()
mock_torch.nn.Module = FakeModule
mock_torch.cuda.is_available.return_value = True
mock_torch.cuda.device_count.return_value = 1
mock_torch.cuda.get_device_capability.return_value = (8, 6)
mock_torch.cuda.get_device_name.return_value = "Mock GPU"
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
sys.modules.setdefault('torch', mock_torch)
sys.modules.setdefault('torch.cuda', mock_torch.cuda)
sys.modules.setdefault('torch.nn', mock_torch.nn)
sys.modules.setdefault('torch.utils', MagicMock())
sys.modules.setdefault('torch.utils.data', MagicMock())

class FakeTokenizerBase:
    pass

mock_transformers = MagicMock()
mock_transformers.PreTrainedTokenizerBase = FakeTokenizerBase
sys.modules.setdefault('transformers', mock_transformers)

for module in [
    'peft', 'accelerate', 'accelerate.utils', 'bitsandbytes', 'wandb',
    'psutil', 'pynvml', 'huggingface_hub',
    'grimoire', 'grimoire.losses', 'grimoire.data',
    'grimoire.trainer', 'grimoire.callbacks', 'grimoire.config',
    'flash_attn', 'datasets',
]:
    sys.modules.setdefault(module, MagicMock())

sys.modules.setdefault('dataset_handlers', MagicMock())

from fastapi.testclient import TestClient


class TestPresetsAPI:
    """Test the /presets endpoints."""

    @pytest.fixture
    def client(self):
        from merlina import app
        return TestClient(app)

    def test_get_preset_returns_200(self, client):
        response = client.get("/presets/orpo")
        assert response.status_code == 200

    def test_get_preset_has_settings(self, client):
        data = client.get("/presets/dpo").json()
        assert "settings" in data
        assert "learning_rate" in data["settings"]
        assert "beta" in data["settings"]

    def test_get_preset_has_notes(self, client):
        data = client.get("/presets/simpo").json()
        assert "notes" in data
        assert len(data["notes"]) > 0

    def test_unknown_mode_returns_404(self, client):
        response = client.get("/presets/nonexistent")
        assert response.status_code == 404

    def test_list_all_presets(self, client):
        data = client.get("/presets").json()
        for mode in ALL_MODES:
            assert mode in data, f"'{mode}' missing from /presets response"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_each_mode_endpoint(self, client, mode):
        response = client.get(f"/presets/{mode}")
        assert response.status_code == 200
        data = response.json()
        assert "settings" in data
        assert "label" in data
        assert "notes" in data
