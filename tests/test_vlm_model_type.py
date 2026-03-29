"""
Tests for VLM (Vision-Language Model) model type detection and loading.

Verifies that _detect_is_vlm() and _get_auto_model_class() correctly identify
VLMs vs text-only LLMs and return the appropriate AutoModel class.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Mock GPU-dependent imports BEFORE importing training_runner
# ============================================================================

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
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
sys.modules.setdefault('torch', mock_torch)
sys.modules.setdefault('torch.cuda', mock_torch.cuda)
sys.modules.setdefault('torch.nn', mock_torch.nn)
sys.modules.setdefault('torch.utils', MagicMock())
sys.modules.setdefault('torch.utils.data', MagicMock())

# Mock transformers — keep AutoConfig, AutoModelForCausalLM, AutoModelForVLM
# as distinct sentinel objects so we can assert identity
mock_AutoConfig = MagicMock()
mock_AutoModelForCausalLM = MagicMock()
mock_AutoModelForVLM = MagicMock()
mock_AutoModelForVLM.__name__ = "AutoModelForImageTextToText"

mock_transformers = MagicMock()
mock_transformers.AutoConfig = mock_AutoConfig
mock_transformers.AutoModelForCausalLM = mock_AutoModelForCausalLM
# Code imports AutoModelForImageTextToText (v5+) with fallback to AutoModelForVision2Seq (v4)
mock_transformers.AutoModelForImageTextToText = mock_AutoModelForVLM
sys.modules.setdefault('transformers', mock_transformers)

# Mock other heavy dependencies
for module in [
    'peft', 'accelerate', 'accelerate.utils', 'bitsandbytes', 'wandb',
    'psutil', 'pynvml', 'huggingface_hub',
    'grimoire', 'grimoire.losses', 'grimoire.data',
    'grimoire.trainer', 'grimoire.callbacks', 'grimoire.config',
    'flash_attn', 'datasets',
]:
    sys.modules.setdefault(module, MagicMock())

# Mock dataset_handlers to avoid cascading imports
mock_dataset_handlers = MagicMock()
sys.modules.setdefault('dataset_handlers', mock_dataset_handlers)

# Now import the functions under test
from src.training_runner import _detect_is_vlm, _get_auto_model_class, AutoConfig, AutoModelForCausalLM, AutoModelForVLM

# Also import TrainingConfig (doesn't need heavy mocks)
from merlina import TrainingConfig


# ---------------------------------------------------------------------------
# 1. Pydantic model accepts model_type field
# ---------------------------------------------------------------------------

class TestModelTypeConfig:
    """Ensure TrainingConfig handles model_type correctly."""

    def test_model_type_default_is_auto(self):
        config = TrainingConfig(output_name="test")
        assert config.model_type == "auto"

    def test_model_type_causal_lm(self):
        config = TrainingConfig(output_name="test", model_type="causal_lm")
        assert config.model_type == "causal_lm"

    def test_model_type_vlm(self):
        config = TrainingConfig(output_name="test", model_type="vlm")
        assert config.model_type == "vlm"

    def test_model_type_in_model_fields(self):
        """model_type must exist as a real field on TrainingConfig."""
        assert "model_type" in TrainingConfig.model_fields


# ---------------------------------------------------------------------------
# 2. VLM detection logic
# ---------------------------------------------------------------------------

def _make_mock_config(architectures=None, has_vision_config=False, has_visual=False):
    """Create a mock AutoConfig with given attributes."""
    mock_config = MagicMock()
    mock_config.architectures = architectures
    if has_vision_config:
        mock_config.vision_config = MagicMock()
    else:
        del mock_config.vision_config
    if has_visual:
        mock_config.visual = MagicMock()
    else:
        del mock_config.visual
    return mock_config


class TestDetectIsVLM:
    """Test _detect_is_vlm() with mocked AutoConfig responses."""

    def test_detects_vlm_by_vision_config(self):
        """Models with vision_config attribute should be detected as VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=["Qwen3_5ForConditionalGeneration"],
            has_vision_config=True,
        )
        assert _detect_is_vlm("Qwen/Qwen3.5-VL-7B") is True

    def test_detects_vlm_by_visual_attr(self):
        """Models with visual attribute should be detected as VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=["SomeVLModel"],
            has_visual=True,
        )
        assert _detect_is_vlm("org/some-vl-model") is True

    def test_detects_vlm_by_architecture_conditional_generation(self):
        """ForConditionalGeneration architecture should be detected as VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
        )
        assert _detect_is_vlm("Qwen/Qwen2.5-VL-7B") is True

    def test_detects_vlm_by_architecture_for_vision(self):
        """ForVision architecture pattern should be detected as VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=["SomeModelForVision2Seq"],
        )
        assert _detect_is_vlm("org/some-vision-model") is True

    def test_detects_vlm_by_architecture_image_text(self):
        """ImageText architecture pattern should be detected as VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=["LlavaForImageTextGeneration"],
        )
        assert _detect_is_vlm("org/llava-model") is True

    def test_text_only_causal_lm_not_vlm(self):
        """Standard CausalLM models should NOT be detected as VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=["LlamaForCausalLM"],
        )
        assert _detect_is_vlm("meta-llama/Llama-3-8B") is False

    def test_text_only_qwen_not_vlm(self):
        """Qwen text-only models should NOT be detected as VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=["Qwen2ForCausalLM"],
        )
        assert _detect_is_vlm("Qwen/Qwen2.5-7B") is False

    def test_no_architectures_not_vlm(self):
        """Models with no architectures field should default to not VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=None,
        )
        assert _detect_is_vlm("unknown/model") is False

    def test_empty_architectures_not_vlm(self):
        """Models with empty architectures list should default to not VLM."""
        AutoConfig.from_pretrained.return_value = _make_mock_config(
            architectures=[],
        )
        assert _detect_is_vlm("unknown/model") is False

    def test_config_load_failure_defaults_not_vlm(self):
        """If AutoConfig.from_pretrained fails, default to not VLM."""
        AutoConfig.from_pretrained.side_effect = Exception("Network error")
        assert _detect_is_vlm("nonexistent/model") is False
        # Reset side_effect for other tests
        AutoConfig.from_pretrained.side_effect = None


# ---------------------------------------------------------------------------
# 3. AutoModel class selection
# ---------------------------------------------------------------------------

class TestGetAutoModelClass:
    """Test _get_auto_model_class() returns the correct class."""

    def test_explicit_causal_lm_returns_causal(self):
        """model_type='causal_lm' should return AutoModelForCausalLM."""
        cls, is_vlm = _get_auto_model_class("any/model", model_type="causal_lm")
        assert cls is AutoModelForCausalLM
        assert is_vlm is False

    def test_explicit_vlm_returns_vision2seq(self):
        """model_type='vlm' should return AutoModelForVLM."""
        cls, is_vlm = _get_auto_model_class("any/model", model_type="vlm")
        assert cls is AutoModelForVLM
        assert is_vlm is True

    @patch("src.training_runner._detect_is_vlm", return_value=True)
    def test_auto_delegates_to_detection_vlm(self, mock_detect):
        """model_type='auto' should return VLM class when detected as VLM."""
        cls, is_vlm = _get_auto_model_class("Qwen/Qwen3.5-VL-7B", model_type="auto")
        mock_detect.assert_called_once_with("Qwen/Qwen3.5-VL-7B")
        assert cls is AutoModelForVLM
        assert is_vlm is True

    @patch("src.training_runner._detect_is_vlm", return_value=False)
    def test_auto_delegates_to_detection_causal(self, mock_detect):
        """model_type='auto' should return CausalLM when not detected as VLM."""
        cls, is_vlm = _get_auto_model_class("meta-llama/Llama-3-8B", model_type="auto")
        mock_detect.assert_called_once_with("meta-llama/Llama-3-8B")
        assert cls is AutoModelForCausalLM
        assert is_vlm is False

    def test_explicit_vlm_skips_detection(self):
        """model_type='vlm' should NOT call _detect_is_vlm."""
        with patch("src.training_runner._detect_is_vlm") as mock_detect:
            _get_auto_model_class("any/model", model_type="vlm")
            mock_detect.assert_not_called()

    def test_explicit_causal_lm_skips_detection(self):
        """model_type='causal_lm' should NOT call _detect_is_vlm."""
        with patch("src.training_runner._detect_is_vlm") as mock_detect:
            _get_auto_model_class("any/model", model_type="causal_lm")
            mock_detect.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Frontend config includes model_type
# ---------------------------------------------------------------------------

class TestFrontendModelType:
    """Verify model_type is wired through frontend JS files."""

    @pytest.fixture
    def app_js(self):
        js_path = Path(__file__).parent.parent / "frontend" / "js" / "app.js"
        if not js_path.exists():
            pytest.skip("frontend/js/app.js not found")
        return js_path.read_text()

    @pytest.fixture
    def config_js(self):
        js_path = Path(__file__).parent.parent / "frontend" / "js" / "config.js"
        if not js_path.exists():
            pytest.skip("frontend/js/config.js not found")
        return js_path.read_text()

    @pytest.fixture
    def index_html(self):
        html_path = Path(__file__).parent.parent / "frontend" / "index.html"
        if not html_path.exists():
            pytest.skip("frontend/index.html not found")
        return html_path.read_text()

    def test_app_js_sends_model_type(self, app_js):
        """app.js must include model_type in the config object."""
        assert "model_type" in app_js, "app.js should send model_type in config"

    def test_config_js_saves_model_type(self, config_js):
        """config.js must include model_type when saving config."""
        assert "model_type" in config_js, "config.js should save model_type"

    def test_index_html_has_model_type_select(self, index_html):
        """index.html must have a model-type select element."""
        assert 'id="model-type"' in index_html, "index.html should have model-type select"

    def test_index_html_has_vlm_option(self, index_html):
        """index.html must have a VLM option in the model-type select."""
        assert 'value="vlm"' in index_html, "index.html should have vlm option"

    def test_index_html_has_auto_option(self, index_html):
        """index.html must have an auto option in the model-type select."""
        assert 'value="auto"' in index_html, "index.html should have auto option"
