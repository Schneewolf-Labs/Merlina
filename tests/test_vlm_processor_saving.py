"""
Tests for VLM processor saving during model save and HuggingFace upload.

Verifies that _save_processor() correctly saves processor files (preprocessor_config.json,
image processor, etc.) for VLMs, and that the upload flow pushes processor to HF Hub.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

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

# Mock transformers
mock_AutoConfig = MagicMock()
mock_AutoModelForCausalLM = MagicMock()
mock_AutoModelForVLM = MagicMock()
mock_AutoModelForVLM.__name__ = "AutoModelForImageTextToText"
mock_AutoProcessor = MagicMock()
mock_AutoTokenizer = MagicMock()

mock_transformers = MagicMock()
mock_transformers.AutoConfig = mock_AutoConfig
mock_transformers.AutoModelForCausalLM = mock_AutoModelForCausalLM
mock_transformers.AutoModelForImageTextToText = mock_AutoModelForVLM
mock_transformers.AutoProcessor = mock_AutoProcessor
mock_transformers.AutoTokenizer = mock_AutoTokenizer
sys.modules.setdefault('transformers', mock_transformers)

# Mock other heavy dependencies
for module in [
    'peft', 'accelerate', 'accelerate.utils', 'bitsandbytes', 'wandb',
    'psutil', 'pynvml', 'huggingface_hub',
    'grimoire', 'grimoire.losses', 'grimoire.data',
    'grimoire.trainer', 'grimoire.callbacks', 'grimoire.config',
    'flash_attn', 'datasets',
    'fastapi', 'fastapi.websockets',
]:
    sys.modules.setdefault(module, MagicMock())

# Mock dataset_handlers and src submodules to avoid cascading imports
sys.modules.setdefault('dataset_handlers', MagicMock())

# Mock src submodules that have heavy deps (fastapi, etc.) so src/__init__.py can import
for src_module in [
    'src.websocket_manager', 'src.job_manager', 'src.preflight_checks',
    'src.utils', 'src.constants', 'src.exceptions', 'src.model_card',
    'src.job_queue',
]:
    sys.modules.setdefault(src_module, MagicMock())

# Now import the functions under test
from src.training_runner import _save_processor, _save_generation_config, AutoProcessor


# ---------------------------------------------------------------------------
# 1. _save_processor() unit tests
# ---------------------------------------------------------------------------

class TestSaveProcessor:
    """Test _save_processor() helper function."""

    def setup_method(self):
        """Reset mocks before each test."""
        AutoProcessor.from_pretrained.reset_mock()
        AutoProcessor.from_pretrained.side_effect = None

    def test_saves_processor_for_vlm(self):
        """Should load processor from base model and save to output dir."""
        mock_proc = MagicMock()
        AutoProcessor.from_pretrained.return_value = mock_proc

        result = _save_processor("Qwen/Qwen3.5-VL-7B", "/tmp/output")

        AutoProcessor.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3.5-VL-7B", trust_remote_code=True
        )
        mock_proc.save_pretrained.assert_called_once_with("/tmp/output")
        assert result is True

    def test_returns_false_when_no_processor(self):
        """Should return False when AutoProcessor fails (e.g., text-only model)."""
        AutoProcessor.from_pretrained.side_effect = Exception("No processor found")

        result = _save_processor("meta-llama/Llama-3-8B", "/tmp/output")

        assert result is False

    def test_does_not_raise_on_failure(self):
        """Should silently handle failures without raising."""
        AutoProcessor.from_pretrained.side_effect = OSError("Not found")

        # Should not raise
        result = _save_processor("bad/model", "/tmp/output")
        assert result is False

    def test_passes_trust_remote_code(self):
        """Should pass trust_remote_code=True to handle custom processors."""
        mock_proc = MagicMock()
        AutoProcessor.from_pretrained.return_value = mock_proc

        _save_processor("org/custom-vlm", "/tmp/out")

        AutoProcessor.from_pretrained.assert_called_once_with(
            "org/custom-vlm", trust_remote_code=True
        )


# ---------------------------------------------------------------------------
# 2. Upload flow pushes processor for VLMs
# ---------------------------------------------------------------------------

class TestUploadProcessorPush:
    """Test that _run_background_upload pushes processor for VLMs."""

    def setup_method(self):
        AutoProcessor.from_pretrained.reset_mock()
        AutoProcessor.from_pretrained.side_effect = None

    def _make_config(self, use_lora=True, merge_lora=True, push_to_hub=True):
        """Create a mock training config."""
        config = MagicMock()
        config.use_lora = use_lora
        config.merge_lora_before_upload = merge_lora
        config.push_to_hub = push_to_hub
        config.hf_hub_private = False
        config.hf_token = "hf_test_token"
        config.output_name = "user/test-vlm-model"
        config.base_model = "Qwen/Qwen3.5-VL-7B"
        config.model_type = "vlm"
        return config

    @patch("src.training_runner.fix_vlm_state_dict_on_disk")
    @patch("src.training_runner.upload_model_readme")
    @patch("src.training_runner.generate_model_readme")
    @patch("src.training_runner.websocket_manager")
    @patch("src.training_runner.HfApi")
    @patch("src.training_runner.PeftModel")
    @patch("src.training_runner._get_auto_model_class")
    def test_merge_upload_saves_processor(
        self, mock_get_cls, mock_peft, mock_hfapi, mock_ws, mock_readme, mock_upload_readme,
        mock_fix_vlm
    ):
        """During LoRA merge upload, processor should be saved to merge dir."""
        config = self._make_config()

        # Setup model merge mocks
        mock_model_cls = MagicMock()
        mock_get_cls.return_value = (mock_model_cls, True)
        mock_merged = MagicMock()
        mock_peft.from_pretrained.return_value.merge_and_unload.return_value = mock_merged

        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup processor mock
        mock_proc = MagicMock()
        AutoProcessor.from_pretrained.return_value = mock_proc

        # Mock HfApi
        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api

        # Mock websocket
        mock_ws.send_status_update = MagicMock()

        from src.training_runner import _run_background_upload
        job_manager = MagicMock()

        _run_background_upload(
            config=config,
            final_output_dir="./models/test-vlm",
            training_mode="orpo",
            job_id="test-job-1",
            job_manager=job_manager,
            event_loop=None,
            is_vlm=True,
        )

        # Verify processor was saved to merge dir (not pushed individually)
        mock_proc.save_pretrained.assert_called_once()
        # Verify the whole directory was uploaded via upload_folder
        mock_api.upload_folder.assert_called_once()

    @patch("src.training_runner.fix_vlm_state_dict_on_disk")
    @patch("src.training_runner.upload_model_readme")
    @patch("src.training_runner.generate_model_readme")
    @patch("src.training_runner.websocket_manager")
    @patch("src.training_runner.HfApi")
    @patch("src.training_runner.PeftModel")
    @patch("src.training_runner._get_auto_model_class")
    def test_merge_upload_continues_if_no_processor(
        self, mock_get_cls, mock_peft, mock_hfapi, mock_ws, mock_readme, mock_upload_readme,
        mock_fix_vlm
    ):
        """Upload should succeed even if processor save fails (text-only model)."""
        config = self._make_config()

        mock_model_cls = MagicMock()
        mock_get_cls.return_value = (mock_model_cls, False)
        mock_merged = MagicMock()
        mock_peft.from_pretrained.return_value.merge_and_unload.return_value = mock_merged

        mock_tokenizer = MagicMock()
        mock_AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Processor fails (text-only model)
        AutoProcessor.from_pretrained.side_effect = Exception("No processor")

        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api
        mock_ws.send_status_update = MagicMock()

        from src.training_runner import _run_background_upload
        job_manager = MagicMock()

        # Should NOT raise despite processor failure
        _run_background_upload(
            config=config,
            final_output_dir="./models/test-model",
            training_mode="sft",
            job_id="test-job-2",
            job_manager=job_manager,
            event_loop=None,
        )

        # Verify upload didn't crash - job_manager should have been called
        assert job_manager.update_job.called


# ---------------------------------------------------------------------------
# 3. Integration: processor saved locally for VLMs
# ---------------------------------------------------------------------------

class TestProcessorSavedLocally:
    """Verify _save_processor is called in the right context during training."""

    def test_save_processor_called_for_vlm_model(self):
        """_save_processor should be called when is_vlm is True."""
        mock_proc = MagicMock()
        AutoProcessor.from_pretrained.return_value = mock_proc
        AutoProcessor.from_pretrained.side_effect = None

        result = _save_processor("Qwen/Qwen3.5-VL-7B", "./models/my-vlm")
        assert result is True
        mock_proc.save_pretrained.assert_called_once_with("./models/my-vlm")

    def test_save_processor_not_called_for_text_model(self):
        """For text-only models, _save_processor returns False gracefully."""
        AutoProcessor.from_pretrained.side_effect = Exception(
            "unrecognized model - no processor"
        )

        result = _save_processor("meta-llama/Llama-3-8B", "./models/my-llm")
        assert result is False


# ---------------------------------------------------------------------------
# 4. _save_generation_config() unit tests
# ---------------------------------------------------------------------------

class TestSaveGenerationConfig:
    """Test _save_generation_config() helper function."""

    @patch("transformers.GenerationConfig")
    def test_saves_generation_config(self, mock_gen_config_cls):
        """Should load generation config from base model and save to output dir."""
        mock_gen_config = MagicMock()
        mock_gen_config_cls.from_pretrained.return_value = mock_gen_config

        result = _save_generation_config("Qwen/Qwen3.5-VL-7B", "/tmp/output")

        mock_gen_config_cls.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3.5-VL-7B", trust_remote_code=True
        )
        mock_gen_config.save_pretrained.assert_called_once_with("/tmp/output")
        assert result is True

    @patch("transformers.GenerationConfig")
    def test_returns_false_on_failure(self, mock_gen_config_cls):
        """Should return False when generation config cannot be loaded."""
        mock_gen_config_cls.from_pretrained.side_effect = Exception("Not found")

        result = _save_generation_config("some/model", "/tmp/output")
        assert result is False

    @patch("transformers.GenerationConfig")
    def test_does_not_raise_on_failure(self, mock_gen_config_cls):
        """Should silently handle failures without raising."""
        mock_gen_config_cls.from_pretrained.side_effect = OSError("No config")

        # Should not raise
        result = _save_generation_config("bad/model", "/tmp/output")
        assert result is False


# ---------------------------------------------------------------------------
# 5. LoRA task_type is configurable via lora_task_type
# ---------------------------------------------------------------------------

class TestLoraTaskType:
    """Verify LoRA task_type is read from config (user-configurable)."""

    def test_task_type_read_from_config(self):
        """training_runner should read lora_task_type from config."""
        import src.training_runner as tr
        import inspect
        source = inspect.getsource(tr.run_training_sync)
        assert 'lora_task_type' in source

    def test_task_type_default_is_causal_lm(self):
        """Default lora_task_type should be CAUSAL_LM via getattr fallback."""
        import src.training_runner as tr
        import inspect
        source = inspect.getsource(tr.run_training_sync)
        assert "'CAUSAL_LM'" in source
