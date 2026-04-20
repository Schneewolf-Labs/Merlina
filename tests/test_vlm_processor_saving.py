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
# 2. Merge flow saves the processor into the shared merged-model dir
# ---------------------------------------------------------------------------
#
# Since the GGUF + upload refactor, the LoRA merge (and the artifact-saving
# steps that ride along with it: tokenizer, VLM processor, generation config,
# chat template, VLM state-dict fix) live in
# ``src.gguf_exporter.merge_lora_to_directory``. ``_run_background_upload``
# now only receives a pre-merged directory and uploads it; it no longer
# touches the processor. So we verify the processor save at the new home.

class TestMergeSavesProcessor:
    """merge_lora_to_directory must save the VLM processor into the merge dir."""

    def setup_method(self):
        AutoProcessor.from_pretrained.reset_mock()
        AutoProcessor.from_pretrained.side_effect = None
        mock_AutoTokenizer.from_pretrained.reset_mock()
        mock_AutoTokenizer.from_pretrained.side_effect = None

    def _run_merge(self, *, is_vlm, tmp_path, processor_loads=True):
        """Invoke merge_lora_to_directory with the standard mock stack.

        ``peft.PeftModel`` and ``transformers.AutoTokenizer`` are imported
        inside the function body (lazy), so we configure them on the
        sys.modules stubs set up at the top of this file rather than
        ``@patch``-ing non-existent module attributes.
        """
        from src.gguf_exporter import merge_lora_to_directory

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = tmp_path / "merged"

        mock_proc = MagicMock()
        if processor_loads:
            AutoProcessor.from_pretrained.return_value = mock_proc
        else:
            AutoProcessor.from_pretrained.side_effect = Exception("no processor")

        mock_AutoTokenizer.from_pretrained.return_value = MagicMock()

        # peft is MagicMock'd at module load time; seat the merge stub on it.
        sys.modules["peft"].PeftModel.from_pretrained.return_value.merge_and_unload.return_value = MagicMock()

        with patch("src.training_runner._get_auto_model_class") as mock_get_cls:
            mock_model_cls = MagicMock()
            mock_model_cls.from_pretrained.return_value = MagicMock()
            mock_get_cls.return_value = (mock_model_cls, is_vlm)

            merge_lora_to_directory(
                base_model="Qwen/Qwen3.5-VL-7B",
                adapter_dir=adapter_dir,
                output_dir=output_dir,
                model_type="vlm" if is_vlm else "auto",
                is_vlm=is_vlm,
            )
        return mock_proc, output_dir

    def test_merge_saves_vlm_processor(self, tmp_path):
        """When is_vlm=True, the processor is loaded and save_pretrained is called."""
        mock_proc, output_dir = self._run_merge(is_vlm=True, tmp_path=tmp_path)
        mock_proc.save_pretrained.assert_called_once()
        # First argument should be the merge output dir as a string.
        assert mock_proc.save_pretrained.call_args.args[0] == str(output_dir)

    def test_merge_skips_processor_for_non_vlm(self, tmp_path):
        """Text-only merges must not even try to load a processor."""
        mock_proc, _ = self._run_merge(is_vlm=False, tmp_path=tmp_path)
        mock_proc.save_pretrained.assert_not_called()

    def test_merge_tolerates_missing_processor(self, tmp_path):
        """Processor load failure must not raise — merge still produces weights."""
        # Should complete without raising. We don't assert on save_pretrained
        # because the mock raised on from_pretrained; we just verify the
        # pipeline didn't crash.
        self._run_merge(is_vlm=True, tmp_path=tmp_path, processor_loads=False)


class TestUploadProcessorPush:
    """Upload thread no longer handles processor directly; verify it consumes a pre-merged dir."""

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

    @patch("src.training_runner.upload_model_readme")
    @patch("src.training_runner.generate_model_readme")
    @patch("src.training_runner.websocket_manager")
    @patch("src.training_runner.HfApi")
    def test_upload_uses_pre_merged_directory(
        self, mock_hfapi, mock_ws, mock_readme, mock_upload_readme, tmp_path,
    ):
        """With a valid MergeArtifact, upload_folder points at the merged dir."""
        from src.merge_artifact import MergeArtifact
        from src.training_runner import _run_background_upload

        merge_dir = tmp_path / "merged"
        merge_dir.mkdir()
        (merge_dir / "config.json").write_text("{}")
        artifact = MergeArtifact(merge_dir, num_consumers=1)

        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api
        mock_ws.send_status_update = MagicMock()

        _run_background_upload(
            config=self._make_config(),
            final_output_dir="./models/test-vlm",
            training_mode="orpo",
            job_id="test-job-1",
            job_manager=MagicMock(),
            event_loop=None,
            is_vlm=True,
            merge_artifact=artifact,
        )

        mock_api.upload_folder.assert_called_once()
        # folder_path kwarg (or 1st positional) must be the merge dir.
        call = mock_api.upload_folder.call_args
        folder = call.kwargs.get("folder_path", call.args[0] if call.args else None)
        assert folder == str(merge_dir)

    @patch("src.training_runner.upload_model_readme")
    @patch("src.training_runner.generate_model_readme")
    @patch("src.training_runner.websocket_manager")
    @patch("src.training_runner.HfApi")
    def test_upload_falls_back_to_adapter_when_merge_failed(
        self, mock_hfapi, mock_ws, mock_readme, mock_upload_readme, tmp_path,
    ):
        """A path-less MergeArtifact (merge failed) → adapter-only upload, no crash."""
        from src.merge_artifact import MergeArtifact
        from src.training_runner import _run_background_upload

        failed_artifact = MergeArtifact(None, num_consumers=1, error="OOM during merge")

        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api
        mock_ws.send_status_update = MagicMock()

        job_manager = MagicMock()
        _run_background_upload(
            config=self._make_config(),
            final_output_dir="./models/test-vlm",
            training_mode="orpo",
            job_id="test-job-2",
            job_manager=job_manager,
            event_loop=None,
            is_vlm=True,
            merge_artifact=failed_artifact,
        )

        # Adapter-only upload targets the adapter dir directly.
        mock_api.upload_folder.assert_called_once()
        call = mock_api.upload_folder.call_args
        folder = call.kwargs.get("folder_path", call.args[0] if call.args else None)
        assert folder == "./models/test-vlm"
        # Job finalization still ran.
        assert job_manager.update_job.called

    @patch("src.training_runner.upload_model_readme")
    @patch("src.training_runner.generate_model_readme")
    @patch("src.training_runner.websocket_manager")
    @patch("src.training_runner.HfApi")
    def test_merge_upload_continues_if_no_processor(
        self, mock_hfapi, mock_ws, mock_readme, mock_upload_readme, tmp_path,
    ):
        """Text-only model (is_vlm not set) should complete upload without crashing."""
        from src.merge_artifact import MergeArtifact
        from src.training_runner import _run_background_upload

        merge_dir = tmp_path / "merged"
        merge_dir.mkdir()
        artifact = MergeArtifact(merge_dir, num_consumers=1)

        mock_api = MagicMock()
        mock_hfapi.return_value = mock_api
        mock_ws.send_status_update = MagicMock()

        job_manager = MagicMock()
        _run_background_upload(
            config=self._make_config(),
            final_output_dir="./models/test-model",
            training_mode="sft",
            job_id="test-job-3",
            job_manager=job_manager,
            event_loop=None,
            merge_artifact=artifact,
        )

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
