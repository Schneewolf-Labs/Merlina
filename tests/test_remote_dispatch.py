"""
Integration tests for remote-mode wiring in merlina.py and the
remote-aware preflight branch.

Run standalone: python -m pytest tests/test_remote_dispatch.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import merlina
from src.preflight_checks import PreflightValidator


def make_training_config(**overrides):
    base = {
        "base_model": "some-org/some-model",
        "output_name": "test-model",
        "hf_token": "hf_test",
        "dataset": {
            "source": {"source_type": "huggingface", "repo_id": "org/data", "split": "train"},
            "format": {"format_type": "chatml"},
        },
    }
    base.update(overrides)
    return merlina.TrainingConfig(**base)


class TestRemoteConfigModel:
    def test_remote_block_accepted(self):
        config = make_training_config(remote={"enabled": True, "gpu_type_id": "NVIDIA H200"})
        assert config.remote.enabled
        assert config.remote.provider == "runpod"
        assert config.remote.merge_strategy == "auto"

    def test_remote_defaults_to_none(self):
        assert make_training_config().remote is None

    def test_remote_survives_model_dump_roundtrip(self):
        from pydantic import TypeAdapter
        config = make_training_config(remote={"enabled": True, "max_runtime_hours": 12})
        dumped = config.model_dump()
        rebuilt = TypeAdapter(merlina.TrainingConfig).validate_python(dumped)
        assert rebuilt.remote.max_runtime_hours == 12


class TestDispatch:
    def test_remote_config_routes_to_remote_runner(self, monkeypatch):
        calls = []
        import src.remote.orchestrator as orch
        monkeypatch.setattr(orch, "run_training_remote",
                            lambda *args, **kwargs: calls.append(args))

        callback = merlina._make_training_callback(None)
        config = make_training_config(remote={"enabled": True})
        callback("job_remote", config.model_dump())

        assert len(calls) == 1
        assert calls[0][0] == "job_remote"

    def test_disabled_remote_stays_local(self, monkeypatch):
        remote_calls = []
        local_calls = []
        import src.remote.orchestrator as orch
        import src.training_runner as tr
        monkeypatch.setattr(orch, "run_training_remote",
                            lambda *a, **k: remote_calls.append(a))
        # Local jobs go in-thread or (whenever a GPU is present) through the
        # subprocess path — either one counts as "stayed local".
        monkeypatch.setattr(tr, "run_training_sync",
                            lambda *a, **k: local_calls.append(a))
        monkeypatch.setattr(tr, "run_training_distributed",
                            lambda *a, **k: local_calls.append(a))
        monkeypatch.setattr(tr, "_get_distributed_gpu_count", lambda c: 1)

        callback = merlina._make_training_callback(None)
        config = make_training_config(remote={"enabled": False})
        callback("job_local", config.model_dump())

        assert remote_calls == []
        assert len(local_calls) == 1


class TestRemotePreflight:
    def _validate(self, config):
        return PreflightValidator().validate_all(config)

    def test_remote_job_skips_local_gpu_checks(self):
        config = make_training_config(remote={"enabled": True})
        _, results = self._validate(config)
        assert "GPU" not in results["checks"]
        assert "VRAM" not in results["checks"]
        assert "Remote Config" in results["checks"]

    def test_missing_provider_key_is_an_error(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "runpod_api_key", None, raising=False)
        config = make_training_config(remote={"enabled": True})
        is_valid, results = self._validate(config)
        assert not is_valid
        assert any("RUNPOD_API_KEY" in e for e in results["errors"])

    def test_with_key_and_hf_dataset_passes_remote_check(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "runpod_api_key", "rp_key", raising=False)
        config = make_training_config(remote={"enabled": True})
        _, results = self._validate(config)
        assert not any("RUNPOD_API_KEY" in e for e in results["errors"])

    def test_local_model_path_rejected(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "runpod_api_key", "rp_key", raising=False)
        config = make_training_config(base_model="./models/my-local-model",
                                      remote={"enabled": True})
        is_valid, results = self._validate(config)
        assert not is_valid
        assert any("local path" in e for e in results["errors"])

    def test_existing_local_file_dataset_can_be_staged_remotely(self, monkeypatch, tmp_path):
        from config import settings
        monkeypatch.setattr(settings, "runpod_api_key", "rp_key", raising=False)
        monkeypatch.setattr(settings, "merlina_artifact_store", "s3")
        monkeypatch.setattr(settings, "merlina_artifact_s3_bucket", "artifacts")
        monkeypatch.setattr(settings, "remote_local_data_roots", str(tmp_path))
        dataset_file = tmp_path / "training.jsonl"
        dataset_file.write_text('{"prompt":"Hi","chosen":"Hello","rejected":"No"}\n')
        config = make_training_config(
            remote={"enabled": True},
            dataset={
                "source": {"source_type": "local_file", "file_path": str(dataset_file)},
                "format": {"format_type": "chatml"},
            },
        )
        is_valid, results = self._validate(config)
        assert is_valid, results["errors"]
        assert results["checks"]["Remote Config"]["status"] == "pass"
        assert results["checks"]["Dataset Config"]["status"] == "pass"

    def test_non_remote_job_unchanged(self):
        config = make_training_config()
        _, results = self._validate(config)
        assert "GPU" in results["checks"]
        assert "Remote Config" not in results["checks"]


class TestQueues:
    def test_remote_queue_exists_and_is_separate(self):
        assert merlina.remote_job_queue is not merlina.job_queue


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestRemoteStoragePreflight:
    @pytest.fixture(autouse=True)
    def remote_settings(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "runpod_api_key", "rp_key")
        monkeypatch.setattr(settings, "merlina_artifact_store", "s3")
        monkeypatch.setattr(settings, "merlina_artifact_s3_bucket", "artifacts")
        monkeypatch.setattr(settings, "merlina_artifact_s3_endpoint", None)
        monkeypatch.setattr(settings, "merlina_artifact_s3_prefix", "runs")
        # These tests validate storage, not Hub availability or offline model cache.
        monkeypatch.setattr(PreflightValidator, "_check_model_access", lambda *_: {"model": "public/model"})

    def _config(self, source, **dataset_fields):
        return make_training_config(hf_token=None, push_to_hub=False,
                                    remote={"enabled": True},
                                    dataset={"source": source, **dataset_fields})

    def _internal(self, monkeypatch, *, private=False):
        from types import SimpleNamespace
        from dataset_handlers.internal_store import Manifest
        manifest = Manifest("demo", "rev", "parquet", ["s3://internal/demo/rev/part.parquet"], private=private)
        monkeypatch.setattr("dataset_handlers.factory.default_internal_store", lambda: SimpleNamespace(
            get_manifest=lambda *_: manifest))
        return {"source_type": "internal", "uri": "swl://demo@rev"}

    def test_s3_output_and_local_input_need_no_hf_token_or_repo(self, tmp_path, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "remote_local_data_roots", str(tmp_path))
        path = tmp_path / "data.jsonl"
        path.write_text('{"prompt":"hi","chosen":"hello"}\n')
        valid, results = PreflightValidator().validate_all(self._config(
            {"source_type": "local_file", "file_path": str(path)}))
        assert valid, results["errors"]

    def test_pinned_public_internal_input_and_s3_output_pass(self, monkeypatch):
        source = self._internal(monkeypatch)
        valid, results = PreflightValidator().validate_all(self._config(source))
        assert valid, results["errors"]

    @pytest.mark.parametrize("location", ["source", "additional_sources", "eval_source"])
    def test_private_internal_input_is_rejected_everywhere(self, monkeypatch, location):
        source = self._internal(monkeypatch, private=True)
        dataset = {"source": {"source_type": "upload", "dataset_id": "uploaded"}}
        dataset[location] = [source] if location == "additional_sources" else source
        cfg = self._config(**dataset)
        valid, results = PreflightValidator().validate_all(cfg)
        assert not valid
        assert any("Private internal" in error for error in results["errors"])

    def test_store_failure_does_not_expose_credentials(self, monkeypatch):
        from types import SimpleNamespace
        def fail(*_):
            raise RuntimeError("key=SECRET /private/path")
        monkeypatch.setattr("dataset_handlers.factory.default_internal_store", lambda: SimpleNamespace(get_manifest=fail))
        valid, results = PreflightValidator().validate_all(self._config(
            {"source_type": "internal", "uri": "swl://demo@rev"}))
        assert not valid
        assert "SECRET" not in str(results)
        assert "/private/path" not in str(results)
        assert "Internal dataset revision is unavailable" in results["errors"]

    def test_s3_requires_bucket(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "merlina_artifact_s3_bucket", None)
        valid, results = PreflightValidator().validate_all(self._config(
            {"source_type": "upload", "dataset_id": "uploaded"}))
        assert not valid
        assert any("MERLINA_ARTIFACT_S3_BUCKET" in error for error in results["errors"])

    def test_hf_output_still_requires_hf_configuration(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "merlina_artifact_store", "hf_hub")
        valid, results = PreflightValidator().validate_all(self._config(
            {"source_type": "upload", "dataset_id": "uploaded"}))
        assert not valid
        assert any("HF token" in error for error in results["errors"])

    def test_public_internal_eval_passes_without_hf_output(self, monkeypatch):
        source = self._internal(monkeypatch)
        valid, results = PreflightValidator().validate_all(self._config(
            {"source_type": "upload", "dataset_id": "uploaded"}, eval_source=source))
        assert valid, results["errors"]

    @pytest.mark.parametrize("uri", ["swl://demo", "swl://private/customer@rev"])
    def test_remote_internal_revision_and_name_policy_applies_in_preflight(self, monkeypatch, uri):
        self._internal(monkeypatch)
        valid, results = PreflightValidator().validate_all(self._config(
            {"source_type": "internal", "uri": uri}))
        assert not valid
        assert any("pinned" in error for error in results["errors"])

    def test_s3_does_not_remove_explicit_hf_publication_auth_requirement(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cfg = self._config({"source_type": "upload", "dataset_id": "uploaded"})
        cfg.push_to_hub = True
        valid, results = PreflightValidator().validate_all(cfg)
        assert not valid
        assert any("Push to HuggingFace Hub" in error for error in results["errors"])
