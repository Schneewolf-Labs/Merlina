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
        monkeypatch.setattr(tr, "run_training_sync",
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

    def test_local_file_dataset_rejected(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "runpod_api_key", "rp_key", raising=False)
        config = make_training_config(
            remote={"enabled": True},
            dataset={
                "source": {"source_type": "local_file", "file_path": "/data/x.json"},
                "format": {"format_type": "chatml"},
            },
        )
        is_valid, results = self._validate(config)
        assert not is_valid
        assert any("local_file" in e for e in results["errors"])

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
