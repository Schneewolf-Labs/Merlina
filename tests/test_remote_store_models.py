"""store: base models on remote runs: the worker payload names the model
store, workers resolve against it, and adapter serving pulls the base too."""
import json
from types import SimpleNamespace

import pytest

from src import model_refs
from src.model_refs import ModelRefError, register_store, remote_store_description
from src.remote.s3_artifacts import S3ArtifactStore
from src.remote.spec import StagePlan
from tests.test_model_refs import CONFIG, InfoS3


@pytest.fixture
def store(tmp_path):
    s = S3ArtifactStore("merlina", "models", endpoint_url="https://r2.example", fs=InfoS3())
    s.tier = "external"
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(CONFIG))
    (src / "model.safetensors").write_bytes(b"w" * 32)
    s.push_dir(src, "C0", snapshot=False)
    return s


@pytest.fixture(autouse=True)
def reset():
    model_refs._MEMO.clear()
    register_store(None)
    yield
    model_refs._MEMO.clear()
    register_store(None)


def test_remote_description_requires_reachable_tier(store, monkeypatch):
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    assert remote_store_description("Qwen/Qwen3-8B") is None
    desc = remote_store_description("store:C0")
    assert (desc["bucket"], desc["prefix"]) == ("merlina", "models")
    store.tier = "internal"
    with pytest.raises(ModelRefError, match="LAN"):
        remote_store_description("store:C0")


def test_stage_payload_carries_model_store(store, monkeypatch):
    from src.remote.orchestrator import _stage_payload
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    cfg = SimpleNamespace(model_dump=lambda: {"base_model": "store:C0", "remote": {"enabled": True}})
    payload = _stage_payload("job_1", StagePlan(name="train", target="remote"), cfg, store)
    assert payload["model_store"]["prefix"] == "models"
    cfg_hf = SimpleNamespace(model_dump=lambda: {"base_model": "org/model", "remote": {"enabled": True}})
    assert "model_store" not in _stage_payload("job_1", StagePlan(name="train", target="remote"), cfg_hf, store)


def test_worker_resolves_against_registered_store(store, tmp_path, monkeypatch):
    def unconfigured(settings=None):
        raise RuntimeError("workers have no ARTIFACT_STORE")
    monkeypatch.setattr("src.remote.artifacts.model_store", unconfigured)
    monkeypatch.setattr(model_refs, "_cache_root", lambda settings=None: tmp_path / "cache")
    register_store(store)
    path = model_refs.resolve_model_ref("store:C0")
    assert json.loads(open(f"{path}/config.json").read()) == CONFIG


def test_serve_worker_pulls_store_base_for_adapter(store, tmp_path, monkeypatch):
    from src.remote.serve_worker import _resolve_model
    monkeypatch.setattr(model_refs, "_cache_root", lambda settings=None: tmp_path / "cache")
    register_store(store)
    adapter_src = tmp_path / "adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text("{}")
    store.push_dir(adapter_src, "adapter", snapshot=False)
    monkeypatch.setattr("src.remote.worker_entry._resolve_store", lambda _p: store)
    serve = {"source_kind": "artifact", "model_id": "store:C0", "artifact": "adapter", "lora": True}
    base, lora = _resolve_model({"config": {}}, serve, tmp_path / "work")
    assert base.startswith(str(tmp_path / "cache")) and lora == str(tmp_path / "work" / "adapter")


def test_remote_preflight(store, monkeypatch):
    from config import settings
    from src.preflight_checks import PreflightValidator
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    monkeypatch.setattr("src.remote.plan.check_remote_compatible", lambda config: [])
    monkeypatch.setattr("src.remote.plan.remote_artifact_location",
                        lambda settings: SimpleNamespace(error=None, kind="other"))
    monkeypatch.setattr(settings, "runpod_api_key", "test-only")
    cfg = SimpleNamespace(base_model="store:C0", remote=SimpleNamespace(provider="runpod"))

    v = PreflightValidator()
    v._check_remote_config(cfg)
    assert v.errors == []
    store.tier = "internal"
    v = PreflightValidator()
    v._check_remote_config(cfg)
    assert len(v.errors) == 1 and "LAN" in v.errors[0]
    v = PreflightValidator()
    v._check_remote_config(SimpleNamespace(base_model="/mnt/models/x", remote=cfg.remote))
    assert "merlina models push" in v.errors[0]
