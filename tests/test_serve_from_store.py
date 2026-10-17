"""Serving a model from object storage (`merlina models push` + model.store_model).
Fake S3 only; no network."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from merlina_cli import main as cli_main
from src.remote.artifacts import ArtifactStoreError, model_store, store_from_description
from src.remote.inference import RemoteInferenceRequest, build_serve_plan, serve_spec
from src.remote.plan import RemotePlanError
from src.remote.s3_artifacts import S3ArtifactStore
from src.remote.serve_worker import _resolve_model
from tests.test_remote_inference_plan import (
    FP8_MLA_MOE_CONFIG,
    FP8_TOTAL_SIZE,
    SETTINGS,
)
from tests.test_s3_artifacts import FakeS3


class InfoS3(FakeS3):
    def info(self, key):
        return {"size": len(self.objects[key])}


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "C0-flat"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(FP8_MLA_MOE_CONFIG))
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": FP8_TOTAL_SIZE}, "weight_map": {}}))
    for i in range(3):
        (d / f"model-0000{i + 1}-of-00003.safetensors").write_bytes(bytes([i]) * 1000)
    return d


@pytest.fixture
def store():
    s = S3ArtifactStore("merlina", "models", endpoint_url="https://r2.example", fs=InfoS3())
    s.tier = "external"
    return s


def req(**model):
    return RemoteInferenceRequest(model=model or {"store_model": "C0"})


def test_in_place_push_roundtrip_parallel(store, model_dir, tmp_path):
    store.push_dir(model_dir, "C0", snapshot=False, workers=4)
    assert {f["path"] for f in store.files("C0")} == {p.name for p in model_dir.iterdir()}
    out = store.pull_dir("C0", tmp_path / "pulled", workers=4)
    for p in model_dir.iterdir():
        assert (out / p.name).read_bytes() == p.read_bytes()
    assert json.loads(store.read_file("C0", "config.json")) == FP8_MLA_MOE_CONFIG


def test_in_place_push_rejects_size_mismatch(store, model_dir):
    store.fs.corrupt_upload = True
    with pytest.raises(ArtifactStoreError):
        store.push_dir(model_dir, "C0", snapshot=False)
    assert not store.exists("C0")


def test_read_file_verifies_digest(store, model_dir):
    store.push_dir(model_dir, "C0", snapshot=False)
    gen = store._manifest("C0")["generation"]
    store.fs.objects[f"merlina/models/C0/generations/{gen}/config.json"] = b"{}"
    with pytest.raises(ArtifactStoreError):
        store.read_file("C0", "config.json")
    with pytest.raises(FileNotFoundError):
        store.read_file("C0", "missing.json")


def test_plan_sizes_from_stored_config(store, model_dir):
    store.push_dir(model_dir, "C0", snapshot=False)
    plan = build_serve_plan(req(), settings=SETTINGS, model_store=store)
    assert plan.model_id == "store:models/C0"
    assert plan.served_model_name == "C0"
    assert plan.artifact == "C0" and not plan.lora
    assert plan.artifact_store["bucket"] == "merlina" and plan.artifact_store["prefix"] == "models"
    assert plan.model_specs.weight_bytes_total == FP8_TOTAL_SIZE
    assert "<store:models/C0>" in plan.vllm_command
    spec = serve_spec(plan)
    assert spec["source_kind"] == "artifact" and spec["artifact"] == "C0"


def test_plan_missing_model(store):
    with pytest.raises(RemotePlanError, match="merlina models push"):
        build_serve_plan(req(), settings=SETTINGS, model_store=store)


def test_plan_refuses_internal_tier(store, model_dir):
    store.push_dir(model_dir, "C0", snapshot=False)
    store.tier = "internal"
    with pytest.raises(RemotePlanError, match="internal"):
        build_serve_plan(req(), settings=SETTINGS, model_store=store)


def test_request_validation():
    with pytest.raises(ValidationError):
        req(store_model="C0", hf_repo="a/b")
    with pytest.raises(ValidationError):
        req(store_model="../C0")
    with pytest.raises(ValidationError):
        req(store_model="models/C0")


def test_worker_pulls_stored_model(store, model_dir, tmp_path, monkeypatch):
    store.push_dir(model_dir, "C0", snapshot=False)
    plan = build_serve_plan(req(), settings=SETTINGS, model_store=store)
    payload = {"artifact_store": plan.artifact_store, "config": {}}
    monkeypatch.setattr("src.remote.worker_entry._resolve_store", lambda _p: store)
    model_ref, lora = _resolve_model(payload, serve_spec(plan), tmp_path / "work")
    assert lora is None
    assert (tmp_path / "work" / "C0" / "config.json").is_file()
    assert model_ref == str(tmp_path / "work" / "C0")
    # The description alone rebuilds the same store on a worker.
    rebuilt = store_from_description(plan.artifact_store)
    assert (rebuilt.bucket, rebuilt.prefix) == ("merlina", "models")


def test_model_store_requires_s3():
    hub = SimpleNamespace(artifact_location=lambda: SimpleNamespace(kind="hf_hub", error=None))
    with pytest.raises(ArtifactStoreError, match="ARTIFACT_STORE"):
        model_store(hub)


def test_cli_push_dry_run_and_push(store, model_dir, monkeypatch, capsys):
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    assert cli_main(["models", "push", str(model_dir), "C0", "--dry-run"]) == 0
    assert not store.exists("C0")
    assert cli_main(["models", "push", str(model_dir), "C0", "--workers", "2"]) == 0
    assert store.exists("C0")
    assert cli_main(["models", "info", "C0"]) == 0
    assert "files:      5" in capsys.readouterr().out


def test_cli_push_refuses_symlinks(store, model_dir, monkeypatch, tmp_path):
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    (model_dir / "link.safetensors").symlink_to(model_dir / "config.json")
    assert cli_main(["models", "push", str(model_dir), "C0"]) == 2
