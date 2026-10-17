"""Publishing models to the model store (publish_to_store / POST /models/{name}/publish)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.remote.s3_artifacts import S3ArtifactStore
from src.store_publish import publish_to_store
from src.upload_state import read_upload_state
from tests.test_model_refs import InfoS3


@pytest.fixture
def store():
    s = S3ArtifactStore("merlina", "models", fs=InfoS3())
    s.tier = "external"
    return s


@pytest.fixture
def adapter(tmp_path):
    d = tmp_path / "models" / "my-tune"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "store:C0"}))
    (d / "adapter_model.safetensors").write_bytes(b"a" * 16)
    (d / "checkpoint-100").mkdir()
    (d / "checkpoint-100" / "optimizer.pt").write_bytes(b"o")
    (d / "upload_state.json").write_text("{}")
    (d / "gguf").mkdir()
    (d / "gguf" / "my-tune-Q4_K_M.gguf").write_bytes(b"g" * 8)
    return d


def test_adapter_published_without_clutter(store, adapter):
    result = publish_to_store(adapter, store=store)
    assert result["ref"] == "store:my-tune" and result["kind"] == "adapter"
    paths = {f["path"] for f in store.files("my-tune")}
    assert paths == {"adapter_config.json", "adapter_model.safetensors", "gguf/my-tune-Q4_K_M.gguf"}
    event = read_upload_state(adapter)["uploads"][-1]
    assert event["destination"] == "store" and event["repo_url"] == "s3://merlina/models/my-tune"
    assert event["artifacts"] == ["adapter", "gguf:my-tune-Q4_K_M.gguf"]


def test_merged_dir_and_readme(store, adapter, tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_bytes(b"m" * 32)
    cfg = SimpleNamespace(output_name="my-tune", base_model="store:hf/org/base", use_lora=True,
                          dataset=SimpleNamespace(source=SimpleNamespace(source_type="internal", uri="swl://data@r1"),
                                                  additional_sources=[]))
    result = publish_to_store(adapter, "tunes/my-tune", merged_dir=merged, config=cfg,
                              include_gguf=False, store=store)
    assert result["kind"] == "full"
    paths = {f["path"] for f in store.files("tunes/my-tune")}
    assert paths == {"config.json", "model.safetensors", "README.md"}
    assert (adapter / "README.md").is_file(), "the card is also written locally"
    meta = store._manifest("tunes/my-tune")["metadata"]
    assert meta["kind"] == "full" and meta["base_model"] == "store:hf/org/base"


def test_failure_is_recorded(store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        publish_to_store(empty, store=store)
    assert read_upload_state(empty)["uploads"][-1]["status"] == "failed"


def test_cli_push_accepts_adapters(store, adapter, monkeypatch):
    from merlina_cli import main
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    (adapter / "checkpoint-100" / "optimizer.pt").unlink()
    assert main(["models", "push", str(adapter), "my-tune", "--dry-run"]) == 0


def test_remote_merge_payload_and_worker_publish(store, monkeypatch, tmp_path):
    from src.remote.orchestrator import _stage_payload
    from src.remote.spec import StagePlan
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    cfg = SimpleNamespace(model_dump=lambda: {"base_model": "org/m", "publish_to_store": True,
                                              "output_name": "my-tune", "remote": {}})
    train = _stage_payload("j", StagePlan(name="train", target="remote"), cfg, store)
    assert train["config"]["publish_to_store"] is False and "publish_store" not in train
    merge = _stage_payload("j", StagePlan(name="merge", target="remote"), cfg, store)
    assert merge["publish_store"]["prefix"] == "models"
    store.tier = "internal"
    with pytest.raises(Exception, match="external"):
        _stage_payload("j", StagePlan(name="merge", target="remote"), cfg, store)


def test_publish_endpoint(store, adapter, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import merlina
    from src.job_manager import JobManager
    jobs = JobManager(db_path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(merlina, "job_manager", jobs)
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    monkeypatch.setattr(merlina, "_resolve_model_dir", lambda name: adapter)

    def inline(job_id, config, *, needs_merge, merge, work, on_error, thread_name):
        assert not needs_merge
        work(None)
        return None
    monkeypatch.setattr(merlina, "_dispatch_merge_gated", inline)
    client = TestClient(merlina.app)
    resp = client.post("/models/my-tune/publish", json={"store_name": "my-tune"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ref"] == "store:my-tune"
    job = jobs.get_job(body["job_id"])
    assert job.status == "completed" and job.metrics["store_publish"]["status"] == "success"
    assert client.post("/models/my-tune/publish", json={"store_name": "../x"}).status_code == 400


def test_post_training_publish_uses_merge_and_releases(store, adapter, monkeypatch, tmp_path):
    from src.job_manager import JobManager
    from src.training_runner import _run_background_store_publish
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    jobs = JobManager(db_path=str(tmp_path / "jobs.db"))
    jobs.create_job("j1", {})
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    released = []
    artifact = SimpleNamespace(path=merged, error=None, release=lambda: released.append(1))
    cfg = SimpleNamespace(output_name="my-tune", store_model_name=None, use_lora=True,
                          merge_lora_before_upload=True, base_model="org/b",
                          dataset=SimpleNamespace(source=SimpleNamespace(source_type="upload"),
                                                  additional_sources=[]))
    _run_background_store_publish(cfg, str(adapter), "sft", "j1", jobs, merge_artifact=artifact)
    assert released == [1]
    assert jobs.get_job("j1").metrics["store_publish"]["kind"] == "full"
    # A failed merge is recorded, not raised, and the artifact is still released.
    bad = SimpleNamespace(path=None, error="OOM", release=lambda: released.append(2))
    _run_background_store_publish(cfg, str(adapter), "sft", "j1", jobs, merge_artifact=bad)
    assert released == [1, 2]
    assert "OOM" in jobs.get_job("j1").metrics["store_publish"]["error"]
