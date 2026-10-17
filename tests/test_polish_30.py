"""3.0 polish: write probe before renting, ARTIFACT_STORE default, store
models in the base-model list, MCP internal sources."""
import json
from types import SimpleNamespace

import pytest

from src.remote.artifacts import ArtifactStoreError, HFHubArtifactStore
from src.remote.s3_artifacts import S3ArtifactStore
from tests.test_model_refs import InfoS3


class ProbeS3(InfoS3):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.removed = []

    def pipe_file(self, key, data):
        if self.fail:
            raise PermissionError("AccessDenied secret-ish detail")
        super().pipe_file(key, data)

    def rm_file(self, key):
        self.removed.append(key)
        self.objects.pop(key, None)


def test_s3_write_probe_cleans_up_and_fails_loudly():
    fs = ProbeS3()
    S3ArtifactStore("merlina", "runs/job", fs=fs).verify_writable()
    assert fs.removed and not fs.objects
    with pytest.raises(ArtifactStoreError, match="can't be written") as e:
        S3ArtifactStore("merlina", "runs/job", fs=ProbeS3(fail=True)).verify_writable()
    assert "secret-ish" not in str(e.value), "provider error text never surfaces"


def test_hf_write_probe(monkeypatch):
    calls = []

    class Api:
        def create_repo(self, *a, **k): calls.append("create")
        def upload_file(self, **k): calls.append("upload")
        def delete_file(self, *a, **k): calls.append("delete")
    store = HFHubArtifactStore("me/merlina-run-x", token="t")
    monkeypatch.setattr(store, "_api", lambda: Api())
    store.verify_writable()
    assert calls == ["create", "upload", "delete"]

    class Full(Api):
        def upload_file(self, **k): raise RuntimeError("403 storage quota exceeded")
    store = HFHubArtifactStore("me/merlina-run-x", token="t")
    monkeypatch.setattr(store, "_api", lambda: Full())
    with pytest.raises(ArtifactStoreError, match="ARTIFACT_STORE=external"):
        store.verify_writable()


def test_remote_training_probes_before_provisioning(monkeypatch, tmp_path):
    from src.remote import orchestrator
    probed = []

    class Store:
        kind = "s3"
        def verify_writable(self):
            probed.append(1)
            raise ArtifactStoreError("bucket gone")
        def describe(self): return {"kind": "s3"}
    provisioned = []
    provider = SimpleNamespace(list_gpu_offers=lambda: [], provision=lambda *a, **k: provisioned.append(1))
    monkeypatch.setattr(orchestrator, "build_remote_plan", lambda *a, **k: SimpleNamespace(warnings=[], to_dict=lambda: {}))
    from src.job_manager import JobManager
    jobs = JobManager(db_path=str(tmp_path / "j.db"))
    jobs.create_job("job_p", {})
    cfg = SimpleNamespace(remote=SimpleNamespace(max_runtime_hours=1, provider="runpod"),
                          model_dump=lambda: {})
    orchestrator.run_training_remote("job_p", cfg, jobs, {}, store=Store(), provider=provider)
    assert probed == [1] and provisioned == []
    job = jobs.get_job("job_p")
    assert job.status == "failed" and "bucket gone" in (job.error or "")


def test_artifact_store_defaults_to_external_when_configured():
    from config import resolve_artifact_location
    base = dict(artifact_store=None, merlina_artifact_store=None, artifact_prefix=None,
                merlina_artifact_s3_prefix=None, artifact_bucket=None)
    configured = SimpleNamespace(**base, storage_external_endpoint="https://r2.example",
                                 storage_external_bucket="merlina", storage_external_access_key="a",
                                 storage_external_secret_key="b", storage_external_region=None)
    assert resolve_artifact_location(configured).tier == "external"
    bare = SimpleNamespace(**base, storage_external_endpoint=None, storage_external_bucket=None)
    assert resolve_artifact_location(bare).kind == "hf_hub"
    explicit = SimpleNamespace(**{**base, "artifact_store": "hf_hub"}, storage_external_endpoint="https://r2.example",
                               storage_external_bucket="merlina")
    assert resolve_artifact_location(explicit).kind == "hf_hub"


def test_store_models_in_base_model_list(tmp_path):
    from src.local_models import list_local_models, list_store_models
    store = S3ArtifactStore("merlina", "models", fs=InfoS3())
    full = tmp_path / "full"
    full.mkdir()
    (full / "config.json").write_text("{}")
    store.push_dir(full, "C0", snapshot=False)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    store.push_dir(adapter, "my-lora", snapshot=False)
    store.fs.ls = lambda root, detail=False: [f"{root}/C0", f"{root}/my-lora"]
    assert [m["model_id"] for m in list_store_models(_store=store)] == ["store:C0"]
    assert list_store_models(_store=SimpleNamespace(list_artifacts=lambda: 1 / 0)) == []
    assert all(m["source"] != "store" for m in list_local_models(tmp_path, _scan=lambda: [])["models"])


def test_mcp_internal_sources(monkeypatch):
    import asyncio
    import mcp_server
    sent = []

    async def fake_request(method, path, params=None, json_body=None):
        sent.append(json_body)
        return {}
    monkeypatch.setattr(mcp_server, "_request", fake_request)
    asyncio.run(mcp_server.preview_dataset("swl://b3/pr@abc", source_type="internal"))
    assert sent[-1]["source"] == {"source_type": "internal", "split": "train", "uri": "swl://b3/pr@abc"}
