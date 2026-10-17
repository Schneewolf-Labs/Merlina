"""Dataset paths that used to need the Hub: swl:// dataset_name, remote runs
with DATASET_MIRROR, streaming from the internal store, `datasets put`."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset

from dataset_handlers.internal_store import InternalStore, InternalStoreError
from dataset_handlers.loaders import InternalStoreLoader
from dataset_handlers.put import put_dataset
from tests.test_mirror import CFG, FakeFS


class ExistsFS(FakeFS):
    """FakeFS + the bits put_dataset's store calls touch."""


def _store():
    return InternalStore(CFG, fs=ExistsFS())


# ── swl:// in dataset_name ────────────────────────────────────────────────────
def test_dataset_name_swl_maps_to_internal_source():
    from merlina import TrainingConfig
    cfg = TrainingConfig(base_model="org/m", output_name="x", dataset_name="swl://b3/pr@abc")
    assert cfg.dataset.source.source_type == "internal"
    assert cfg.dataset.source.uri == "swl://b3/pr@abc"
    hf = TrainingConfig(base_model="org/m", output_name="x", dataset_name="org/ds")
    assert hf.dataset.source.source_type == "huggingface"


# ── remote runs honor DATASET_MIRROR ──────────────────────────────────────────
def test_remote_staging_mirrors_hf_sources(monkeypatch, tmp_path):
    from config import settings
    from src.remote import inputs
    from src.remote.artifacts import LocalArtifactStore
    monkeypatch.setattr(inputs, "OPERATOR_STORE_KINDS", {"s3", "local"})
    monkeypatch.setattr(settings, "merlina_artifact_store", "s3")
    monkeypatch.setattr("dataset_handlers.factory.default_prefer_internal", lambda: True)
    staged = []
    monkeypatch.setattr(inputs, "_stage_internal",
                        lambda source, work, store: (staged.append(dict(source)) or (_parquet(work), {"source_type": "internal"})))
    monkeypatch.setattr("dataset_handlers.mirror.ensure_mirror",
                        lambda store, repo_id, split, config_name, token: (f"hf/{repo_id}/{split}", "sha1"))
    cfg = {"hf_token": "t", "dataset": {"source": {"source_type": "huggingface", "repo_id": "org/ds",
                                                   "split": "test", "column_mapping": {"prompt": "q"}},
                                        "additional_sources": []}}
    prepared = inputs.stage_remote_inputs(cfg, {}, LocalArtifactStore(tmp_path / "store"),
                                          internal_store=object())
    assert staged[0]["uri"] == "swl://hf/org/ds/test@sha1"
    src = prepared["dataset"]["source"]
    assert src["source_type"] == "local_file" and src["file_path"].startswith("artifact://")
    assert src["split"] == "train" and src["column_mapping"] == {"prompt": "q"}
    # Streaming sources stay direct.
    cfg["dataset"]["source"]["streaming"] = True
    staged.clear()
    out = inputs.stage_remote_inputs(cfg, {}, LocalArtifactStore(tmp_path / "store2"), internal_store=object())
    assert staged == [] and out["dataset"]["source"]["source_type"] == "huggingface"


def _parquet(work):
    path = Path(work) / "data.parquet"
    Dataset.from_list([{"prompt": "a"}]).to_parquet(str(path))
    return path


# ── streaming from the internal store ─────────────────────────────────────────
def test_internal_loader_streams_only_max_samples(monkeypatch):
    store = _store()
    manifest = SimpleNamespace(rev="r1", hf_revision=None, format="parquet", data_files=["s3://x"])
    monkeypatch.setattr(store, "get_manifest", lambda name, rev: manifest)
    monkeypatch.setattr(store, "storage_options", lambda: {})
    seen = {}

    def fake_load(fmt, streaming=False, **kw):
        seen["streaming"] = streaming
        return iter([{"q": i} for i in range(1000)])
    monkeypatch.setattr("dataset_handlers.loaders.load_dataset", fake_load)
    ds = InternalStoreLoader("swl://x", store, max_samples=10, streaming=True, batch_size=4).load()
    assert seen["streaming"] is True and len(ds) == 10


def test_factory_streams_from_an_existing_mirror(monkeypatch):
    from dataset_handlers.factory import create_loader
    from dataset_handlers.loaders import StreamingHuggingFaceLoader
    store = _store()
    monkeypatch.setattr(store, "exists", lambda name, rev="latest": name == "hf/org/ds/train")
    loader = create_loader("huggingface", repo_id="org/ds", streaming=True, store=store, prefer_internal=True)
    assert isinstance(loader, InternalStoreLoader) and loader.streaming and loader.uri == "swl://hf/org/ds/train"
    loader = create_loader("huggingface", repo_id="org/other", streaming=True, store=store, prefer_internal=True)
    assert isinstance(loader, StreamingHuggingFaceLoader), "no mirror: stream from the Hub, don't snapshot"


# ── datasets put ──────────────────────────────────────────────────────────────
def test_put_dataset_is_content_addressed(tmp_path):
    store = _store()
    f = tmp_path / "pairs.jsonl"
    f.write_text("\n".join(json.dumps({"prompt": f"q{i}", "chosen": "a", "rejected": "b"}) for i in range(3)))
    name, rev, created = put_dataset(store, f, "b3/pr-review", private=True)
    assert created and len(rev) == 16
    m = store.get_manifest(name, "latest")
    assert m.rev == rev and m.num_rows == 3 and m.private and m.source["kind"] == "put"
    assert put_dataset(store, f, "b3/pr-review")[2] is False, "same bytes: no new revision"
    with pytest.raises(InternalStoreError):
        put_dataset(store, f, "hf/org/x")
    with pytest.raises(InternalStoreError):
        put_dataset(store, tmp_path / "x.txt", "ok")


def test_cli_put_refuses_private_to_external(tmp_path, capsys):
    from merlina_cli import main
    f = tmp_path / "d.jsonl"
    f.write_text('{"a": 1}\n')
    assert main(["datasets", "put", str(f), "vps-incident/x", "--store", "external"]) == 2
    assert main(["datasets", "put", str(f), "ok", "--store", "external", "--private"]) == 2


def test_upload_to_store_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    import merlina
    store = _store()
    monkeypatch.setattr("dataset_handlers.factory.get_store", lambda which: store)
    with merlina._uploaded_datasets_lock:
        merlina.uploaded_datasets["u1"] = {"content": b'{"prompt": "q"}\n', "filename": "d.jsonl"}
    try:
        client = TestClient(merlina.app)
        resp = client.post("/dataset/uploads/u1/store", json={"name": "mine/d"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["uri"].startswith("swl://mine/d@")
        assert client.post("/dataset/uploads/nope/store", json={"name": "x"}).status_code == 404
        assert client.post("/dataset/uploads/u1/store", json={"name": "hf/x"}).status_code == 400
    finally:
        with merlina._uploaded_datasets_lock:
            merlina.uploaded_datasets.pop("u1", None)
