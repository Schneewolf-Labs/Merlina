"""Hub model mirroring into the model store (merlina models mirror / MODEL_MIRROR)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import model_refs
from src.model_mirror import ModelMirrorError, ensure_model_mirror
from src.remote.s3_artifacts import S3ArtifactStore
from tests.test_model_refs import CONFIG, InfoS3


class Hub:
    def __init__(self):
        self.sha = "a" * 40
        self.down = False
        self.downloads = []

    def model_info(self, repo_id, revision=None, token=None, files_metadata=False):
        if self.down:
            raise ConnectionError("hub down")
        return SimpleNamespace(sha=self.sha, siblings=[SimpleNamespace(rfilename=f) for f in
                                                       ("config.json", "model.safetensors", "pytorch_model.bin")])

    def snapshot_download(self, repo_id, revision=None, token=None, local_dir=None, ignore_patterns=None):
        self.downloads.append((repo_id, revision, tuple(ignore_patterns or ())))
        Path(local_dir, "config.json").write_text(json.dumps(CONFIG))
        Path(local_dir, "model.safetensors").write_bytes(revision.encode())
        Path(local_dir, ".cache").mkdir()
        Path(local_dir, ".cache", "x.lock").write_text("")
        return local_dir


@pytest.fixture
def hub(monkeypatch):
    h = Hub()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: h)
    monkeypatch.setattr("huggingface_hub.snapshot_download", h.snapshot_download)
    return h


@pytest.fixture
def store():
    s = S3ArtifactStore("merlina", "models", fs=InfoS3())
    s.tier = "internal"
    return s


@pytest.fixture(autouse=True)
def reset():
    model_refs._MEMO.clear()
    model_refs.register_store(None)
    yield
    model_refs._MEMO.clear()


def test_mirror_miss_hit_and_upstream_move(hub, store):
    ref = ensure_model_mirror("org/model", store=store)
    assert ref == "store:hf/org/model"
    assert {f["path"] for f in store.files("hf/org/model")} == {"config.json", "model.safetensors"}
    assert "*.bin" in hub.downloads[0][2], ".bin skipped when safetensors exist"
    meta = store._manifest("hf/org/model")["metadata"]["source"]
    assert meta == {"hf_repo": "org/model", "hf_sha": "a" * 40, "revision": None}

    ensure_model_mirror("org/model", store=store)
    assert len(hub.downloads) == 1, "same upstream commit: no re-download"
    ensure_model_mirror("org/model", store=store, force=True)
    assert len(hub.downloads) == 2
    hub.sha = "b" * 40
    ensure_model_mirror("org/model", store=store)
    assert hub.downloads[-1][1] == "b" * 40
    assert store._manifest("hf/org/model")["metadata"]["source"]["hf_sha"] == "b" * 40


def test_hub_down_serves_existing_mirror(hub, store):
    with pytest.raises(ModelMirrorError):
        hub.down = True
        ensure_model_mirror("org/model", store=store)
    hub.down = False
    ensure_model_mirror("org/model", store=store)
    hub.down = True
    assert ensure_model_mirror("org/model", store=store) == "store:hf/org/model"


def test_model_mirror_setting_routes_hub_ids_through_the_store(hub, store, monkeypatch, tmp_path):
    from config import settings
    monkeypatch.setattr(settings, "model_mirror", True)
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    monkeypatch.setattr(model_refs, "_cache_root", lambda settings=None: tmp_path / "cache")
    path = model_refs.resolve_model_ref("org/model")
    assert path.startswith(str(tmp_path / "cache"))
    assert json.loads(Path(path, "config.json").read_text()) == CONFIG
    # Local paths and non-Hub strings are never mirrored.
    assert model_refs.resolve_model_ref("./models/x") == "./models/x"
    monkeypatch.setattr(settings, "model_mirror", False)
    model_refs._MEMO.clear()
    assert model_refs.resolve_model_ref("org/model") == "org/model"


def test_mirror_failure_falls_back(hub, store, monkeypatch, tmp_path):
    from config import settings
    monkeypatch.setattr(settings, "model_mirror", True)
    monkeypatch.setattr(model_refs, "_cache_root", lambda settings=None: tmp_path / "cache")

    def no_store(settings=None):
        raise RuntimeError("no artifact store configured")
    monkeypatch.setattr("src.remote.artifacts.model_store", no_store)
    assert model_refs.resolve_model_ref("org/model") == "org/model", "falls back to the Hub id"

    # With a cached mirror, the cache wins even with Hub and store both down.
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    model_refs._MEMO.clear()
    cached = model_refs.resolve_model_ref("org/model")
    model_refs._MEMO.clear()
    hub.down = True
    monkeypatch.setattr("src.remote.artifacts.model_store", no_store)
    assert model_refs.resolve_model_ref("org/model") == cached


def test_cli_mirror(hub, store, monkeypatch, capsys):
    from merlina_cli import main
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    assert main(["models", "mirror", "org/model"]) == 0
    out = capsys.readouterr().out
    assert "store:hf/org/model" in out and "aaaaaaaaaaaa" in out
