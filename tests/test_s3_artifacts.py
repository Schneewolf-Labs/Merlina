"""Storage-contract tests. Fake S3 is not a live AWS/R2 compatibility claim."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.remote.artifacts import ArtifactStoreError, artifact_worker_env, store_for_job, store_from_description
from src.remote.s3_artifacts import S3ArtifactStore, artifact_metadata


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail_upload = False
        self.corrupt_upload = False

    def put_file(self, local, key):
        if self.fail_upload:
            raise OSError("secret credential error")
        self.objects[key] = b"corrupt" if self.corrupt_upload else Path(local).read_bytes()

    def get_file(self, key, local):
        Path(local).write_bytes(self.cat_file(key))

    def cat_file(self, key, start=None, end=None):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key][start:end]

    def pipe_file(self, key, data):
        self.objects[key] = data


@pytest.fixture
def setup(tmp_path):
    fs = FakeS3()
    store = S3ArtifactStore("bucket", "runs/job", fs=fs)
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_bytes(b"weights")
    (source / "checkpoint-5").mkdir()
    (source / "checkpoint-5" / "state.json").write_text("{}")
    return store, fs, source


def test_roundtrip_manifest_and_secret_free_description(setup, tmp_path):
    store, fs, source = setup
    store.metadata = artifact_metadata({"base_model": "model", "hf_token": "secret", "nested": {"api_key": "secret"}})
    assert not store.exists("adapter")
    store.push_dir(source, "adapter")
    assert store.exists("adapter")
    manifest = store._manifest("adapter")
    assert manifest["version"] == 1
    assert manifest["checkpoints"] == ["checkpoint-5"]
    assert "secret" not in json.dumps(manifest)
    assert "metadata" not in store.describe()
    output = store.pull_dir("adapter", tmp_path / "out")
    assert (output / "weights").read_bytes() == b"weights"
    assert (output / "checkpoint-5/state.json").read_text() == "{}"
    rebuilt = store_from_description(store.describe())
    assert rebuilt.describe() == store.describe()


def test_failed_overwrite_keeps_previous_commit_and_retry_succeeds(setup, tmp_path):
    store, fs, source = setup
    store.push_dir(source, "adapter")
    manifest = store._manifest("adapter")
    (source / "weights").write_bytes(b"updated")
    fs.fail_upload = True
    with pytest.raises(ArtifactStoreError, match="upload failed") as failure:
        store.push_dir(source, "adapter")
    assert "secret" not in str(failure.value)
    assert store._manifest("adapter") == manifest
    fs.fail_upload = False
    assert (store.pull_dir("adapter", tmp_path / "out") / "weights").read_bytes() == b"weights"
    store.push_dir(source, "adapter")
    assert (store.pull_dir("adapter", tmp_path / "out") / "weights").read_bytes() == b"updated"


def test_corrupt_upload_never_commits(setup):
    store, fs, source = setup
    fs.corrupt_upload = True
    with pytest.raises(ArtifactStoreError, match="integrity"):
        store.push_dir(source, "adapter")
    assert not store.exists("adapter")


def test_corrupt_download_preserves_existing_destination(setup, tmp_path):
    store, fs, source = setup
    store.push_dir(source, "adapter")
    key = next(k for k in fs.objects if k.endswith("/weights"))
    fs.objects[key] = b"truncated"
    output = tmp_path / "out"
    output.mkdir()
    (output / "original").write_text("keep")
    with pytest.raises(ArtifactStoreError, match="integrity"):
        store.pull_dir("adapter", output)
    assert (output / "original").read_text() == "keep"
    assert not (output / "weights").exists()


@pytest.mark.parametrize("path", ["../escape", "/abs", "a/../b", "a//b", "a\\b", "a/./b", "s3:foo"])
def test_untrusted_paths_rejected(setup, tmp_path, path):
    store, fs, source = setup
    with pytest.raises(ArtifactStoreError, match="path"):
        store.push_dir(source, path)
    store.push_dir(source, "adapter")
    manifest = store._manifest("adapter")
    manifest["files"][0]["path"] = path
    fs.objects[store._key("adapter", "manifest.json")] = json.dumps(manifest).encode()
    with pytest.raises(ArtifactStoreError, match="path"):
        store.pull_dir("adapter", tmp_path / "out")


def test_symlinks_rejected(setup, tmp_path):
    store, fs, source = setup
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ArtifactStoreError, match="links"):
        store.push_dir(source, "adapter")
    assert not store.exists("adapter")


def test_permission_error_is_not_missing(setup):
    store, fs, source = setup
    fs.cat_file = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("secret"))
    with pytest.raises(ArtifactStoreError, match="manifest"):
        store.exists("adapter")


def test_generation_and_duplicate_files_validated(setup, tmp_path):
    store, fs, source = setup
    store.push_dir(source, "adapter")
    manifest = store._manifest("adapter")
    manifest["generation"] = "../escape"
    fs.objects[store._key("adapter", "manifest.json")] = json.dumps(manifest).encode()
    with pytest.raises(ArtifactStoreError, match="generation"):
        store.pull_dir("adapter", tmp_path / "out")
    store.push_dir(source, "adapter")
    manifest = store._manifest("adapter")
    manifest["files"].append(manifest["files"][0])
    fs.objects[store._key("adapter", "manifest.json")] = json.dumps(manifest).encode()
    with pytest.raises(ArtifactStoreError, match="file entry"):
        store.exists("adapter")


def test_endpoint_credentials_rejected():
    for endpoint in ["https://key:secret@example.com", "https://example.com?token=secret", "file:///tmp"]:
        with pytest.raises(ArtifactStoreError, match="endpoint"):
            S3ArtifactStore("bucket", endpoint_url=endpoint)


def test_factory_s3_needs_no_hf_and_worker_credentials_are_separate(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "merlina_artifact_store", "s3")
    monkeypatch.setattr(settings, "merlina_artifact_s3_bucket", "bucket")
    monkeypatch.setattr(settings, "merlina_artifact_access_key", "scoped-key")
    monkeypatch.setattr(settings, "merlina_artifact_secret_key", "scoped-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "platform-secret")
    config = SimpleNamespace(hf_token=None, model_dump=lambda: {"base_model": "public/model"})
    store = store_for_job(config, "job_123")
    assert store.prefix == "merlina/runs/job_123"
    assert "scoped" not in json.dumps(store.describe())
    env = artifact_worker_env(store.describe())
    assert env["MERLINA_ARTIFACT_SECRET_KEY"] == "scoped-secret"
    assert "platform-secret" not in json.dumps(env)
    assert artifact_worker_env({"kind": "hf_hub"}) == {}


def test_broker_never_receives_static_credentials(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "merlina_artifact_access_key", "broad-key")
    monkeypatch.setattr(settings, "merlina_artifact_secret_key", "broad-secret")
    env = artifact_worker_env({"kind": "s3", "credential_transport": "broker"})
    assert "broad" not in json.dumps(env)
    assert set(env) == {"MERLINA_CHECKPOINT_TIMEOUT_SECONDS"}


def test_checkpoint_timeout_reaches_s3_workers(monkeypatch):
    from config import Settings, settings
    monkeypatch.setattr(settings, "merlina_checkpoint_timeout_seconds", 300)
    env = artifact_worker_env({"kind": "s3"})
    assert env["MERLINA_CHECKPOINT_TIMEOUT_SECONDS"] == "300"
    assert artifact_worker_env({"kind": "hf_hub"}) == {}
    # The worker's own settings pick the forwarded value up from its env.
    monkeypatch.setenv("MERLINA_CHECKPOINT_TIMEOUT_SECONDS", env["MERLINA_CHECKPOINT_TIMEOUT_SECONDS"])
    assert Settings(_env_file=None).merlina_checkpoint_timeout_seconds == 300


def test_empty_and_invalid_manifests_never_count_as_committed(setup):
    store, fs, source = setup
    for data in [b"{}", b"not-json", b"null", b"[]"]:
        fs.objects[store._key("adapter", "manifest.json")] = data
        with pytest.raises(ArtifactStoreError):
            store.exists("adapter")


def test_reader_pins_generation_during_concurrent_overwrite(setup, tmp_path):
    store, fs, source = setup
    store.push_dir(source, "adapter")
    old_manifest = store._manifest("adapter")
    original_get = fs.get_file
    replaced = False

    def concurrent_get(key, local):
        nonlocal replaced
        if not replaced:
            replaced = True
            (source / "weights").write_bytes(b"new weights")
            store.push_dir(source, "adapter")
        original_get(key, local)

    fs.get_file = concurrent_get
    output = store.pull_dir("adapter", tmp_path / "out")
    assert (output / "weights").read_bytes() == b"weights"
    assert store._manifest("adapter")["generation"] != old_manifest["generation"]
