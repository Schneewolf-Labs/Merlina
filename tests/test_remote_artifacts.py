"""
Tests for artifact stores (src/remote/artifacts.py).

Run standalone: python -m pytest tests/test_remote_artifacts.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from types import SimpleNamespace

from src.remote.artifacts import (
    ARTIFACT_S3_ACCESS_KEY_ENV,
    ARTIFACT_S3_SECRET_KEY_ENV,
    ArtifactStoreError,
    HFHubArtifactStore,
    LocalArtifactStore,
    S3ArtifactStore,
    store_from_description,
)


def make_dir(path: Path, files: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (path / name).write_text(content)
    return path


class TestLocalArtifactStore:
    def test_roundtrip(self, tmp_path):
        store = LocalArtifactStore(tmp_path / "store")
        src = make_dir(tmp_path / "src", {"adapter_model.safetensors": "weights",
                                          "adapter_config.json": "{}"})
        store.push_dir(src, "adapter")
        assert store.exists("adapter")

        dest = store.pull_dir("adapter", tmp_path / "dest")
        assert (dest / "adapter_model.safetensors").read_text() == "weights"

    def test_push_overwrites(self, tmp_path):
        store = LocalArtifactStore(tmp_path / "store")
        store.push_dir(make_dir(tmp_path / "v1", {"f": "one"}), "adapter")
        store.push_dir(make_dir(tmp_path / "v2", {"f": "two"}), "adapter")
        dest = store.pull_dir("adapter", tmp_path / "out")
        assert (dest / "f").read_text() == "two"

    def test_pull_missing_raises(self, tmp_path):
        store = LocalArtifactStore(tmp_path / "store")
        with pytest.raises(ArtifactStoreError, match="not found"):
            store.pull_dir("nope", tmp_path / "out")

    def test_describe_roundtrip(self, tmp_path):
        store = LocalArtifactStore(tmp_path / "store")
        rebuilt = store_from_description(store.describe())
        assert isinstance(rebuilt, LocalArtifactStore)
        assert rebuilt.base_dir == store.base_dir


class FakeHfApi:
    instances = []

    def __init__(self, token=None):
        self.token = token
        self.created = []
        self.uploaded = []
        self.repo_files = []
        FakeHfApi.instances.append(self)

    def create_repo(self, repo_id, private=True, exist_ok=False):
        self.created.append((repo_id, private))

    def upload_folder(self, folder_path=None, path_in_repo=None, repo_id=None,
                      commit_message=None):
        self.uploaded.append((folder_path, path_in_repo, repo_id))

    def list_repo_files(self, repo_id):
        return self.repo_files


class TestHFHubArtifactStore:
    @pytest.fixture(autouse=True)
    def patch_hf(self, monkeypatch):
        FakeHfApi.instances = []
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)

    def test_push_creates_private_repo_and_uploads(self, tmp_path):
        store = HFHubArtifactStore("user/merlina-run-job_1", token="hf_x")
        src = make_dir(tmp_path / "adapter", {"adapter_config.json": "{}"})
        store.push_dir(src, "adapter")

        api = FakeHfApi.instances[0]
        assert ("user/merlina-run-job_1", True) in api.created
        # find the upload call across instances (one HfApi per _api() call)
        uploads = [u for inst in FakeHfApi.instances for u in inst.uploaded]
        assert uploads == [(str(src), "adapter", "user/merlina-run-job_1")]

    def test_exists_checks_prefix(self):
        store = HFHubArtifactStore("u/r", token="t")
        FakeHfApi.repo_files = []  # class default unused; patch per instance

        # exists() builds a fresh api; make every instance report the files
        def fake_list(self, repo_id):
            return ["adapter/adapter_config.json", "README.md"]
        FakeHfApi.list_repo_files = fake_list

        assert store.exists("adapter")
        assert not store.exists("merged")

    def test_pull_via_snapshot_download(self, tmp_path, monkeypatch):
        def fake_snapshot(repo_id, token=None, allow_patterns=None, local_dir=None):
            make_dir(Path(local_dir) / "adapter", {"adapter_config.json": "{}"})
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

        store = HFHubArtifactStore("u/r", token="t")
        dest = store.pull_dir("adapter", tmp_path / "out")
        assert (dest / "adapter_config.json").exists()

    def test_describe_excludes_token(self):
        store = HFHubArtifactStore("u/r", token="hf_secret")
        desc = store.describe()
        assert "hf_secret" not in str(desc)
        rebuilt = store_from_description(desc, token="hf_other")
        assert isinstance(rebuilt, HFHubArtifactStore)
        assert rebuilt.repo_id == "u/r"
        assert rebuilt.token == "hf_other"


def test_unknown_store_kind_raises():
    with pytest.raises(ArtifactStoreError, match="Unknown"):
        store_from_description({"kind": "ftp"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class FakeS3FS:
    """Dict-backed stand-in for s3fs.S3FileSystem (keys are 'bucket/path')."""

    def __init__(self, fail_writes=False):
        self.objects = {}
        self.fail_writes = fail_writes

    def invalidate_cache(self):
        pass

    def find(self, path):
        path = path.rstrip("/")
        return sorted(k for k in self.objects if k == path or k.startswith(path + "/"))

    def rm(self, keys):
        for k in [keys] if isinstance(keys, str) else keys:
            self.objects.pop(k, None)

    def put_file(self, local, key):
        if self.fail_writes:
            raise PermissionError("403 Forbidden")
        self.objects[key] = Path(local).read_bytes()

    def pipe_file(self, key, data):
        if self.fail_writes:
            raise PermissionError("403 Forbidden")
        self.objects[key] = data

    def get_file(self, key, local):
        Path(local).write_bytes(self.objects[key])


def make_s3_store(fs=None, **kw):
    kw.setdefault("access_key", "ak")
    kw.setdefault("secret_key", "sk")
    return S3ArtifactStore("merlina", "artifacts/job_1",
                           endpoint_url="https://r2.example", fs=fs or FakeS3FS(), **kw)


class TestS3ArtifactStore:
    def test_roundtrip_nested(self, tmp_path):
        store = make_s3_store()
        src = make_dir(tmp_path / "src", {"adapter_model.safetensors": "weights",
                                          "adapter_config.json": "{}"})
        make_dir(src / "checkpoint", {"state.json": "s"})
        store.push_dir(src, "adapter")
        assert "merlina/artifacts/job_1/adapter/checkpoint/state.json" in store.fs.objects
        assert store.exists("adapter")

        dest = store.pull_dir("adapter", tmp_path / "dest")
        assert (dest / "adapter_model.safetensors").read_text() == "weights"
        assert (dest / "checkpoint" / "state.json").read_text() == "s"

    def test_push_overwrites_and_drops_stale_files(self, tmp_path):
        store = make_s3_store()
        store.push_dir(make_dir(tmp_path / "v1", {"f": "one", "old": "x"}), "adapter")
        store.push_dir(make_dir(tmp_path / "v2", {"f": "two"}), "adapter")
        dest = store.pull_dir("adapter", tmp_path / "out")
        assert (dest / "f").read_text() == "two"
        assert not (dest / "old").exists()

    def test_sibling_prefix_not_confused(self, tmp_path):
        store = make_s3_store()
        store.push_dir(make_dir(tmp_path / "a", {"f": "x"}), "adapter-old")
        assert not store.exists("adapter")
        with pytest.raises(ArtifactStoreError, match="not found"):
            store.pull_dir("adapter", tmp_path / "out")

    def test_push_failure_wrapped(self, tmp_path):
        store = make_s3_store(fs=FakeS3FS(fail_writes=True))
        with pytest.raises(ArtifactStoreError, match="Failed to push artifact 'adapter'"):
            store.push_dir(make_dir(tmp_path / "src", {"f": "x"}), "adapter")

    def test_verify_writable_leaves_no_probe(self):
        store = make_s3_store()
        store.verify_writable()
        assert store.fs.objects == {}

    def test_verify_writable_raises_when_denied(self):
        store = make_s3_store(fs=FakeS3FS(fail_writes=True))
        with pytest.raises(ArtifactStoreError, match="not writable"):
            store.verify_writable()

    def test_describe_excludes_credentials(self):
        desc = make_s3_store(access_key="AKSECRET", secret_key="SKSECRET").describe()
        assert desc["kind"] == "s3"
        assert "AKSECRET" not in str(desc) and "SKSECRET" not in str(desc)

    def test_worker_env_carries_credentials(self):
        env = make_s3_store().worker_env()
        assert env == {ARTIFACT_S3_ACCESS_KEY_ENV: "ak", ARTIFACT_S3_SECRET_KEY_ENV: "sk"}

    def test_rebuilt_from_description_and_env(self):
        store = make_s3_store()
        rebuilt = store_from_description(store.describe(), env=store.worker_env())
        assert isinstance(rebuilt, S3ArtifactStore)
        assert (rebuilt.bucket, rebuilt.prefix, rebuilt.endpoint_url) == \
            ("merlina", "artifacts/job_1", "https://r2.example")
        assert (rebuilt.access_key, rebuilt.secret_key) == ("ak", "sk")

    def test_rebuild_without_credentials_raises(self):
        with pytest.raises(ArtifactStoreError, match=ARTIFACT_S3_ACCESS_KEY_ENV):
            store_from_description(make_s3_store().describe(), env={})


def r2_settings(**overrides):
    base = dict(r2_endpoint_url="https://r2.example", r2_dataset_bucket="merlina",
                r2_artifact_bucket=None, r2_access_key="ak", r2_secret_key="sk",
                r2_region="auto")
    base.update(overrides)
    return SimpleNamespace(**base)


class TestFromR2Settings:
    def test_builds_job_scoped_store(self):
        store = S3ArtifactStore.from_r2_settings(r2_settings(), "job_9")
        assert (store.bucket, store.prefix) == ("merlina", "artifacts/job_9")

    def test_artifact_bucket_overrides_dataset_bucket(self):
        store = S3ArtifactStore.from_r2_settings(r2_settings(r2_artifact_bucket="arts"), "j")
        assert store.bucket == "arts"

    @pytest.mark.parametrize("missing", ["r2_endpoint_url", "r2_dataset_bucket",
                                         "r2_access_key", "r2_secret_key"])
    def test_incomplete_config_is_none(self, missing):
        assert S3ArtifactStore.from_r2_settings(r2_settings(**{missing: None}), "j") is None
