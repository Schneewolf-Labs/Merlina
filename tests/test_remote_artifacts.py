"""
Tests for artifact stores (src/remote/artifacts.py).

Run standalone: python -m pytest tests/test_remote_artifacts.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.remote.artifacts import (
    ArtifactStoreError,
    HFHubArtifactStore,
    LocalArtifactStore,
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
