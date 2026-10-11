"""
Mirror-on-first-use: name scheme, the mirror/hit/miss/offline decision logic,
the snapshot write path, and factory routing of huggingface sources through the
mirror when prefer_internal is set.

HF calls (revision resolution + load_dataset) are patched and the store uses an
in-memory fake filesystem — no network or real S3.
"""

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

from datasets import Dataset
import dataset_handlers.factory as factory
from dataset_handlers.internal_store import InternalStore, InternalStoreConfig, InternalStoreError
from dataset_handlers.loaders import MirroredHuggingFaceLoader, HuggingFaceLoader
from dataset_handlers import mirror as mirror_mod


# ── a writable in-memory fsspec-like filesystem ──────────────────────────────
class FakeFS:
    def __init__(self):
        self.files: dict[str, str] = {}

    def open(self, path, mode="r"):
        if "r" in mode:
            if path not in self.files:
                raise FileNotFoundError(path)
            return io.StringIO(self.files[path])
        # write modes
        fs, key = self, path

        class _W(io.StringIO):
            def close(self_inner):
                fs.files[key] = self_inner.getvalue()
                super().close()
        return _W()

    def put(self, local_path, key):  # upload_file
        with open(local_path, "rb") as f:
            self.files[key] = f"<parquet:{len(f.read())}bytes>"


CFG = InternalStoreConfig(endpoint_url="http://x:9000", bucket="datasets",
                          access_key="ak", secret_key="sk")


def _store():
    return InternalStore(CFG, fs=FakeFS())


# ── mirror_name ──────────────────────────────────────────────────────────────
def test_mirror_name_scheme():
    assert mirror_mod.mirror_name("org/ds", None, "train") == "hf/org/ds/train"
    assert mirror_mod.mirror_name("org/ds", "sub", "test") == "hf/org/ds/sub/test"


# ── ensure_mirror: miss → snapshot ───────────────────────────────────────────
def test_ensure_mirror_miss_snapshots(tmp_path):
    store = _store()
    rows = [{"prompt": f"q{i}", "chosen": "a", "rejected": "r"} for i in range(4)]
    with patch.object(mirror_mod, "resolve_hf_revision", return_value="sha123"), \
         patch("datasets.load_dataset", return_value=Dataset.from_list(rows)):
        name, rev = mirror_mod.ensure_mirror(store, "org/ds", split="train")

    assert name == "hf/org/ds/train" and rev == "sha123"
    # manifest + latest pointer were written, pinned to the sha
    m = store.get_manifest(name, "latest")
    assert m.rev == "sha123"
    assert m.hf_revision == "sha123"
    assert m.num_rows == 4
    assert m.source["hf_repo"] == "org/ds"


# ── ensure_mirror: hit → no download ─────────────────────────────────────────
def test_ensure_mirror_hit_skips_download():
    store = _store()
    rows = [{"prompt": "q"}]
    # first call snapshots
    with patch.object(mirror_mod, "resolve_hf_revision", return_value="shaA"), \
         patch("datasets.load_dataset", return_value=Dataset.from_list(rows)):
        mirror_mod.ensure_mirror(store, "org/ds")
    # second call: sha unchanged -> load_dataset must NOT be called
    with patch.object(mirror_mod, "resolve_hf_revision", return_value="shaA"), \
         patch("datasets.load_dataset", side_effect=AssertionError("should not download")):
        name, rev = mirror_mod.ensure_mirror(store, "org/ds")
    assert rev == "shaA"


# ── ensure_mirror: offline with existing mirror → serve cached ───────────────
def test_ensure_mirror_offline_falls_back_to_cache():
    store = _store()
    rows = [{"prompt": "q"}]
    with patch.object(mirror_mod, "resolve_hf_revision", return_value="shaA"), \
         patch("datasets.load_dataset", return_value=Dataset.from_list(rows)):
        mirror_mod.ensure_mirror(store, "org/ds")
    # Hub unreachable now (sha None) but a mirror exists -> serve latest
    with patch.object(mirror_mod, "resolve_hf_revision", return_value=None), \
         patch("datasets.load_dataset", side_effect=AssertionError("offline, no download")):
        name, rev = mirror_mod.ensure_mirror(store, "org/ds")
    assert rev == "shaA"


# ── ensure_mirror: offline with nothing cached → error ───────────────────────
def test_ensure_mirror_offline_no_cache_errors():
    store = _store()
    with patch.object(mirror_mod, "resolve_hf_revision", return_value=None):
        with pytest.raises(InternalStoreError):
            mirror_mod.ensure_mirror(store, "org/ds")


# ── MirroredHuggingFaceLoader end-to-end (mocked HF, fake store) ──────────────
def test_mirrored_loader_loads_and_reports_source():
    store = _store()
    rows = [{"prompt": f"q{i}", "chosen": "a", "rejected": "r"} for i in range(3)]
    # Two distinct load_dataset symbols: datasets.load_dataset is the HF pull in
    # the mirror's snapshot; loaders.load_dataset is the parquet read-back from
    # the store (which would otherwise hit real S3). Patch both.
    with patch.object(mirror_mod, "resolve_hf_revision", return_value="shaZ"), \
         patch("datasets.load_dataset", return_value=Dataset.from_list(rows)), \
         patch("dataset_handlers.loaders.load_dataset", return_value=Dataset.from_list(rows)):
        loader = MirroredHuggingFaceLoader("org/ds", store, split="train")
        ds = loader.load()
    assert len(ds) == 3
    info = loader.get_source_info()
    assert info["mirrored"] is True
    assert info["repo_id"] == "org/ds"
    assert info["rev"] == "shaZ"
    assert info["hf_revision"] == "shaZ"


# ── factory routing ──────────────────────────────────────────────────────────
def test_factory_routes_hf_through_mirror_when_preferred():
    store = _store()
    loader = factory.create_loader(
        source_type="huggingface", repo_id="org/ds", store=store, prefer_internal=True
    )
    assert isinstance(loader, MirroredHuggingFaceLoader)


def test_factory_hf_direct_without_prefer_internal():
    store = _store()
    loader = factory.create_loader(
        source_type="huggingface", repo_id="org/ds", store=store, prefer_internal=False
    )
    assert isinstance(loader, HuggingFaceLoader)


def test_factory_streaming_not_mirrored():
    """A streaming source is left direct even with prefer_internal (a mirror
    would materialize the split, defeating streaming)."""
    from dataset_handlers.loaders import StreamingHuggingFaceLoader
    store = _store()
    loader = factory.create_loader(
        source_type="huggingface", repo_id="org/ds", store=store,
        prefer_internal=True, streaming=True,
    )
    assert isinstance(loader, StreamingHuggingFaceLoader)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
