"""
Tests for the internal dataset store: swl:// addressing, manifest (de)serialization,
the read path against a fake filesystem, the loader, and factory dispatch.

No real S3/MinIO is needed — the store takes an injected fsspec-like filesystem,
and the loader's load_dataset call is patched, following the existing loader tests.
"""

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

from datasets import Dataset
from dataset_handlers.internal_store import (
    InternalStore,
    InternalStoreConfig,
    InternalStoreError,
    Manifest,
    parse_swl_uri,
)
from dataset_handlers.loaders import InternalStoreLoader
from dataset_handlers.factory import create_loader, create_loader_from_config, LoaderCreationError


# ── a minimal in-memory fsspec-like filesystem ───────────────────────────────
class FakeFS:
    """Just enough of the fsspec filesystem surface for the store's read path."""

    def __init__(self, files: dict[str, str]):
        self.files = dict(files)  # key -> text content

    def open(self, path, mode="r"):
        if "r" in mode:
            if path not in self.files:
                raise FileNotFoundError(path)
            return io.StringIO(self.files[path])
        raise NotImplementedError("FakeFS is read-only")


CFG = InternalStoreConfig(
    endpoint_url="http://sabre:9000",
    bucket="datasets",
    access_key="ak",
    secret_key="sk",
)


def _store(files):
    return InternalStore(CFG, fs=FakeFS(files))


# ── parse_swl_uri ────────────────────────────────────────────────────────────
def test_parse_swl_uri_defaults_to_latest():
    assert parse_swl_uri("swl://athanorlite-dpo") == ("athanorlite-dpo", "latest")


def test_parse_swl_uri_with_revision():
    assert parse_swl_uri("swl://athanorlite-dpo@2026-10-08") == (
        "athanorlite-dpo",
        "2026-10-08",
    )


def test_parse_swl_uri_name_may_contain_slashes():
    assert parse_swl_uri("swl://org/athanorlite-dpo@v2") == ("org/athanorlite-dpo", "v2")


def test_parse_swl_uri_rejects_non_swl():
    with pytest.raises(InternalStoreError):
        parse_swl_uri("hf://org/name")


def test_parse_swl_uri_rejects_empty_name():
    with pytest.raises(InternalStoreError):
        parse_swl_uri("swl://")
    with pytest.raises(InternalStoreError):
        parse_swl_uri("swl://@rev")


# ── Manifest ─────────────────────────────────────────────────────────────────
def test_manifest_roundtrip_and_hf_revision():
    doc = {
        "name": "d",
        "rev": "r1",
        "format": "parquet",
        "data_files": ["s3://datasets/datasets/d/r1/data-0.parquet"],
        "num_rows": 10,
        "source": {"hf_repo": "schneewolflabs/d", "hf_revision": "abc123"},
        "created_at": "2026-10-08T00:00:00Z",
    }
    m = Manifest.from_json("d", "r1", doc)
    assert m.hf_revision == "abc123"
    assert m.num_rows == 10
    assert m.to_json()["data_files"] == doc["data_files"]
    # roundtrip preserves provenance
    assert Manifest.from_json("d", "r1", m.to_json()).hf_revision == "abc123"


def test_manifest_missing_data_files_is_error():
    with pytest.raises(InternalStoreError):
        Manifest.from_json("d", "r1", {"format": "parquet"})


def test_manifest_empty_data_files_is_error():
    with pytest.raises(InternalStoreError):
        Manifest.from_json("d", "r1", {"data_files": []})


# ── InternalStore read path ──────────────────────────────────────────────────
def _manifest_files(name, rev, data_files, source=None):
    key = f"datasets/datasets/{name}/{rev}/manifest.json"
    doc = {"name": name, "rev": rev, "format": "parquet", "data_files": data_files}
    if source:
        doc["source"] = source
    return {key: json.dumps(doc)}


def test_resolve_rev_follows_latest_pointer():
    files = {"datasets/datasets/d/latest": "2026-10-08\n"}
    store = _store(files)
    assert store.resolve_rev("d", "latest") == "2026-10-08"


def test_resolve_rev_passes_through_concrete():
    store = _store({})
    assert store.resolve_rev("d", "r5") == "r5"


def test_resolve_rev_missing_latest_pointer_errors():
    store = _store({})
    with pytest.raises(InternalStoreError):
        store.resolve_rev("d", "latest")


def test_get_manifest_latest():
    data = ["s3://datasets/datasets/d/2026-10-08/data-0.parquet"]
    files = {"datasets/datasets/d/latest": "2026-10-08"}
    files.update(_manifest_files("d", "2026-10-08", data, source={"hf_revision": "xyz"}))
    store = _store(files)
    m = store.get_manifest("d", "latest")
    assert m.rev == "2026-10-08"
    assert m.data_files == data
    assert m.hf_revision == "xyz"


def test_get_manifest_missing_is_error():
    store = _store({"datasets/datasets/d/latest": "r1"})  # pointer but no manifest
    with pytest.raises(InternalStoreError):
        store.get_manifest("d", "latest")


def test_get_manifest_bad_json_is_error():
    key = "datasets/datasets/d/r1/manifest.json"
    store = _store({key: "{not json"})
    with pytest.raises(InternalStoreError):
        store.get_manifest("d", "r1")


def test_exists():
    data = ["s3://datasets/datasets/d/r1/data-0.parquet"]
    store = _store(_manifest_files("d", "r1", data))
    assert store.exists("d", "r1") is True
    assert store.exists("d", "nope") is False


def test_storage_options_shape():
    store = _store({})
    opts = store.storage_options()
    assert opts["key"] == "ak"
    assert opts["secret"] == "sk"
    assert opts["client_kwargs"]["endpoint_url"] == "http://sabre:9000"


# ── InternalStoreConfig.from_settings ────────────────────────────────────────
class _Settings:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_config_from_settings_complete():
    s = _Settings(
        s3_endpoint_url="http://sabre:9000",
        s3_dataset_bucket="datasets",
        s3_access_key="ak",
        s3_secret_key="sk",
        s3_region="auto",
    )
    cfg = InternalStoreConfig.from_settings(s)
    assert cfg is not None and cfg.bucket == "datasets"


def test_config_from_settings_missing_returns_none():
    s = _Settings(s3_endpoint_url="http://sabre:9000", s3_dataset_bucket=None,
                  s3_access_key=None, s3_secret_key=None)
    assert InternalStoreConfig.from_settings(s) is None


# ── InternalStoreLoader ──────────────────────────────────────────────────────
def test_loader_resolves_rev_and_passes_data_files():
    data = ["s3://datasets/datasets/d/2026-10-08/data-0.parquet"]
    files = {"datasets/datasets/d/latest": "2026-10-08"}
    files.update(_manifest_files("d", "2026-10-08", data, source={"hf_revision": "rev99"}))
    store = _store(files)

    rows = [{"prompt": "q", "chosen": "a", "rejected": "b"}]
    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = Dataset.from_list(rows)
        loader = InternalStoreLoader("swl://d", store)
        ds = loader.load()

    assert len(ds) == 1
    # load_dataset was called with parquet + the manifest's data_files + storage_options
    args, kwargs = mock_load.call_args
    assert args[0] == "parquet"
    assert kwargs["data_files"] == data
    assert kwargs["storage_options"]["client_kwargs"]["endpoint_url"] == "http://sabre:9000"
    # provenance reflects the resolved concrete revision, not "latest"
    info = loader.get_source_info()
    assert info["source_type"] == "internal"
    assert info["rev"] == "2026-10-08"
    assert info["hf_revision"] == "rev99"


def test_loader_applies_max_samples():
    data = ["s3://datasets/datasets/d/r1/data-0.parquet"]
    store = _store(_manifest_files("d", "r1", data))
    rows = [{"prompt": f"q{i}"} for i in range(10)]
    with patch("dataset_handlers.loaders.load_dataset") as mock_load:
        mock_load.return_value = Dataset.from_list(rows)
        loader = InternalStoreLoader("swl://d@r1", store, max_samples=3)
        ds = loader.load()
    assert len(ds) == 3


def test_loader_missing_manifest_raises_valueerror():
    store = _store({})  # nothing
    loader = InternalStoreLoader("swl://d@r1", store)
    with pytest.raises(ValueError):
        loader.load()


# ── factory dispatch ─────────────────────────────────────────────────────────
def test_factory_internal_requires_uri():
    with pytest.raises(LoaderCreationError):
        create_loader(source_type="internal", store=_store({}))


def test_factory_internal_requires_store():
    with pytest.raises(LoaderCreationError):
        create_loader(source_type="internal", uri="swl://d")


def test_factory_internal_builds_loader():
    store = _store({})
    loader = create_loader(source_type="internal", uri="swl://d@r1", store=store)
    assert isinstance(loader, InternalStoreLoader)
    assert loader.name == "d" and loader.rev == "r1"


def test_factory_from_config_swl_shorthand():
    """A repo_id starting with swl:// is treated as an internal source."""
    store = _store({})
    cfg = {"repo_id": "swl://d@r1"}  # no explicit source_type
    loader = create_loader_from_config(cfg, store=store)
    assert isinstance(loader, InternalStoreLoader)
    assert loader.rev == "r1"


def test_factory_from_config_explicit_internal():
    store = _store({})
    cfg = {"source_type": "internal", "uri": "swl://d"}
    loader = create_loader_from_config(cfg, store=store)
    assert isinstance(loader, InternalStoreLoader)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
