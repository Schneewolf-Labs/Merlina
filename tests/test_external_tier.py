"""
External store tier: a second S3 backend (R2) alongside the internal MinIO
store, selected per process by DATASET_STORE, with a one-way `promote` to copy
revisions internal -> external and a guard that keeps private data internal.

No network or real S3: both stores use an in-memory fake filesystem, and
promotion streams through real temp files (the only on-disk I/O).
"""

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import dataset_handlers.factory as factory
from dataset_handlers.internal_store import (
    InternalStore,
    InternalStoreConfig,
    InternalStoreError,
    Manifest,
    is_private_name,
)
from dataset_handlers.promote import promote


# ── a writable in-memory fsspec-like filesystem (text + binary) ──────────────
class FakeFS:
    def __init__(self):
        self.files: dict[str, object] = {}  # str for text writes, bytes for puts

    def open(self, path, mode="r"):
        if "r" in mode:
            if path not in self.files:
                raise FileNotFoundError(path)
            data = self.files[path]
            if "b" in mode:
                return io.BytesIO(data if isinstance(data, bytes) else data.encode())
            return io.StringIO(data if isinstance(data, str) else data.decode())
        binary = "b" in mode
        fs, key = self, path
        base = io.BytesIO if binary else io.StringIO

        class _W(base):
            def close(self_inner):
                fs.files[key] = self_inner.getvalue()
                super().close()

        return _W()

    def put(self, local_path, key):
        with open(local_path, "rb") as f:
            self.files[key] = f.read()

    def get(self, key, local_path):
        if key not in self.files:
            raise FileNotFoundError(key)
        data = self.files[key]
        with open(local_path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode())

    def size(self, key):
        if key not in self.files:
            raise FileNotFoundError(key)
        return len(self.files[key])


def _store(bucket="datasets"):
    cfg = InternalStoreConfig(endpoint_url="http://x:9000", bucket=bucket,
                              access_key="ak", secret_key="sk")
    return InternalStore(cfg, fs=FakeFS())


def _seed(store, name, rev, shards, *, num_rows=None, source=None, private=False):
    """Write shards (filename -> bytes) + manifest + latest directly into a store."""
    for fn, data in shards.items():
        key = f"{store.config.bucket}/datasets/{name}/{rev}/{fn}"
        store.fs.files[key] = data
    m = Manifest(
        name=name, rev=rev, format="parquet",
        data_files=[store.data_uri(name, rev, fn) for fn in shards],
        num_rows=num_rows, source=source or {}, private=private,
    )
    store.put_manifest(m)
    store.set_latest(name, rev)
    return m


# ── config: the r2_* prefix ──────────────────────────────────────────────────
class _Settings:
    pass


def test_from_settings_reads_r2_prefix():
    s = _Settings()
    s.r2_endpoint_url = "https://acct.r2.cloudflarestorage.com"
    s.r2_dataset_bucket = "merlina-datasets"
    s.r2_access_key = "ra"
    s.r2_secret_key = "rs"
    s.r2_region = "auto"
    cfg = InternalStoreConfig.from_settings(s, prefix="r2")
    assert cfg is not None
    assert cfg.endpoint_url.endswith("r2.cloudflarestorage.com")
    assert cfg.bucket == "merlina-datasets"


def test_from_settings_r2_missing_returns_none():
    s = _Settings()
    s.r2_endpoint_url = None
    assert InternalStoreConfig.from_settings(s, prefix="r2") is None


def test_from_settings_default_prefix_is_s3():
    s = _Settings()
    s.s3_endpoint_url = "http://127.0.0.1:9000"
    s.s3_dataset_bucket = "datasets"
    s.s3_access_key = "a"
    s.s3_secret_key = "b"
    cfg = InternalStoreConfig.from_settings(s)  # default prefix
    assert cfg is not None and cfg.bucket == "datasets"


# ── factory backend selection by DATASET_STORE ───────────────────────────────
@pytest.fixture(autouse=True)
def _reset_cache():
    factory._reset_store_cache()
    yield
    factory._reset_store_cache()


def _configure(monkeypatch, *, internal=True, external=True, active="internal"):
    import config
    pairs = {}
    if internal:
        pairs.update(s3_endpoint_url="http://127.0.0.1:9000", s3_dataset_bucket="datasets",
                     s3_access_key="a", s3_secret_key="b")
    else:
        pairs.update(s3_endpoint_url=None)
    if external:
        pairs.update(r2_endpoint_url="https://acct.r2.cloudflarestorage.com",
                     r2_dataset_bucket="ext", r2_access_key="ra", r2_secret_key="rs")
    else:
        pairs.update(r2_endpoint_url=None)
    pairs["dataset_store"] = active
    for k, v in pairs.items():
        monkeypatch.setattr(config.settings, k, v, raising=False)


def test_active_store_defaults_internal(monkeypatch):
    _configure(monkeypatch, active="internal")
    assert factory.active_store_name() == "internal"
    assert factory.default_internal_store().config.bucket == "datasets"


def test_active_store_external_selects_r2(monkeypatch):
    _configure(monkeypatch, active="external")
    assert factory.active_store_name() == "external"
    assert factory.default_internal_store().config.bucket == "ext"


def test_get_store_each_backend(monkeypatch):
    _configure(monkeypatch)
    assert factory.get_store("internal").config.bucket == "datasets"
    assert factory.get_store("external").config.bucket == "ext"


def test_get_store_unconfigured_backend_is_none(monkeypatch):
    _configure(monkeypatch, external=False)
    assert factory.get_store("external") is None
    # cached negative result returns None again without re-probing
    assert factory.get_store("external") is None


# ── private markers + manifest flag ──────────────────────────────────────────
def test_is_private_name_markers():
    assert is_private_name("vps-incident-schneewolf-west1")
    assert is_private_name("hf/org/private/thing")
    assert is_private_name("some-INCIDENT-notes")  # case-insensitive
    assert not is_private_name("hf/org/athanorlite-dpo/train")


def test_manifest_private_round_trips():
    m = Manifest("x", "r1", "parquet", ["s3://b/datasets/x/r1/d.parquet"], private=True)
    doc = m.to_json()
    assert doc["private"] is True
    back = Manifest.from_json("x", "r1", doc)
    assert back.private is True and back.is_private()


def test_manifest_default_not_private():
    m = Manifest("x", "r1", "parquet", ["s3://b/datasets/x/r1/d.parquet"])
    assert "private" not in m.to_json()
    assert m.is_private() is False


# ── promote: the happy path ───────────────────────────────────────────────────
def test_promote_copies_shards_manifest_and_latest():
    src, dst = _store("datasets"), _store("ext")
    payload = b"PARQUET-BYTES-abc"
    _seed(src, "hf/org/ds/train", "sha1", {"data-0.parquet": payload},
          num_rows=10, source={"hf_repo": "org/ds", "hf_revision": "sha1"})

    plan = promote(src, dst, "hf/org/ds/train", "latest")

    assert plan.num_shards == 1 and plan.skipped is False
    # destination now resolves the same rev...
    assert dst.exists("hf/org/ds/train", "sha1")
    dm = dst.get_manifest("hf/org/ds/train", "latest")
    assert dm.rev == "sha1" and dm.num_rows == 10
    # ...with the shard bytes byte-identical and the URI rewritten to dst bucket
    assert dm.data_files[0] == "s3://ext/datasets/hf/org/ds/train/sha1/data-0.parquet"
    assert dst.fs.files["ext/datasets/hf/org/ds/train/sha1/data-0.parquet"] == payload


def test_promote_skips_when_already_present():
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "ds", "r1", {"d.parquet": b"x"})
    promote(src, dst, "ds")  # first copy
    plan = promote(src, dst, "ds")  # second: already there
    assert plan.skipped is True and plan.already_present is True


def test_promote_overwrite_recopies():
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "ds", "r1", {"d.parquet": b"v1"})
    promote(src, dst, "ds")
    # change the source bytes for the same rev, then overwrite
    src.fs.files["datasets/datasets/ds/r1/d.parquet"] = b"v2"
    plan = promote(src, dst, "ds", overwrite=True)
    assert plan.skipped is False
    assert dst.fs.files["ext/datasets/ds/r1/d.parquet"] == b"v2"


# ── promote: the private guard ────────────────────────────────────────────────
def test_promote_refuses_private_manifest():
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "sensitive", "r1", {"d.parquet": b"x"}, private=True)
    with pytest.raises(InternalStoreError):
        promote(src, dst, "sensitive")
    assert not dst.exists("sensitive", "r1")


def test_promote_refuses_private_name():
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "vps-incident-west", "r1", {"d.parquet": b"x"})
    with pytest.raises(InternalStoreError):
        promote(src, dst, "vps-incident-west")


def test_promote_allow_private_overrides():
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "sensitive", "r1", {"d.parquet": b"x"}, private=True)
    plan = promote(src, dst, "sensitive", allow_private=True)
    assert plan.skipped is False
    assert dst.exists("sensitive", "r1")


# ── promote: dry run writes nothing ───────────────────────────────────────────
def test_promote_dry_run_does_not_write():
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "ds", "r1", {"a.parquet": b"xxxxx", "b.parquet": b"yyy"})
    plan = promote(src, dst, "ds", dry_run=True)
    assert plan.num_shards == 2
    assert plan.total_bytes == 8  # 5 + 3
    assert "8" in plan.human_bytes() or plan.human_bytes().endswith("B")
    assert not dst.exists("ds", "r1")  # nothing transferred


# ── CLI wiring (get_store patched to fakes; no config/network) ────────────────
def _patch_cli_stores(monkeypatch, src, dst):
    mapping = {"internal": src, "external": dst}
    monkeypatch.setattr(factory, "get_store", lambda which: mapping.get(which))


def test_cli_promote_dry_run(monkeypatch, capsys):
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "hf/org/ds/train", "sha1", {"data-0.parquet": b"PARQ"})
    _patch_cli_stores(monkeypatch, src, dst)
    import merlina_cli
    rc = merlina_cli.main(["datasets", "promote", "hf/org/ds/train", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN" in out
    assert not dst.exists("hf/org/ds/train", "sha1")


def test_cli_promote_real(monkeypatch, capsys):
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "ds", "sha1", {"data-0.parquet": b"PARQ"})
    _patch_cli_stores(monkeypatch, src, dst)
    import merlina_cli
    rc = merlina_cli.main(["datasets", "promote", "ds@sha1"])
    assert rc == 0
    assert dst.exists("ds", "sha1")
    assert "Promoted" in capsys.readouterr().out


def test_cli_promote_refuses_private(monkeypatch, capsys):
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "vps-incident-notes", "r1", {"d.parquet": b"x"})
    _patch_cli_stores(monkeypatch, src, dst)
    import merlina_cli
    rc = merlina_cli.main(["datasets", "promote", "vps-incident-notes"])
    assert rc == 1
    assert "Cannot promote" in capsys.readouterr().err
    assert not dst.exists("vps-incident-notes", "r1")


def test_cli_info(monkeypatch, capsys):
    src, dst = _store("datasets"), _store("ext")
    _seed(src, "ds", "sha1", {"d.parquet": b"x"}, num_rows=7,
          source={"hf_repo": "org/ds", "hf_revision": "sha1", "split": "train"})
    _patch_cli_stores(monkeypatch, src, dst)
    import merlina_cli
    rc = merlina_cli.main(["datasets", "info", "ds", "--store", "internal"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "swl://ds@sha1" in out and "rows:" in out and "hf:org/ds@sha1" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
