"""
Storage tiers: registry, browse (ls/du), sync (copy / skip / verify / move),
the private -> shared guard, delete guards, the /storage/* API and the
`merlina storage` CLI.

No network or real S3: every tier gets an in-memory fake of the slice of the
s3fs surface the module uses (find/ls/info/open/metadata/setxattr/rm/call_s3).
"""

import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import storage_tiers as st  # noqa: E402


# ── an in-memory s3fs-like filesystem ───────────────────────────────────────
class FakeS3:
    """Objects keyed by fsspec path ('bucket/key') with bytes + user metadata."""

    def __init__(self, bucket, *, can_list_buckets=True, can_create_buckets=False):
        self.bucket = bucket
        self.objects: dict[str, bytes] = {}
        self.meta: dict[str, dict] = {}
        self.setxattr_calls = []
        self.fail_writes: set[str] = set()  # paths whose write "succeeds" truncated
        self.can_list_buckets = can_list_buckets
        self.can_create_buckets = can_create_buckets
        self.buckets = {bucket}

    # helpers
    def put(self, key, data, meta=None):
        path = f"{self.bucket}/{key}"
        self.objects[path] = data if isinstance(data, bytes) else data.encode()
        self.meta[path] = dict(meta or {})

    def keys(self):
        return sorted(p.split("/", 1)[1] for p in self.objects)

    # fsspec surface
    def find(self, path, detail=False):
        path = path.rstrip("/")
        out = {}
        for p, data in self.objects.items():
            if p == path or p.startswith(path + "/"):
                out[p] = {"name": p, "type": "file", "size": len(data), "LastModified": "2026-01-01T00:00:00"}
        return out if detail else sorted(out)

    def ls(self, path, detail=False, refresh=False):
        path = path.rstrip("/")
        entries = {}
        for p, data in self.objects.items():
            if p == path:
                entries[p] = {"name": p, "type": "file", "size": len(data)}
            elif p.startswith(path + "/"):
                rest = p[len(path) + 1:]
                if "/" in rest:
                    d = f"{path}/{rest.split('/', 1)[0]}"
                    entries[d] = {"name": d, "type": "directory", "size": 0}
                else:
                    entries[p] = {"name": p, "type": "file", "size": len(data),
                                  "LastModified": "2026-01-01T00:00:00"}
        if not entries and path != self.bucket:
            raise FileNotFoundError(path)
        return list(entries.values()) if detail else sorted(entries)

    def info(self, path):
        if path not in self.objects:
            raise FileNotFoundError(path)
        return {"name": path, "type": "file", "size": len(self.objects[path])}

    def cat_file(self, path, start=None, end=None):
        if path not in self.objects:
            raise FileNotFoundError(path)
        return self.objects[path][start:end]

    def metadata(self, path):
        if path not in self.objects:
            raise FileNotFoundError(path)
        return dict(self.meta.get(path, {}))

    def setxattr(self, path, **kw):
        self.setxattr_calls.append(path)
        self.meta.setdefault(path, {}).update(kw)

    def open(self, path, mode="rb", block_size=None, Metadata=None, **kw):
        if "r" in mode:
            if path not in self.objects:
                raise FileNotFoundError(path)
            return io.BytesIO(self.objects[path])
        fs = self

        class _W(io.BytesIO):
            discarded = False

            def discard(self_inner):
                self_inner.discarded = True

            def close(self_inner):
                if not self_inner.closed and not self_inner.discarded:
                    data = self_inner.getvalue()
                    if path in fs.fail_writes:
                        data = data[:-1]
                    fs.objects[path] = data
                    fs.meta[path] = dict(Metadata or {})
                super().close()

        return _W()

    def rm(self, paths):
        for p in ([paths] if isinstance(paths, str) else paths):
            self.objects.pop(p, None)
            self.meta.pop(p, None)

    def call_s3(self, method, **kw):
        if method == "list_buckets":
            if not self.can_list_buckets:
                raise PermissionError("AccessDenied: Access Denied")
            return {"Buckets": [{"Name": b} for b in sorted(self.buckets)]}
        if method == "create_bucket":
            if not self.can_create_buckets:
                raise PermissionError("AccessDenied: Access Denied")
            self.buckets.add(kw["Bucket"])
            return {}
        raise NotImplementedError(method)

    def invalidate_cache(self, path=""):
        pass


def _settings(**over):
    base = dict(
        s3_endpoint_url="http://minio.lan:9000", s3_dataset_bucket="datasets",
        s3_access_key="INTERNAL_AK", s3_secret_key="INTERNAL_SECRET", s3_region="auto",
        r2_endpoint_url="https://acct.r2.example.com", r2_dataset_bucket="shared",
        r2_access_key="EXTERNAL_AK", r2_secret_key="EXTERNAL_SECRET", r2_region="auto",
        merlina_artifact_s3_bucket=None, merlina_artifact_s3_endpoint=None,
        merlina_artifact_s3_prefix="runs", merlina_artifact_s3_region=None,
        merlina_artifact_access_key=None, merlina_artifact_secret_key=None,
        merlina_artifact_session_token=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def fakes():
    return {"internal": FakeS3("datasets"), "external": FakeS3("shared")}


@pytest.fixture
def tiers(fakes):
    return st.build_tiers(_settings(), fs_overrides=fakes)


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ── registry ────────────────────────────────────────────────────────────────
class TestRegistry:
    def test_internal_private_external_shared(self, tiers):
        assert tiers["internal"].private and not tiers["external"].private
        assert tiers["internal"].configured and tiers["external"].configured

    def test_artifacts_deduped_when_same_as_external(self):
        t = st.build_tiers(_settings(
            merlina_artifact_s3_bucket="shared",
            merlina_artifact_s3_endpoint="https://ACCT.r2.example.com/",
            merlina_artifact_access_key="A", merlina_artifact_secret_key="S"))
        assert "artifacts" not in t
        assert t["external"].hint_prefix == "runs/"

    def test_artifacts_listed_when_distinct(self):
        t = st.build_tiers(_settings(
            merlina_artifact_s3_bucket="artifact-bucket",
            merlina_artifact_s3_endpoint="https://acct.r2.example.com",
            merlina_artifact_access_key="A", merlina_artifact_secret_key="S"))
        assert "artifacts" in t and not t["artifacts"].private
        assert t["artifacts"].hint_prefix == "runs/"

    def test_artifacts_absent_when_unconfigured(self, tiers):
        assert "artifacts" not in tiers

    def test_unconfigured_tier(self):
        t = st.build_tiers(_settings(r2_dataset_bucket=None, r2_endpoint_url=None,
                                     r2_access_key=None, r2_secret_key=None))
        assert not t["external"].configured
        with pytest.raises(st.StorageTierError, match="not configured"):
            st.get_tier("external", t)
        with pytest.raises(st.StorageTierError, match="Unknown"):
            st.get_tier("nope", t)

    def test_describe_and_repr_never_expose_credentials(self):
        t = st.build_tiers(_settings())
        for tier in t.values():
            blob = json.dumps(tier.describe()) + repr(tier)
            for secret in ("INTERNAL_AK", "INTERNAL_SECRET", "EXTERNAL_AK", "EXTERNAL_SECRET"):
                assert secret not in blob
        assert t["internal"].describe()["endpoint_host"] == "minio.lan:9000"


# ── browse ──────────────────────────────────────────────────────────────────
class TestBrowse:
    def test_ls_levels(self, tiers, fakes):
        fakes["external"].put("runs/job1/adapter/a.bin", b"x" * 10)
        fakes["external"].put("runs/job1/manifest.json", b"{}")
        fakes["external"].put("top.txt", b"hi")
        root = st.ls(tiers["external"], "")
        assert [d["prefix"] for d in root["dirs"]] == ["runs/"]
        assert [o["key"] for o in root["objects"]] == ["top.txt"]
        lvl = st.ls(tiers["external"], "runs/job1")
        assert lvl["prefix"] == "runs/job1/"
        assert [d["name"] for d in lvl["dirs"]] == ["adapter/"]
        assert lvl["objects"][0]["key"] == "runs/job1/manifest.json"
        assert lvl["objects"][0]["size"] == 2

    def test_ls_missing_prefix_is_empty(self, tiers):
        assert st.ls(tiers["external"], "nope/")["objects"] == []

    def test_du_and_prefix_boundary(self, tiers, fakes):
        fakes["external"].put("a/b/1", b"12345")
        fakes["external"].put("a/b/2", b"123")
        fakes["external"].put("a/bc/3", b"1")
        u = st.du(tiers["external"], "a/b")
        assert (u["objects"], u["bytes"]) == (2, 8)
        assert st.du(tiers["external"])["objects"] == 3

    def test_rejects_dotdot(self, tiers):
        with pytest.raises(st.StorageTierError):
            st.ls(tiers["external"], "a/../b")


# ── buckets ─────────────────────────────────────────────────────────────────
class TestBuckets:
    def test_list(self, tiers):
        r = st.list_buckets(tiers["external"])
        assert r["ok"] and r["buckets"] == ["shared"]

    def test_list_access_denied_is_reported(self, tiers, fakes):
        fakes["internal"].can_list_buckets = False
        r = st.list_buckets(tiers["internal"])
        assert not r["ok"] and "access denied" in r["message"] and "datasets" in r["message"]

    def test_make_bucket_denied_and_invalid(self, tiers, fakes):
        r = st.make_bucket(tiers["internal"], "new-bucket")
        assert not r["ok"] and "may not create" in r["message"]
        assert not st.make_bucket(tiers["internal"], "Bad_Name")["ok"]
        fakes["internal"].can_create_buckets = True
        assert st.make_bucket(tiers["internal"], "new-bucket")["ok"]


# ── sync ────────────────────────────────────────────────────────────────────
class TestSync:
    def test_copy_sets_sha_metadata(self, tiers, fakes):
        fakes["external"].put("runs/j/a.bin", b"hello")
        res = st.sync(tiers["external"], tiers["internal"], "runs/")
        assert res.ok and [c["key"] for c in res.copied] == ["runs/j/a.bin"]
        assert fakes["internal"].objects["datasets/runs/j/a.bin"] == b"hello"
        assert fakes["internal"].meta["datasets/runs/j/a.bin"]["sha256"] == sha(b"hello")

    def test_dry_run_writes_nothing(self, tiers, fakes):
        fakes["external"].put("runs/j/a.bin", b"hello")
        res = st.sync(tiers["external"], tiers["internal"], "runs/", dry_run=True)
        assert len(res.to_copy) == 1 and res.bytes_to_copy == 5
        assert fakes["internal"].objects == {}

    def test_second_run_skips_identical(self, tiers, fakes):
        fakes["external"].put("runs/j/a.bin", b"hello")
        st.sync(tiers["external"], tiers["internal"], "runs/")
        # Source has no metadata, dst has: skipped on size (no sha to compare).
        res = st.sync(tiers["external"], tiers["internal"], "runs/")
        assert not res.to_copy and res.skipped[0]["reason"].startswith("same size")
        # Reverse direction: both sides now carry sha256 metadata.
        res = st.sync(tiers["internal"], tiers["external"], "runs/", dry_run=True)
        assert res.skipped and "sha256" in res.skipped[0]["reason"]

    def test_same_size_sha_metadata_mismatch_recopies(self, tiers, fakes):
        fakes["external"].put("k", b"AAAA", {"sha256": sha(b"AAAA")})
        fakes["internal"].put("k", b"BBBB", {"sha256": sha(b"BBBB")})
        res = st.sync(tiers["external"], tiers["internal"], "k")
        assert [c["key"] for c in res.copied] == ["k"]
        assert fakes["internal"].objects["datasets/k"] == b"AAAA"

    def test_same_size_no_metadata_skips_unless_verify(self, tiers, fakes):
        fakes["external"].put("k", b"AAAA")
        fakes["internal"].put("k", b"BBBB")
        assert st.sync(tiers["external"], tiers["internal"], "k").skipped
        res = st.sync(tiers["external"], tiers["internal"], "k", verify=True)
        assert res.copied and res.copied[0]["key"] == "k"
        assert fakes["internal"].objects["datasets/k"] == b"AAAA"
        # identical content + verify -> hashed and skipped
        res = st.sync(tiers["external"], tiers["internal"], "k", verify=True)
        assert res.skipped[0]["reason"] in ("identical (hashed)", "identical (sha256 metadata)")

    def test_size_differs_and_overwrite(self, tiers, fakes):
        fakes["external"].put("k", b"AAAA")
        fakes["internal"].put("k", b"AA")
        assert st.sync(tiers["external"], tiers["internal"], "k", dry_run=True).to_copy[0]["reason"] == "size differs"
        fakes["internal"].put("k", b"AAAA")
        assert st.sync(tiers["external"], tiers["internal"], "k", dry_run=True, overwrite=True).to_copy

    def test_large_object_streams_and_attaches_metadata(self, tiers, fakes, monkeypatch):
        monkeypatch.setattr(st, "CHUNK", 4)
        data = b"0123456789abcdef!"
        fakes["external"].put("big", data)
        seen = []
        res = st.sync(tiers["external"], tiers["internal"], "big", progress_cb=seen.append)
        assert res.ok and fakes["internal"].objects["datasets/big"] == data
        assert fakes["internal"].setxattr_calls == ["datasets/big"]
        assert fakes["internal"].meta["datasets/big"]["sha256"] == sha(data)
        assert max(p["bytes_done"] for p in seen) == len(data)
        assert seen[-1]["phase"] == "done"

    def test_large_object_over_copy_limit_reports_meta_missing(self, tiers, fakes, monkeypatch):
        monkeypatch.setattr(st, "CHUNK", 4)
        monkeypatch.setattr(st, "MAX_COPY_OBJECT", 8)
        fakes["external"].put("big", b"x" * 20)
        res = st.sync(tiers["external"], tiers["internal"], "big")
        assert res.ok and res.meta_missing == ["big"]

    def test_size_verified_after_write(self, tiers, fakes):
        fakes["external"].put("k", b"hello")
        fakes["internal"].fail_writes.add("datasets/k")
        res = st.sync(tiers["external"], tiers["internal"], "k")
        assert not res.ok and "size mismatch" in res.errors[0]["error"]

    def test_same_tier_rejected(self, tiers):
        with pytest.raises(st.StorageTierError):
            st.sync(tiers["external"], tiers["external"], "")


class TestMove:
    def test_move_deletes_source_after_verified_copy(self, tiers, fakes):
        fakes["external"].put("runs/j/a", b"aaa")
        fakes["external"].put("runs/j/b", b"bbbb")
        res = st.sync(tiers["external"], tiers["internal"], "runs/j", move=True)
        assert res.ok and res.verify
        assert sorted(res.deleted) == ["runs/j/a", "runs/j/b"]
        assert fakes["external"].keys() == []
        assert fakes["internal"].keys() == ["runs/j/a", "runs/j/b"]

    def test_move_deletes_nothing_when_any_copy_fails(self, tiers, fakes):
        fakes["external"].put("runs/j/a", b"aaa")
        fakes["external"].put("runs/j/b", b"bbbb")
        fakes["internal"].fail_writes.add("datasets/runs/j/b")
        res = st.sync(tiers["external"], tiers["internal"], "runs/j", move=True)
        assert not res.ok and res.deleted == [] and res.move_aborted
        assert fakes["external"].keys() == ["runs/j/a", "runs/j/b"]

    def test_move_hash_verifies_already_present_objects(self, tiers, fakes):
        fakes["external"].put("k", b"same")
        fakes["internal"].put("k", b"same")  # no metadata anywhere
        res = st.sync(tiers["external"], tiers["internal"], "k", move=True)
        assert res.skipped and res.deleted == ["k"]

    def test_move_recopies_differing_object_before_deleting(self, tiers, fakes):
        fakes["external"].put("k", b"AAAA")
        fakes["internal"].put("k", b"BBBB")
        res = st.sync(tiers["external"], tiers["internal"], "k", move=True)
        # differing content is recopied (verified) before the source goes
        assert res.ok and res.copied and res.deleted == ["k"]
        assert fakes["internal"].objects["datasets/k"] == b"AAAA"

    def test_cancel_before_copy(self, tiers, fakes):
        import threading
        fakes["external"].put("k", b"x")
        ev = threading.Event()
        ev.set()
        res = st.sync(tiers["external"], tiers["internal"], "k", move=True, cancel=ev)
        assert not res.ok and fakes["external"].keys() == ["k"]


# ── privacy guard ───────────────────────────────────────────────────────────
def _dataset(fs, name, rev, private=False, extra=None):
    doc = {"name": name, "rev": rev, "format": "parquet",
           "data_files": [f"s3://datasets/datasets/{name}/{rev}/data-0.parquet"]}
    if private:
        doc["private"] = True
    fs.put(f"datasets/{name}/{rev}/manifest.json", json.dumps(doc))
    fs.put(f"datasets/{name}/{rev}/data-0.parquet", b"PAR1" + name.encode())
    fs.put(f"datasets/{name}/latest", rev)


class TestPrivacyGuard:
    def test_refuses_private_manifest_and_marker(self, tiers, fakes):
        _dataset(fakes["internal"], "public-ds", "r1")
        _dataset(fakes["internal"], "secret-ds", "r1", private=True)
        _dataset(fakes["internal"], "vps-incident-notes", "r1")  # name marker
        fakes["internal"].put("runs/job/incident-report.txt", b"x")  # marker in a non-dataset key
        res = st.sync(tiers["internal"], tiers["external"], "")
        refused = {r["key"] for r in res.refused}
        assert res.guarded
        assert all(k.startswith(("datasets/secret-ds/", "datasets/vps-incident-notes/")) or "incident" in k
                   for k in refused)
        assert "datasets/secret-ds/r1/data-0.parquet" in refused
        assert "datasets/secret-ds/latest" in refused
        assert "datasets/vps-incident-notes/r1/manifest.json" in refused
        assert "runs/job/incident-report.txt" in refused
        shared = fakes["external"].keys()
        assert shared == ["datasets/public-ds/latest", "datasets/public-ds/r1/data-0.parquet",
                          "datasets/public-ds/r1/manifest.json"]

    def test_private_revision_taints_whole_dataset(self, tiers, fakes):
        _dataset(fakes["internal"], "mixed", "r1")
        _dataset(fakes["internal"], "mixed", "r2", private=True)
        res = st.sync(tiers["internal"], tiers["external"], "datasets/mixed/r1/", dry_run=True)
        assert not res.to_copy and len(res.refused) == 2

    def test_slashy_names(self, tiers, fakes):
        _dataset(fakes["internal"], "org/secret", "r1", private=True)
        _dataset(fakes["internal"], "org/open", "r1")
        res = st.sync(tiers["internal"], tiers["external"], "datasets/org", dry_run=True)
        assert {r["key"].split("/")[2] for r in res.refused} == {"secret"}
        assert {c["key"].split("/")[2] for c in res.to_copy} == {"open"}

    def test_unreadable_manifest_fails_closed(self, tiers, fakes):
        fakes["internal"].put("datasets/odd/r1/manifest.json", "{not json")
        fakes["internal"].put("datasets/odd/r1/data-0.parquet", b"x")
        res = st.sync(tiers["internal"], tiers["external"], "datasets/", dry_run=True)
        assert len(res.refused) == 2 and not res.to_copy

    def test_move_never_deletes_refused(self, tiers, fakes):
        _dataset(fakes["internal"], "secret-ds", "r1", private=True)
        res = st.sync(tiers["internal"], tiers["external"], "datasets/", move=True)
        assert res.refused and not res.deleted
        assert fakes["internal"].keys() and fakes["external"].keys() == []

    def test_allow_private_override(self, tiers, fakes):
        _dataset(fakes["internal"], "secret-ds", "r1", private=True)
        res = st.sync(tiers["internal"], tiers["external"], "", allow_private=True)
        assert not res.refused and len(res.copied) == 3

    def test_no_guard_into_private_tier(self, tiers, fakes):
        _dataset(fakes["external"], "secret-ds", "r1", private=True)
        res = st.sync(tiers["external"], tiers["internal"], "")
        assert not res.guarded and not res.refused and len(res.copied) == 3


# ── privacy guard: remote-training artifacts ────────────────────────────────
GEN = "0123456789abcdef0123456789abcdef"


def _hf(repo="org/public-dpo"):
    return {"source_type": "huggingface", "repo_id": repo, "split": "train"}


def _artifact(fs, root, sources=None, *, eval_source=None, in_training_config=False,
              model="org/base-model", raw_manifest=None):
    """Write an artifact the way src/remote/s3_artifacts.py lays it out."""
    files = ["adapter_model.safetensors", "adapter_config.json"]
    for f in files:
        fs.put(f"{root}/generations/{GEN}/{f}", b"weights-" + f.encode())
    if raw_manifest is not None:
        fs.put(f"{root}/manifest.json", raw_manifest)
        return
    dataset = {}
    if sources is not None:
        dataset = {"source": sources[0], "additional_sources": sources[1:], "eval_source": eval_source}
    metadata = {"model": {"id": model, "revision": None}, "runtime": {"python": "3.10"}}
    if in_training_config:
        metadata.update(datasets={}, training_config={"dataset": dataset or None})
    else:
        metadata.update(datasets=dataset, training_config={"dataset": dataset or None})
    doc = {"version": 1, "name": root.split("/", 1)[1], "generation": GEN,
           "created_at": "2026-01-01T00:00:00+00:00", "checkpoints": [],
           "files": [{"path": f, "size": 1, "sha256": "0" * 64} for f in files],
           "metadata": metadata}
    fs.put(f"{root}/manifest.json", json.dumps(doc))


def _keys_under(fs, root):
    return {k for k in fs.keys() if k.startswith(root + "/")}


class TestArtifactGuard:
    def _run(self, tiers, prefix="runs/", **kw):
        return st.sync(tiers["internal"], tiers["external"], prefix, **kw)

    def test_public_hf_artifact_is_allowed(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/job1/adapter", [_hf()])
        res = self._run(tiers)
        assert not res.refused and len(res.copied) == 3
        assert fakes["external"].keys() == sorted(_keys_under(fakes["internal"], "runs/job1/adapter"))

    def test_dataset_metadata_only_in_training_config(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/job1/adapter", [_hf()], in_training_config=True)
        assert not self._run(tiers, dry_run=True).refused

    def _assert_refused(self, tiers, fakes, root, needle):
        res = self._run(tiers, dry_run=True)
        refused = {r["key"]: r["reason"] for r in res.refused}
        assert set(refused) == _keys_under(fakes["internal"], root), refused
        assert f"{root}/manifest.json" in refused
        assert any("/generations/" in k for k in refused)
        assert all(needle in r for r in refused.values()), refused
        assert not res.to_copy

    def test_private_marker_in_hf_repo(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/adapter", [_hf("lab/schneewolf-west-logs")])
        self._assert_refused(tiers, fakes, "runs/j/adapter", "private marker")

    def test_private_marker_in_additional_and_eval_sources(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/a/adapter", [_hf(), _hf("lab/incident-notes")])
        _artifact(fakes["internal"], "runs/b/adapter", [_hf()], eval_source=_hf("lab/private/eval"))
        res = self._run(tiers, dry_run=True)
        assert {r["key"].split("/")[1] for r in res.refused} == {"a", "b"}

    @pytest.mark.parametrize("kind", ["local_file", "upload"])
    def test_local_and_upload_sources_fail_closed(self, tiers, fakes, kind):
        _artifact(fakes["internal"], "runs/j/merged", [_hf(), {"source_type": kind, "file_format": "jsonl"}])
        self._assert_refused(tiers, fakes, "runs/j/merged", "cannot be verified")

    def test_internal_source_private_dataset(self, tiers, fakes):
        _dataset(fakes["internal"], "secret-ds", "r1", private=True)
        _artifact(fakes["internal"], "runs/j/adapter", [{"source_type": "internal", "uri": "swl://secret-ds@r1"}])
        self._assert_refused(tiers, fakes, "runs/j/adapter", "private internal dataset")

    def test_internal_source_public_dataset_allowed(self, tiers, fakes):
        _dataset(fakes["internal"], "open-ds", "r1")
        _artifact(fakes["internal"], "runs/j/adapter", [{"source_type": "internal", "uri": "swl://open-ds"}])
        res = self._run(tiers, dry_run=True)
        assert not res.refused and len(res.to_copy) == 3

    def test_internal_source_missing_dataset_fails_closed(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/adapter", [{"source_type": "internal", "uri": "swl://gone@r1"}])
        self._assert_refused(tiers, fakes, "runs/j/adapter", "not in this tier")

    def test_no_dataset_metadata_fails_closed(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/merged", None)
        self._assert_refused(tiers, fakes, "runs/j/merged", "no dataset metadata")

    def test_unreadable_manifest_fails_closed(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/merged", raw_manifest="{oops")
        self._assert_refused(tiers, fakes, "runs/j/merged", "unreadable")

    def test_orphan_generation_fails_closed(self, tiers, fakes):
        fakes["internal"].put(f"runs/j/adapter/generations/{GEN}/w.bin", b"x")
        res = self._run(tiers, dry_run=True)
        assert [r["key"] for r in res.refused] == [f"runs/j/adapter/generations/{GEN}/w.bin"]
        assert "without a manifest" in res.refused[0]["reason"]

    def test_private_base_model(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/adapter", [_hf()], model="lab/vps-incident-tuned")
        self._assert_refused(tiers, fakes, "runs/j/adapter", "base model")

    def test_narrow_prefix_still_finds_root(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/merged", None)
        res = self._run(tiers, prefix=f"runs/j/merged/generations/{GEN}/", dry_run=True)
        assert len(res.refused) == 2 and not res.to_copy

    def test_mixed_run_copies_only_shareable_artifact(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/adapter", [_hf()])
        _artifact(fakes["internal"], "runs/j/merged", None)
        res = self._run(tiers, move=True)
        assert {k.split("/")[2] for k in fakes["external"].keys()} == {"adapter"}
        assert {r["key"].split("/")[2] for r in res.refused} == {"merged"}
        # move deletes only the verified, shareable copies
        assert _keys_under(fakes["internal"], "runs/j/merged") and not _keys_under(fakes["internal"], "runs/j/adapter")

    def test_allow_private_bypasses_artifact_guard(self, tiers, fakes):
        _artifact(fakes["internal"], "runs/j/merged", [{"source_type": "upload"}])
        res = self._run(tiers, allow_private=True)
        assert not res.refused and len(res.copied) == 3

    def test_no_artifact_guard_into_private_tier(self, tiers, fakes):
        _artifact(fakes["external"], "runs/j/merged", None)
        res = st.sync(tiers["external"], tiers["internal"], "runs/")
        assert not res.refused and len(res.copied) == 3

    def test_api_has_no_override_and_reports_refusal(self, api, fakes):
        client, _ = api
        _artifact(fakes["internal"], "runs/j/merged", None)
        r = client.post("/storage/sync", json={"src": "internal", "dst": "external", "prefix": "runs/"})
        assert r.json()["plan"]["counts"]["refused"] == 3
        r = client.post("/storage/sync", json={"src": "internal", "dst": "external", "prefix": "runs/",
                                               "dry_run": False, "allow_private": True})
        assert r.status_code == 422 and fakes["external"].keys() == []


# ── delete ──────────────────────────────────────────────────────────────────
class TestDelete:
    def test_refuses_empty_prefix(self, tiers):
        for p in ("", "/", "  "):
            with pytest.raises(st.StorageTierError, match="entire bucket"):
                st.delete(tiers["external"], p, confirm=True)

    def test_requires_confirm(self, tiers, fakes):
        fakes["external"].put("tmp/a", b"1")
        with pytest.raises(st.StorageTierError, match="confirm"):
            st.delete(tiers["external"], "tmp/")
        assert fakes["external"].keys() == ["tmp/a"]

    def test_dry_run_then_delete_respects_boundary(self, tiers, fakes):
        fakes["external"].put("tmp/a", b"1")
        fakes["external"].put("tmp/b/c", b"22")
        fakes["external"].put("tmpx/keep", b"3")
        plan = st.delete(tiers["external"], "tmp", dry_run=True)
        assert plan["objects"] == 2 and plan["bytes"] == 3 and not plan["deleted"]
        out = st.delete(tiers["external"], "tmp/", confirm=True)
        assert out["deleted"] and fakes["external"].keys() == ["tmpx/keep"]


# ── API ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def api(tiers):
    from fastapi.testclient import TestClient
    import merlina
    st.set_tiers(tiers)
    merlina._storage_active_pairs.clear()
    try:
        yield TestClient(merlina.app), merlina
    finally:
        st.set_tiers(None)
        merlina._storage_active_pairs.clear()


class TestAPI:
    def test_tiers_no_secrets(self, api, fakes):
        client, _ = api
        fakes["external"].put("a", b"123")
        r = client.get("/storage/tiers?usage=true")
        assert r.status_code == 200
        body = r.text
        for secret in ("INTERNAL_AK", "INTERNAL_SECRET", "EXTERNAL_AK", "EXTERNAL_SECRET"):
            assert secret not in body
        tiers = {t["name"]: t for t in r.json()["tiers"]}
        assert tiers["internal"]["privacy"] == "private"
        assert tiers["external"]["usage"]["bytes"] == 3

    def test_ls_du_and_unknown_tier(self, api, fakes):
        client, _ = api
        fakes["external"].put("runs/j/a", b"12")
        assert client.get("/storage/ls", params={"tier": "external", "prefix": "runs/"}).json()["dirs"][0]["prefix"] == "runs/j/"
        assert client.get("/storage/du", params={"tier": "external", "prefix": "runs"}).json()["bytes"] == 2
        assert client.get("/storage/ls", params={"tier": "bogus"}).status_code == 400

    def test_sync_dry_run_plan_and_guard(self, api, fakes):
        client, _ = api
        _dataset(fakes["internal"], "secret-ds", "r1", private=True)
        _dataset(fakes["internal"], "open-ds", "r1")
        r = client.post("/storage/sync", json={"src": "internal", "dst": "external", "prefix": "datasets/"})
        assert r.status_code == 200
        plan = r.json()["plan"]
        assert plan["counts"]["refused"] == 3 and plan["counts"]["to_copy"] == 3
        assert fakes["external"].keys() == []

    def test_no_allow_private_over_api(self, api):
        client, _ = api
        r = client.post("/storage/sync", json={"src": "internal", "dst": "external",
                                               "allow_private": True})
        assert r.status_code == 422

    def test_real_sync_job_and_progress(self, api, fakes):
        client, _ = api
        fakes["external"].put("runs/j/a", b"hello")
        r = client.post("/storage/sync", json={"src": "external", "dst": "internal",
                                               "prefix": "runs/", "dry_run": False})
        assert r.status_code == 200
        job_id = r.json()["id"]
        for _ in range(100):
            job = client.get(f"/storage/sync/{job_id}").json()
            if job["state"] != "running":
                break
            time.sleep(0.02)
        assert job["state"] == "completed", job
        assert job["result"]["counts"]["copied"] == 1
        assert job["progress"]["phase"] == "done"
        assert "_cancel" not in job
        assert fakes["internal"].keys() == ["runs/j/a"]
        assert client.get("/storage/sync/nope").status_code == 404

    def test_one_sync_per_pair(self, api):
        client, merlina = api
        merlina._storage_active_pairs[("external", "internal")] = "sync_busy"
        r = client.post("/storage/sync", json={"src": "external", "dst": "internal", "dry_run": False})
        assert r.status_code == 409
        # deletes on an involved tier are refused while it runs
        r = client.post("/storage/delete", json={"tier": "internal", "prefix": "x", "confirm": True})
        assert r.status_code == 409

    def test_delete_guards(self, api, fakes):
        client, _ = api
        fakes["external"].put("tmp/a", b"1")
        assert client.post("/storage/delete", json={"tier": "external", "prefix": "tmp/"}).status_code == 400
        assert client.post("/storage/delete", json={"tier": "external", "prefix": "", "confirm": True}).status_code == 400
        prev = client.post("/storage/delete", json={"tier": "external", "prefix": "tmp/", "dry_run": True}).json()
        assert prev["objects"] == 1 and fakes["external"].keys() == ["tmp/a"]
        r = client.post("/storage/delete", json={"tier": "external", "prefix": "tmp/", "confirm": True})
        assert r.status_code == 200 and r.json()["deleted"] and fakes["external"].keys() == []


# ── CLI ─────────────────────────────────────────────────────────────────────
class TestCLI:
    @pytest.fixture(autouse=True)
    def _registry(self, tiers):
        st.set_tiers(tiers)
        yield
        st.set_tiers(None)

    def test_smoke(self, fakes, capsys):
        import merlina_cli
        fakes["external"].put("runs/j/a", b"hello")
        assert merlina_cli.main(["storage", "tiers", "--usage"]) == 0
        assert merlina_cli.main(["storage", "ls", "external", "runs/"]) == 0
        assert merlina_cli.main(["storage", "du", "external"]) == 0
        assert merlina_cli.main(["storage", "sync", "external", "internal", "runs/", "--dry-run"]) == 0
        assert fakes["internal"].keys() == []
        assert merlina_cli.main(["storage", "sync", "external", "internal", "runs/"]) == 0
        assert fakes["internal"].keys() == ["runs/j/a"]
        assert merlina_cli.main(["storage", "rm", "internal", "runs/"]) == 1  # preview only
        assert fakes["internal"].keys() == ["runs/j/a"]
        assert merlina_cli.main(["storage", "rm", "internal", "runs/", "--yes"]) == 0
        assert fakes["internal"].keys() == []
        assert merlina_cli.main(["storage", "rm", "internal", "/", "--yes"]) == 2
        out = capsys.readouterr().out
        assert "private" in out and "shared" in out and "DRY RUN" in out and "5 B" in out

    def test_buckets_access_denied(self, fakes, capsys):
        import merlina_cli
        fakes["internal"].can_list_buckets = False
        assert merlina_cli.main(["storage", "buckets", "internal"]) == 1
        assert merlina_cli.main(["storage", "buckets", "internal", "--create", "new-bucket"]) == 1
        assert "access denied" in capsys.readouterr().err

    def test_refused_listed_and_allow_private(self, fakes, capsys):
        import merlina_cli
        _dataset(fakes["internal"], "secret-ds", "r1", private=True)
        assert merlina_cli.main(["storage", "sync", "internal", "external", "datasets/"]) == 0
        assert "refused: 3" in capsys.readouterr().out and fakes["external"].keys() == []
        assert merlina_cli.main(["storage", "sync", "internal", "external", "datasets/", "--allow-private"]) == 0
        assert len(fakes["external"].keys()) == 3

    def test_cli_stays_import_light(self):
        code = ("import sys, merlina_cli; merlina_cli.main(['storage', 'tiers']); "
                "print('HEAVY' if any(m in sys.modules for m in ('torch', 'datasets', 'transformers')) else 'LIGHT')")
        out = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).parent.parent),
                             capture_output=True, text=True, timeout=60)
        assert out.stdout.strip().endswith("LIGHT"), out.stdout + out.stderr
