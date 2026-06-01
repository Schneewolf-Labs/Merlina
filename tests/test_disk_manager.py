#!/usr/bin/env python3
"""Tests for src/disk_manager.py — analysis + safe checkpoint cleanup.

Runnable standalone (`python tests/test_disk_manager.py`) or via pytest.
Builds a throwaway results/ tree under tmp so nothing real is touched.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.disk_manager import (
    analyze_artifacts,
    analyze_disk,
    analyze_hf_cache,
    clear_wandb_runs,
    delete_gguf_files,
    delete_hf_repos,
    delete_models,
    human_size,
    plan_cleanup,
    run_cleanup,
)


# ── Lightweight fakes mirroring huggingface_hub's scan_cache_dir objects ──────

class _FakeRev:
    def __init__(self, commit_hash):
        self.commit_hash = commit_hash


class _FakeRepo:
    def __init__(self, repo_id, repo_type, size, last_accessed, revisions, nb_files=3):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.size_on_disk = size
        self.last_accessed = last_accessed
        self.last_modified = last_accessed
        self.nb_files = nb_files
        self.revisions = {_FakeRev(h) for h in revisions}


class _FakeStrategy:
    def __init__(self, hashes):
        self.hashes = hashes
        self.expected_freed_size = 1000 * len(hashes)
        self.executed = False

    def execute(self):
        self.executed = True


class _FakeCacheInfo:
    def __init__(self, repos):
        self.repos = set(repos)
        self.size_on_disk = sum(r.size_on_disk for r in repos)
        self.last_strategy = None

    def delete_revisions(self, *hashes):
        self.last_strategy = _FakeStrategy(hashes)
        return self.last_strategy


_NOW = 1_000_000_000.0  # fixed clock so staleness is deterministic


def _fake_scan():
    # recent repo (accessed ~1 day ago) + stale repo (~200 days ago)
    return _FakeCacheInfo([
        _FakeRepo("org/recent-model", "model", 5000, _NOW - 86400, ["aaa"]),
        _FakeRepo("org/old-model", "model", 70000, _NOW - 200 * 86400, ["bbb", "ccc"]),
        _FakeRepo("org/old-dataset", "dataset", 30000, _NOW - 300 * 86400, ["ddd"]),
    ])


def _make_ckpt(job_dir: Path, name: str, size_bytes: int = 1024):
    ck = job_dir / name
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"\0" * size_bytes)


def _fixture(root: Path):
    """A results tree with: a completed multi-ckpt job, a single-ckpt job,
    a failed job, and an *active* job that must never be pruned."""
    results = root / "results"
    # completed: 3 checkpoints — newest is checkpoint-300
    done = results / "job_done"
    _make_ckpt(done, "checkpoint-100", 1000)
    _make_ckpt(done, "checkpoint-200", 1000)
    _make_ckpt(done, "checkpoint-300", 1000)
    # single checkpoint — nothing to prune
    _make_ckpt(results / "job_single", "checkpoint-50", 500)
    # failed job
    _make_ckpt(results / "job_failed", "checkpoint-10", 700)
    # active job with two checkpoints — OFF LIMITS
    active = results / "job_active"
    _make_ckpt(active, "checkpoint-1", 1000)
    _make_ckpt(active, "checkpoint-2", 1000)

    statuses = {
        "job_done": "completed",
        "job_single": "completed",
        "job_failed": "failed",
        "job_active": "training",
    }
    return results, statuses


def test_plan_keeps_latest_and_protects_active():
    with tempfile.TemporaryDirectory() as tmp:
        results, statuses = _fixture(Path(tmp))
        deletions, skipped = plan_cleanup(results, statuses, keep=1)

        paths = {Path(d["path"]).name for d in deletions}
        # The two older checkpoints of the completed job go; newest stays.
        assert paths == {"checkpoint-100", "checkpoint-200"}, paths
        # Active job and single-checkpoint job are skipped, never deleted.
        assert (results / "job_active" / "checkpoint-1").exists()
        skipped_ids = {s["job_id"] for s in skipped}
        assert "job_active" in skipped_ids
        assert "job_single" in skipped_ids
        print("✓ plan keeps latest, protects active + single-ckpt jobs")


def test_keep_two():
    with tempfile.TemporaryDirectory() as tmp:
        results, statuses = _fixture(Path(tmp))
        deletions, _ = plan_cleanup(results, statuses, keep=2)
        paths = {Path(d["path"]).name for d in deletions}
        assert paths == {"checkpoint-100"}, paths
        print("✓ keep=2 prunes only the oldest")


def test_purge_failed():
    with tempfile.TemporaryDirectory() as tmp:
        results, statuses = _fixture(Path(tmp))
        deletions, _ = plan_cleanup(results, statuses, keep=1, purge_failed=True)
        reasons = {Path(d["path"]).name: d["reason"] for d in deletions}
        assert reasons.get("job_failed", "").startswith("failed job"), reasons
        print("✓ purge_failed flags the whole failed-job dir")


def test_run_cleanup_dry_then_apply():
    with tempfile.TemporaryDirectory() as tmp:
        results, statuses = _fixture(Path(tmp))

        dry = run_cleanup(results, statuses, keep=1, apply=False)
        assert dry["count"] == 2
        assert dry["freed_bytes"] >= 2000
        # Dry run deletes nothing.
        assert (results / "job_done" / "checkpoint-100").exists()
        assert all(not d["deleted"] for d in dry["deletions"])

        applied = run_cleanup(results, statuses, keep=1, apply=True)
        assert applied["count"] == 2
        assert not (results / "job_done" / "checkpoint-100").exists()
        assert (results / "job_done" / "checkpoint-300").exists()  # newest kept
        assert (results / "job_active" / "checkpoint-1").exists()  # active untouched
        print("✓ dry run is non-destructive; apply deletes only planned items")


def test_analyze_disk_shape():
    with tempfile.TemporaryDirectory() as tmp:
        results, statuses = _fixture(Path(tmp))
        models = Path(tmp) / "models"
        (models / "my-model").mkdir(parents=True)
        (models / "my-model" / "w.bin").write_bytes(b"\0" * 2048)

        a = analyze_disk(results, models, statuses, keep=1)
        assert a["results"]["job_count"] == 4
        # Reclaimable counts only the completed job's two old ckpts.
        assert a["results"]["reclaimable_bytes"] == 2000, a["results"]["reclaimable_bytes"]
        # Active job reports reclaimable 0 even though it has 2 checkpoints.
        active = next(j for j in a["results"]["jobs"] if j["job_id"] == "job_active")
        assert active["active"] is True
        assert active["reclaimable_bytes"] == 0
        # Models breakdown present.
        assert a["models"]["count"] == 1
        assert a["models"]["total_bytes"] == 2048
        # Jobs sorted largest-first.
        sizes = [j["total_bytes"] for j in a["results"]["jobs"]]
        assert sizes == sorted(sizes, reverse=True)
        print("✓ analyze_disk shape, reclaimable math, active-job guard, sort order")


def test_human_size():
    assert human_size(0) == "0.0 B"
    assert human_size(1536) == "1.5 KB"
    assert human_size(int(2.5 * 1024**3)) == "2.5 GB"
    print("✓ human_size formatting")


def _models_fixture(root: Path):
    models = root / "models"
    for name, size in [("old-model", 4096), ("keep-model", 2048), ("loaded-model", 1024)]:
        d = models / name
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"\0" * size)
    return models


def test_delete_models_dry_run_and_apply():
    with tempfile.TemporaryDirectory() as tmp:
        models = _models_fixture(Path(tmp))

        dry = delete_models(models, ["old-model"], apply=False)
        assert dry["count"] == 1
        assert dry["freed_bytes"] >= 4096
        assert (models / "old-model").exists()  # dry run keeps it
        assert all(not d["deleted"] for d in dry["deleted"])

        applied = delete_models(models, ["old-model"], apply=True)
        assert applied["count"] == 1
        assert not (models / "old-model").exists()
        assert (models / "keep-model").exists()
        print("✓ delete_models: dry-run non-destructive, apply removes the dir")


def test_delete_models_protects_in_use():
    with tempfile.TemporaryDirectory() as tmp:
        models = _models_fixture(Path(tmp))
        res = delete_models(models, ["loaded-model", "old-model"], apply=True,
                            protected={"loaded-model"})
        deleted_names = {d["name"] for d in res["deleted"]}
        skipped_names = {s["name"] for s in res["skipped"]}
        assert deleted_names == {"old-model"}
        assert "loaded-model" in skipped_names
        assert (models / "loaded-model").exists()  # protected, untouched
        print("✓ delete_models refuses protected (in-use) models")


def test_delete_models_rejects_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        models = _models_fixture(Path(tmp))
        res = delete_models(models, ["../models", "a/b", ".."], apply=True)
        assert res["count"] == 0
        assert {s["reason"] for s in res["skipped"]} == {"invalid name"}
        print("✓ delete_models rejects path-traversal names")


def test_analyze_disk_models_have_dates():
    with tempfile.TemporaryDirectory() as tmp:
        results, statuses = _fixture(Path(tmp))
        models = _models_fixture(Path(tmp))
        a = analyze_disk(results, models, statuses, keep=1)
        assert a["models"]["count"] == 3
        assert all(m.get("modified_date") for m in a["models"]["items"])
        print("✓ analyze_disk model items carry modified_date")


def _gguf_fixture(root: Path):
    models = root / "models"
    for model, fname, size in [
        ("big-model", "big-model.q8_0.gguf", 8000),
        ("big-model", "big-model.q4_k_m.gguf", 4000),
        ("small-model", "small-model.q8_0.gguf", 2000),
    ]:
        gdir = models / model / "gguf"
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / fname).write_bytes(b"\0" * size)
    # A model dir with no gguf/ — must be ignored.
    (models / "no-gguf-model").mkdir(parents=True)
    return models


def test_analyze_artifacts_gguf():
    with tempfile.TemporaryDirectory() as tmp:
        models = _gguf_fixture(Path(tmp))
        a = analyze_artifacts(models, Path(tmp) / "wandb")
        g = a["gguf"]
        assert g["count"] == 3
        assert g["total_bytes"] == 14000
        # Largest-first; quant parsed from the filename convention.
        assert g["files"][0]["bytes"] == 8000
        assert g["files"][0]["quant"] == "q8_0"
        assert all(not f["loaded"] for f in g["files"])
        print("✓ analyze_artifacts: gguf listing, totals, quant parse, sort")


def test_analyze_artifacts_marks_loaded_gguf():
    with tempfile.TemporaryDirectory() as tmp:
        models = _gguf_fixture(Path(tmp))
        loaded = models / "big-model" / "gguf" / "big-model.q8_0.gguf"
        a = analyze_artifacts(models, Path(tmp) / "wandb", loaded_gguf=str(loaded))
        loaded_flags = {(f["model"], f["file"]): f["loaded"] for f in a["gguf"]["files"]}
        assert loaded_flags[("big-model", "big-model.q8_0.gguf")] is True
        assert loaded_flags[("small-model", "small-model.q8_0.gguf")] is False
        print("✓ analyze_artifacts flags the loaded GGUF")


def test_delete_gguf_protects_loaded_and_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        models = _gguf_fixture(Path(tmp))
        loaded = models / "big-model" / "gguf" / "big-model.q8_0.gguf"

        # Loaded file refused; traversal refused; the other deleted.
        res = delete_gguf_files(models, [
            {"model": "big-model", "file": "big-model.q8_0.gguf"},   # loaded
            {"model": "../etc", "file": "x.gguf"},                    # traversal
            {"model": "big-model", "file": "big-model.q4_k_m.gguf"},  # ok
        ], apply=True, loaded_gguf=str(loaded))
        deleted = {(d["model"], d["file"]) for d in res["deleted"]}
        reasons = {(s["model"], s["file"]): s["reason"] for s in res["skipped"]}
        assert deleted == {("big-model", "big-model.q4_k_m.gguf")}
        assert loaded.exists()  # loaded one untouched
        assert reasons[("big-model", "big-model.q8_0.gguf")] == "loaded for inference"
        assert reasons[("../etc", "x.gguf")] == "invalid name"
        print("✓ delete_gguf refuses loaded + traversal, deletes the rest")


def test_delete_gguf_tidies_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        models = _gguf_fixture(Path(tmp))
        # small-model has exactly one gguf — deleting it should remove gguf/.
        delete_gguf_files(models, [{"model": "small-model", "file": "small-model.q8_0.gguf"}], apply=True)
        assert not (models / "small-model" / "gguf").exists()
        print("✓ delete_gguf removes an emptied gguf/ dir")


def test_clear_wandb_keeps_active_run():
    with tempfile.TemporaryDirectory() as tmp:
        wandb = Path(tmp) / "wandb"
        wandb.mkdir()
        for name in ["run-A", "run-B", "run-C"]:
            d = wandb / name
            d.mkdir()
            (d / "log.txt").write_bytes(b"\0" * 100)
        # latest-run -> run-C (the active run).
        os.symlink("run-C", wandb / "latest-run")

        dry = clear_wandb_runs(wandb, apply=False)
        assert dry["count"] == 2  # A and B, not C
        assert (wandb / "run-A").exists()  # dry run keeps all

        applied = clear_wandb_runs(wandb, apply=True)
        assert applied["count"] == 2
        assert not (wandb / "run-A").exists()
        assert (wandb / "run-C").exists()  # active run preserved
        print("✓ clear_wandb deletes inactive runs, preserves the active run")


def test_analyze_artifacts_wandb_reclaimable():
    with tempfile.TemporaryDirectory() as tmp:
        wandb = Path(tmp) / "wandb"
        wandb.mkdir()
        for name in ["run-A", "run-B"]:
            (wandb / name).mkdir()
            (wandb / name / "f").write_bytes(b"\0" * 500)
        os.symlink("run-B", wandb / "latest-run")
        a = analyze_artifacts(Path(tmp) / "models", wandb)
        assert a["wandb"]["run_count"] == 2
        assert a["wandb"]["active_run"] == "run-B"
        assert a["wandb"]["reclaimable_bytes"] == 500  # only run-A
        print("✓ analyze_artifacts wandb: counts, active run, reclaimable")


def test_hf_cache_analysis():
    a = analyze_hf_cache(stale_days=90, now=_NOW, _scan=_fake_scan)
    assert a["available"] is True
    assert a["repo_count"] == 3
    assert a["total_bytes"] == 105000
    # Largest-first: old-model (70k) leads.
    assert a["repos"][0]["repo_id"] == "org/old-model"
    # Stale = old-model + old-dataset (70k + 30k); recent excluded.
    assert a["stale_reclaimable_bytes"] == 100000, a["stale_reclaimable_bytes"]
    recent = next(r for r in a["repos"] if r["repo_id"] == "org/recent-model")
    assert recent["stale"] is False
    assert recent["last_accessed_date"]  # ISO date populated
    print("✓ hf cache analysis: sizes, sort, staleness, dates")


def test_hf_cache_analysis_unavailable():
    def _boom():
        raise RuntimeError("no cache here")
    a = analyze_hf_cache(_scan=_boom)
    assert a["available"] is False
    assert a["repos"] == []
    assert "no cache" in a["error"]
    print("✓ hf cache missing → non-fatal available:false")


def test_hf_cache_delete_dry_run_and_apply():
    info_box = {}

    def _scan_capture():
        info = _fake_scan()
        info_box["info"] = info
        return info

    targets = [{"repo_id": "org/old-model", "repo_type": "model"}]

    dry = delete_hf_repos(targets, apply=False, _scan=_scan_capture)
    assert dry["count"] == 1
    assert dry["deleted"][0]["repo_id"] == "org/old-model"
    # old-model has 2 revisions → expected_freed_size = 1000*2.
    assert dry["freed_bytes"] == 2000
    assert info_box["info"].last_strategy.executed is False  # dry run executes nothing

    applied = delete_hf_repos(targets, apply=True, _scan=_scan_capture)
    assert applied["applied"] is True
    assert info_box["info"].last_strategy.executed is True
    print("✓ hf delete: dry-run plans without executing; apply executes")


def test_hf_cache_delete_matches_type():
    # Same-id collision across types must not over-match.
    a = delete_hf_repos([{"repo_id": "org/old-dataset", "repo_type": "model"}],
                        apply=False, _scan=_fake_scan)
    assert a["count"] == 0, "model filter must not match the dataset repo"
    b = delete_hf_repos([{"repo_id": "org/old-dataset", "repo_type": "dataset"}],
                        apply=False, _scan=_fake_scan)
    assert b["count"] == 1
    print("✓ hf delete matches on (repo_id, repo_type)")


if __name__ == "__main__":
    test_plan_keeps_latest_and_protects_active()
    test_keep_two()
    test_purge_failed()
    test_run_cleanup_dry_then_apply()
    test_analyze_disk_shape()
    test_human_size()
    test_delete_models_dry_run_and_apply()
    test_delete_models_protects_in_use()
    test_delete_models_rejects_traversal()
    test_analyze_disk_models_have_dates()
    test_analyze_artifacts_gguf()
    test_analyze_artifacts_marks_loaded_gguf()
    test_delete_gguf_protects_loaded_and_traversal()
    test_delete_gguf_tidies_empty_dir()
    test_clear_wandb_keeps_active_run()
    test_analyze_artifacts_wandb_reclaimable()
    test_hf_cache_analysis()
    test_hf_cache_analysis_unavailable()
    test_hf_cache_delete_dry_run_and_apply()
    test_hf_cache_delete_matches_type()
    print("\nAll disk_manager tests passed ✨")
