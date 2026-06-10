#!/usr/bin/env python3
"""Tests for src/local_models.py — offline model discovery (issue #80).

Runnable standalone (`python tests/test_local_models.py`) or via pytest.
Uses fake HF cache scans and throwaway tmp directories so no real cache
or network is touched.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.local_models import (
    is_model_cached,
    list_hf_cached_models,
    list_local_models,
    list_models_dir_models,
    offline_mode_active,
)


# ── Lightweight fakes mirroring huggingface_hub's scan_cache_dir objects ──────

class _FakeRev:
    def __init__(self, commit_hash):
        self.commit_hash = commit_hash


class _FakeRepo:
    def __init__(self, repo_id, repo_type, size=1000, revisions=("aaa",)):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.size_on_disk = size
        self.last_modified = 1_000_000_000.0
        self.revisions = {_FakeRev(h) for h in revisions}


class _FakeCacheInfo:
    def __init__(self, repos):
        self.repos = set(repos)


def _fake_scan():
    return _FakeCacheInfo([
        _FakeRepo("org/model-b", "model", size=5000),
        _FakeRepo("org/model-a", "model", size=3000),
        _FakeRepo("org/some-dataset", "dataset"),
        _FakeRepo("org/incomplete-model", "model", revisions=()),
    ])


def _failing_scan():
    raise RuntimeError("no cache here")


def _make_models_dir(root: Path) -> Path:
    """models/ with one full model, one LoRA-only adapter, and a stray file."""
    models = root / "models"
    full = models / "my-merged-model"
    full.mkdir(parents=True)
    (full / "config.json").write_text("{}")
    (full / "model.safetensors").write_bytes(b"\0" * 16)
    adapter = models / "my-lora-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (models / "stray.txt").write_text("not a model")
    return models


# ── HF cache listing ──────────────────────────────────────────────────────────

def test_hf_cache_lists_models_only():
    models = list_hf_cached_models(_scan=_fake_scan)
    ids = [m["model_id"] for m in models]
    # Sorted, model-type only, no snapshot-less repos
    assert ids == ["org/model-a", "org/model-b"], ids
    assert all(m["source"] == "hf_cache" for m in models)
    assert models[0]["size_bytes"] == 3000
    print("✓ HF cache listing filters to complete model repos")


def test_hf_cache_unavailable_is_empty_not_fatal():
    assert list_hf_cached_models(_scan=_failing_scan) == []
    print("✓ missing HF cache degrades to empty list")


# ── models/ directory listing ─────────────────────────────────────────────────

def test_models_dir_lists_full_models_only():
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = _make_models_dir(Path(tmp))
        found = list_models_dir_models(models_dir)
        assert len(found) == 1, found
        assert found[0]["name"] == "my-merged-model"
        assert found[0]["source"] == "models_dir"
        # model_id is the path that goes straight into base_model
        assert found[0]["model_id"].endswith("my-merged-model")
    print("✓ models dir lists full models, skips LoRA-only adapters")


def test_models_dir_missing_is_empty():
    assert list_models_dir_models(Path("/nonexistent/never/here")) == []
    print("✓ missing models dir degrades to empty list")


# ── Combined endpoint payload ─────────────────────────────────────────────────

def test_list_local_models_combines_sources():
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = _make_models_dir(Path(tmp))
        result = list_local_models(models_dir, _scan=_fake_scan)
        assert result["count"] == 3
        sources = {m["source"] for m in result["models"]}
        assert sources == {"hf_cache", "models_dir"}
        assert isinstance(result["offline_mode"], bool)
    print("✓ list_local_models combines HF cache and models dir")


# ── Cache presence check (used by offline preflight) ─────────────────────────

def test_is_model_cached_checks_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        snap = cache / "models--org--cached-model" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        assert is_model_cached("org/cached-model", cache_dir=cache)
        assert not is_model_cached("org/never-downloaded", cache_dir=cache)
        # Repo dir with empty snapshots is not loadable offline
        (cache / "models--org--empty" / "snapshots").mkdir(parents=True)
        assert not is_model_cached("org/empty", cache_dir=cache)
    print("✓ is_model_cached requires a non-empty snapshots dir")


# ── Offline mode detection ────────────────────────────────────────────────────

def test_offline_mode_env_vars():
    old = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        assert offline_mode_active()
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "true"
        assert offline_mode_active()
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        # With both env vars falsy, falls through to settings.offline_mode
        # (default False) — must not raise even if config isn't importable.
        offline_mode_active()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("✓ offline mode respects HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE")


# ── Preflight integration: offline model-access check ────────────────────────

def test_preflight_offline_model_access():
    from types import SimpleNamespace
    from src.preflight_checks import PreflightValidator
    import src.local_models as lm

    old_env = os.environ.get("HF_HUB_OFFLINE")
    orig_cached = lm.is_model_cached
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        lm.is_model_cached = lambda mid: mid == "org/cached-model"

        v = PreflightValidator()
        res = v._check_model_access(
            SimpleNamespace(base_model="org/cached-model", hf_token=None))
        assert res["offline_mode"] and res["cached"]
        assert not v.errors, v.errors

        v = PreflightValidator()
        res = v._check_model_access(
            SimpleNamespace(base_model="org/never-downloaded", hf_token=None))
        assert not res["cached"]
        assert any("Offline mode" in e for e in v.errors), v.errors
    finally:
        lm.is_model_cached = orig_cached
        if old_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_env
    print("✓ preflight blocks uncached HF models in offline mode")


if __name__ == "__main__":
    test_hf_cache_lists_models_only()
    test_hf_cache_unavailable_is_empty_not_fatal()
    test_models_dir_lists_full_models_only()
    test_models_dir_missing_is_empty()
    test_list_local_models_combines_sources()
    test_is_model_cached_checks_snapshots()
    test_offline_mode_env_vars()
    test_preflight_offline_model_access()
    print("\nAll local_models tests passed ✨")
