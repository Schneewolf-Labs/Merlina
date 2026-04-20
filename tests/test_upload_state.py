"""
Tests for the per-model ``upload_state.json`` sidecar.

Verifies the read/write round-trip, failure-mode representation, and the
"local is newer than last upload" heuristic that drives the Export UI's
"✓ uploaded" staleness hint.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.upload_state import (
    SIDECAR_FILENAME,
    UploadEvent,
    last_successful_upload,
    local_is_newer_than_last_upload,
    read_upload_state,
    record_upload,
    summary_for_api,
)


def _mk_model_dir(tmp_path: Path, name: str = "mymodel") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"x" * 16)
    return d


def test_read_missing_state_returns_empty_scaffold(tmp_path):
    d = _mk_model_dir(tmp_path)
    state = read_upload_state(d)

    assert state["format_version"] == 1
    assert state["model_name"] == "mymodel"
    assert state["uploads"] == []


def test_record_upload_persists_event(tmp_path):
    d = _mk_model_dir(tmp_path)

    event = record_upload(
        d,
        repo_id="user/mymodel",
        repo_url="https://huggingface.co/user/mymodel",
        private=True,
        commit_message="first push",
        artifacts=["adapter", "readme"],
    )

    assert event.status == "success"
    assert event.repo_id == "user/mymodel"

    path = d / SIDECAR_FILENAME
    data = json.loads(path.read_text())
    assert len(data["uploads"]) == 1
    assert data["uploads"][0]["repo_id"] == "user/mymodel"
    assert data["uploads"][0]["artifacts"] == ["adapter", "readme"]


def test_multiple_uploads_are_ordered(tmp_path):
    d = _mk_model_dir(tmp_path)
    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="first", artifacts=[])
    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="second", artifacts=[])
    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="third", artifacts=[])

    state = read_upload_state(d)
    commits = [u["commit_message"] for u in state["uploads"]]
    assert commits == ["first", "second", "third"]


def test_last_successful_skips_failures(tmp_path):
    d = _mk_model_dir(tmp_path)
    record_upload(d, repo_id="u/m", repo_url="https://hf.co/u/m", private=True,
                  commit_message="ok1", artifacts=[])
    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="failure", artifacts=[], status="failed",
                  error="401 unauthorized")

    last = last_successful_upload(d)
    assert last is not None
    assert last.commit_message == "ok1"
    assert last.status == "success"


def test_last_successful_returns_none_when_only_failures(tmp_path):
    d = _mk_model_dir(tmp_path)
    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="nope", artifacts=[], status="failed",
                  error="boom")

    assert last_successful_upload(d) is None


def test_local_is_newer_when_never_uploaded(tmp_path):
    d = _mk_model_dir(tmp_path)
    assert local_is_newer_than_last_upload(d) is True


def test_local_is_not_newer_right_after_upload(tmp_path):
    d = _mk_model_dir(tmp_path)
    # Age the existing files so the upload timestamp is newer than them.
    old = time.time() - 60
    for p in d.rglob("*"):
        if p.is_file():
            import os
            os.utime(p, (old, old))

    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="fresh", artifacts=[])

    assert local_is_newer_than_last_upload(d) is False


def test_local_is_newer_after_file_modified(tmp_path):
    d = _mk_model_dir(tmp_path)
    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="old", artifacts=[])

    # Simulate the user re-training and overwriting the adapter.
    import os
    adapter = d / "adapter_model.safetensors"
    future = time.time() + 5
    os.utime(adapter, (future, future))

    assert local_is_newer_than_last_upload(d) is True


def test_sidecar_edits_itself_do_not_confuse_freshness(tmp_path):
    """
    Writing upload_state.json updates its own mtime; that write must not
    make "local is newer" immediately flip back to True.
    """
    d = _mk_model_dir(tmp_path)
    import os
    old = time.time() - 60
    for p in d.rglob("*"):
        if p.is_file():
            os.utime(p, (old, old))

    record_upload(d, repo_id="u/m", repo_url="", private=True,
                  commit_message="fresh", artifacts=[])
    assert local_is_newer_than_last_upload(d) is False


def test_summary_for_api_shape(tmp_path):
    d = _mk_model_dir(tmp_path)
    record_upload(d, repo_id="u/m", repo_url="https://hf.co/u/m", private=False,
                  commit_message="first", artifacts=["adapter"])

    summary = summary_for_api(d)
    assert summary["has_uploads"] is True
    assert summary["last_upload"]["repo_id"] == "u/m"
    assert summary["last_upload"]["private"] is False
    assert len(summary["history"]) == 1


def test_tolerates_corrupt_sidecar(tmp_path):
    d = _mk_model_dir(tmp_path)
    (d / SIDECAR_FILENAME).write_text("{this is not json")

    state = read_upload_state(d)
    assert state["uploads"] == []  # treated as empty, no exception


def test_upload_event_from_dict_preserves_fields():
    raw = {
        "timestamp": "2026-04-20T00:00:00+00:00",
        "repo_id": "u/m",
        "repo_url": "",
        "private": True,
        "commit_message": "msg",
        "artifacts": ["adapter"],
        "status": "success",
        "error": None,
    }
    event = UploadEvent.from_dict(raw)
    assert event.repo_id == "u/m"
    assert event.artifacts == ["adapter"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
