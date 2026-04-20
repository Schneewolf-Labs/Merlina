"""
Tests for the reference-counted MergeArtifact helper.

The class itself has no ML dependencies — these tests exercise the
refcount semantics in isolation so we can verify cleanup ordering and
the failure-mode contract (path=None when merge fails) without touching
torch / peft / transformers.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# MergeArtifact lives in its own module specifically so we can test it
# without importing training_runner (which pulls torch/peft/grimoire).
from src.merge_artifact import MergeArtifact as _MergeArtifact


def _scratch_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "weights.safetensors").write_bytes(b"x" * 16)
    return d


# ---------------------------------------------------------------------------
# Refcount semantics
# ---------------------------------------------------------------------------

def test_single_consumer_release_removes_directory(tmp_path):
    merge_dir = _scratch_dir(tmp_path, "merge1")
    artifact = _MergeArtifact(merge_dir, num_consumers=1)
    assert merge_dir.is_dir()

    artifact.release()
    assert not merge_dir.exists()


def test_directory_survives_until_last_consumer_releases(tmp_path):
    merge_dir = _scratch_dir(tmp_path, "merge_two")
    artifact = _MergeArtifact(merge_dir, num_consumers=2)

    artifact.release()
    assert merge_dir.is_dir(), "first release must not delete the dir"

    artifact.release()
    assert not merge_dir.exists(), "second release should clean up"


def test_extra_releases_are_noops(tmp_path):
    merge_dir = _scratch_dir(tmp_path, "extra")
    artifact = _MergeArtifact(merge_dir, num_consumers=1)

    artifact.release()
    # Extra releases past zero must not raise or recreate state.
    artifact.release()
    artifact.release()
    assert not merge_dir.exists()


def test_failed_merge_artifact_carries_error_and_no_path(tmp_path):
    artifact = _MergeArtifact(None, num_consumers=2, error="OOM during merge")
    assert artifact.path is None
    assert artifact.error == "OOM during merge"

    # Releases on a path-less artifact must be safe and do nothing.
    artifact.release()
    artifact.release()


def test_release_is_thread_safe(tmp_path):
    """Concurrent releases must collectively delete the dir exactly once."""
    merge_dir = _scratch_dir(tmp_path, "concurrent")
    n = 16
    artifact = _MergeArtifact(merge_dir, num_consumers=n)

    barrier = threading.Barrier(n)

    def worker():
        barrier.wait()
        artifact.release()

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not merge_dir.exists()


def test_zero_consumer_release_is_safe(tmp_path):
    merge_dir = _scratch_dir(tmp_path, "zero")
    artifact = _MergeArtifact(merge_dir, num_consumers=0)

    # No-one ever called release; cleanup never triggered. That's fine —
    # the orchestrator won't construct a 0-consumer artifact in practice.
    artifact.release()
    assert merge_dir.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
