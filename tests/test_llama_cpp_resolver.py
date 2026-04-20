"""
Tests for the llama.cpp binary resolver.

Exercises each tier of the resolution precedence in isolation using a
synthetic directory tree, so the tests don't need a real llama.cpp build.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llama_cpp_resolver import (
    ALL_BINARIES,
    CONVERT_SCRIPT_NAME,
    REQUIRED_BINARIES,
    resolve_llama_cpp,
)


def _touch_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_fake_repo(root: Path, *, with_convert: bool = True) -> Path:
    """Create a llama.cpp-shaped directory with fake binaries. Returns bin dir."""
    bin_dir = root / "build" / "bin"
    for name in ALL_BINARIES:
        _touch_executable(bin_dir / name)
    if with_convert:
        (root / CONVERT_SCRIPT_NAME).write_text("# fake convert script\n")
    return bin_dir


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip any inherited llama.cpp env vars so tests stay deterministic."""
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.delenv("LLAMA_CPP_BIN_DIR", raising=False)
    # Prevent the fallback PATH scan from finding anything by default.
    monkeypatch.setenv("PATH", "")
    yield


def test_returns_unavailable_when_nothing_is_configured(tmp_path):
    result = resolve_llama_cpp(cwd=tmp_path)
    assert result.available is False
    assert result.source is None
    assert result.binaries == {}
    assert result.warnings, "missing llama.cpp should produce a user-facing hint"


def test_explicit_repo_root_takes_precedence(tmp_path, monkeypatch):
    repo = tmp_path / "explicit_repo"
    bin_dir = _make_fake_repo(repo)

    # Also set env and vendor to ensure explicit wins.
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "env_repo"))
    _make_fake_repo(tmp_path / "env_repo")
    _make_fake_repo(tmp_path / "vendor" / "llama.cpp")

    result = resolve_llama_cpp(str(repo), cwd=tmp_path)

    assert result.available is True
    assert result.source == "explicit"
    assert result.bin_dir == bin_dir.resolve()
    assert result.root == repo.resolve()
    assert result.convert_script == (repo / CONVERT_SCRIPT_NAME).resolve()
    for name in REQUIRED_BINARIES:
        assert name in result.binaries


def test_explicit_bin_only_path_still_resolves(tmp_path):
    bin_dir = tmp_path / "loose_bin"
    for name in ALL_BINARIES:
        _touch_executable(bin_dir / name)

    result = resolve_llama_cpp(str(bin_dir), cwd=tmp_path)

    assert result.available is True
    assert result.source == "explicit"
    assert result.bin_dir == bin_dir.resolve()
    assert result.convert_script is None
    # A bin-only resolution should warn that the convert script is absent.
    assert any("convert_hf_to_gguf.py" in w for w in result.warnings)


def test_env_dir_beats_env_bin_and_vendor(tmp_path, monkeypatch):
    env_repo = tmp_path / "env_repo"
    env_bin = tmp_path / "env_bin"
    _make_fake_repo(env_repo)
    for name in ALL_BINARIES:
        _touch_executable(env_bin / name)
    _make_fake_repo(tmp_path / "vendor" / "llama.cpp")

    monkeypatch.setenv("LLAMA_CPP_DIR", str(env_repo))
    monkeypatch.setenv("LLAMA_CPP_BIN_DIR", str(env_bin))

    result = resolve_llama_cpp(cwd=tmp_path)

    assert result.available is True
    assert result.source == "env_root"
    assert result.root == env_repo.resolve()


def test_env_bin_used_when_only_bin_is_configured(tmp_path, monkeypatch):
    env_bin = tmp_path / "env_bin"
    for name in ALL_BINARIES:
        _touch_executable(env_bin / name)

    monkeypatch.setenv("LLAMA_CPP_BIN_DIR", str(env_bin))

    result = resolve_llama_cpp(cwd=tmp_path)

    assert result.available is True
    assert result.source == "env_bin"
    assert result.bin_dir == env_bin.resolve()


def test_path_fallback_discovers_binaries(tmp_path, monkeypatch):
    path_dir = tmp_path / "system_bin"
    for name in ALL_BINARIES:
        _touch_executable(path_dir / name)
    monkeypatch.setenv("PATH", str(path_dir))

    result = resolve_llama_cpp(cwd=tmp_path)

    assert result.available is True
    assert result.source == "path"
    assert result.bin_dir == path_dir.resolve()


def test_vendor_checkout_is_last_resort(tmp_path):
    vendor_repo = tmp_path / "vendor" / "llama.cpp"
    _make_fake_repo(vendor_repo)

    result = resolve_llama_cpp(cwd=tmp_path)

    assert result.available is True
    assert result.source == "vendor"
    assert result.root == vendor_repo.resolve()


def test_invalid_explicit_path_reports_error(tmp_path):
    result = resolve_llama_cpp(str(tmp_path / "does_not_exist"), cwd=tmp_path)
    assert result.available is False
    assert result.errors, "missing explicit dir should be an error, not a silent miss"


def test_missing_required_binary_flags_unavailable(tmp_path):
    repo = tmp_path / "partial_repo"
    # Only build llama-server, skip llama-quantize (which is required).
    _touch_executable(repo / "build" / "bin" / "llama-server")

    result = resolve_llama_cpp(str(repo), cwd=tmp_path)

    # Partial tree has binaries but not the required one.
    assert result.available is False
    # Either discovery failed outright, or a warning called out the missing binary.
    assert result.errors or any("missing required" in w for w in result.warnings)


def test_to_dict_is_json_safe(tmp_path):
    repo = tmp_path / "repo"
    _make_fake_repo(repo)
    result = resolve_llama_cpp(str(repo), cwd=tmp_path)

    import json
    payload = json.dumps(result.to_dict())
    parsed = json.loads(payload)
    assert parsed["available"] is True
    assert parsed["source"] == "explicit"
    assert "llama-quantize" in parsed["binaries"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
