"""
Tests for the GGUF export pipeline and the llama-server helper.

These tests don't touch real llama.cpp binaries — they stub out the
subprocess invocation and feed fake success/failure return codes so we
can verify orchestration, manifest shape, quant normalization, and
fallback behavior deterministically.
"""

import json
import stat
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import gguf_exporter
from src.gguf_exporter import (
    DEFAULT_QUANT_TYPES,
    SUPPORTED_QUANT_TYPES,
    GGUFExportError,
    ProgressEvent,
    export_gguf_artifacts,
    normalize_quant_types,
    read_manifest,
    write_manifest,
)
from src.llama_cpp_resolver import LlamaCppResolution
from src.llama_server import resolve_gguf_for_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source_dir(tmp_path: Path) -> Path:
    """Create a minimal HF-shaped source directory."""
    src = tmp_path / "merged_model"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"")
    return src


def _fake_resolution(tmp_path: Path) -> LlamaCppResolution:
    """Resolution pointing at stub binaries/script that we'll monkeypatch over."""
    bin_dir = tmp_path / "llamabin"
    bin_dir.mkdir()
    quantize = bin_dir / "llama-quantize"
    quantize.write_text("#!/bin/sh\nexit 0\n")
    quantize.chmod(quantize.stat().st_mode | stat.S_IXUSR)

    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# fake\n")

    return LlamaCppResolution(
        available=True,
        source="explicit",
        root=tmp_path,
        bin_dir=bin_dir,
        convert_script=script,
        binaries={"llama-quantize": quantize},
    )


class _FakeSubprocess:
    """Stand-in for subprocess invocations the exporter makes."""

    def __init__(self, *, fail_for: set[str] | None = None):
        # Quant types (or "F16" for convert) that should fail.
        self.fail_for = fail_for or set()
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        # args = [python, convert.py, source, --outfile, out, --outtype, f16]
        # or    [llama-quantize, fp16.gguf, out.gguf, QUANT]
        op_name = args[-1] if args[0] != sys.executable else "F16"

        # Identify the target output path. For convert, it's the arg
        # after --outfile. For quantize, it's args[2].
        if args[0] == sys.executable:
            out_index = args.index("--outfile") + 1
            target = Path(args[out_index])
        else:
            target = Path(args[2])

        should_fail = op_name in self.fail_for

        class _FakeProc:
            def __init__(self, lines, returncode):
                self._lines = lines
                self.returncode = returncode
                self.stdout = iter(lines)

            def wait(self):
                return self.returncode

        if should_fail:
            return _FakeProc(
                [f"[fake] simulated failure for {op_name}\n"],
                returncode=1,
            )

        # Create the fake output file so the exporter's existence checks pass.
        target.write_bytes(b"gguf-bytes")
        return _FakeProc([f"[fake] produced {target.name}\n"], returncode=0)


# ---------------------------------------------------------------------------
# normalize_quant_types
# ---------------------------------------------------------------------------

def test_normalize_quant_types_accepts_known_values():
    result = normalize_quant_types(["q4_k_m", "F16", "Q8_0"])
    assert result == ["Q4_K_M", "F16", "Q8_0"]


def test_normalize_quant_types_deduplicates_preserving_order():
    result = normalize_quant_types(["Q4_K_M", "q4_k_m", "Q8_0"])
    assert result == ["Q4_K_M", "Q8_0"]


def test_normalize_quant_types_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported GGUF quant types"):
        normalize_quant_types(["Q4_K_M", "Q42_FUNKY"])


def test_normalize_quant_types_ignores_empty_strings():
    assert normalize_quant_types(["", "Q4_K_M", None or ""]) == ["Q4_K_M"]


# ---------------------------------------------------------------------------
# Exporter orchestration
# ---------------------------------------------------------------------------

def test_export_happy_path_writes_manifest_and_artifacts(tmp_path, monkeypatch):
    source = _make_source_dir(tmp_path)
    output_dir = tmp_path / "out"
    resolution = _fake_resolution(tmp_path)

    fake = _FakeSubprocess()
    monkeypatch.setattr(gguf_exporter.subprocess, "Popen", fake)

    events: list[ProgressEvent] = []
    result = export_gguf_artifacts(
        source,
        output_dir,
        output_name="mini",
        quant_types=["Q4_K_M", "Q8_0"],
        resolution=resolution,
        callback=events.append,
    )

    assert result.succeeded
    # 2 quants produced + fp16 was intermediate (deleted).
    assert [a.quant_type for a in result.artifacts] == ["Q4_K_M", "Q8_0"]
    for artifact in result.artifacts:
        assert artifact.path.is_file()

    # Manifest written and readable.
    assert result.manifest_path is not None
    manifest = read_manifest(output_dir)
    assert manifest is not None
    assert manifest["source_model"] == "mini"
    assert [a["quant_type"] for a in manifest["artifacts"]] == ["Q4_K_M", "Q8_0"]

    # Events include a complete stage at the end.
    stages = [e.stage for e in events]
    assert "converting" in stages
    assert "quantizing" in stages
    assert stages[-1] == "complete"

    # Intermediate fp16 is cleaned up when F16 wasn't requested and keep_fp16=False.
    fp16_path = output_dir / "mini.f16.gguf"
    assert not fp16_path.exists()


def test_export_keeps_fp16_when_requested_in_quant_list(tmp_path, monkeypatch):
    source = _make_source_dir(tmp_path)
    output_dir = tmp_path / "out"
    resolution = _fake_resolution(tmp_path)

    monkeypatch.setattr(gguf_exporter.subprocess, "Popen", _FakeSubprocess())

    result = export_gguf_artifacts(
        source,
        output_dir,
        output_name="mini",
        quant_types=["F16", "Q4_K_M"],
        resolution=resolution,
    )

    fp16_path = output_dir / "mini.f16.gguf"
    assert fp16_path.is_file(), "F16 must remain on disk when listed explicitly"
    quants = [a.quant_type for a in result.artifacts]
    assert quants == ["F16", "Q4_K_M"]


def test_export_keeps_fp16_when_keep_flag_set(tmp_path, monkeypatch):
    source = _make_source_dir(tmp_path)
    output_dir = tmp_path / "out"
    resolution = _fake_resolution(tmp_path)

    monkeypatch.setattr(gguf_exporter.subprocess, "Popen", _FakeSubprocess())

    export_gguf_artifacts(
        source,
        output_dir,
        output_name="mini",
        quant_types=["Q4_K_M"],
        resolution=resolution,
        keep_fp16=True,
    )
    assert (output_dir / "mini.f16.gguf").is_file()


def test_export_records_failed_quant_without_raising(tmp_path, monkeypatch):
    source = _make_source_dir(tmp_path)
    output_dir = tmp_path / "out"
    resolution = _fake_resolution(tmp_path)

    monkeypatch.setattr(
        gguf_exporter.subprocess, "Popen",
        _FakeSubprocess(fail_for={"Q8_0"}),
    )

    result = export_gguf_artifacts(
        source,
        output_dir,
        output_name="mini",
        quant_types=["Q4_K_M", "Q8_0"],
        resolution=resolution,
    )

    assert [a.quant_type for a in result.artifacts] == ["Q4_K_M"]
    assert result.failed_quants == ["Q8_0"]
    assert not result.succeeded
    # Manifest still written so partial results are discoverable.
    assert read_manifest(output_dir)["artifacts"][0]["quant_type"] == "Q4_K_M"


def test_export_raises_when_convert_fails(tmp_path, monkeypatch):
    source = _make_source_dir(tmp_path)
    output_dir = tmp_path / "out"
    resolution = _fake_resolution(tmp_path)

    monkeypatch.setattr(
        gguf_exporter.subprocess, "Popen",
        _FakeSubprocess(fail_for={"F16"}),
    )

    with pytest.raises(GGUFExportError, match="converting failed"):
        export_gguf_artifacts(
            source,
            output_dir,
            output_name="mini",
            quant_types=["Q4_K_M"],
            resolution=resolution,
        )


def test_export_rejects_missing_source_dir(tmp_path):
    output_dir = tmp_path / "out"
    with pytest.raises(GGUFExportError, match="Source directory"):
        export_gguf_artifacts(
            tmp_path / "does_not_exist",
            output_dir,
            output_name="mini",
            quant_types=["Q4_K_M"],
            resolution=_fake_resolution(tmp_path),
        )


def test_export_rejects_source_without_config_json(tmp_path):
    source = tmp_path / "incomplete"
    source.mkdir()
    with pytest.raises(GGUFExportError, match="config.json"):
        export_gguf_artifacts(
            source,
            tmp_path / "out",
            output_name="mini",
            quant_types=["Q4_K_M"],
            resolution=_fake_resolution(tmp_path),
        )


def test_export_honours_cancel_event_between_quants(tmp_path, monkeypatch):
    source = _make_source_dir(tmp_path)
    output_dir = tmp_path / "out"
    resolution = _fake_resolution(tmp_path)

    monkeypatch.setattr(gguf_exporter.subprocess, "Popen", _FakeSubprocess())

    cancel = threading.Event()
    # Cancel as soon as the first quant finishes (before the second starts).
    events: list[ProgressEvent] = []

    def on_event(event: ProgressEvent):
        events.append(event)
        if (
            event.stage == "quantizing"
            and event.quant_type == "Q4_K_M"
            and event.current is not None  # the post-produce event
        ):
            cancel.set()

    result = export_gguf_artifacts(
        source,
        output_dir,
        output_name="mini",
        quant_types=["Q4_K_M", "Q8_0"],
        resolution=resolution,
        callback=on_event,
        cancel_event=cancel,
    )

    assert [a.quant_type for a in result.artifacts] == ["Q4_K_M"]
    assert any("cancelled" in w for w in result.warnings)


def test_export_fails_fast_if_resolver_unavailable(tmp_path):
    source = _make_source_dir(tmp_path)
    output_dir = tmp_path / "out"
    unavailable = LlamaCppResolution(
        available=False,
        warnings=["llama.cpp not found. Set LLAMA_CPP_DIR..."],
    )
    with pytest.raises(GGUFExportError, match="llama.cpp"):
        export_gguf_artifacts(
            source,
            output_dir,
            output_name="mini",
            quant_types=["Q4_K_M"],
            resolution=unavailable,
        )


def test_manifest_roundtrips(tmp_path):
    from src.gguf_exporter import GGUFArtifact

    out = tmp_path / "gguf"
    out.mkdir()
    fake = out / "x.q4_k_m.gguf"
    fake.write_bytes(b"abc")
    artifact = GGUFArtifact("Q4_K_M", fake, fake.stat().st_size)

    path = write_manifest(out, [artifact], source_model="x", base_model="base/y")
    assert path.is_file()
    loaded = read_manifest(out)
    assert loaded["base_model"] == "base/y"
    assert loaded["artifacts"][0]["filename"] == "x.q4_k_m.gguf"


def test_default_quant_types_are_supported():
    # Guard against typos in the default list drifting from the supported set.
    for name in DEFAULT_QUANT_TYPES:
        assert name in SUPPORTED_QUANT_TYPES


# ---------------------------------------------------------------------------
# llama_server.resolve_gguf_for_model
# ---------------------------------------------------------------------------

def test_resolve_gguf_prefers_requested_quant(tmp_path):
    models_dir = tmp_path / "models"
    gguf_dir = models_dir / "mymodel" / "gguf"
    gguf_dir.mkdir(parents=True)
    q4 = gguf_dir / "mymodel.q4_k_m.gguf"
    q8 = gguf_dir / "mymodel.q8_0.gguf"
    q4.write_bytes(b"x")
    q8.write_bytes(b"x")

    manifest = {
        "artifacts": [
            {"quant_type": "Q4_K_M", "filename": q4.name, "path": str(q4)},
            {"quant_type": "Q8_0", "filename": q8.name, "path": str(q8)},
        ]
    }
    (gguf_dir / "manifest.json").write_text(json.dumps(manifest))

    chosen = resolve_gguf_for_model(models_dir, "mymodel", preferred_quant="Q8_0")
    assert chosen == q8


def test_resolve_gguf_falls_back_to_first_entry(tmp_path):
    models_dir = tmp_path / "models"
    gguf_dir = models_dir / "m2" / "gguf"
    gguf_dir.mkdir(parents=True)
    q4 = gguf_dir / "m2.q4_k_m.gguf"
    q4.write_bytes(b"x")
    manifest = {"artifacts": [{"quant_type": "Q4_K_M", "filename": q4.name, "path": str(q4)}]}
    (gguf_dir / "manifest.json").write_text(json.dumps(manifest))

    chosen = resolve_gguf_for_model(models_dir, "m2", preferred_quant="Q9_NOPE")
    # Requested quant missing → first entry wins.
    assert chosen == q4


def test_resolve_gguf_returns_none_when_no_manifest(tmp_path):
    models_dir = tmp_path / "models"
    (models_dir / "m3").mkdir(parents=True)
    assert resolve_gguf_for_model(models_dir, "m3") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
