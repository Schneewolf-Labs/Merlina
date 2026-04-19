"""
GGUF Export Pipeline

Converts a HuggingFace-format checkpoint into one or more GGUF files by
shelling out to llama.cpp's ``convert_hf_to_gguf.py`` and ``llama-quantize``.

The module is designed as a pure pipeline:

    adapter_dir ─► (optional LoRA merge) ─► fp16 GGUF ─► quantized GGUF(s)

It never imports Merlina-specific singletons (job manager, websocket),
so unit tests can exercise it without the full stack. Training-runner
wiring passes a ``progress_callback`` to stream updates back.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .llama_cpp_resolver import LlamaCppResolution, resolve_llama_cpp

logger = logging.getLogger(__name__)


# Quant types we expose in the UI. ``F16`` is really the raw fp16 GGUF
# (produced directly by convert_hf_to_gguf.py); everything else is a
# llama-quantize target. The list order also drives frontend presentation.
SUPPORTED_QUANT_TYPES: tuple[str, ...] = (
    "F16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q4_K_M",
    "Q4_K_S",
    "Q3_K_M",
    "Q2_K",
)

DEFAULT_QUANT_TYPES: tuple[str, ...] = ("Q4_K_M",)

MANIFEST_FILENAME = "manifest.json"


class GGUFExportError(RuntimeError):
    """Raised when an export stage fails hard. Callers treat this as advisory."""


@dataclass
class ProgressEvent:
    """Structured update emitted to ``progress_callback`` during export."""

    stage: str        # "merging" | "converting" | "quantizing" | "complete" | "error"
    message: str
    quant_type: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None
    artifact_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class GGUFArtifact:
    """A single GGUF file produced by the pipeline."""

    quant_type: str
    path: Path
    size_bytes: int

    def to_dict(self) -> dict:
        return {
            "quant_type": self.quant_type,
            "path": str(self.path),
            "filename": self.path.name,
            "size_bytes": self.size_bytes,
        }


@dataclass
class GGUFExportResult:
    """Summary returned from :func:`export_gguf_artifacts`."""

    output_dir: Path
    artifacts: List[GGUFArtifact] = field(default_factory=list)
    failed_quants: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    manifest_path: Optional[Path] = None

    @property
    def succeeded(self) -> bool:
        return bool(self.artifacts) and not self.failed_quants

    def to_dict(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "failed_quants": list(self.failed_quants),
            "warnings": list(self.warnings),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "succeeded": self.succeeded,
        }


def _emit(callback: Optional[ProgressCallback], event: ProgressEvent) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # pragma: no cover — never let a faulty callback break the pipeline
        logger.exception("GGUF progress callback raised; continuing export")


def normalize_quant_types(requested: Iterable[str]) -> List[str]:
    """Deduplicate + validate quant type names, preserving user order."""
    seen: set[str] = set()
    out: List[str] = []
    invalid: List[str] = []
    for raw in requested:
        if not raw:
            continue
        name = raw.strip().upper()
        if name not in SUPPORTED_QUANT_TYPES:
            invalid.append(raw)
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    if invalid:
        raise ValueError(
            f"Unsupported GGUF quant types: {invalid}. "
            f"Supported: {list(SUPPORTED_QUANT_TYPES)}"
        )
    return out


def _run_subprocess(
    args: List[str],
    *,
    stage: str,
    quant_type: Optional[str],
    callback: Optional[ProgressCallback],
    cwd: Optional[Path] = None,
) -> None:
    """Run a subprocess and forward stderr-ish lines via progress events."""
    logger.info("GGUF %s: %s", stage, " ".join(args))
    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise GGUFExportError(f"Binary not found: {args[0]} ({exc})") from exc

    assert process.stdout is not None
    last_line = ""
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        last_line = line
        # Forward every ~512 bytes worth of output as a heartbeat. Parsing
        # llama-quantize's exact progress format isn't worth the churn — the
        # UI only needs "still working".
        _emit(
            callback,
            ProgressEvent(
                stage=stage,
                quant_type=quant_type,
                message=line[-400:],
            ),
        )

    returncode = process.wait()
    if returncode != 0:
        raise GGUFExportError(
            f"{stage} failed (exit {returncode}) for {quant_type or 'fp16'}. "
            f"Last output: {last_line!r}"
        )


def convert_to_fp16_gguf(
    source_dir: Path,
    output_path: Path,
    *,
    resolution: LlamaCppResolution,
    callback: Optional[ProgressCallback] = None,
) -> Path:
    """Run ``convert_hf_to_gguf.py`` to produce an fp16 GGUF."""
    if not resolution.convert_script or not resolution.convert_script.is_file():
        raise GGUFExportError(
            "convert_hf_to_gguf.py not available — set LLAMA_CPP_DIR to a full "
            "llama.cpp checkout to enable GGUF export."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        str(resolution.convert_script),
        str(source_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        "f16",
    ]
    _emit(
        callback,
        ProgressEvent(
            stage="converting",
            message=f"Converting {source_dir.name} to fp16 GGUF",
            quant_type="F16",
        ),
    )
    _run_subprocess(args, stage="converting", quant_type="F16", callback=callback)
    if not output_path.is_file():
        raise GGUFExportError(
            f"Conversion reported success but {output_path} was not created."
        )
    return output_path


def quantize_gguf(
    fp16_path: Path,
    quant_type: str,
    output_path: Path,
    *,
    resolution: LlamaCppResolution,
    callback: Optional[ProgressCallback] = None,
) -> Path:
    """Run ``llama-quantize`` to produce a single quantized GGUF."""
    quantize_bin = resolution.binary("llama-quantize")
    if quantize_bin is None:
        raise GGUFExportError("llama-quantize binary not resolved")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [str(quantize_bin), str(fp16_path), str(output_path), quant_type]
    _emit(
        callback,
        ProgressEvent(
            stage="quantizing",
            quant_type=quant_type,
            message=f"Quantizing → {quant_type}",
        ),
    )
    _run_subprocess(args, stage="quantizing", quant_type=quant_type, callback=callback)
    if not output_path.is_file():
        raise GGUFExportError(
            f"Quantize reported success but {output_path} was not created."
        )
    return output_path


def write_manifest(
    output_dir: Path,
    artifacts: List[GGUFArtifact],
    *,
    source_model: str,
    base_model: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Write a ``manifest.json`` describing the produced GGUF files."""
    payload = {
        "format_version": 1,
        "source_model": source_model,
        "base_model": base_model,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": [a.to_dict() for a in artifacts],
    }
    if extra:
        payload.update(extra)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return manifest_path


def read_manifest(output_dir: Path) -> Optional[dict]:
    """Read a previously written GGUF manifest, if any."""
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read GGUF manifest at %s: %s", manifest_path, exc)
        return None


def _filename_for(output_name: str, quant_type: str) -> str:
    return f"{output_name}.{quant_type.lower()}.gguf"


def export_gguf_artifacts(
    source_dir: Path,
    output_dir: Path,
    output_name: str,
    quant_types: Iterable[str],
    *,
    resolution: Optional[LlamaCppResolution] = None,
    keep_fp16: bool = False,
    base_model: Optional[str] = None,
    callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> GGUFExportResult:
    """Convert ``source_dir`` into one or more GGUF files under ``output_dir``.

    Args:
        source_dir: Directory containing an un-adapted (merged or full) model
            in HuggingFace format. Must contain ``config.json`` plus weights.
        output_dir: Destination directory. Created if missing.
        output_name: Base filename stem (usually ``config.output_name``).
        quant_types: Iterable of requested quant names; order preserved.
        resolution: Pre-computed :class:`LlamaCppResolution`. If omitted, the
            resolver runs with the current environment.
        keep_fp16: If False, the intermediate fp16 GGUF is deleted after all
            quants are produced (unless ``F16`` was explicitly requested).
        base_model: Optional metadata written into the manifest.
        callback: Optional progress sink; receives :class:`ProgressEvent`s.
        cancel_event: Optional threading.Event; if set between stages, the
            pipeline stops and returns what it has so far.

    Returns:
        :class:`GGUFExportResult` summarising artifacts + failures. Never
        raises for per-quant failures — training-runner callers want a
        non-fatal experience when some quants succeed and others don't.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise GGUFExportError(f"Source directory does not exist: {source_dir}")
    if not (source_dir / "config.json").is_file():
        raise GGUFExportError(
            f"Source directory {source_dir} missing config.json — cannot convert."
        )

    resolution = resolution or resolve_llama_cpp()
    if not resolution.available:
        hint = resolution.warnings[0] if resolution.warnings else "llama.cpp not available"
        raise GGUFExportError(hint)

    requested = normalize_quant_types(quant_types)
    if not requested:
        raise GGUFExportError("No GGUF quant types requested.")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = GGUFExportResult(output_dir=output_dir)

    fp16_requested = "F16" in requested
    quant_targets = [q for q in requested if q != "F16"]

    # Step 1: always produce the fp16 GGUF (it's the input to llama-quantize).
    fp16_filename = _filename_for(output_name, "F16")
    fp16_path = output_dir / fp16_filename

    try:
        convert_to_fp16_gguf(
            source_dir, fp16_path, resolution=resolution, callback=callback
        )
    except GGUFExportError as exc:
        _emit(
            callback,
            ProgressEvent(stage="error", message=str(exc), error=str(exc)),
        )
        raise

    if fp16_requested:
        result.artifacts.append(
            GGUFArtifact(
                quant_type="F16",
                path=fp16_path,
                size_bytes=fp16_path.stat().st_size,
            )
        )

    # Step 2: per-quant quantize loop.
    for idx, quant in enumerate(quant_targets, start=1):
        if cancel_event is not None and cancel_event.is_set():
            result.warnings.append(
                f"GGUF export cancelled before {quant} "
                f"(completed {idx - 1} of {len(quant_targets)} quants)."
            )
            break

        _emit(
            callback,
            ProgressEvent(
                stage="quantizing",
                quant_type=quant,
                current=idx,
                total=len(quant_targets),
                message=f"Quantizing {quant} ({idx}/{len(quant_targets)})",
            ),
        )

        out_filename = _filename_for(output_name, quant)
        out_path = output_dir / out_filename
        try:
            quantize_gguf(
                fp16_path,
                quant,
                out_path,
                resolution=resolution,
                callback=callback,
            )
        except GGUFExportError as exc:
            logger.warning("GGUF quantize for %s failed: %s", quant, exc)
            result.failed_quants.append(quant)
            result.warnings.append(f"Failed to produce {quant}: {exc}")
            _emit(
                callback,
                ProgressEvent(
                    stage="error",
                    quant_type=quant,
                    message=f"{quant} failed: {exc}",
                    error=str(exc),
                ),
            )
            continue

        result.artifacts.append(
            GGUFArtifact(
                quant_type=quant,
                path=out_path,
                size_bytes=out_path.stat().st_size,
            )
        )
        _emit(
            callback,
            ProgressEvent(
                stage="quantizing",
                quant_type=quant,
                current=idx,
                total=len(quant_targets),
                message=f"Produced {out_path.name}",
                artifact_path=str(out_path),
            ),
        )

    # Step 3: clean up intermediate fp16 unless user kept it.
    if not fp16_requested and not keep_fp16 and fp16_path.is_file():
        try:
            fp16_path.unlink()
        except OSError as exc:
            result.warnings.append(
                f"Could not remove intermediate fp16 GGUF {fp16_path}: {exc}"
            )

    # Step 4: write manifest and emit final event.
    result.manifest_path = write_manifest(
        output_dir,
        result.artifacts,
        source_model=output_name,
        base_model=base_model,
    )

    _emit(
        callback,
        ProgressEvent(
            stage="complete",
            message=(
                f"GGUF export complete: {len(result.artifacts)} artifact(s), "
                f"{len(result.failed_quants)} failure(s)"
            ),
        ),
    )

    return result


# ---------------------------------------------------------------------------
# Merge helper shared with the HuggingFace Hub upload path
# ---------------------------------------------------------------------------

def merge_lora_to_directory(
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
    *,
    model_type: str = "auto",
    is_vlm: bool = False,
) -> Path:
    """
    Produce a merged HuggingFace-format checkpoint on disk.

    Mirrors the CPU bf16 merge used by ``_run_background_upload`` but
    writes to a caller-owned directory so multiple consumers (HF upload,
    GGUF export) can share a single merge. Moved here so the training
    runner can call it for GGUF without duplicating the merge code.
    """
    # Imports are deferred so ``src.gguf_exporter`` remains lightweight
    # enough to import in CI / resolver tests that mock heavy ML deps.
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    from .training_runner import _get_auto_model_class  # circular-safe lazy import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reload_cls, _ = _get_auto_model_class(base_model, model_type)
    logger.info("Reloading base model on CPU for merge: %s", base_model)
    base_model_reload = reload_cls.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    merged = PeftModel.from_pretrained(
        base_model_reload,
        str(adapter_dir),
        device_map="cpu",
    ).merge_and_unload()

    merged.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))

    if is_vlm:
        try:
            from transformers import AutoProcessor  # type: ignore
            processor = AutoProcessor.from_pretrained(str(adapter_dir), trust_remote_code=True)
            processor.save_pretrained(str(output_dir))
        except Exception as exc:
            logger.warning("Could not save VLM processor to %s: %s", output_dir, exc)

    del merged, base_model_reload
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return output_dir
