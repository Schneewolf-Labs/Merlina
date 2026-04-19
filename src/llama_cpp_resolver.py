"""
llama.cpp Binary Resolver

Locates llama.cpp binaries and helper scripts for Merlina's optional
GGUF export and synthetic-data features.

Resolution precedence (highest first):
1. Explicit override passed to ``resolve_llama_cpp``
2. ``LLAMA_CPP_DIR`` env var / settings (treated as the llama.cpp repo root)
3. ``LLAMA_CPP_BIN_DIR`` env var / settings (treated as a directory of built binaries)
4. ``shutil.which()`` lookup of the binary names on ``PATH``
5. Conventional vendored checkout at ``./vendor/llama.cpp``

Missing binaries are never fatal: callers receive a
:class:`LlamaCppResolution` object whose ``available`` flag indicates
whether the feature should be exposed in the UI. Pre-flight surfaces the
absence as a warning so jobs don't silently break.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Binaries Merlina actually uses. The first list is what gates
# ``available``; the second list is best-effort discovery.
REQUIRED_BINARIES = ("llama-quantize",)
OPTIONAL_BINARIES = (
    "llama-server",
    "llama-cli",
    "llama-gguf-split",
    "llama-imatrix",
    "llama-perplexity",
)
ALL_BINARIES = REQUIRED_BINARIES + OPTIONAL_BINARIES

CONVERT_SCRIPT_NAME = "convert_hf_to_gguf.py"

# Candidate bin subdirectories relative to a llama.cpp repo root.
# Modern builds put binaries under ``build/bin``; older Makefile builds
# drop them at the repo root.
BIN_SUBDIRS = ("build/bin", "build", "bin", "")

# Conventional vendored checkout location (relative to CWD).
VENDOR_PATH = Path("vendor/llama.cpp")


@dataclass
class LlamaCppResolution:
    """Result of a llama.cpp discovery attempt."""

    available: bool = False
    source: Optional[str] = None  # "explicit" | "env_root" | "env_bin" | "path" | "vendor"
    root: Optional[Path] = None
    bin_dir: Optional[Path] = None
    convert_script: Optional[Path] = None
    binaries: Dict[str, Path] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def binary(self, name: str) -> Optional[Path]:
        """Return the resolved path for ``name`` or ``None`` if missing."""
        return self.binaries.get(name)

    def to_dict(self) -> dict:
        """Serializable form for API responses and pre-flight output."""
        return {
            "available": self.available,
            "source": self.source,
            "root": str(self.root) if self.root else None,
            "bin_dir": str(self.bin_dir) if self.bin_dir else None,
            "convert_script": str(self.convert_script) if self.convert_script else None,
            "binaries": {name: str(path) for name, path in self.binaries.items()},
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _find_binaries_in(bin_dir: Path) -> Dict[str, Path]:
    """Return the subset of known llama.cpp binaries present in ``bin_dir``."""
    found: Dict[str, Path] = {}
    if not bin_dir.is_dir():
        return found
    for name in ALL_BINARIES:
        candidate = bin_dir / name
        if _executable(candidate):
            found[name] = candidate.resolve()
    return found


def _derive_bin_dir_from_root(root: Path) -> Optional[Path]:
    """Locate the build output directory inside a llama.cpp checkout."""
    for sub in BIN_SUBDIRS:
        candidate = (root / sub) if sub else root
        if _find_binaries_in(candidate):
            return candidate.resolve()
    return None


def _derive_root_from_bin(bin_dir: Path) -> Optional[Path]:
    """Walk upwards from a bin dir to find a plausible llama.cpp repo root."""
    current = bin_dir.resolve()
    for _ in range(4):  # bin_dir itself + up to 3 parents
        if (current / CONVERT_SCRIPT_NAME).is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _resolve_path_env() -> Dict[str, Path]:
    """Find llama.cpp binaries on the system PATH."""
    found: Dict[str, Path] = {}
    for name in ALL_BINARIES:
        located = shutil.which(name)
        if located:
            found[name] = Path(located).resolve()
    return found


def _settings_values() -> tuple[Optional[str], Optional[str]]:
    """Read llama.cpp paths from Merlina settings, tolerating import-order issues."""
    try:
        from config import settings  # type: ignore
    except Exception:  # pragma: no cover - settings may not be importable in workers
        return None, None
    root = getattr(settings, "llama_cpp_dir", None)
    bin_dir = getattr(settings, "llama_cpp_bin_dir", None)
    return (str(root) if root else None, str(bin_dir) if bin_dir else None)


def _resolve_from_root(
    root_str: str, *, source: str, result: LlamaCppResolution
) -> bool:
    """Populate ``result`` from an explicit repo-root path. Returns True on success."""
    root = Path(root_str).expanduser()
    if not root.is_dir():
        result.errors.append(
            f"Configured llama.cpp root '{root}' does not exist or is not a directory."
        )
        return False

    bin_dir = _derive_bin_dir_from_root(root)
    if not bin_dir:
        result.errors.append(
            f"Configured llama.cpp root '{root}' has no built binaries. "
            "Build llama.cpp first (e.g. `cmake -B build && cmake --build build -j`)."
        )
        return False

    binaries = _find_binaries_in(bin_dir)
    convert = root / CONVERT_SCRIPT_NAME
    result.source = source
    result.root = root.resolve()
    result.bin_dir = bin_dir
    result.binaries = binaries
    result.convert_script = convert.resolve() if convert.is_file() else None
    return True


def _resolve_from_bin(
    bin_str: str, *, source: str, result: LlamaCppResolution
) -> bool:
    bin_dir = Path(bin_str).expanduser()
    if not bin_dir.is_dir():
        result.errors.append(
            f"Configured llama.cpp bin dir '{bin_dir}' does not exist or is not a directory."
        )
        return False

    binaries = _find_binaries_in(bin_dir)
    if not binaries:
        result.errors.append(
            f"Configured llama.cpp bin dir '{bin_dir}' contains no recognized binaries."
        )
        return False

    derived_root = _derive_root_from_bin(bin_dir)
    result.source = source
    result.bin_dir = bin_dir.resolve()
    result.root = derived_root
    result.binaries = binaries
    if derived_root:
        convert = derived_root / CONVERT_SCRIPT_NAME
        result.convert_script = convert.resolve() if convert.is_file() else None
    return True


def resolve_llama_cpp(
    explicit_dir: Optional[str] = None,
    *,
    cwd: Optional[Path] = None,
) -> LlamaCppResolution:
    """Locate llama.cpp binaries following the documented precedence.

    Args:
        explicit_dir: Optional path supplied by a caller. Treated as a repo
            root if it contains ``convert_hf_to_gguf.py``, otherwise as a
            bin directory.
        cwd: Override the working directory used to resolve the vendored
            checkout. Useful for tests.

    Returns:
        A :class:`LlamaCppResolution` describing what was found. Callers
        should consult ``available`` before invoking any binaries.
    """
    result = LlamaCppResolution()
    cwd = cwd or Path.cwd()

    env_root = os.environ.get("LLAMA_CPP_DIR")
    env_bin = os.environ.get("LLAMA_CPP_BIN_DIR")
    settings_root, settings_bin = _settings_values()

    # 1. Explicit override. Try root-shaped interpretation first.
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if (path / CONVERT_SCRIPT_NAME).is_file() and _resolve_from_root(
            str(path), source="explicit", result=result
        ):
            pass
        elif _resolve_from_bin(str(path), source="explicit", result=result):
            pass

    # 2. LLAMA_CPP_DIR (env or settings).
    if not result.binaries:
        root_value = env_root or settings_root
        if root_value:
            _resolve_from_root(root_value, source="env_root", result=result)

    # 3. LLAMA_CPP_BIN_DIR (env or settings).
    if not result.binaries:
        bin_value = env_bin or settings_bin
        if bin_value:
            _resolve_from_bin(bin_value, source="env_bin", result=result)

    # 4. System PATH lookup.
    if not result.binaries:
        found = _resolve_path_env()
        if found:
            # All binaries on PATH should generally share a directory; pick
            # the one for llama-quantize if present, else the first.
            anchor = found.get("llama-quantize") or next(iter(found.values()))
            bin_dir = anchor.parent
            result.source = "path"
            result.bin_dir = bin_dir
            result.binaries = found
            result.root = _derive_root_from_bin(bin_dir)
            if result.root:
                convert = result.root / CONVERT_SCRIPT_NAME
                result.convert_script = convert if convert.is_file() else None

    # 5. Vendored checkout.
    if not result.binaries:
        vendor_root = (cwd / VENDOR_PATH).resolve()
        if vendor_root.is_dir():
            _resolve_from_root(str(vendor_root), source="vendor", result=result)

    # Finalize availability + warnings.
    missing_required = [b for b in REQUIRED_BINARIES if b not in result.binaries]
    result.available = not missing_required and bool(result.binaries)

    if result.binaries and missing_required:
        result.warnings.append(
            f"llama.cpp found at {result.bin_dir} but missing required binaries: "
            f"{', '.join(missing_required)}."
        )
    if result.binaries:
        missing_optional = [b for b in OPTIONAL_BINARIES if b not in result.binaries]
        if missing_optional:
            result.warnings.append(
                f"Optional llama.cpp binaries not found at {result.bin_dir}: "
                f"{', '.join(missing_optional)}. Some features may be disabled."
            )
        if not result.convert_script:
            result.warnings.append(
                "llama.cpp binaries located but 'convert_hf_to_gguf.py' was not found. "
                "GGUF export from HuggingFace checkpoints will be unavailable until "
                "LLAMA_CPP_DIR points at a full llama.cpp checkout."
            )

    if not result.binaries and not result.errors:
        result.warnings.append(
            "llama.cpp not found. Set LLAMA_CPP_DIR (repo root) or "
            "LLAMA_CPP_BIN_DIR, install binaries on PATH, or clone to "
            f"./{VENDOR_PATH} to enable GGUF export and synthetic-data features."
        )

    if result.available:
        logger.info(
            "Resolved llama.cpp via %s: bin_dir=%s, convert_script=%s",
            result.source,
            result.bin_dir,
            result.convert_script,
        )
    else:
        logger.debug("llama.cpp not available: %s", result.warnings or result.errors)

    return result
