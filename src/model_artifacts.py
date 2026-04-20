"""
Model artifact inventory.

Walks a model directory under ``./models/`` and categorizes what it
finds so the frontend Artifacts panel can render a tidy grouped list
with sizes, and so delete / upload endpoints know what's safe to touch.

Categories (see :data:`PROTECTED_FILES` for undeletable core files):
  - ``adapter``      — PEFT LoRA adapter shards + ``adapter_config.json``
  - ``merged``       — full-precision model weights at the top level
                       (safetensors / pytorch_model.bin variants)
  - ``gguf``         — any ``*.gguf`` under ``gguf/`` + its manifest
  - ``tokenizer``    — tokenizer / chat template / vocab files
  - ``processor``    — VLM processor & preprocessor_config.json
  - ``config``       — model ``config.json`` and generation_config.json
  - ``readme``       — README.md, upload_state.json sidecars
  - ``other``        — everything else

The categorization uses filename / path patterns — no file contents are
parsed. Sidecars (``upload_state.json``, ``manifest.json``) are surfaced
under ``readme`` and ``gguf`` respectively so the UI can show them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Deletion guardrails — anything matching one of these (by path relative
# to the model dir) is refused by the delete endpoint. Users can still
# rm them from the CLI if they really want to.
PROTECTED_FILES = frozenset({
    "config.json",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
})


# Suffix/pattern buckets.
_ADAPTER_FILES = frozenset({"adapter_config.json"})
_ADAPTER_SUFFIXES = (".bin", ".safetensors", ".pt")  # filtered below when inside a LoRA dir
_TOKENIZER_FILES = frozenset({
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
})
_PROCESSOR_FILES = frozenset({
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor.json",
})
_CONFIG_FILES = frozenset({"config.json", "generation_config.json"})
_README_FILES = frozenset({"README.md", "readme.md", "upload_state.json"})


@dataclass
class ArtifactFile:
    """A single file with metadata for the UI."""

    path: str              # path relative to the model dir
    filename: str
    size_bytes: int
    mtime: float
    category: str
    quant_type: Optional[str] = None   # populated for GGUF files
    protected: bool = False

    def to_dict(self) -> dict:
        payload = {
            "path": self.path,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "category": self.category,
            "protected": self.protected,
        }
        if self.quant_type is not None:
            payload["quant_type"] = self.quant_type
        return payload


@dataclass
class ArtifactInventory:
    """Grouped inventory for a single model directory."""

    model_dir: Path
    total_bytes: int = 0
    categories: Dict[str, List[ArtifactFile]] = field(default_factory=dict)
    is_lora: bool = False
    has_merged: bool = False
    has_gguf: bool = False

    def to_dict(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "total_bytes": self.total_bytes,
            "is_lora": self.is_lora,
            "has_merged": self.has_merged,
            "has_gguf": self.has_gguf,
            "categories": {
                name: [f.to_dict() for f in files]
                for name, files in self.categories.items()
            },
        }

    def add(self, file: ArtifactFile) -> None:
        self.categories.setdefault(file.category, []).append(file)
        self.total_bytes += file.size_bytes


def _categorize(model_dir: Path, file_path: Path, is_lora: bool) -> tuple[str, Optional[str]]:
    """
    Return (category, quant_type?) for a given file.
    ``quant_type`` is only set for GGUF files.
    """
    rel = file_path.relative_to(model_dir)
    rel_parts = rel.parts
    name = file_path.name

    # GGUF files live under gguf/ — manifest too.
    if len(rel_parts) >= 2 and rel_parts[0] == "gguf":
        if name.endswith(".gguf"):
            # Filename convention: {model}.{quant_lower}.gguf
            stem = name[: -len(".gguf")]
            quant = stem.rsplit(".", 1)[-1].upper() if "." in stem else None
            return "gguf", quant
        if name == "manifest.json":
            return "gguf", None
        return "gguf", None

    if name in _TOKENIZER_FILES:
        return "tokenizer", None
    if name in _PROCESSOR_FILES:
        return "processor", None
    if name in _CONFIG_FILES:
        return "config", None
    if name in _README_FILES:
        return "readme", None
    if name in _ADAPTER_FILES:
        return "adapter", None

    # Weight shards — adapter files are named with "adapter" in the stem
    # (e.g. adapter_model.safetensors). Everything else is merged/base.
    if name.endswith(_ADAPTER_SUFFIXES) or name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json"):
        if "adapter" in name.lower():
            return "adapter", None
        # When the model is LoRA-only, a stray shard at the top level is
        # unusual — still categorize it as merged so the UI shows it.
        return "merged", None

    return "other", None


def _looks_like_lora(model_dir: Path) -> bool:
    return (model_dir / "adapter_config.json").is_file()


def inventory_model(model_dir: Path) -> ArtifactInventory:
    """
    Walk ``model_dir`` recursively and produce a categorized inventory.

    Raises ``FileNotFoundError`` if the directory doesn't exist.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    inv = ArtifactInventory(model_dir=model_dir.resolve())
    inv.is_lora = _looks_like_lora(model_dir)

    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue

        category, quant = _categorize(model_dir, path, is_lora=inv.is_lora)
        rel = path.relative_to(model_dir)
        protected = str(rel) in PROTECTED_FILES or path.name in PROTECTED_FILES

        artifact = ArtifactFile(
            path=str(rel),
            filename=path.name,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            category=category,
            quant_type=quant,
            protected=protected,
        )
        inv.add(artifact)

        if category == "merged":
            inv.has_merged = True
        elif category == "gguf" and path.name.endswith(".gguf"):
            inv.has_gguf = True

    return inv


def resolve_artifact_path(model_dir: Path, rel_path: str) -> Path:
    """
    Resolve a user-supplied relative path against ``model_dir`` with
    directory-traversal protection. Raises ``ValueError`` for unsafe
    inputs so the delete endpoint can return 400.
    """
    model_dir = Path(model_dir).resolve()
    if not rel_path:
        raise ValueError("artifact path must not be empty")

    # Normalize and reject absolute or upward-traversing paths.
    candidate = (model_dir / rel_path).resolve()
    try:
        candidate.relative_to(model_dir)
    except ValueError as exc:
        raise ValueError(f"path escapes model directory: {rel_path}") from exc

    return candidate


def is_protected(model_dir: Path, rel_path: str) -> bool:
    """Return True when the file matches the hardcoded protect list."""
    name = Path(rel_path).name
    return rel_path in PROTECTED_FILES or name in PROTECTED_FILES
