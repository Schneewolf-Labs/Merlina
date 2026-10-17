"""
Publish a trained model to the model store instead of (or as well as) the Hub.

``publish_to_store`` pushes a model directory to ``models/<name>`` in the
artifact bucket, where ``store:<name>`` loads it as a base model and
``model.store_model`` serves it. What goes up:

- the full model: the merged checkpoint when a merge was made, else the
  output directory as is (an adapter directory stays an adapter, with its
  ``adapter_config.json`` naming the base);
- ``README.md``, written into the output directory first so the model card
  also exists locally;
- ``gguf/`` exports, when present and ``include_gguf``.

Run-local clutter (``checkpoint-*``, the upload sidecar, merge scratch)
never goes up. Each publish is recorded in the model's ``upload_state.json``
with ``destination: "store"``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SKIP_DIRS = ("_base_index", "_base_shards", "__pycache__")
_SKIP_FILES = ("upload_state.json",)


def _wanted(relative: str, include_gguf: bool) -> bool:
    parts = relative.split("/")
    if parts[0].startswith("checkpoint-") or any(p in _SKIP_DIRS for p in parts):
        return False
    if parts[-1] in _SKIP_FILES or parts[-1].endswith(".lock"):
        return False
    if parts[0] == "gguf" and not include_gguf:
        return False
    return True


def _files(root: Path, include_gguf: bool, prefix: str = ""):
    for f in sorted(root.rglob("*")):
        if f.is_file() and not f.is_symlink():
            rel = prefix + f.relative_to(root).as_posix()
            if _wanted(rel, include_gguf):
                yield rel, f


def publish_to_store(
    model_dir: Any,
    name: Optional[str] = None,
    *,
    merged_dir: Any = None,
    config: Any = None,
    training_mode: str = "sft",
    is_vlm: bool = False,
    is_diffusion: bool = False,
    include_gguf: bool = True,
    store=None,
    workers: int = 8,
) -> dict:
    """Push a model to ``models/<name>``; returns ``{ref, uri, files, bytes, kind}``."""
    from src.model_refs import store_name
    from src.upload_state import record_upload

    model_dir = Path(model_dir)
    name = name or model_dir.name
    store_name(f"store:{name}")             # validates the name
    if store is None:
        from src.remote.artifacts import model_store
        store = model_store()
    uri = f"s3://{store.bucket}/{store.prefix}/{name}"

    try:
        if config is not None:
            from src.model_card import generate_model_readme
            readme = generate_model_readme(config, training_mode, is_vlm=is_vlm,
                                           is_diffusion=is_diffusion)
            (model_dir / "README.md").write_text(readme, encoding="utf-8")

        source = Path(merged_dir) if merged_dir else model_dir
        files = list(_files(source, include_gguf=False))
        if merged_dir:
            if (model_dir / "README.md").is_file():
                files = [f for f in files if f[0] != "README.md"] + [("README.md", model_dir / "README.md")]
        if include_gguf and (model_dir / "gguf").is_dir():
            files += list(_files(model_dir / "gguf", include_gguf=True, prefix="gguf/"))
        if not files:
            raise ValueError(f"Nothing to publish in {source}")

        has_adapter = any(rel == "adapter_config.json" for rel, _ in files)
        kind = "adapter" if has_adapter else ("full" if any(rel == "config.json" for rel, _ in files) else "files")
        base = getattr(config, "base_model", None) if config is not None else None
        store.metadata = {"kind": kind, "base_model": base,
                          "output_name": getattr(config, "output_name", None) if config else None}
        store.push_files(name, files, workers=workers)
    except Exception as e:
        record_upload(model_dir, repo_id=name, repo_url=uri, private=True,
                      commit_message="Publish to model store", artifacts=[],
                      status="failed", error=str(e), destination="store")
        raise

    artifacts = [kind] + sorted({f"gguf:{rel.rsplit('/', 1)[-1]}" for rel, _ in files if rel.startswith("gguf/") and rel.endswith(".gguf")})
    record_upload(model_dir, repo_id=name, repo_url=uri, private=True,
                  commit_message="Publish to model store", artifacts=artifacts,
                  destination="store")
    total = sum(f.stat().st_size for _, f in files)
    logger.info("Published %s to %s (%d files, %.1f GiB, %s)", model_dir, uri, len(files),
                total / 1024 ** 3, kind)
    return {"ref": f"store:{name}", "uri": uri, "files": len(files), "bytes": total, "kind": kind}
