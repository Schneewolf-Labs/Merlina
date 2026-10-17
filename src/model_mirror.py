"""
Mirror HuggingFace Hub models into the model store.

``merlina models mirror <org/repo>`` (or ``MODEL_MIRROR=true``, on first use)
downloads a Hub model once, at a pinned commit, and publishes it to the
model store as ``hf/<org>/<repo>``, i.e. ``store:hf/<org>/<repo>``. After
that the stored copy is what gets loaded: a model that disappears from the
Hub, or a Hub outage, no longer stops a job, and every job loads the same
bytes.

The manifest records the Hub repo and commit sha. A later mirror call
compares the Hub's current sha with it and re-publishes only when the
upstream moved (or ``force``). When the Hub can't be reached, the existing
mirror is served as-is. This is the model-side counterpart of
``dataset_handlers/mirror.py``.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Never mirror formats nothing in Merlina loads; .bin is skipped only when
# the repo also ships safetensors.
_ALWAYS_IGNORE = ["*.pth", "*.pt", "*.gguf", "*.onnx", "*.h5", "*.msgpack", "*.ot",
                  "*.tflite", "original/*", ".gitattributes"]


class ModelMirrorError(RuntimeError):
    pass


def mirror_name(repo_id: str) -> str:
    return f"hf/{repo_id.strip('/')}"


def _hub_sha(repo_id: str, revision: Optional[str], token: Optional[str]):
    """``(sha, files)`` for the repo at ``revision``, or ``(None, None)`` if the
    Hub can't be reached / doesn't have it."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, revision=revision, token=token, files_metadata=False)
        return info.sha, [s.rfilename for s in (info.siblings or [])]
    except Exception as e:
        logger.warning(f"Could not resolve {repo_id} on the Hub: {e}")
        return None, None


def _mirrored_sha(store, name: str) -> Optional[str]:
    try:
        meta = (store._manifest(name).get("metadata") or {}).get("source") or {}
        return meta.get("hf_sha")
    except FileNotFoundError:
        return None


def ensure_model_mirror(repo_id: str, *, store=None, revision: Optional[str] = None,
                        token: Optional[str] = None, force: bool = False,
                        workers: int = 8) -> str:
    """Make sure ``repo_id`` is mirrored and return its store ref
    (``store:hf/<org>/<repo>``)."""
    from src.model_refs import store_name
    if store is None:
        from src.remote.artifacts import model_store
        store = model_store()
    name = mirror_name(repo_id)
    store_name(name)            # validates the shape
    ref = f"store:{name}"

    sha, files = _hub_sha(repo_id, revision, token)
    current = _mirrored_sha(store, name)
    if sha is None:
        if current is not None:
            logger.info(f"Hub unreachable; using the existing mirror {ref} ({current[:12]})")
            return ref
        raise ModelMirrorError(
            f"Can't mirror {repo_id}: the Hub can't be reached (or has no such repo) "
            "and no mirror exists yet.")
    if current == sha and not force:
        logger.info(f"Mirror hit: {ref} @ {sha[:12]}")
        return ref

    ignore = list(_ALWAYS_IGNORE)
    if files and any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")
    logger.info(f"Mirroring {repo_id}@{sha[:12]} -> models/{name}")
    from huggingface_hub import snapshot_download
    with tempfile.TemporaryDirectory(prefix="merlina-model-mirror-") as tmp:
        # local_dir gives real files (no cache symlinks), which push_dir requires.
        snapshot_download(repo_id, revision=sha, token=token, local_dir=tmp,
                          ignore_patterns=ignore)
        from pathlib import Path
        cache = Path(tmp) / ".cache"
        if cache.exists():
            import shutil
            shutil.rmtree(cache)        # huggingface_hub's download bookkeeping
        store.metadata = {"source": {"hf_repo": repo_id, "hf_sha": sha, "revision": revision}}
        store.push_dir(Path(tmp), name, snapshot=False, workers=workers)
    return ref


def mirror_enabled(settings=None) -> bool:
    if settings is None:
        from config import settings
    return bool(getattr(settings, "model_mirror", False))
