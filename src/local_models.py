"""
Local model discovery for offline training support.

Lets users pick base models that are already on disk — either in the
HuggingFace hub cache (downloaded by a previous run) or in Merlina's own
models directory (full/merged models produced by earlier training jobs) —
without needing internet access.

Like the disk tooling, the HF cache scan is injectable (``_scan``) so this
is unit-testable without a real cache, and every failure path degrades to
an empty list rather than an error: offline discovery is a convenience,
never a blocker.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def offline_mode_active() -> bool:
    """True when Merlina should avoid all HuggingFace Hub network access.

    Active when the ``OFFLINE_MODE`` setting is enabled (which exports the
    env vars below at startup) or when the standard HF offline env vars are
    set externally.
    """
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.getenv(var, "").strip().lower() in _TRUTHY:
            return True
    try:
        from config import settings  # type: ignore
        return bool(settings.offline_mode)
    except Exception:
        return False


def _default_scan():
    """Import + call ``huggingface_hub.scan_cache_dir`` lazily."""
    from huggingface_hub import scan_cache_dir
    return scan_cache_dir()


def list_hf_cached_models(_scan=None) -> list:
    """List model repos available in the local HuggingFace cache.

    Returns ``[{model_id, source: "hf_cache", size_bytes, last_modified}]``
    sorted by repo id. Returns ``[]`` if huggingface_hub or the cache is
    missing — callers treat that as "nothing downloaded yet".
    """
    scan = _scan or _default_scan
    try:
        info = scan()
    except Exception as exc:  # noqa: BLE001 — discovery must never be fatal
        logger.debug("HF cache scan unavailable: %s", exc)
        return []

    models = []
    try:
        for repo in info.repos:
            if getattr(repo, "repo_type", "model") != "model":
                continue
            # Repos with no completed snapshot can't be loaded offline.
            if not getattr(repo, "revisions", None):
                continue
            models.append({
                "model_id": repo.repo_id,
                "source": "hf_cache",
                "size_bytes": getattr(repo, "size_on_disk", None),
                "last_modified": getattr(repo, "last_modified", None),
            })
    except Exception as exc:  # noqa: BLE001
        logger.debug("HF cache scan returned unexpected shape: %s", exc)
        return []

    models.sort(key=lambda m: m["model_id"].lower())
    return models


def list_models_dir_models(models_dir: Path) -> list:
    """List full (non-adapter) models in Merlina's models directory.

    Only directories with a top-level ``config.json`` qualify — LoRA
    adapter outputs (``adapter_config.json`` without ``config.json``)
    can't serve as a base model on their own.
    """
    models = []
    try:
        if not models_dir.exists():
            return []
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if not (model_dir / "config.json").exists():
                continue
            models.append({
                "model_id": str(model_dir),
                "name": model_dir.name,
                "source": "models_dir",
            })
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not scan models dir %s: %s", models_dir, exc)
        return []
    return models


def list_local_models(models_dir: Path, _scan=None) -> dict:
    """All locally available base models, for the offline model picker.

    Returns ``{offline_mode, models}`` where each model has at least
    ``model_id`` (what goes into TrainingConfig.base_model) and ``source``
    (``hf_cache`` or ``models_dir``).
    """
    models = list_hf_cached_models(_scan=_scan) + list_models_dir_models(models_dir)
    return {
        "offline_mode": offline_mode_active(),
        "models": models,
        "count": len(models),
    }


def is_model_cached(model_id: str, cache_dir: Optional[Path] = None) -> bool:
    """Check whether a HuggingFace model id has a usable local cache entry.

    Used by preflight checks in offline mode: an uncached HF id is a
    guaranteed download failure, better caught before the job starts.
    """
    try:
        if cache_dir is None:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = Path(HF_HUB_CACHE)
        repo_dir = cache_dir / ("models--" + model_id.replace("/", "--"))
        snapshots = repo_dir / "snapshots"
        return snapshots.is_dir() and any(snapshots.iterdir())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not check cache for %s: %s", model_id, exc)
        return False
