"""
Model references that point into object storage instead of the Hub.

A base model can be given as ``store:<name>``. ``<name>`` is a model pushed
with ``merlina models push <dir> <name>`` (or mirrored from the Hub with
``merlina models mirror``) and lives in the artifact bucket under
``models/<name>``. Names may contain ``/``: Hub mirrors are stored as
``hf/<org>/<repo>``.

:func:`resolve_model_ref` turns such a reference into a local directory that
``from_pretrained`` can load: the committed generation is pulled once into
``<data_dir>/model-cache/<bucket>/<name>/<generation>`` and reused after
that. When the store can't be reached, the newest complete cached
generation is used, so a machine that has pulled a model once keeps working
offline. Anything that isn't a ``store:`` reference (an HF id, a local
path) is returned unchanged, so callers can resolve unconditionally.

Configs keep the ``store:`` reference; only the code that actually loads
weights resolves it, so model cards, uploads and job records name the
reference rather than this machine's cache path. PEFT records the path it
loaded from in ``adapter_config.json``; :func:`restore_adapter_base_ref`
puts the reference back after an adapter is saved.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STORE_PREFIX = "store:"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}(/[A-Za-z0-9][A-Za-z0-9._-]{0,127}){0,3}$")
_COMPLETE = ".merlina-complete"


# ref -> (path, resolved_at). Loaders resolve the same ref many times per
# job; re-reading the manifest each time would be a store round trip per call.
_MEMO: dict = {}
_MEMO_LOCK = threading.Lock()
MEMO_TTL_S = 300.0


class ModelRefError(RuntimeError):
    """A ``store:`` reference could not be resolved."""


def is_store_ref(ref: Any) -> bool:
    return isinstance(ref, str) and ref.startswith(STORE_PREFIX)


def store_name(ref: str) -> str:
    """The ``models/`` name of a ``store:`` reference, validated."""
    name = ref[len(STORE_PREFIX):] if is_store_ref(ref) else ref
    if not _NAME_RE.match(name or ""):
        raise ModelRefError(
            f"Invalid store model reference {ref!r}: expected store:<name> where <name> is "
            "letters, digits, '.', '_' or '-' (with up to three '/' separators)")
    return name


def _cache_root(settings=None) -> Path:
    if settings is None:
        from config import settings
    return Path(getattr(settings, "data_dir", "./data")) / "model-cache"


def _cached_generations(base: Path) -> list:
    if not base.is_dir():
        return []
    done = [d for d in base.iterdir() if d.is_dir() and (d / _COMPLETE).is_file()]
    return sorted(done, key=lambda d: (d / _COMPLETE).stat().st_mtime, reverse=True)


def resolve_model_ref(ref: str, *, store=None, cache_root: Optional[Path] = None,
                      workers: int = 8) -> str:
    """A local directory for ``ref`` (pulling it if needed), or ``ref`` itself
    when it isn't a ``store:`` reference."""
    if not is_store_ref(ref):
        return ref
    if store is None and cache_root is None:
        with _MEMO_LOCK:
            hit = _MEMO.get(ref)
        if hit and time.monotonic() - hit[1] < MEMO_TTL_S and Path(hit[0], _COMPLETE).is_file():
            return hit[0]
        path = _resolve(ref, None, None, workers)
        with _MEMO_LOCK:
            _MEMO[ref] = (path, time.monotonic())
        return path
    return _resolve(ref, store, cache_root, workers)


def _resolve(ref: str, store, cache_root, workers: int) -> str:
    name = store_name(ref)
    store_error = None
    if store is None:
        from src.remote.artifacts import model_store
        try:
            store = model_store()
        except Exception as e:
            store, store_error = None, e
    cache_root = Path(cache_root) if cache_root is not None else _cache_root()
    bucket = getattr(store, "bucket", None) or "default"
    base = cache_root / bucket / name

    generation = None
    if store is not None:
        try:
            generation = store._manifest(name)["generation"]
        except FileNotFoundError:
            raise ModelRefError(
                f"No model {name!r} in the store (models/{name}). Push it with "
                f"`merlina models push <dir> {name}`"
                + (f" or mirror it with `merlina models mirror {name[3:]}`" if name.startswith("hf/") else "")
            ) from None
        except Exception as e:
            store_error = e
    if generation is None:
        # Any bucket: the store may be unconfigured on this machine.
        cached = cached_ref_path(ref, cache_root)
        if cached:
            logger.warning("Model store unreachable (%s); using cached %s generation %s",
                           store_error, ref, Path(cached).name)
            return cached
        raise ModelRefError(f"Can't resolve {ref}: the model store is unavailable "
                            f"({store_error}) and nothing is cached") from None

    target = base / generation
    if (target / _COMPLETE).is_file():
        return str(target)
    base.mkdir(parents=True, exist_ok=True)
    with open(base / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)     # one pull per model, across jobs and DDP ranks
        if not (target / _COMPLETE).is_file():
            logger.info("Pulling %s (generation %s) into %s", ref, generation, target)
            if target.exists():
                shutil.rmtree(target)
            kwargs = {"workers": workers} if getattr(store, "kind", None) == "s3" else {}
            store.pull_dir(name, target, **kwargs)
            (target / _COMPLETE).write_text(json.dumps({"ref": ref, "generation": generation}))
            for old in _cached_generations(base):
                if old != target:
                    logger.info("Removing superseded cached generation %s", old)
                    shutil.rmtree(old, ignore_errors=True)
    return str(target)


def cached_ref_path(ref: str, cache_root: Optional[Path] = None) -> Optional[str]:
    """The newest complete cached generation of ``ref`` in any bucket, or None."""
    if not is_store_ref(ref):
        return None
    name = store_name(ref)
    root = Path(cache_root) if cache_root is not None else _cache_root()
    found = [d for b in (root.iterdir() if root.is_dir() else []) for d in _cached_generations(b / name)]
    found.sort(key=lambda d: (d / _COMPLETE).stat().st_mtime, reverse=True)
    return str(found[0]) if found else None


def restore_adapter_base_ref(output_dir: Any, ref: Optional[str]) -> None:
    """Point a saved adapter's ``base_model_name_or_path`` back at the store
    reference instead of this machine's cache directory."""
    if not ref:
        return
    path = Path(output_dir) / "adapter_config.json"
    if not path.is_file():
        return
    try:
        cfg = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if cfg.get("base_model_name_or_path") != ref:
        cfg["base_model_name_or_path"] = ref
        path.write_text(json.dumps(cfg, indent=2))
