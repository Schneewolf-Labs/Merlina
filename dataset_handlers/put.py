"""
Write our own data into the internal dataset store as ``swl://name@rev``.

The mirror only writes revisions copied from the Hub. ``put_dataset`` adds
any local JSON / JSONL / CSV / Parquet file (an upload, a pipeline output,
a hand-built preference set) under a name of our choosing. The revision is
the first 16 hex digits of the file's sha256, so putting the same bytes
twice is a no-op and the revision pins exactly what was trained on.

``private=True`` (or a private-looking name) keeps the revision out of
``merlina datasets promote`` to the external tier, the same guard the
mirror's manifests use.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .internal_store import LATEST, InternalStore, InternalStoreError, Manifest

logger = logging.getLogger(__name__)

_FORMATS = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put_dataset(store: InternalStore, path: os.PathLike, name: str, *,
                private: bool = False, file_format: Optional[str] = None,
                source: Optional[dict] = None) -> tuple[str, str, bool]:
    """Store ``path`` as ``swl://name@rev``; returns ``(name, rev, created)``."""
    path = Path(path)
    if not _NAME_RE.match(name or "") or name.startswith("hf/"):
        raise InternalStoreError(
            f"Invalid dataset name {name!r} (letters, digits, '.', '_', '-', '/'; "
            "hf/ is reserved for mirrors)")
    fmt = file_format or _FORMATS.get(path.suffix.lower())
    if fmt not in ("json", "csv", "parquet"):
        raise InternalStoreError(f"Unsupported dataset file {path.name} (json, jsonl, csv, parquet)")
    rev = _sha256(path)[:16]
    if store.exists(name, rev):
        store.set_latest(name, rev)
        logger.info(f"swl://{name}@{rev} already stored; latest points at it")
        return name, rev, False

    from datasets import load_dataset
    dataset = load_dataset(fmt, data_files=str(path), split="train")
    with tempfile.TemporaryDirectory(prefix="merlina-put-") as tmp:
        local = os.path.join(tmp, "data-0.parquet")
        dataset.to_parquet(local)
        data_uri = store.upload_file(local, name, rev, "data-0.parquet")
    manifest = Manifest(
        name=name, rev=rev, format="parquet", data_files=[data_uri], num_rows=len(dataset),
        source={"kind": "put", "file_sha256": _sha256(path), **(source or {})},
        created_at=datetime.now(timezone.utc).isoformat(), private=private,
    )
    store.put_manifest(manifest)
    store.set_latest(name, rev)
    logger.info(f"Stored {path.name} as swl://{name}@{rev} ({len(dataset)} rows)")
    return name, rev, True


__all__ = ["put_dataset", "LATEST"]
