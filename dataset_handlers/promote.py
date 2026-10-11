"""
Promote a dataset revision from one store tier to another.

The internal MinIO store is LAN-only; a rented GPU box (RunPod) can't reach it.
Promotion copies a dataset revision to the external tier (R2) so such a box can
train from the same ``swl://name@rev`` — the copy is byte-identical because a
revision is content-addressed by its pinned upstream sha.

Promotion is one-way by design: the internal store is the source of truth, and
the external tier holds only the subset explicitly pushed there. Anything whose
manifest is flagged private, or whose name matches a private marker (west/IR
data), is refused — that data never leaves the private store.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

from .internal_store import InternalStore, InternalStoreError, Manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotePlan:
    """What a promotion would do — returned by a dry run, carried by a real run."""

    name: str
    rev: str
    num_shards: int
    total_bytes: Optional[int]  # None if any shard's size couldn't be read
    already_present: bool  # the rev already exists in the destination
    skipped: bool  # nothing was copied (already present and not overwriting)

    def human_bytes(self) -> str:
        if self.total_bytes is None:
            return "unknown"
        n = float(self.total_bytes)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024 or unit == "TiB":
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{self.total_bytes} B"


def promote(
    src: InternalStore,
    dst: InternalStore,
    name: str,
    rev: str = "latest",
    *,
    overwrite: bool = False,
    allow_private: bool = False,
    dry_run: bool = False,
) -> PromotePlan:
    """
    Copy ``swl://name@rev`` from ``src`` to ``dst``.

    Resolves ``latest`` against the source, refuses private data unless
    ``allow_private`` is set, skips a revision already present in the
    destination unless ``overwrite`` is set, and — on a real run — streams each
    shard through a temp file, writes a destination manifest with rewritten
    ``data_files``, and moves the destination's ``latest`` pointer.

    ``dry_run`` plans and validates (including the private guard and size
    estimate) without writing anything, so cost can be checked before spend.
    """
    manifest = src.get_manifest(name, rev)  # resolves latest -> concrete rev
    concrete = manifest.rev

    if manifest.is_private() and not allow_private:
        raise InternalStoreError(
            f"Refusing to promote swl://{name}@{concrete}: it is marked private "
            f"(or its name matches a private marker). Private data stays in the "
            f"internal store. Pass allow_private=True only if you are certain."
        )

    total = 0
    have_all_sizes = True
    for uri in manifest.data_files:
        size = src.file_size(uri)
        if size is None:
            have_all_sizes = False
        else:
            total += size
    total_bytes = total if have_all_sizes else None

    already = dst.exists(name, concrete)
    if already and not overwrite:
        logger.info(
            f"swl://{name}@{concrete} already present in destination — skipping "
            f"(pass overwrite=True to recopy)"
        )
        return PromotePlan(name, concrete, len(manifest.data_files), total_bytes,
                           already_present=True, skipped=True)

    if dry_run:
        return PromotePlan(name, concrete, len(manifest.data_files), total_bytes,
                           already_present=already, skipped=False)

    new_data_files = []
    with tempfile.TemporaryDirectory() as tmp:
        for uri in manifest.data_files:
            filename = uri.rstrip("/").rsplit("/", 1)[-1]
            local = os.path.join(tmp, filename)
            src.download_file(uri, local)
            new_uri = dst.upload_file(local, name, concrete, filename)
            new_data_files.append(new_uri)

    dst_manifest = dataclasses.replace(manifest, data_files=new_data_files)
    dst.put_manifest(dst_manifest)
    dst.set_latest(name, concrete)
    logger.info(
        f"Promoted swl://{name}@{concrete} -> destination "
        f"({len(new_data_files)} shard(s), {PromotePlan(name, concrete, len(new_data_files), total_bytes, False, False).human_bytes()})"
    )
    return PromotePlan(name, concrete, len(new_data_files), total_bytes,
                       already_present=already, skipped=False)
