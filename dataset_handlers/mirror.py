"""
Mirror-on-first-use: snapshot a HuggingFace dataset into the internal store the
first time it's requested, then serve every later run from the store.

The win is resilience + reproducibility. A mirror is keyed by the upstream
commit sha, so a run pins exactly what it trained on and is unaffected if the
Hub later rate-limits, goes down, or the dataset is pulled or changed.

Flow (``ensure_mirror``):
  1. Resolve the current upstream commit sha (cheap metadata call).
  2. If that sha is already mirrored → return it (no download).
  3. Otherwise pull from HF once, write parquet + a sha-pinned manifest, move the
     'latest' pointer, and return the sha.
  4. If the Hub is unreachable but a mirror already exists → fall back to the
     stored 'latest' so training still runs offline.

Scoped to non-streaming HF sources: a mirror materializes the split to parquet,
which is the whole point (a cached, local copy).
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

from .internal_store import LATEST, InternalStore, InternalStoreError, Manifest

logger = logging.getLogger(__name__)


def mirror_name(repo_id: str, config_name: Optional[str], split: str) -> str:
    """
    The internal dataset name a HF source mirrors to. Split and config are part
    of the name (not the revision) because they select different row sets of the
    same upstream commit.
    """
    name = f"hf/{repo_id.strip('/')}"
    if config_name:
        name += f"/{config_name}"
    name += f"/{split}"
    return name


def resolve_hf_revision(
    repo_id: str, revision: Optional[str] = None, token: Optional[str] = None
) -> Optional[str]:
    """
    Resolve a HF dataset ref to its commit sha, or None if the Hub can't be
    reached. A lightweight metadata call — no dataset download.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id, revision=revision, token=token)
        return info.sha
    except Exception as e:
        logger.warning(f"Could not resolve HF revision for {repo_id}: {e}")
        return None


def ensure_mirror(
    store: InternalStore,
    repo_id: str,
    split: str = "train",
    config_name: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> tuple[str, str]:
    """
    Ensure the dataset is mirrored in the store and return ``(name, rev)`` to
    load from it.

    Raises InternalStoreError if the dataset is neither reachable on the Hub nor
    already mirrored.
    """
    name = mirror_name(repo_id, config_name, split)
    sha = resolve_hf_revision(repo_id, revision=revision, token=token)

    if sha is None:
        # Hub unreachable — serve an existing mirror if we have one.
        if store.exists(name, LATEST):
            resolved = store.resolve_rev(name, LATEST)
            logger.info(f"HF unreachable; serving cached mirror swl://{name}@{resolved}")
            return name, resolved
        raise InternalStoreError(
            f"Cannot mirror {repo_id}: the HuggingFace Hub is unreachable and no "
            f"mirror exists in the store yet."
        )

    if store.exists(name, sha):
        logger.info(f"Mirror hit: swl://{name}@{sha}")
        return name, sha

    logger.info(f"Mirror miss for {repo_id}@{sha} — snapshotting to the internal store")
    _snapshot(store, name, sha, repo_id, split, config_name, token, revision)
    return name, sha


def _snapshot(
    store: InternalStore,
    name: str,
    sha: str,
    repo_id: str,
    split: str,
    config_name: Optional[str],
    token: Optional[str],
    revision: Optional[str],
) -> None:
    """Pull the split from HF, write parquet + a sha-pinned manifest, move latest."""
    from datasets import load_dataset

    # Pin the download to the resolved sha so the bytes match the manifest.
    kwargs = {"split": split, "token": token, "revision": revision or sha}
    if config_name:
        kwargs["name"] = config_name
    dataset = load_dataset(repo_id, **kwargs)

    filename = "data-0.parquet"
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, filename)
        dataset.to_parquet(local)
        data_uri = store.upload_file(local, name, sha, filename)

    manifest = Manifest(
        name=name,
        rev=sha,
        format="parquet",
        data_files=[data_uri],
        num_rows=len(dataset),
        source={
            "hf_repo": repo_id,
            "hf_revision": sha,
            "split": split,
            "config_name": config_name,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store.put_manifest(manifest)
    store.set_latest(name, sha)
    logger.info(f"Mirrored {repo_id}@{sha} -> swl://{name} ({len(dataset)} rows)")
