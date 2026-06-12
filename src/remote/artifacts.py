"""
Artifact stores — the durable handoff between pipeline stages.

Stages never copy files to each other directly; they push named artifact
directories ("adapter", "merged", ...) to a store and later stages pull
them. Because the store is provider-neutral, the train stage can run on
RunPod while the merge stage runs on a different provider, a home server,
or the control plane itself.

The default store is a private HuggingFace Hub repo: every Merlina user
already has an HF token, transfers are resumable, and a pushed adapter is
instantly usable from anywhere.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ArtifactStoreError(RuntimeError):
    """An artifact push/pull failed."""


class ArtifactStore(ABC):
    @abstractmethod
    def push_dir(self, local_dir: Path, name: str) -> None:
        """Upload a directory as artifact ``name`` (overwrites)."""

    @abstractmethod
    def pull_dir(self, name: str, local_dir: Path) -> Path:
        """Download artifact ``name`` into ``local_dir``; returns the dir."""

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Whether artifact ``name`` has been pushed."""

    @abstractmethod
    def describe(self) -> dict:
        """JSON-safe description, used to reconstruct the store on workers."""


class LocalArtifactStore(ArtifactStore):
    """Directory-backed store, for tests and shared-volume handoff."""

    kind = "local"

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def push_dir(self, local_dir: Path, name: str) -> None:
        dest = self.base_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local_dir, dest)
        logger.info("Pushed artifact %r to %s", name, dest)

    def pull_dir(self, name: str, local_dir: Path) -> Path:
        src = self.base_dir / name
        if not src.is_dir():
            raise ArtifactStoreError(f"Artifact {name!r} not found in {self.base_dir}")
        local_dir = Path(local_dir)
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(src, local_dir)
        return local_dir

    def exists(self, name: str) -> bool:
        return (self.base_dir / name).is_dir()

    def describe(self) -> dict:
        return {"kind": self.kind, "base_dir": str(self.base_dir)}


class HFHubArtifactStore(ArtifactStore):
    """
    Artifacts as folders inside one private HF Hub repo per run
    (e.g. ``you/merlina-run-<job_id>`` containing ``adapter/``, ``merged/``).
    """

    kind = "hf_hub"

    def __init__(self, repo_id: str, token: Optional[str] = None, private: bool = True):
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self._repo_ensured = False

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def _ensure_repo(self) -> None:
        if self._repo_ensured:
            return
        self._api().create_repo(self.repo_id, private=self.private, exist_ok=True)
        self._repo_ensured = True

    def push_dir(self, local_dir: Path, name: str) -> None:
        try:
            self._ensure_repo()
            self._api().upload_folder(
                folder_path=str(local_dir),
                path_in_repo=name,
                repo_id=self.repo_id,
                commit_message=f"merlina: push artifact '{name}'",
            )
            logger.info("Pushed artifact %r to hf://%s/%s", name, self.repo_id, name)
        except Exception as e:
            raise ArtifactStoreError(f"Failed to push artifact {name!r} to {self.repo_id}: {e}") from e

    def pull_dir(self, name: str, local_dir: Path) -> Path:
        import tempfile
        try:
            from huggingface_hub import snapshot_download
            local_dir = Path(local_dir)
            with tempfile.TemporaryDirectory(prefix="merlina_artifact_") as tmp:
                snapshot_download(
                    self.repo_id,
                    token=self.token,
                    allow_patterns=[f"{name}/**"],
                    local_dir=tmp,
                )
                src = Path(tmp) / name
                if not src.is_dir():
                    raise ArtifactStoreError(f"Artifact {name!r} not found in {self.repo_id}")
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                shutil.copytree(src, local_dir)
            return local_dir
        except ArtifactStoreError:
            raise
        except Exception as e:
            raise ArtifactStoreError(f"Failed to pull artifact {name!r} from {self.repo_id}: {e}") from e

    def exists(self, name: str) -> bool:
        try:
            files: List[str] = self._api().list_repo_files(self.repo_id)
        except Exception:
            return False
        prefix = name.rstrip("/") + "/"
        return any(f.startswith(prefix) for f in files)

    def describe(self) -> dict:
        # Token intentionally excluded — workers receive it via their own env.
        return {"kind": self.kind, "repo_id": self.repo_id, "private": self.private}


def store_from_description(desc: dict, token: Optional[str] = None) -> ArtifactStore:
    """Reconstruct a store on a worker from its JSON description."""
    kind = desc.get("kind")
    if kind == "local":
        return LocalArtifactStore(Path(desc["base_dir"]))
    if kind == "hf_hub":
        return HFHubArtifactStore(desc["repo_id"], token=token, private=desc.get("private", True))
    raise ArtifactStoreError(f"Unknown artifact store kind: {kind!r}")
