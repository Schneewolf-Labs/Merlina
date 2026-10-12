"""
Artifact stores — the durable handoff between pipeline stages.

Stages never copy files to each other directly; they push named artifact
directories ("adapter", "merged", ...) to a store and later stages pull
them. Because the store is provider-neutral, the train stage can run on
RunPod while the merge stage runs on a different provider, a home server,
or the control plane itself.

Two durable backends:

- ``S3ArtifactStore`` — a prefix in an S3-API bucket (Cloudflare R2 in
  practice). Used when the server has R2 configured: no HF storage quota
  involved, and the adapter lands in the bucket the operator owns.
- ``HFHubArtifactStore`` — a private HuggingFace Hub repo per run. Needs
  nothing beyond the HF token every Merlina user already has, but is
  subject to the account's HF storage quota.

Credentials never travel in a store's ``describe()`` payload; a worker gets
them through its instance env (see ``ArtifactStore.worker_env``).
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Instance env vars carrying S3 artifact-store credentials to a worker.
ARTIFACT_S3_ACCESS_KEY_ENV = "MERLINA_ARTIFACT_S3_ACCESS_KEY"
ARTIFACT_S3_SECRET_KEY_ENV = "MERLINA_ARTIFACT_S3_SECRET_KEY"


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

    def worker_env(self) -> dict:
        """Credentials a worker needs to rebuild this store, as instance env vars."""
        return {}

    def verify_writable(self) -> None:
        """Cheap pre-flight write check; raises ArtifactStoreError if the store
        can't take a push. Run before renting compute — a store that fails
        only after training loses the trained artifact."""


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


class S3ArtifactStore(ArtifactStore):
    """
    Artifacts as folders under one prefix of an S3-API bucket
    (e.g. ``merlina/artifacts/<job_id>/adapter/``). Backend-agnostic:
    Cloudflare R2, MinIO, or AWS — only the endpoint differs.
    """

    kind = "s3"
    _PROBE = ".merlina-write-probe"

    def __init__(self, bucket: str, prefix: str, *, endpoint_url: Optional[str] = None,
                 access_key: Optional[str] = None, secret_key: Optional[str] = None,
                 region: str = "auto", fs=None):
        self.bucket = bucket.strip("/")
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self._fs = fs

    @classmethod
    def from_r2_settings(cls, settings, job_id: str) -> Optional["S3ArtifactStore"]:
        """The server's R2 bucket, under ``artifacts/<job_id>``; None if R2 isn't configured."""
        bucket = (getattr(settings, "r2_artifact_bucket", None)
                  or getattr(settings, "r2_dataset_bucket", None))
        endpoint = getattr(settings, "r2_endpoint_url", None)
        key = getattr(settings, "r2_access_key", None)
        secret = getattr(settings, "r2_secret_key", None)
        if not (bucket and endpoint and key and secret):
            return None
        return cls(bucket, f"artifacts/{job_id}", endpoint_url=endpoint,
                   access_key=key, secret_key=secret,
                   region=getattr(settings, "r2_region", "auto") or "auto")

    @property
    def fs(self):
        if self._fs is None:
            import s3fs  # lazy: only S3-backed runs need it
            client_kwargs = {"region_name": self.region}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            self._fs = s3fs.S3FileSystem(key=self.access_key, secret=self.secret_key,
                                         client_kwargs=client_kwargs)
        return self._fs

    def _root(self, name: str) -> str:
        return f"{self.bucket}/{self.prefix}/{name.strip('/')}"

    def _keys(self, name: str) -> List[str]:
        root = self._root(name) + "/"
        self.fs.invalidate_cache()
        return [k for k in self.fs.find(root) if k.startswith(root)]

    def uri(self, name: str) -> str:
        return f"s3://{self._root(name)}"

    def push_dir(self, local_dir: Path, name: str) -> None:
        local_dir = Path(local_dir)
        root = self._root(name)
        try:
            stale = self._keys(name)
            if stale:
                self.fs.rm(stale)
            for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
                self.fs.put_file(str(path), f"{root}/{path.relative_to(local_dir).as_posix()}")
            logger.info("Pushed artifact %r to %s", name, self.uri(name))
        except Exception as e:
            raise ArtifactStoreError(f"Failed to push artifact {name!r} to {self.uri(name)}: {e}") from e

    def pull_dir(self, name: str, local_dir: Path) -> Path:
        local_dir = Path(local_dir)
        root = self._root(name) + "/"
        try:
            keys = self._keys(name)
            if not keys:
                raise ArtifactStoreError(f"Artifact {name!r} not found at {self.uri(name)}")
            if local_dir.exists():
                shutil.rmtree(local_dir)
            for key in keys:
                dest = local_dir / key[len(root):]
                dest.parent.mkdir(parents=True, exist_ok=True)
                self.fs.get_file(key, str(dest))
            return local_dir
        except ArtifactStoreError:
            raise
        except Exception as e:
            raise ArtifactStoreError(f"Failed to pull artifact {name!r} from {self.uri(name)}: {e}") from e

    def exists(self, name: str) -> bool:
        try:
            return bool(self._keys(name))
        except Exception:
            return False

    def verify_writable(self) -> None:
        key = f"{self.bucket}/{self.prefix}/{self._PROBE}"
        try:
            self.fs.pipe_file(key, b"ok")
            self.fs.rm(key)
        except Exception as e:
            raise ArtifactStoreError(
                f"Artifact store s3://{self.bucket}/{self.prefix} is not writable: {e}") from e

    def worker_env(self) -> dict:
        env = {}
        if self.access_key:
            env[ARTIFACT_S3_ACCESS_KEY_ENV] = self.access_key
        if self.secret_key:
            env[ARTIFACT_S3_SECRET_KEY_ENV] = self.secret_key
        return env

    def describe(self) -> dict:
        # Credentials intentionally excluded — workers receive them via env.
        return {"kind": self.kind, "bucket": self.bucket, "prefix": self.prefix,
                "endpoint_url": self.endpoint_url, "region": self.region}


def store_from_description(desc: dict, token: Optional[str] = None,
                           env: Optional[dict] = None) -> ArtifactStore:
    """Reconstruct a store on a worker from its JSON description."""
    kind = desc.get("kind")
    if kind == "local":
        return LocalArtifactStore(Path(desc["base_dir"]))
    if kind == "hf_hub":
        return HFHubArtifactStore(desc["repo_id"], token=token, private=desc.get("private", True))
    if kind == "s3":
        env = env if env is not None else os.environ
        key = env.get(ARTIFACT_S3_ACCESS_KEY_ENV)
        secret = env.get(ARTIFACT_S3_SECRET_KEY_ENV)
        if not (key and secret):
            raise ArtifactStoreError(
                f"S3 artifact store needs {ARTIFACT_S3_ACCESS_KEY_ENV} and "
                f"{ARTIFACT_S3_SECRET_KEY_ENV} in the worker environment")
        return S3ArtifactStore(desc["bucket"], desc["prefix"],
                               endpoint_url=desc.get("endpoint_url"),
                               access_key=key, secret_key=secret,
                               region=desc.get("region", "auto"))
    raise ArtifactStoreError(f"Unknown artifact store kind: {kind!r}")
