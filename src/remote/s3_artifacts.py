"""Verified S3-compatible artifacts; one final manifest PUT publishes a generation."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from typing import Protocol

from .artifacts import ArtifactStore, ArtifactStoreError

MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_SECRET = re.compile(r"token|secret|password|credential|api[_-]?key|access[_-]?key|wandb_key", re.I)


def safe_path(value: str) -> str:
    """Require a canonical relative POSIX path, never a filesystem escape."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or "%" in value:
        raise ArtifactStoreError("Invalid artifact path")
    if any(part in ("", ".", "..") for part in value.split("/")) or ":" in value:
        raise ArtifactStoreError("Invalid artifact path")
    return value


def safe_metadata(value):
    """Remove credential-bearing keys and URL credentials from provenance."""
    if isinstance(value, dict):
        return {str(k): safe_metadata(v) for k, v in value.items()
                if not _SECRET.search(str(k))
                and not str(k).lower().endswith(("_path", "_dir"))
                and str(k).lower() != "path"}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(v) for v in value]
    if isinstance(value, str) and (value.startswith(("/", "./", "../", "~/")) or re.match(r"^[A-Za-z]:[\\/]", value)):
        return "[local path omitted]"
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return "[invalid URL omitted]"
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return "[credential-bearing URL omitted]"
    return value


def artifact_metadata(config: dict) -> dict:
    runtime = {"python": platform.python_version()}
    for package in ("merlina", "torch", "transformers", "peft", "grimoire-rl", "s3fs"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            pass
    return safe_metadata({
        "training_config": config,
        "model": {"id": config.get("base_model"), "revision": config.get("model_revision")},
        "datasets": config.get("dataset", {}),
        "runtime": runtime,
    })


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


class StorageClient(Protocol):
    """Narrow transport contract, also implementable by scoped worker gateways."""

    def put_file(self, local_path: str, key: str) -> None: ...
    def get_file(self, key: str, local_path: str) -> None: ...
    def cat_file(self, key: str, start: int | None = None, end: int | None = None) -> bytes: ...
    def pipe_file(self, key: str, data: bytes) -> None: ...


class S3ArtifactStore(ArtifactStore):
    """Immutable data generations, checksummed downloads and atomic publication.

    ``fs`` is an optional s3fs-compatible client for tests. Credentials use the
    dedicated artifact env variables, or the standard AWS credential chain.
    No credentials or config provenance are included in describe().
    """
    kind = "s3"

    def __init__(self, bucket: str, prefix: str = "merlina", *, endpoint_url=None,
                 region_name=None, fs=None, metadata=None):
        if not isinstance(bucket, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", bucket):
            raise ArtifactStoreError("Invalid artifact bucket")
        self.bucket = bucket
        self.prefix = safe_path(prefix) if prefix else ""
        if endpoint_url:
            url = urlsplit(endpoint_url)
            if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ArtifactStoreError("Artifact endpoint must be an HTTP(S) URL without credentials")
        self.endpoint_url, self.region_name = endpoint_url, region_name
        self._fs = fs
        self.metadata = safe_metadata(metadata or {})

    @property
    def fs(self):
        if self._fs is None:
            import s3fs
            from config import settings
            self._fs = s3fs.S3FileSystem(
                key=settings.merlina_artifact_access_key,
                secret=settings.merlina_artifact_secret_key,
                token=settings.merlina_artifact_session_token,
                client_kwargs={k: v for k, v in {
                    "endpoint_url": self.endpoint_url, "region_name": self.region_name,
                }.items() if v},
                config_kwargs={"retries": {"max_attempts": 5, "mode": "standard"}},
            )
        return self._fs

    def _key(self, name, suffix):
        safe_path(name)
        return "/".join(part for part in (self.bucket, self.prefix, name, suffix) if part)

    def _manifest(self, name):
        try:
            # A bounded GET prevents a corrupt manifest exhausting worker memory.
            raw = self.fs.cat_file(self._key(name, "manifest.json"), start=0, end=MAX_MANIFEST_BYTES + 1)
            if len(raw) > MAX_MANIFEST_BYTES:
                raise ArtifactStoreError("Artifact manifest exceeds size limit")
            doc = json.loads(raw)
            if doc.get("version") != MANIFEST_VERSION or doc.get("name") != name:
                raise ArtifactStoreError("Unsupported or mismatched artifact manifest")
            if not re.fullmatch(r"[0-9a-f]{32}", doc["generation"]):
                raise ArtifactStoreError("Invalid artifact generation")
            paths = set()
            if not isinstance(doc["files"], list) or not doc["files"]:
                raise ArtifactStoreError("Empty artifact manifest")
            for entry in doc["files"]:
                path = safe_path(entry["path"])
                if path in paths or type(entry["size"]) is not int or entry["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                    raise ArtifactStoreError("Invalid artifact file entry")
                paths.add(path)
            # Reject file/directory collisions before touching the destination.
            for path in paths:
                if any("/".join(path.split("/")[:i]) in paths for i in range(1, len(path.split("/")))):
                    raise ArtifactStoreError("Conflicting artifact paths")
            return doc
        except ArtifactStoreError:
            raise
        except FileNotFoundError:
            raise
        except Exception:
            raise ArtifactStoreError("Cannot read artifact manifest") from None

    def exists(self, name):
        try:
            self._manifest(name)
            return True
        except FileNotFoundError:
            return False

    def push_dir(self, local_dir: Path, name: str) -> None:
        safe_path(name)
        local_dir = Path(local_dir)
        if not local_dir.is_dir() or local_dir.is_symlink():
            raise ArtifactStoreError("Artifact source must be a real directory")
        generation = uuid.uuid4().hex
        files = []
        try:
            # Snapshot before upload: source mutation cannot publish a mismatched digest.
            with tempfile.TemporaryDirectory(prefix="merlina-artifact-upload-") as temp:
                snapshot = Path(temp) / "snapshot"
                snapshot.mkdir()
                for source in sorted(local_dir.rglob("*")):
                    if source.is_symlink() or (not source.is_file() and not source.is_dir()):
                        raise ArtifactStoreError("Artifact sources cannot contain links or special files")
                    if not source.is_file():
                        continue
                    relative = safe_path(source.relative_to(local_dir).as_posix())
                    target = snapshot / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    size, sha = _digest(target)
                    key = self._key(name, f"generations/{generation}/{relative}")
                    self.fs.put_file(str(target), key)
                    # Verify actual uploaded bytes; multipart ETags are not checksums.
                    verify = Path(temp) / "verify-upload"
                    self.fs.get_file(key, str(verify))
                    if _digest(verify) != (size, sha):
                        raise ArtifactStoreError("Artifact upload integrity check failed")
                    verify.unlink()
                    files.append({"path": relative, "size": size, "sha256": sha})
            if not files:
                raise ArtifactStoreError("Cannot publish an empty artifact")
            manifest = {"version": MANIFEST_VERSION, "name": name, "generation": generation,
                        "created_at": datetime.now(timezone.utc).isoformat(), "files": files,
                        "metadata": safe_metadata(self.metadata),
                        "checkpoints": sorted({f["path"].split("/")[0] for f in files if f["path"].startswith("checkpoint-")})}
            data = json.dumps(manifest, sort_keys=True).encode()
            if len(data) > MAX_MANIFEST_BYTES:
                raise ArtifactStoreError("Artifact manifest exceeds size limit")
            # Only this PUT makes the new generation visible; interrupted retries
            # leave old committed data untouched, even while readers download it.
            self.fs.pipe_file(self._key(name, "manifest.json"), data)
        except ArtifactStoreError:
            raise
        except Exception:
            raise ArtifactStoreError("Artifact upload failed; previous committed generation is preserved") from None

    def pull_dir(self, name: str, local_dir: Path) -> Path:
        safe_path(name)
        local_dir = Path(local_dir)
        if local_dir.is_symlink():
            raise ArtifactStoreError("Artifact destination cannot be a symlink")
        try:
            manifest = self._manifest(name)
            local_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".merlina-artifact-", dir=local_dir.parent) as temp:
                staged = Path(temp) / "download"
                staged.mkdir()
                for entry in manifest["files"]:
                    target = staged / entry["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    key = self._key(name, f"generations/{manifest['generation']}/{entry['path']}")
                    self.fs.get_file(key, str(target))
                    if _digest(target) != (entry["size"], entry["sha256"]):
                        raise ArtifactStoreError("Artifact download integrity check failed")
                # Keep the previous destination until every file is verified.
                backup = Path(temp) / "previous"
                had_previous = local_dir.exists()
                if had_previous:
                    local_dir.rename(backup)
                try:
                    staged.rename(local_dir)
                except Exception:
                    if had_previous:
                        backup.rename(local_dir)
                    raise
            return local_dir
        except ArtifactStoreError:
            raise
        except FileNotFoundError:
            raise ArtifactStoreError("Artifact or committed file not found") from None
        except Exception:
            raise ArtifactStoreError("Artifact download failed") from None

    def describe(self):
        return {"kind": self.kind, "bucket": self.bucket, "prefix": self.prefix,
                "endpoint_url": self.endpoint_url, "region_name": self.region_name}
