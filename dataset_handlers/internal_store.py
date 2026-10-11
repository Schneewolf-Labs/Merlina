"""
Internal dataset store — an S3-compatible object store that Merlina can train
from without depending on the HuggingFace Hub.

The store is backend-agnostic: the same code talks to a self-hosted MinIO
(the private primary) or to Cloudflare R2 (a later, shareable tier), because
both speak the S3 API. Which one is decided entirely by the endpoint URL and
credentials in :class:`config.Settings`.

Datasets are addressed by an ``swl://`` URI::

    swl://athanorlite-dpo            # the 'latest' revision
    swl://athanorlite-dpo@2026-10-08 # a specific, pinned revision

Layout in the bucket::

    datasets/<name>/<rev>/manifest.json   # describes the revision
    datasets/<name>/<rev>/data-*.parquet  # the shards
    datasets/<name>/latest                # text file holding the current <rev>

A manifest records where the data came from (an HF repo + its revision hash, or
our own pipeline) so every training run can prove exactly what it trained on.
Writing manifests (mirror-on-first-use) lands in a later phase; this module is
the read path plus the shared addressing/serialization that the writer reuses.

``s3fs`` is imported lazily so importing this module — and the rest of Merlina —
never requires the S3 stack to be installed; only actually touching the store
does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

SWL_SCHEME = "swl://"
LATEST = "latest"


class InternalStoreError(Exception):
    """Raised when the internal store is misconfigured or a lookup fails."""


def parse_swl_uri(uri: str) -> tuple[str, str]:
    """
    Parse an ``swl://name`` or ``swl://name@rev`` URI into ``(name, rev)``.

    ``rev`` defaults to ``"latest"``. The name is the dataset's logical id and
    may itself contain slashes (``swl://org/name@rev``); only the final ``@``
    separates the revision, so revisions must not contain ``@``.
    """
    if not uri.startswith(SWL_SCHEME):
        raise InternalStoreError(f"Not an swl:// URI: {uri!r}")
    body = uri[len(SWL_SCHEME):].strip("/")
    if not body:
        raise InternalStoreError(f"swl:// URI has no dataset name: {uri!r}")
    if "@" in body:
        name, rev = body.rsplit("@", 1)
        name, rev = name.strip("/"), rev.strip()
        if not name or not rev:
            raise InternalStoreError(f"Malformed swl:// URI: {uri!r}")
        return name, rev
    return body, LATEST


@dataclass(frozen=True)
class Manifest:
    """A single revision of a stored dataset."""

    name: str
    rev: str
    format: str  # "parquet" (the only format the writer emits today)
    data_files: list[str]  # fully-qualified s3://bucket/key shard URIs
    num_rows: Optional[int] = None
    source: dict[str, Any] = field(default_factory=dict)  # provenance (hf repo+revision, etc.)
    created_at: Optional[str] = None

    @property
    def hf_revision(self) -> Optional[str]:
        """The upstream HF commit hash this was mirrored from, if any."""
        return self.source.get("hf_revision")

    @classmethod
    def from_json(cls, name: str, rev: str, doc: dict[str, Any]) -> "Manifest":
        try:
            data_files = list(doc["data_files"])
        except (KeyError, TypeError) as e:
            raise InternalStoreError(
                f"Manifest for swl://{name}@{rev} is missing 'data_files'"
            ) from e
        if not data_files:
            raise InternalStoreError(f"Manifest for swl://{name}@{rev} lists no data files")
        return cls(
            name=name,
            rev=rev,
            format=doc.get("format", "parquet"),
            data_files=data_files,
            num_rows=doc.get("num_rows"),
            source=doc.get("source", {}) or {},
            created_at=doc.get("created_at"),
        )

    def to_json(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "name": self.name,
            "rev": self.rev,
            "format": self.format,
            "data_files": list(self.data_files),
        }
        if self.num_rows is not None:
            doc["num_rows"] = self.num_rows
        if self.source:
            doc["source"] = self.source
        if self.created_at is not None:
            doc["created_at"] = self.created_at
        return doc


@dataclass(frozen=True)
class InternalStoreConfig:
    """Connection settings for the S3-compatible backend."""

    endpoint_url: str
    bucket: str
    access_key: str
    secret_key: str
    region: str = "auto"

    @classmethod
    def from_settings(cls, settings: Any) -> Optional["InternalStoreConfig"]:
        """
        Build a config from :class:`config.Settings`, or return ``None`` when the
        store is not configured. A missing store is not an error — Merlina simply
        falls back to loading straight from HuggingFace.
        """
        endpoint = getattr(settings, "s3_endpoint_url", None)
        bucket = getattr(settings, "s3_dataset_bucket", None)
        access = getattr(settings, "s3_access_key", None)
        secret = getattr(settings, "s3_secret_key", None)
        if not (endpoint and bucket and access and secret):
            return None
        return cls(
            endpoint_url=str(endpoint),
            bucket=str(bucket),
            access_key=str(access),
            secret_key=str(secret),
            region=str(getattr(settings, "s3_region", "auto") or "auto"),
        )


class InternalStore:
    """
    Read access to the internal dataset store.

    The filesystem handle (``s3fs``) is built lazily and cached, so constructing
    a store is cheap and import-safe; the S3 dependency is only required once a
    method actually reaches the backend.
    """

    def __init__(self, config: InternalStoreConfig, *, fs: Any = None) -> None:
        self.config = config
        self._fs = fs  # injectable for tests (an fsspec-like filesystem)

    @property
    def fs(self) -> Any:
        if self._fs is None:
            try:
                import s3fs
            except ImportError as e:  # pragma: no cover - exercised only without the dep
                raise InternalStoreError(
                    "s3fs is required to use the internal dataset store. "
                    "Install it with `pip install s3fs`."
                ) from e
            self._fs = s3fs.S3FileSystem(
                key=self.config.access_key,
                secret=self.config.secret_key,
                client_kwargs={
                    "endpoint_url": self.config.endpoint_url,
                    "region_name": self.config.region,
                },
            )
        return self._fs

    def storage_options(self) -> dict[str, Any]:
        """
        ``storage_options`` for ``datasets.load_dataset(..., storage_options=...)``
        so it can read the parquet shards directly from the backend.
        """
        return {
            "key": self.config.access_key,
            "secret": self.config.secret_key,
            "client_kwargs": {
                "endpoint_url": self.config.endpoint_url,
                "region_name": self.config.region,
            },
        }

    # ── key layout ────────────────────────────────────────────────────────────
    def _rev_prefix(self, name: str, rev: str) -> str:
        return f"{self.config.bucket}/datasets/{name}/{rev}"

    def _manifest_key(self, name: str, rev: str) -> str:
        return f"{self._rev_prefix(name, rev)}/manifest.json"

    def _latest_pointer_key(self, name: str) -> str:
        return f"{self.config.bucket}/datasets/{name}/latest"

    # ── reads ─────────────────────────────────────────────────────────────────
    def resolve_rev(self, name: str, rev: str) -> str:
        """Resolve ``latest`` to the concrete revision it points at; else echo it."""
        if rev != LATEST:
            return rev
        pointer = self._latest_pointer_key(name)
        try:
            with self.fs.open(pointer, "r") as f:
                resolved = f.read().strip()
        except FileNotFoundError as e:
            raise InternalStoreError(
                f"swl://{name} has no 'latest' pointer at {pointer}"
            ) from e
        if not resolved:
            raise InternalStoreError(f"'latest' pointer for swl://{name} is empty")
        return resolved

    def get_manifest(self, name: str, rev: str = LATEST) -> Manifest:
        """Fetch and parse the manifest for a (possibly ``latest``) revision."""
        concrete = self.resolve_rev(name, rev)
        key = self._manifest_key(name, concrete)
        try:
            with self.fs.open(key, "r") as f:
                doc = json.load(f)
        except FileNotFoundError as e:
            raise InternalStoreError(
                f"No manifest for swl://{name}@{concrete} (looked at {key})"
            ) from e
        except json.JSONDecodeError as e:
            raise InternalStoreError(
                f"Manifest for swl://{name}@{concrete} is not valid JSON: {e}"
            ) from e
        return Manifest.from_json(name, concrete, doc)

    def exists(self, name: str, rev: str = LATEST) -> bool:
        """True if a manifest can be resolved and read for this address."""
        try:
            self.get_manifest(name, rev)
            return True
        except InternalStoreError:
            return False
