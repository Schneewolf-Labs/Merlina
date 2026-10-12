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

# Substrings that mark a dataset name as private — never to be promoted to the
# external tier, regardless of its manifest flag. A belt-and-braces net for
# sensitive material (the west VPS incident / IR data) even if someone forgets
# to set ``private`` on the manifest. Matched case-insensitively against the
# dataset name.
PRIVATE_NAME_MARKERS = (
    "vps-incident",
    "schneewolf-west",
    "west1",
    "incident",
    "/private/",
    "private/",
)


def is_private_name(name: str) -> bool:
    """True if a dataset name matches a private marker and must stay internal."""
    low = name.lower()
    return any(marker in low for marker in PRIVATE_NAME_MARKERS)


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
    private: bool = False  # if True, never promote to the external tier

    @property
    def hf_revision(self) -> Optional[str]:
        """The upstream HF commit hash this was mirrored from, if any."""
        return self.source.get("hf_revision")

    def is_private(self) -> bool:
        """True if this revision must stay in the internal store — either its
        manifest is flagged private or its name matches a private marker."""
        return self.private or is_private_name(self.name)

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
            private=bool(doc.get("private", False)),
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
        if self.private:
            doc["private"] = True
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
    def from_settings(
        cls, settings: Any, prefix: str = "internal"
    ) -> Optional["InternalStoreConfig"]:
        """
        Build a config from :class:`config.Settings`, or return ``None`` when the
        store is not configured. A missing store is not an error — Merlina simply
        falls back to loading straight from HuggingFace.

        ``prefix`` selects the tier: ``"internal"`` (default) or ``"external"``
        (the legacy ``"s3"`` / ``"r2"`` spellings are accepted). Settings are
        read from the role-named ``storage_<tier>_*`` attributes, falling back
        to the deprecated ``s3_*`` / ``r2_*`` ones on plain settings objects.
        """
        tier = {"s3": "internal", "r2": "external"}.get(prefix, prefix)
        legacy = {"internal": "s3", "external": "r2"}.get(tier, prefix)

        def pick(new: str, old: str):
            value = getattr(settings, f"storage_{tier}_{new}", None)
            return value if value is not None else getattr(settings, f"{legacy}_{old}", None)

        endpoint = pick("endpoint", "endpoint_url")
        bucket = pick("bucket", "dataset_bucket")
        access = pick("access_key", "access_key")
        secret = pick("secret_key", "secret_key")
        if not (endpoint and bucket and access and secret):
            return None
        return cls(
            endpoint_url=str(endpoint),
            bucket=str(bucket),
            access_key=str(access),
            secret_key=str(secret),
            region=str(pick("region", "region") or "auto"),
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

    # ── writes (mirror path) ────────────────────────────────────────────────
    def data_uri(self, name: str, rev: str, filename: str) -> str:
        """The fully-qualified ``s3://`` URI of a shard, for a manifest's data_files."""
        return f"s3://{self._rev_prefix(name, rev)}/{filename}"

    def upload_file(self, local_path: str, name: str, rev: str, filename: str) -> str:
        """Upload a local file as a shard of a revision; returns its s3:// URI."""
        key = f"{self._rev_prefix(name, rev)}/{filename}"
        self.fs.put(local_path, key)
        return f"s3://{key}"

    def download_file(self, data_uri: str, local_path: str) -> None:
        """Download a shard (an ``s3://`` URI from a manifest) to a local file.

        The ``s3://`` scheme is stripped to the fsspec key (``bucket/key``),
        mirroring how :meth:`upload_file` writes it.
        """
        key = data_uri[len("s3://"):] if data_uri.startswith("s3://") else data_uri
        self.fs.get(key, local_path)

    def file_size(self, data_uri: str) -> Optional[int]:
        """Size in bytes of a shard, or ``None`` if it can't be determined
        (used for dry-run cost estimates, so a miss is non-fatal)."""
        key = data_uri[len("s3://"):] if data_uri.startswith("s3://") else data_uri
        try:
            return int(self.fs.size(key))
        except Exception:  # pragma: no cover - best-effort
            return None

    def put_manifest(self, manifest: Manifest) -> None:
        """Write a revision's manifest.json."""
        key = self._manifest_key(manifest.name, manifest.rev)
        with self.fs.open(key, "w") as f:
            json.dump(manifest.to_json(), f)

    def set_latest(self, name: str, rev: str) -> None:
        """Point the dataset's ``latest`` at a concrete revision."""
        with self.fs.open(self._latest_pointer_key(name), "w") as f:
            f.write(rev)
