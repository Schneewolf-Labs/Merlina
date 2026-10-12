"""
Storage tiers — browse, measure, sync and prune Merlina's S3-compatible buckets.

Merlina talks to up to three S3-API object stores, each configured in the
server's ``.env``:

``internal``  (``STORAGE_INTERNAL_*``)  the private, LAN-only store (e.g. MinIO)
``external``  (``STORAGE_EXTERNAL_*``)  the shared tier a rented GPU box can reach (e.g. R2)

Remote-run artifacts (``ARTIFACT_STORE=<tier>``) live in one of those tiers —
under ``ARTIFACT_PREFIX`` in its bucket, or in ``ARTIFACT_BUCKET`` on the same
endpoint. Only a legacy custom location (``MERLINA_ARTIFACT_*`` with its own
endpoint/keys) appears as a separate ``artifacts`` tier.

This module is the backend-agnostic core used by ``merlina storage`` (CLI),
the ``/storage/*`` API and the web UI's Storage section. It deliberately stays
import-light — no torch, no ``datasets`` — so the CLI is fast; ``s3fs`` is only
imported when a tier is actually touched, and the dataset privacy helpers are
only imported when a private -> shared sync needs the guard.

Integrity model for :func:`sync`
--------------------------------
Objects are streamed in fixed-size chunks (never loaded whole, except objects
smaller than one chunk) while a SHA-256 is computed. The digest is stored on the
destination object as S3 user metadata (``x-amz-meta-sha256``):

* objects that fit in one chunk are written with a single PUT that carries the
  metadata directly;
* larger objects are written as a multipart upload (whose metadata must be
  fixed before the first part, i.e. before the digest is known), then the
  digest is attached with a server-side metadata-only ``CopyObject``. No data
  is re-transferred. Providers cap single ``CopyObject`` at 5 GiB, so above
  that the object is kept without metadata (reported as ``meta_missing``) and
  later comparisons fall back to size, or to hashing both sides with ``verify``.

No sidecar files are written: a sidecar would be a second object that can
drift from the first and would pollute every listing.

Privacy
-------
When the source tier is private (``internal``) and the destination is shared,
:func:`sync` refuses — and reports, never copies — any key that

* matches a private marker (``PRIVATE_NAME_MARKERS``, e.g. incident data), or
* lives under ``datasets/<name>/`` where any revision manifest of ``<name>`` is
  flagged ``private`` (or cannot be parsed — the guard fails closed).
* lives inside a remote-training artifact (adapter, merged model, checkpoint,
  staged input) whose manifest does not prove its training data shareable:
  every recorded dataset source must be a public HuggingFace repo or a
  non-private internal dataset present in the source tier. Local-file/upload
  sources, missing dataset metadata and unreadable manifests are refused, as
  is artifact generation data with no manifest at all.

The only override is ``allow_private=True``, exposed solely as the CLI's
``--allow-private`` flag (mirroring ``merlina datasets promote``); the API and
web UI never offer it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Stream granularity. Also the multipart part size on writes, so the largest
# object we can write is 10,000 parts x CHUNK (~320 GiB at 32 MiB).
CHUNK = 32 * 1024 * 1024
# A server-side CopyObject (used to attach metadata after a multipart write)
# is limited to 5 GiB by S3 and S3-compatible providers.
MAX_COPY_OBJECT = 5 * 1024 ** 3
SHA_META = "sha256"
# Parallelism for the metadata HEAD requests made while planning a sync.
_HEAD_WORKERS = 8
# Bucket names per the S3 naming rules (lower-case, 3-63 chars) — for create.
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
# Looser rule for *addressing* existing buckets (legacy names may break the
# modern rules); still no '/' so a bucket can never smuggle a path.
_BUCKET_ACCESS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,254}$")

TIER_ORDER = ("internal", "external", "artifacts")


class StorageTierError(Exception):
    """A tier is unknown/unconfigured, or a request was refused."""


class StorageNotFound(StorageTierError):
    """The requested bucket or object does not exist."""


def validate_bucket_name(name: str) -> Optional[str]:
    """Why ``name`` is not a valid new S3 bucket name, or ``None`` if it is.

    Mirrors the S3 general-purpose bucket rules (and ``storage.js``)."""
    if not isinstance(name, str) or not name:
        return "Bucket name is required."
    if not 3 <= len(name) <= 63:
        return "Bucket names must be 3-63 characters long."
    if not _BUCKET_RE.match(name):
        return ("Use only lower-case letters, digits, '.' and '-', starting and "
                "ending with a letter or digit.")
    if ".." in name or ".-" in name or "-." in name:
        return "Bucket names may not contain '..', '.-' or '-.'."
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", name):
        return "Bucket names may not look like an IP address."
    if name.startswith(("xn--", "sthree-")) or name.endswith(("-s3alias", "--ol-s3", "--x-s3")):
        return "Bucket names may not use a reserved prefix or suffix."
    return None


def human_bytes(n: Optional[float]) -> str:
    if n is None:
        return "unknown"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"  # pragma: no cover


# ── tiers ─────────────────────────────────────────────────────────────────────
@dataclass
class Tier:
    """One bucket on one S3-compatible endpoint. Credentials never leave it:
    they are excluded from ``repr`` and from :meth:`describe`.

    A tier is configured with one *default* bucket; :meth:`with_bucket` returns
    a *view* of another bucket on the same endpoint that keeps the tier's name
    and privacy, so every listing/sync/guard rule applies per bucket.

    Optional admin credentials are used only for bucket-level operations
    (list/create/delete buckets) and for reading buckets other than the
    default one; without them the normal key is used."""

    name: str
    bucket: Optional[str]
    endpoint_url: Optional[str]
    private: bool
    region: str = "auto"
    description: str = ""
    hint_prefix: str = ""  # where interesting things live (e.g. the artifact prefix)
    env_prefix: str = ""   # settings prefix ("S3", "R2", ...) — for hints only
    default_buckets: tuple = ()  # configured buckets on this endpoint (never deleted)
    bucket_hints: dict = field(default_factory=dict)  # bucket -> where to start browsing
    access_key: Optional[str] = field(default=None, repr=False)
    secret_key: Optional[str] = field(default=None, repr=False)
    session_token: Optional[str] = field(default=None, repr=False)
    admin_access_key: Optional[str] = field(default=None, repr=False)
    admin_secret_key: Optional[str] = field(default=None, repr=False)
    fs_override: Any = field(default=None, repr=False)  # injectable for tests
    admin_fs_override: Any = field(default=None, repr=False)  # injectable for tests
    parent: Optional["Tier"] = field(default=None, repr=False, compare=False)
    _fs: Any = field(default=None, repr=False)
    _admin_fs: Any = field(default=None, repr=False)

    @property
    def configured(self) -> bool:
        return bool(self.bucket) and (self.fs_override is not None or bool(self.endpoint_url) or bool(self.access_key))

    @property
    def admin_configured(self) -> bool:
        return self.admin_fs_override is not None or bool(self.admin_access_key and self.admin_secret_key)

    @property
    def default_bucket(self) -> Optional[str]:
        return self.parent.bucket if self.parent is not None else self.bucket

    @property
    def endpoint_host(self) -> Optional[str]:
        if not self.endpoint_url:
            return None
        try:
            parts = urlsplit(self.endpoint_url)
            host = parts.hostname or ""
            return f"{host}:{parts.port}" if parts.port else host
        except ValueError:
            return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "bucket": self.default_bucket,
            "endpoint_host": self.endpoint_host,
            "private": self.private,
            "privacy": "private" if self.private else "shared",
            "configured": self.configured,
            "description": self.description,
            "hint_prefix": self.hint_prefix,
            "default_buckets": list(self.default_buckets),
            "bucket_hints": dict(self.bucket_hints),
            "admin_configured": self.admin_configured,
            "env_prefix": self.env_prefix,
        }

    def _make_fs(self, key: Optional[str], secret: Optional[str], token: Optional[str]) -> Any:
        try:
            import s3fs
        except ImportError as e:  # pragma: no cover - only without the dep
            raise StorageTierError("s3fs is required for storage tiers: pip install s3fs") from e
        client_kwargs = {"endpoint_url": self.endpoint_url}
        if self.region:
            client_kwargs["region_name"] = self.region
        return s3fs.S3FileSystem(
            key=key or None, secret=secret or None, token=token or None,
            client_kwargs={k: v for k, v in client_kwargs.items() if v},
            config_kwargs={"retries": {"max_attempts": 5, "mode": "standard"}},
            skip_instance_cache=True,
        )

    @property
    def fs(self) -> Any:
        if self.fs_override is not None:
            return self.fs_override
        if not self.configured:
            raise StorageTierError(f"Storage tier '{self.name}' is not configured")
        if self._fs is None:
            self._fs = self._make_fs(self.access_key, self.secret_key, self.session_token)
        return self._fs

    @property
    def admin_fs(self) -> Any:
        """Filesystem for bucket-level operations: the admin key if configured,
        else the normal key."""
        if self.admin_fs_override is not None:
            return self.admin_fs_override
        if not (self.admin_access_key and self.admin_secret_key):
            return self.fs
        if not self.configured:
            raise StorageTierError(f"Storage tier '{self.name}' is not configured")
        if self._admin_fs is None:
            self._admin_fs = self._make_fs(self.admin_access_key, self.admin_secret_key, None)
        return self._admin_fs

    def with_bucket(self, bucket: Optional[str]) -> "Tier":
        """A view of ``bucket`` on this tier's endpoint (same name and privacy).
        ``None``/empty or the default bucket returns the tier itself."""
        base = self.parent or self
        if not bucket or bucket == base.bucket:
            return base
        if not _BUCKET_ACCESS_RE.match(bucket) or ".." in bucket:
            raise StorageTierError(f"Invalid bucket name: {bucket!r}")
        fs = base.admin_fs if base.admin_configured else base.fs
        return Tier(
            name=base.name, bucket=bucket, endpoint_url=base.endpoint_url, private=base.private,
            region=base.region, description=base.description, hint_prefix="",
            env_prefix=base.env_prefix, default_buckets=base.default_buckets,
            fs_override=fs, admin_fs_override=base.admin_fs, parent=base,
        )

    def path(self, key: str = "") -> str:
        """The fsspec path (``bucket/key``) for a key in this tier."""
        key = key.lstrip("/")
        return f"{self.bucket}/{key}" if key else str(self.bucket)

    def key_of(self, path: str) -> str:
        """Inverse of :meth:`path`."""
        path = path.lstrip("/")
        b = f"{self.bucket}/"
        return path[len(b):] if path.startswith(b) else ("" if path == self.bucket else path)

    def same_location(self, other: "Tier") -> bool:
        return (
            bool(self.bucket) and self.bucket == other.bucket
            and _norm_endpoint(self.endpoint_url) == _norm_endpoint(other.endpoint_url)
        )


def _norm_endpoint(url: Optional[str]) -> str:
    return (url or "").strip().rstrip("/").lower()


def build_tiers(settings: Any, *, fs_overrides: Optional[dict] = None) -> dict[str, Tier]:
    """Build the tier registry from :class:`config.Settings`.

    ``internal`` and ``external`` (``STORAGE_INTERNAL_*`` / ``STORAGE_EXTERNAL_*``)
    are always present (``configured`` says if they are usable). The artifact
    store (``ARTIFACT_STORE=<tier>``) is folded into its tier: its prefix is
    that tier's browse hint and, when ``ARTIFACT_BUCKET`` differs, that bucket
    is listed as a protected default bucket of the tier. Only a legacy custom
    artifact location (``MERLINA_ARTIFACT_*`` with its own endpoint/keys) is
    shown as a separate ``artifacts`` tier.
    """
    from config import resolve_artifact_location, tier_settings  # lazy: app config
    fs_overrides = fs_overrides or {}

    tiers: dict[str, Tier] = {}
    descriptions = {
        "internal": ("Private store (LAN). Source of truth; private data never leaves it.", True),
        "external": ("Shared store reachable from rented GPU boxes.", False),
    }
    for name, (description, private) in descriptions.items():
        t = tier_settings(settings, name)
        tiers[name] = Tier(
            name=name, bucket=t["bucket"], endpoint_url=t["endpoint"], region=t["region"],
            private=private, description=description, env_prefix=f"STORAGE_{name.upper()}",
            access_key=t["access_key"], secret_key=t["secret_key"],
            admin_access_key=t["admin_access_key"], admin_secret_key=t["admin_secret_key"],
            fs_override=fs_overrides.get(name), admin_fs_override=fs_overrides.get(f"{name}_admin"),
        )

    extra: list[Tier] = []  # configured locations that count as default buckets
    loc = resolve_artifact_location(settings)
    if loc.kind == "s3" and not loc.error and loc.bucket:
        hint = f"{loc.prefix.strip('/')}/" if loc.prefix.strip("/") else ""
        if loc.tier in tiers:
            home = tiers[loc.tier]
            if loc.bucket == home.bucket:
                home.hint_prefix = home.hint_prefix or hint
            else:
                home.bucket_hints[loc.bucket] = hint
                extra.append(Tier(name=home.name, bucket=loc.bucket, endpoint_url=home.endpoint_url,
                                  private=home.private))
        else:
            artifacts = Tier(
                name="artifacts", bucket=loc.bucket, endpoint_url=loc.endpoint_url,
                region=str(loc.region or "auto"), private=False,
                description="Custom artifact location (legacy MERLINA_ARTIFACT_* settings).",
                hint_prefix=hint, env_prefix="MERLINA_ARTIFACT",
                access_key=loc.access_key, secret_key=loc.secret_key, session_token=loc.session_token,
                fs_override=fs_overrides.get("artifacts"),
            )
            tiers["artifacts"] = artifacts
    # Every configured bucket is a "default" (protected from deletion) for each
    # tier on the same endpoint.
    configured = [t for t in (*tiers.values(), *extra) if t.bucket and t.endpoint_url]
    for t in tiers.values():
        same = {c.bucket for c in configured if _norm_endpoint(c.endpoint_url) == _norm_endpoint(t.endpoint_url)}
        if t.bucket:
            same.add(t.bucket)
        t.default_buckets = tuple(sorted(same))
    return tiers


_REGISTRY: Optional[dict[str, Tier]] = None
_REGISTRY_LOCK = threading.Lock()


def get_tiers() -> dict[str, Tier]:
    """The process-wide registry built from app settings (cached)."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            from config import settings  # lazy: app config
            _REGISTRY = build_tiers(settings)
        return _REGISTRY


def set_tiers(tiers: Optional[dict[str, Tier]]) -> None:
    """Replace (or with ``None`` reset) the cached registry — for tests."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = tiers


def get_tier(name: str, tiers: Optional[dict[str, Tier]] = None) -> Tier:
    tiers = tiers if tiers is not None else get_tiers()
    tier = tiers.get((name or "").lower())
    if tier is None:
        raise StorageTierError(f"Unknown storage tier '{name}' (have: {', '.join(tiers)})")
    if not tier.configured:
        raise StorageTierError(f"Storage tier '{tier.name}' is not configured")
    return tier


# ── keys & listings ───────────────────────────────────────────────────────────
def norm_prefix(prefix: Optional[str]) -> str:
    """Normalize a user-supplied prefix: no leading slash, no '..' segments."""
    p = (prefix or "").strip().lstrip("/")
    if "\x00" in p or any(seg == ".." for seg in p.split("/")):
        raise StorageTierError(f"Invalid prefix: {prefix!r}")
    return p


def _mtime(info: dict) -> Optional[str]:
    v = info.get("LastModified") or info.get("mtime") or info.get("last_modified")
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v).isoformat()
    return str(v)


LS_SORTS = ("name", "size", "mtime")


def ls(tier: Tier, prefix: str = "", *, search: str = "", sort: str = "name",
       order: str = "asc", offset: int = 0, limit: int = 1000) -> dict:
    """Delimiter-style listing of one level under ``prefix``.

    ``search`` filters names (case-insensitive substring) within this level;
    ``sort`` is name/size/mtime (folders always come first, by name);
    ``offset``/``limit`` page through the combined folders+objects list."""
    prefix = norm_prefix(prefix)
    base = prefix.rstrip("/")
    if sort not in LS_SORTS:
        raise StorageTierError(f"sort must be one of {', '.join(LS_SORTS)}")
    offset, limit = max(0, int(offset)), max(1, min(int(limit), 5000))
    try:
        entries = tier.fs.ls(tier.path(base), detail=True, refresh=True)
    except FileNotFoundError:
        entries = []
    dirs, objects = [], []
    for e in entries:
        key = tier.key_of(e["name"]).rstrip("/")
        if not key or key == base:
            # s3fs lists an exact-file path as itself; show it as an object.
            if e.get("type") != "directory" and key == base and base:
                objects.append({"key": key, "name": key.rsplit("/", 1)[-1],
                                "size": int(e.get("size") or 0), "mtime": _mtime(e)})
            continue
        name = key.rsplit("/", 1)[-1]
        if e.get("type") == "directory":
            dirs.append({"prefix": key + "/", "name": name + "/"})
        else:
            objects.append({"key": key, "name": name, "size": int(e.get("size") or 0),
                            "mtime": _mtime(e)})
    needle = (search or "").strip().lower()
    if needle:
        dirs = [d for d in dirs if needle in d["name"].lower()]
        objects = [o for o in objects if needle in o["name"].lower()]
    reverse = order == "desc"
    dirs.sort(key=lambda d: d["prefix"], reverse=reverse and sort == "name")
    if sort == "name":
        objects.sort(key=lambda o: o["key"], reverse=reverse)
    elif sort == "size":
        objects.sort(key=lambda o: (o["size"], o["key"]), reverse=reverse)
    else:
        objects.sort(key=lambda o: (o["mtime"] or "", o["key"]), reverse=reverse)
    combined = [("d", d) for d in dirs] + [("o", o) for o in objects]
    page = combined[offset:offset + limit]
    return {
        "tier": tier.name, "bucket": tier.bucket, "prefix": f"{base}/" if base else "",
        "dirs": [x for k, x in page if k == "d"], "objects": [x for k, x in page if k == "o"],
        "total_dirs": len(dirs), "total_objects": len(objects), "total": len(combined),
        "offset": offset, "limit": limit, "has_more": offset + limit < len(combined),
        "truncated": offset + limit < len(combined), "object_count": len(objects),
        "search": needle, "sort": sort, "order": "desc" if reverse else "asc",
    }


def iter_objects(tier: Tier, prefix: str = "") -> dict[str, dict]:
    """All objects under ``prefix`` (recursive), as ``{key: {size, mtime}}``.

    ``prefix`` is treated as a directory (``a/b`` matches ``a/b`` itself and
    ``a/b/...`` but not ``a/bc``)."""
    prefix = norm_prefix(prefix).rstrip("/")
    try:
        found = tier.fs.find(tier.path(prefix), detail=True)
    except FileNotFoundError:
        return {}
    out = {}
    for path, info in found.items():
        if info.get("type") == "directory":
            continue
        key = tier.key_of(path)
        if prefix and not (key == prefix or key.startswith(prefix + "/")):
            continue
        out[key] = {"size": int(info.get("size") or 0), "mtime": _mtime(info)}
    return out


def du(tier: Tier, prefix: str = "") -> dict:
    objs = iter_objects(tier, prefix)
    total = sum(o["size"] for o in objs.values())
    p = norm_prefix(prefix)
    return {"tier": tier.name, "bucket": tier.bucket, "prefix": p, "objects": len(objs),
            "bytes": total, "human": human_bytes(total)}


# ── buckets ───────────────────────────────────────────────────────────────────
def _is_access_denied(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    text = f"{type(exc).__name__} {exc}"
    return any(s in text for s in ("AccessDenied", "Access Denied", "Forbidden", "403", "Unauthorized"))


def admin_hint(tier: Tier, action: str) -> str:
    """Which (optional) admin key would allow a bucket-level ``action``."""
    p = tier.env_prefix or tier.name.upper()
    if tier.admin_configured:
        return (f"The admin key ({p}_ADMIN_ACCESS_KEY) is configured but was refused — "
                f"grant it permission to {action}. See docs/storage-tiers.md.")
    host = (tier.endpoint_host or "").lower()
    if host.endswith("r2.cloudflarestorage.com"):
        need = "an R2 account API token with Admin Read & Write"
    elif host.endswith("amazonaws.com"):
        need = ("an IAM key allowing s3:ListAllMyBuckets, s3:CreateBucket, "
                "s3:DeleteBucket and s3:GetBucketLocation")
    else:
        need = ("e.g. a MinIO service account whose policy allows s3:ListAllMyBuckets, "
                "s3:CreateBucket, s3:DeleteBucket and s3:GetBucketLocation")
    return (f"Set {p}_ADMIN_ACCESS_KEY / {p}_ADMIN_SECRET_KEY to a key that may {action} "
            f"({need}). See docs/storage-tiers.md.")


def _bucket_entry(tier: Tier, name: str, created: Any = None) -> dict:
    return {
        "name": name,
        "creation_date": created.isoformat() if isinstance(created, datetime) else (str(created) if created else None),
        "default": name in tier.default_buckets,
        "is_tier_bucket": name == tier.default_bucket,
    }


def list_buckets(tier: Tier) -> dict:
    """List the buckets visible to this tier's (admin, else normal) key.

    Scoped keys usually can't list buckets; then the configured bucket(s) are
    returned anyway with ``ok=False`` and a hint naming the key that would
    allow it — reported, never raised."""
    try:
        resp = tier.admin_fs.call_s3("list_buckets")
        listed = {b.get("Name"): b.get("CreationDate") for b in resp.get("Buckets", []) if b.get("Name")}
        buckets = [_bucket_entry(tier, n, c) for n, c in sorted(listed.items())]
        # A configured bucket the key can't see (e.g. on a different account) still shows.
        for d in tier.default_buckets:
            if d not in listed:
                buckets.append(_bucket_entry(tier, d))
        return {"tier": tier.name, "ok": True, "buckets": buckets, "message": "", "hint": "",
                "admin_configured": tier.admin_configured}
    except Exception as e:  # noqa: BLE001 - provider errors vary
        buckets = [_bucket_entry(tier, d) for d in tier.default_buckets]
        if _is_access_denied(e):
            msg = (f"The '{tier.name}' credentials may not list buckets (access denied). "
                   f"That is normal for a key scoped to one bucket; this tier uses "
                   f"'{tier.default_bucket}'.")
            hint = admin_hint(tier, "list buckets")
        else:
            msg = f"Could not list buckets on '{tier.name}': {type(e).__name__}: {e}"
            hint = ""
        return {"tier": tier.name, "ok": False, "buckets": buckets, "message": msg, "hint": hint,
                "admin_configured": tier.admin_configured}


def bucket_info(tier: Tier, name: str) -> dict:
    """Region, creation date and flags for one bucket (best effort: each field
    is ``None`` when the key may not read it)."""
    view = tier.with_bucket(name)
    fs = (tier.parent or tier).admin_fs
    out = {"tier": tier.name, "bucket": view.bucket, "private": tier.private,
           "default": view.bucket in tier.default_buckets, "region": None,
           "creation_date": None, "exists": None, "notes": []}
    try:
        fs.call_s3("head_bucket", Bucket=view.bucket)
        out["exists"] = True
    except Exception as e:  # noqa: BLE001
        text = f"{type(e).__name__} {e}"
        if isinstance(e, FileNotFoundError) or "404" in text or "NoSuchBucket" in text or "Not Found" in text:
            raise StorageNotFound(f"Bucket '{view.bucket}' does not exist") from None
        out["notes"].append("bucket existence could not be checked (access denied)"
                            if _is_access_denied(e) else f"head_bucket failed: {type(e).__name__}")
    try:
        loc = fs.call_s3("get_bucket_location", Bucket=view.bucket).get("LocationConstraint")
        out["region"] = loc or "us-east-1 (default)"
    except Exception as e:  # noqa: BLE001
        out["notes"].append("region unavailable (access denied)" if _is_access_denied(e)
                            else f"region unavailable ({type(e).__name__})")
    listing = list_buckets(tier)
    for b in listing["buckets"]:
        if b["name"] == view.bucket:
            out["creation_date"] = b["creation_date"]
    if not listing["ok"]:
        out["notes"].append("creation date unavailable (this key may not list buckets)")
    return out


def make_bucket(tier: Tier, name: str) -> dict:
    """Create a bucket on this tier's endpoint (admin key if configured).
    Reports (doesn't raise) when the key isn't allowed to."""
    name = (name or "").strip()
    problem = validate_bucket_name(name)
    if problem:
        return {"tier": tier.name, "ok": False, "bucket": name,
                "message": f"Invalid bucket name {name!r}: {problem}", "hint": ""}
    params: dict[str, Any] = {"Bucket": name}
    if tier.region and tier.region not in ("auto", "us-east-1"):
        params["CreateBucketConfiguration"] = {"LocationConstraint": tier.region}
    fs = (tier.parent or tier).admin_fs
    try:
        fs.call_s3("create_bucket", **params)
        try:
            fs.invalidate_cache("")
        except Exception:  # noqa: BLE001
            pass
        return {"tier": tier.name, "ok": True, "bucket": name, "message": f"Created bucket '{name}'.", "hint": ""}
    except Exception as e:  # noqa: BLE001
        text = f"{type(e).__name__} {e}"
        hint = ""
        if "BucketAlreadyOwnedByYou" in text or "BucketAlreadyExists" in text or isinstance(e, FileExistsError):
            msg = f"Bucket '{name}' already exists."
        elif _is_access_denied(e):
            msg = f"The '{tier.name}' credentials may not create buckets (access denied)."
            hint = admin_hint(tier, "create buckets")
        else:
            msg = f"Could not create bucket '{name}': {type(e).__name__}: {e}"
        return {"tier": tier.name, "ok": False, "bucket": name, "message": msg, "hint": hint}


def delete_bucket(tier: Tier, name: str, *, confirm_name: Optional[str] = None,
                  empty_first: bool = False, dry_run: bool = False) -> dict:
    """Delete a bucket. Refuses configured (dataset/artifact) buckets, requires
    the bucket name typed back as ``confirm_name``, and refuses a non-empty
    bucket unless ``empty_first`` (which deletes its objects first).

    ``dry_run`` only reports the object count and size. Access-denied is
    reported (``ok=False``) rather than raised."""
    base = tier.parent or tier
    if name in base.default_buckets:
        raise StorageTierError(f"Refusing to delete '{name}': it is a configured Merlina bucket")
    view = base.with_bucket(name)
    if view is base:  # pragma: no cover - default bucket is always in default_buckets
        raise StorageTierError("Refusing to delete the tier's configured bucket")
    try:
        objs = iter_objects(view, "")
    except Exception as e:  # noqa: BLE001
        if _is_access_denied(e):
            return {"tier": tier.name, "bucket": name, "ok": False, "objects": None, "bytes": None,
                    "human": "unknown", "emptied": False, "deleted": False,
                    "message": f"The '{tier.name}' credentials may not read bucket '{name}'.",
                    "hint": admin_hint(tier, "read and delete buckets")}
        raise
    total = sum(o["size"] for o in objs.values())
    out = {"tier": tier.name, "bucket": name, "ok": True, "objects": len(objs), "bytes": total,
           "human": human_bytes(total), "emptied": False, "deleted": False, "message": "", "hint": ""}
    if dry_run:
        return out
    if confirm_name != name:
        raise StorageTierError("Type the bucket name exactly to confirm deletion")
    if objs and not empty_first:
        raise StorageTierError(
            f"Bucket '{name}' is not empty ({len(objs)} object(s), {human_bytes(total)}); "
            f"empty it first (empty_first) to delete it")
    try:
        if objs:
            _rm_keys(view, sorted(objs))
            out["emptied"] = True
        base.admin_fs.call_s3("delete_bucket", Bucket=name)
        try:
            base.admin_fs.invalidate_cache("")
        except Exception:  # noqa: BLE001
            pass
        out["deleted"] = True
        out["message"] = f"Deleted bucket '{name}'."
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        if _is_access_denied(e):
            out["message"] = f"The '{tier.name}' credentials may not delete bucket '{name}' (access denied)."
            out["hint"] = admin_hint(tier, "delete buckets")
        else:
            out["message"] = f"Could not delete bucket '{name}': {type(e).__name__}: {e}"
    return out


# ── objects: info, preview, download ──────────────────────────────────────────
PREVIEW_MAX = 256 * 1024
PARQUET_ROWS = 20
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
TEXT_EXTS = {
    ".json", ".jsonl", ".md", ".markdown", ".txt", ".jinja", ".j2", ".yaml", ".yml",
    ".csv", ".tsv", ".py", ".toml", ".cfg", ".ini", ".log", ".xml", ".html", ".sh",
    ".model_card", ".gitattributes", ".gitignore",
}


def _ext(key: str) -> str:
    name = key.rsplit("/", 1)[-1].lower()
    return "." + name.rsplit(".", 1)[-1] if "." in name.lstrip(".") else ""


def _object_key(key: Optional[str]) -> str:
    k = norm_prefix(key)
    if not k or k.endswith("/"):
        raise StorageTierError("An object key is required")
    return k


def object_info(tier: Tier, key: str) -> dict:
    """Size, mtime, content type, ETag and user metadata (incl. our sha256)."""
    key = _object_key(key)
    try:
        # refresh: a listing fills s3fs' cache with entries that lack ContentType.
        info = tier.fs.info(tier.path(key), refresh=True)
    except FileNotFoundError:
        raise StorageNotFound(f"No object '{key}' in {tier.name}:{tier.bucket}") from None
    if info.get("type") == "directory":
        raise StorageTierError(f"'{key}' is a folder, not an object")
    return {
        "tier": tier.name, "bucket": tier.bucket, "key": key, "name": key.rsplit("/", 1)[-1],
        "size": int(info.get("size") or 0), "human": human_bytes(int(info.get("size") or 0)),
        "mtime": _mtime(info), "content_type": info.get("ContentType"),
        "etag": (info.get("ETag") or "").strip('"') or None,
        "metadata": _meta(tier, key), "preview_kind": preview_kind(key),
    }


def preview_kind(key: str) -> str:
    ext = _ext(key)
    if ext == ".parquet":
        return "parquet"
    if ext == ".safetensors":
        return "safetensors"
    if ext in TEXT_EXTS or ext == "":
        return "text"
    return "binary"


def _jsonable(v: Any, limit: int = 300) -> Any:
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + "…"
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    try:
        text = json.dumps(v, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        text = str(v)
    return text if len(text) <= limit else text[:limit] + "…"


def preview(tier: Tier, key: str, *, max_bytes: int = PREVIEW_MAX) -> dict:
    """An inline preview of an object, reading as little as possible:

    * text-like files: the first ``max_bytes`` (JSON pretty-printed when whole);
    * ``.safetensors``: only the header (two range reads) — tensor names,
      dtypes, shapes and the total parameter count;
    * ``.parquet``: schema + the first rows, via pyarrow over range reads
      (footer + the first pages), never the whole file;
    * anything else: no preview (download it)."""
    key = _object_key(key)
    path = tier.path(key)
    kind = preview_kind(key)
    out: dict[str, Any] = {"tier": tier.name, "bucket": tier.bucket, "key": key, "kind": kind}
    try:
        if kind == "text":
            raw = tier.fs.cat_file(path, start=0, end=max_bytes + 1)
            truncated = len(raw) > max_bytes
            raw = raw[:max_bytes]
            if b"\x00" in raw[:8192]:
                out.update(kind="binary", message="Looks like a binary file; download it instead.")
                return out
            text = raw.decode("utf-8", errors="replace")
            fmt = _ext(key).lstrip(".") or "text"
            if fmt == "json" and not truncated:
                try:
                    text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
                except ValueError:
                    pass
            out.update(format=fmt, text=text, truncated=truncated, bytes_shown=len(raw))
        elif kind == "safetensors":
            out.update(_safetensors_header(tier, path))
        elif kind == "parquet":
            out.update(_parquet_preview(tier, path))
        else:
            out["message"] = "No inline preview for this file type; download it instead."
    except FileNotFoundError:
        raise StorageNotFound(f"No object '{key}' in {tier.name}:{tier.bucket}") from None
    return out


def _safetensors_header(tier: Tier, path: str) -> dict:
    head = tier.fs.cat_file(path, start=0, end=8)
    if len(head) < 8:
        raise StorageTierError("Not a safetensors file (too short)")
    n = int.from_bytes(head, "little")
    if n <= 0 or n > MAX_SAFETENSORS_HEADER:
        raise StorageTierError(f"Implausible safetensors header length ({n} bytes)")
    doc = json.loads(tier.fs.cat_file(path, start=8, end=8 + n))
    meta = doc.pop("__metadata__", None)
    tensors, dtypes, total = [], {}, 0
    for name in sorted(doc):
        t = doc[name] or {}
        shape = [int(x) for x in t.get("shape") or []]
        count = 1
        for d in shape:
            count *= d
        total += count
        dtype = str(t.get("dtype"))
        dtypes[dtype] = dtypes.get(dtype, 0) + count
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "params": count})
    return {"header_bytes": n, "metadata": meta, "tensor_count": len(tensors),
            "total_params": total, "dtypes": dtypes, "tensors": tensors[:2000],
            "tensors_truncated": len(tensors) > 2000}


def _parquet_preview(tier: Tier, path: str, rows: int = PARQUET_ROWS) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow ships with `datasets`
        return {"message": "pyarrow is not installed; download the file to inspect it."}
    with tier.fs.open(path, "rb", block_size=1024 * 1024) as f:
        # buffer_size: stream pages instead of pre-loading whole column chunks.
        pf = pq.ParquetFile(f, buffer_size=1024 * 1024, pre_buffer=False)
        schema = [{"name": fld.name, "type": str(fld.type)} for fld in pf.schema_arrow]
        md = pf.metadata
        first = []
        if md.num_rows:
            batch = next(pf.iter_batches(batch_size=rows), None)
            if batch is not None:
                first = [{k: _jsonable(v) for k, v in r.items()} for r in batch.to_pylist()[:rows]]
    return {"schema": schema, "num_rows": md.num_rows, "num_row_groups": md.num_row_groups,
            "num_columns": md.num_columns, "rows": first}


def open_download(tier: Tier, key: str, chunk: int = 8 * 1024 * 1024):
    """``(info, iterator)`` streaming an object in chunks (for downloads)."""
    info = object_info(tier, key)

    def gen():
        with tier.fs.open(tier.path(info["key"]), "rb", block_size=chunk) as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                yield b

    return info, gen()


# ── privacy guard ─────────────────────────────────────────────────────────────
_GENERATION_RE = re.compile(r"(^|/)generations/[0-9a-f]{32}/")
# Artifact manifests are bounded by the artifact store; read no more than this.
_MAX_ARTIFACT_MANIFEST = 16 * 1024 * 1024
_KNOWN_SOURCE_TYPES = {"huggingface", "internal", "local_file", "upload"}


def _is_artifact_manifest(doc: Any) -> bool:
    """True for a manifest in the artifact-store format (src/remote/s3_artifacts.py)."""
    return isinstance(doc, dict) and "files" in doc and ("generation" in doc or "metadata" in doc)


def _dataset_sources(dataset_cfg: Any) -> list[dict]:
    """``source``, ``additional_sources[]`` and ``eval_source`` of a dataset config."""
    if not isinstance(dataset_cfg, dict):
        return []
    out = []
    if isinstance(dataset_cfg.get("source"), dict):
        out.append(dataset_cfg["source"])
    for extra in dataset_cfg.get("additional_sources") or []:
        if isinstance(extra, dict):
            out.append(extra)
    if isinstance(dataset_cfg.get("eval_source"), dict):
        out.append(dataset_cfg["eval_source"])
    return out


class PrivacyGuard:
    """Decides which keys of a private tier must not reach a shared tier.

    Three rules, all failing closed:

    * a path matching a private marker;
    * a key under ``datasets/<name>/`` when any revision manifest of ``<name>`` is
      private or unreadable;
    * a key inside a remote-training artifact (the nearest ancestor directory
      holding an artifact-store ``manifest.json``: adapters, merged models,
      checkpoints, staged inputs) unless that artifact's recorded training data
      is provably shareable — every dataset source a public HuggingFace repo, or
      an internal dataset that exists here and is not private. Local-file and
      upload sources, missing dataset metadata and unreadable manifests are
      refused. Artifact generation data with no manifest is refused too.
    """

    def __init__(self, tier: Tier):
        from dataset_handlers.internal_store import is_private_name  # lazy (pulls `datasets`)
        self.tier = tier
        # swl:// datasets live in the tier's configured bucket, whichever bucket
        # is being synced.
        base = tier.parent or tier
        self._swl = self if base is tier else PrivacyGuard(base)
        self._is_private_name = is_private_name
        self._groups_scanned: set[str] = set()
        self._private_dirs: set[str] = set()
        self._dataset_dirs: set[str] = set()
        # dir -> parsed manifest.json, the string "unreadable", or None (absent)
        self._manifests: dict[str, Any] = {}
        self._artifact_verdicts: dict[str, Optional[str]] = {}

    # ── datasets/<name>/ ──
    def _scan_group(self, group: str) -> None:
        """Read every manifest under ``datasets/<group>`` (``group`` is the first
        name segment; names may contain slashes) and record private dataset dirs."""
        if group in self._groups_scanned:
            return
        self._groups_scanned.add(group)
        for key in iter_objects(self.tier, f"datasets/{group}"):
            if not key.endswith("/manifest.json"):
                continue
            rev_dir = key[: -len("/manifest.json")]
            dataset_dir = rev_dir.rsplit("/", 1)[0] if "/" in rev_dir[len("datasets/"):] else rev_dir
            self._dataset_dirs.add(dataset_dir + "/")
            private = True  # fail closed on unreadable/invalid manifests
            try:
                raw = self.tier.fs.cat_file(self.tier.path(key))
                doc = json.loads(raw)
                name = dataset_dir[len("datasets/"):]
                private = bool(doc.get("private", False)) or self._is_private_name(str(doc.get("name") or name))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"privacy guard: unreadable manifest {key} ({type(e).__name__}); treating as private")
            if private:
                self._private_dirs.add(dataset_dir + "/")

    def _dataset_refusal(self, key: str) -> Optional[str]:
        rest = key[len("datasets/"):]
        if "/" not in rest:
            return None
        self._scan_group(rest.split("/", 1)[0])
        for d in self._private_dirs:
            if key.startswith(d):
                return f"dataset manifest is private ({d})"
        return None

    # ── remote-training artifacts ──
    def _manifest_in(self, d: str) -> Any:
        if d not in self._manifests:
            path = self.tier.path(f"{d}/manifest.json" if d else "manifest.json")
            try:
                raw = self.tier.fs.cat_file(path, start=0, end=_MAX_ARTIFACT_MANIFEST + 1)
                if len(raw) > _MAX_ARTIFACT_MANIFEST:
                    raise ValueError("manifest too large")
                self._manifests[d] = json.loads(raw)
            except FileNotFoundError:
                self._manifests[d] = None
            except Exception as e:  # noqa: BLE001
                logger.warning(f"privacy guard: unreadable {d}/manifest.json ({type(e).__name__})")
                self._manifests[d] = "unreadable"
        return self._manifests[d]

    def _artifact_root(self, key: str) -> Optional[tuple[str, Any]]:
        """Nearest ancestor dir whose manifest.json is an artifact manifest (or
        is unreadable, which fails closed). ``None`` when there is none."""
        parts = key.split("/")[:-1]
        for i in range(len(parts), 0, -1):
            d = "/".join(parts[:i])
            doc = self._manifest_in(d)
            if doc == "unreadable" or _is_artifact_manifest(doc):
                return d, doc
        return None

    def _internal_dataset_refusal(self, uri: str) -> Optional[str]:
        from dataset_handlers.internal_store import InternalStoreError, parse_swl_uri
        try:
            name, _rev = parse_swl_uri(uri)
        except InternalStoreError:
            return f"internal dataset reference {uri!r} is malformed"
        if self._is_private_name(name):
            return f"trained on internal dataset {name!r}, which matches a private marker"
        store = self._swl
        store._scan_group(name.split("/", 1)[0])
        d = f"datasets/{name}/"
        if d not in store._dataset_dirs:
            return f"trained on internal dataset {name!r}, which is not in this tier (cannot verify it is shareable)"
        if d in store._private_dirs:
            return f"trained on private internal dataset {name!r}"
        return None

    def _source_refusal(self, source: dict) -> Optional[str]:
        kind = str(source.get("source_type") or "huggingface")
        for field_name in ("repo_id", "name", "uri", "path", "file_path", "dataset_id"):
            v = source.get(field_name)
            if isinstance(v, str) and v and self._is_private_name(v):
                return f"trained on dataset {v!r}, which matches a private marker"
        if kind in ("local_file", "upload"):
            return (f"trained on a {kind} dataset; its contents cannot be verified "
                    f"from the artifact, so it is treated as private")
        uri = source.get("uri")
        if kind == "internal" or (isinstance(uri, str) and uri.startswith("swl://")):
            if not isinstance(uri, str) or not uri:
                return "trained on an internal dataset with no recorded uri"
            return self._internal_dataset_refusal(uri)
        if kind not in _KNOWN_SOURCE_TYPES:
            return f"trained on an unknown dataset source type {kind!r}"
        if not source.get("repo_id"):
            return "trained on a HuggingFace source with no recorded repo_id"
        return None

    def _artifact_refusal(self, root: str, doc: Any) -> Optional[str]:
        if root in self._artifact_verdicts:
            return self._artifact_verdicts[root]
        verdict = self._judge_artifact(root, doc)
        self._artifact_verdicts[root] = verdict
        return verdict

    def _judge_artifact(self, root: str, doc: Any) -> Optional[str]:
        where = f"artifact {root}/"
        if doc == "unreadable":
            return f"{where}: manifest unreadable (cannot verify its training data)"
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
        if isinstance(model.get("id"), str) and self._is_private_name(model["id"]):
            return f"{where}: base model {model['id']!r} matches a private marker"
        training_config = metadata.get("training_config") if isinstance(metadata.get("training_config"), dict) else {}
        sources = _dataset_sources(metadata.get("datasets")) + _dataset_sources(training_config.get("dataset"))
        if not sources:
            return f"{where}: manifest records no dataset metadata (cannot verify its training data)"
        for source in sources:
            reason = self._source_refusal(source)
            if reason:
                return f"{where}: {reason}"
        return None

    # ── entry point ──
    def check(self, key: str) -> Optional[str]:
        """The refusal reason for ``key``, or ``None`` if it may be shared."""
        if self._is_private_name(key):
            return "path matches a private marker"
        if key.startswith("datasets/"):
            reason = self._dataset_refusal(key)
            if reason:
                return reason
        found = self._artifact_root(key)
        if found is not None:
            return self._artifact_refusal(*found)
        if _GENERATION_RE.search(key):
            return "artifact generation data without a manifest (cannot verify its training data)"
        return None


# ── sync ──────────────────────────────────────────────────────────────────────
ProgressCB = Callable[[dict], None]


@dataclass
class SyncResult:
    src: str
    dst: str
    prefix: str
    dry_run: bool
    move: bool
    verify: bool
    overwrite: bool
    to_copy: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    copied: list[dict] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    meta_missing: list[str] = field(default_factory=list)
    guarded: bool = False  # src private -> dst shared
    src_bucket: Optional[str] = None
    dst_bucket: Optional[str] = None
    move_aborted: str = ""

    @property
    def bytes_to_copy(self) -> int:
        return sum(i["size"] for i in self.to_copy)

    @property
    def bytes_skipped(self) -> int:
        return sum(i["size"] for i in self.skipped)

    @property
    def bytes_copied(self) -> int:
        return sum(i["size"] for i in self.copied)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self, max_items: int = 500) -> dict:
        def cap(items):
            return items[:max_items]
        return {
            "src": self.src, "dst": self.dst, "src_bucket": self.src_bucket,
            "dst_bucket": self.dst_bucket, "prefix": self.prefix,
            "dry_run": self.dry_run, "move": self.move, "verify": self.verify,
            "overwrite": self.overwrite, "guarded": self.guarded, "ok": self.ok,
            "counts": {
                "to_copy": len(self.to_copy), "skipped": len(self.skipped),
                "refused": len(self.refused), "errors": len(self.errors),
                "copied": len(self.copied), "deleted": len(self.deleted),
                "meta_missing": len(self.meta_missing),
            },
            "bytes": {
                "to_copy": self.bytes_to_copy, "to_copy_human": human_bytes(self.bytes_to_copy),
                "skipped": self.bytes_skipped, "skipped_human": human_bytes(self.bytes_skipped),
                "copied": self.bytes_copied, "copied_human": human_bytes(self.bytes_copied),
            },
            "to_copy": cap(self.to_copy), "skipped": cap(self.skipped),
            "refused": cap(self.refused), "errors": cap(self.errors),
            "copied": cap(self.copied), "deleted": cap(self.deleted),
            "meta_missing": cap(self.meta_missing), "move_aborted": self.move_aborted,
            "truncated": any(len(x) > max_items for x in (
                self.to_copy, self.skipped, self.refused, self.errors, self.copied, self.deleted)),
        }


def _meta(tier: Tier, key: str) -> dict:
    try:
        return {str(k).lower(): v for k, v in (tier.fs.metadata(tier.path(key)) or {}).items()}
    except Exception:  # noqa: BLE001 - missing metadata is not an error
        return {}


def _hash_object(tier: Tier, key: str, cancel: Optional[threading.Event] = None) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with tier.fs.open(tier.path(key), "rb", block_size=CHUNK) as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise StorageTierError("cancelled")
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    return n, h.hexdigest()


def _copy_object(src: Tier, dst: Tier, key: str, size: int, *,
                 on_bytes: Callable[[int], None], cancel: Optional[threading.Event]) -> tuple[str, bool]:
    """Stream one object src -> dst, hashing as it goes. Returns
    ``(sha256, metadata_attached)``."""
    h = hashlib.sha256()
    spath, dpath = src.path(key), dst.path(key)
    if size <= CHUNK:
        with src.fs.open(spath, "rb", block_size=CHUNK) as r:
            data = r.read()
        h.update(data)
        digest = h.hexdigest()
        with dst.fs.open(dpath, "wb", block_size=CHUNK, Metadata={SHA_META: digest}) as w:
            w.write(data)
        on_bytes(len(data))
        return digest, True

    with src.fs.open(spath, "rb", block_size=CHUNK) as r, \
            dst.fs.open(dpath, "wb", block_size=CHUNK) as w:
        while True:
            if cancel is not None and cancel.is_set():
                # Abandon the multipart upload rather than committing a partial object.
                if hasattr(w, "discard"):
                    w.discard()
                raise StorageTierError("cancelled")
            b = r.read(CHUNK)
            if not b:
                break
            h.update(b)
            w.write(b)
            on_bytes(len(b))
    digest = h.hexdigest()
    attached = False
    if size <= MAX_COPY_OBJECT:
        try:
            dst.fs.setxattr(dpath, **{SHA_META: digest})
            attached = True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"sync: could not attach sha256 metadata to {dst.name}:{key}: {type(e).__name__}")
    return digest, attached


def sync(
    src: Tier,
    dst: Tier,
    prefix: str = "",
    *,
    dry_run: bool = False,
    move: bool = False,
    verify: bool = False,
    overwrite: bool = False,
    allow_private: bool = False,
    progress_cb: Optional[ProgressCB] = None,
    cancel: Optional[threading.Event] = None,
) -> SyncResult:
    """Copy every object under ``prefix`` in ``src`` to the same key in ``dst``.

    See the module docstring for the integrity and privacy rules. ``move``
    implies ``verify`` and deletes source objects only after every copy (and
    every skipped "already there" object) has been verified by SHA-256 and the
    run had no errors. Refused (private) objects are never copied or deleted.
    """
    if src.same_location(dst) or (src.name == dst.name and src.bucket == dst.bucket):
        raise StorageTierError("Source and destination are the same bucket")
    prefix = norm_prefix(prefix)
    if move:
        verify = True
    res = SyncResult(src=src.name, dst=dst.name, prefix=prefix, dry_run=dry_run, move=move,
                     verify=verify, overwrite=overwrite, guarded=src.private and not dst.private,
                     src_bucket=src.bucket, dst_bucket=dst.bucket)

    state = {"phase": "planning", "objects_done": 0, "objects_total": 0,
             "bytes_done": 0, "bytes_total": 0, "current_key": None}

    def report(**kw):
        state.update(kw)
        if progress_cb:
            try:
                progress_cb(dict(state))
            except Exception:  # noqa: BLE001 - a broken callback must not kill a sync
                logger.debug("progress callback failed", exc_info=True)

    report()
    src_objs = iter_objects(src, prefix)
    dst_objs = iter_objects(dst, prefix)

    guard = PrivacyGuard(src) if (res.guarded and not allow_private) else None
    candidates: list[tuple[str, int]] = []
    for key in sorted(src_objs):
        size = src_objs[key]["size"]
        if guard is not None:
            reason = guard.check(key)
            if reason:
                res.refused.append({"key": key, "size": size, "reason": reason})
                continue
        candidates.append((key, size))

    # Decide copy vs skip. HEADs (for sha256 metadata) only where sizes match.
    def decide(item: tuple[str, int]) -> tuple[str, int, str, Optional[str], bool]:
        """-> (key, size, action, reason, identity_verified)"""
        key, size = item
        d = dst_objs.get(key)
        if d is None:
            return key, size, "copy", "missing", False
        if overwrite:
            return key, size, "copy", "overwrite", False
        if d["size"] != size:
            return key, size, "copy", "size differs", False
        dsha = _meta(dst, key).get(SHA_META)
        ssha = _meta(src, key).get(SHA_META) if (dsha or verify) else None
        if dsha and ssha:
            if dsha == ssha:
                return key, size, "skip", "identical (sha256 metadata)", True
            return key, size, "copy", "sha256 differs", False
        if not verify:
            return key, size, "skip", "same size (no sha256 metadata)", False
        if dry_run:
            # Hashing whole objects is the run's job, not the preview's.
            return key, size, "skip", "same size (will hash-verify)", False
        try:
            s = ssha or _hash_object(src, key, cancel)[1]
            t = dsha or _hash_object(dst, key, cancel)[1]
        except Exception as e:  # noqa: BLE001
            return key, size, "error", f"hash failed: {type(e).__name__}: {e}", False
        if s == t:
            return key, size, "skip", "identical (hashed)", True
        return key, size, "copy", "sha256 differs", False

    # Same-size candidates need HEADs (and maybe hashing): do those in parallel.
    def needs_head(c: tuple[str, int]) -> bool:
        return not overwrite and c[0] in dst_objs and dst_objs[c[0]]["size"] == c[1]
    pending = [c for c in candidates if needs_head(c)]
    quick = [c for c in candidates if not needs_head(c)]
    decisions = [decide(c) for c in quick]
    if pending:
        with ThreadPoolExecutor(max_workers=_HEAD_WORKERS) as pool:
            decisions += list(pool.map(decide, pending))
    decisions.sort(key=lambda d: d[0])

    verified_skips: list[str] = []
    for key, size, action, reason, verified in decisions:
        if action == "copy":
            res.to_copy.append({"key": key, "size": size, "reason": reason})
        elif action == "skip":
            res.skipped.append({"key": key, "size": size, "reason": reason})
            if verified:
                verified_skips.append(key)
        else:
            res.errors.append({"key": key, "error": reason})

    report(objects_total=len(res.to_copy), bytes_total=res.bytes_to_copy)
    if dry_run:
        report(phase="done")
        return res

    # ── copy ──
    report(phase="copying")
    for item in res.to_copy:
        key, size = item["key"], item["size"]
        if cancel is not None and cancel.is_set():
            res.errors.append({"key": key, "error": "cancelled"})
            break
        report(current_key=key)
        try:
            digest, attached = _copy_object(
                src, dst, key, size, cancel=cancel,
                on_bytes=lambda n: report(bytes_done=state["bytes_done"] + n))
            got = int(dst.fs.info(dst.path(key)).get("size") or 0)
            if got != size:
                raise StorageTierError(f"size mismatch after write ({got} != {size})")
            if verify:
                report(phase="verifying")
                _, back = _hash_object(dst, key, cancel)
                report(phase="copying")
                if back != digest:
                    raise StorageTierError("sha256 mismatch after write")
            if not attached:
                res.meta_missing.append(key)
            res.copied.append({"key": key, "size": size, "sha256": digest})
        except Exception as e:  # noqa: BLE001
            res.errors.append({"key": key, "error": f"{type(e).__name__}: {e}"})
            if str(e) == "cancelled":
                break
        report(objects_done=state["objects_done"] + 1)

    # ── move: delete sources only once everything is verified ──
    if move:
        if res.errors:
            res.move_aborted = "errors during copy; no source objects were deleted"
        else:
            unverified = [s["key"] for s in res.skipped if s["key"] not in verified_skips]
            if unverified:
                res.move_aborted = (f"{len(unverified)} skipped object(s) could not be verified; "
                                    f"no source objects were deleted")
            else:
                report(phase="deleting", current_key=None)
                victims = [c["key"] for c in res.copied] + verified_skips
                try:
                    _rm_keys(src, victims)
                    res.deleted = sorted(victims)
                except Exception as e:  # noqa: BLE001
                    res.errors.append({"key": None, "error": f"delete failed: {type(e).__name__}: {e}"})

    report(phase="done", current_key=None)
    return res


def _rm_keys(tier: Tier, keys: Iterable[str]) -> None:
    paths = [tier.path(k) for k in keys]
    for i in range(0, len(paths), 1000):  # DeleteObjects takes <= 1000 keys
        batch = paths[i:i + 1000]
        if batch:
            tier.fs.rm(batch)


# ── delete ────────────────────────────────────────────────────────────────────
def delete(tier: Tier, prefix: str, *, confirm: bool = False, dry_run: bool = False) -> dict:
    """Delete every object under ``prefix``. Refuses an empty prefix (the whole
    bucket) and, unless ``dry_run``, requires ``confirm``."""
    p = norm_prefix(prefix).rstrip("/")
    if not p:
        raise StorageTierError("Refusing to delete an entire bucket: give a non-empty prefix")
    if not dry_run and not confirm:
        raise StorageTierError("Deletion requires confirm=True")
    objs = iter_objects(tier, p)
    total = sum(o["size"] for o in objs.values())
    out = {"tier": tier.name, "bucket": tier.bucket, "prefix": p, "objects": len(objs), "bytes": total,
           "human": human_bytes(total), "dry_run": dry_run, "deleted": False}
    if dry_run or not objs:
        return out
    _rm_keys(tier, sorted(objs))
    out["deleted"] = True
    return out
