"""
Storage tiers — browse, measure, sync and prune Merlina's S3-compatible buckets.

Merlina talks to up to three S3-API object stores, each configured in the
server's ``.env``:

``internal``  (``S3_*``)                the private, LAN-only dataset store (e.g. MinIO)
``external``  (``R2_*``)                the shared tier a rented GPU box can reach (e.g. R2)
``artifacts`` (``MERLINA_ARTIFACT_*``)  where remote runs publish adapters/merged
                                        models — listed only when it is a *different*
                                        endpoint+bucket than ``external`` (when it is the
                                        same, ``external`` already covers its prefix)

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
# Bucket names per the S3 naming rules (lower-case, 3-63 chars).
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")

TIER_ORDER = ("internal", "external", "artifacts")


class StorageTierError(Exception):
    """A tier is unknown/unconfigured, or a request was refused."""


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
    they are excluded from ``repr`` and from :meth:`describe`."""

    name: str
    bucket: Optional[str]
    endpoint_url: Optional[str]
    private: bool
    region: str = "auto"
    description: str = ""
    hint_prefix: str = ""  # where interesting things live (e.g. the artifact prefix)
    access_key: Optional[str] = field(default=None, repr=False)
    secret_key: Optional[str] = field(default=None, repr=False)
    session_token: Optional[str] = field(default=None, repr=False)
    fs_override: Any = field(default=None, repr=False)  # injectable for tests
    _fs: Any = field(default=None, repr=False)

    @property
    def configured(self) -> bool:
        return bool(self.bucket) and (self.fs_override is not None or bool(self.endpoint_url) or bool(self.access_key))

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
            "bucket": self.bucket,
            "endpoint_host": self.endpoint_host,
            "private": self.private,
            "privacy": "private" if self.private else "shared",
            "configured": self.configured,
            "description": self.description,
            "hint_prefix": self.hint_prefix,
        }

    @property
    def fs(self) -> Any:
        if self.fs_override is not None:
            return self.fs_override
        if not self.configured:
            raise StorageTierError(f"Storage tier '{self.name}' is not configured")
        if self._fs is None:
            try:
                import s3fs
            except ImportError as e:  # pragma: no cover - only without the dep
                raise StorageTierError("s3fs is required for storage tiers: pip install s3fs") from e
            client_kwargs = {"endpoint_url": self.endpoint_url}
            if self.region:
                client_kwargs["region_name"] = self.region
            self._fs = s3fs.S3FileSystem(
                key=self.access_key or None,
                secret=self.secret_key or None,
                token=self.session_token or None,
                client_kwargs={k: v for k, v in client_kwargs.items() if v},
                config_kwargs={"retries": {"max_attempts": 5, "mode": "standard"}},
                skip_instance_cache=True,
            )
        return self._fs

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

    ``internal`` and ``external`` are always present (``configured`` says if
    they are usable); ``artifacts`` is added only when it is configured AND not
    the same endpoint+bucket as ``external``.
    """
    fs_overrides = fs_overrides or {}
    g = lambda attr, default=None: getattr(settings, attr, default)  # noqa: E731

    tiers: dict[str, Tier] = {}
    tiers["internal"] = Tier(
        name="internal", bucket=g("s3_dataset_bucket"), endpoint_url=g("s3_endpoint_url"),
        region=str(g("s3_region", "auto") or "auto"), private=True,
        description="Private store (LAN). Source of truth; private data never leaves it.",
        access_key=g("s3_access_key"), secret_key=g("s3_secret_key"),
        fs_override=fs_overrides.get("internal"),
    )
    tiers["external"] = Tier(
        name="external", bucket=g("r2_dataset_bucket"), endpoint_url=g("r2_endpoint_url"),
        region=str(g("r2_region", "auto") or "auto"), private=False,
        description="Shared store reachable from rented GPU boxes.",
        access_key=g("r2_access_key"), secret_key=g("r2_secret_key"),
        fs_override=fs_overrides.get("external"),
    )
    artifact_prefix = str(g("merlina_artifact_s3_prefix", "") or "").strip("/")
    artifacts = Tier(
        name="artifacts", bucket=g("merlina_artifact_s3_bucket"),
        endpoint_url=g("merlina_artifact_s3_endpoint"),
        region=str(g("merlina_artifact_s3_region", "") or "auto"), private=False,
        description="Remote-training artifacts (adapters, merged models).",
        hint_prefix=f"{artifact_prefix}/" if artifact_prefix else "",
        access_key=g("merlina_artifact_access_key"), secret_key=g("merlina_artifact_secret_key"),
        session_token=g("merlina_artifact_session_token"),
        fs_override=fs_overrides.get("artifacts"),
    )
    if artifacts.bucket:
        if tiers["external"].bucket and artifacts.same_location(tiers["external"]):
            # Same bucket: point external at where runs land so it's discoverable.
            if artifacts.hint_prefix and not tiers["external"].hint_prefix:
                tiers["external"].hint_prefix = artifacts.hint_prefix
        else:
            tiers["artifacts"] = artifacts
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


def ls(tier: Tier, prefix: str = "", *, limit: int = 1000) -> dict:
    """Delimiter-style listing of one level under ``prefix``."""
    prefix = norm_prefix(prefix)
    base = prefix.rstrip("/")
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
    dirs.sort(key=lambda d: d["prefix"])
    objects.sort(key=lambda o: o["key"])
    truncated = len(objects) > limit
    return {
        "tier": tier.name, "prefix": f"{base}/" if base else "",
        "dirs": dirs, "objects": objects[:limit], "truncated": truncated,
        "object_count": len(objects),
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
    return {"tier": tier.name, "prefix": p, "objects": len(objs), "bytes": total,
            "human": human_bytes(total)}


# ── buckets ───────────────────────────────────────────────────────────────────
def _is_access_denied(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    text = f"{type(exc).__name__} {exc}"
    return any(s in text for s in ("AccessDenied", "Access Denied", "Forbidden", "403", "Unauthorized"))


def list_buckets(tier: Tier) -> dict:
    """List the buckets visible to this tier's credentials. Scoped keys
    (limited to one bucket) usually can't — that is reported, not raised."""
    try:
        resp = tier.fs.call_s3("list_buckets")
        names = sorted(b.get("Name") for b in resp.get("Buckets", []) if b.get("Name"))
        return {"tier": tier.name, "ok": True, "buckets": names, "message": ""}
    except Exception as e:  # noqa: BLE001 - provider errors vary
        if _is_access_denied(e):
            msg = (f"The '{tier.name}' credentials may not list buckets (access denied). "
                   f"That is normal for a key scoped to one bucket; this tier uses "
                   f"'{tier.bucket}'.")
        else:
            msg = f"Could not list buckets on '{tier.name}': {type(e).__name__}: {e}"
        return {"tier": tier.name, "ok": False, "buckets": [], "message": msg}


def make_bucket(tier: Tier, name: str) -> dict:
    """Create a bucket on this tier's endpoint. Reports (doesn't raise) when the
    credentials aren't allowed to — the usual case for scoped keys."""
    name = (name or "").strip()
    if not _BUCKET_RE.match(name) or ".." in name:
        return {"tier": tier.name, "ok": False, "bucket": name,
                "message": f"Invalid bucket name {name!r} (3-63 chars: a-z, 0-9, '.', '-')."}
    params: dict[str, Any] = {"Bucket": name}
    if tier.region and tier.region not in ("auto", "us-east-1"):
        params["CreateBucketConfiguration"] = {"LocationConstraint": tier.region}
    try:
        tier.fs.call_s3("create_bucket", **params)
        try:
            tier.fs.invalidate_cache("")
        except Exception:  # noqa: BLE001
            pass
        return {"tier": tier.name, "ok": True, "bucket": name, "message": f"Created bucket '{name}'."}
    except Exception as e:  # noqa: BLE001
        text = f"{type(e).__name__} {e}"
        if "BucketAlreadyOwnedByYou" in text or "BucketAlreadyExists" in text or isinstance(e, FileExistsError):
            msg = f"Bucket '{name}' already exists."
        elif _is_access_denied(e):
            msg = (f"The '{tier.name}' credentials may not create buckets (access denied). "
                   f"Create it with an admin key in the provider's console/CLI, then grant "
                   f"this key access to it.")
        else:
            msg = f"Could not create bucket '{name}': {type(e).__name__}: {e}"
        return {"tier": tier.name, "ok": False, "bucket": name, "message": msg}


# ── privacy guard ─────────────────────────────────────────────────────────────
class PrivacyGuard:
    """Decides which keys of a private tier must not reach a shared tier."""

    def __init__(self, tier: Tier):
        from dataset_handlers.internal_store import is_private_name  # lazy (pulls `datasets`)
        self.tier = tier
        self._is_private_name = is_private_name
        self._private_dataset_dirs: Optional[set[str]] = None
        self._groups_scanned: set[str] = set()
        self._private_dirs: set[str] = set()

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

    def check(self, key: str) -> Optional[str]:
        """The refusal reason for ``key``, or ``None`` if it may be shared."""
        if self._is_private_name(key):
            return "path matches a private marker"
        if key.startswith("datasets/"):
            rest = key[len("datasets/"):]
            if "/" in rest:
                self._scan_group(rest.split("/", 1)[0])
                for d in self._private_dirs:
                    if key.startswith(d):
                        return f"dataset manifest is private ({d})"
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
            "src": self.src, "dst": self.dst, "prefix": self.prefix,
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
    if src.same_location(dst) or src.name == dst.name:
        raise StorageTierError("Source and destination are the same bucket")
    prefix = norm_prefix(prefix)
    if move:
        verify = True
    res = SyncResult(src=src.name, dst=dst.name, prefix=prefix, dry_run=dry_run, move=move,
                     verify=verify, overwrite=overwrite, guarded=src.private and not dst.private)

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
    out = {"tier": tier.name, "prefix": p, "objects": len(objs), "bytes": total,
           "human": human_bytes(total), "dry_run": dry_run, "deleted": False}
    if dry_run or not objs:
        return out
    _rm_keys(tier, sorted(objs))
    out["deleted"] = True
    return out
