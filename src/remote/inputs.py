"""Portable dataset snapshots shared through the job's artifact store.

Only the control plane reads upload memory, local paths and internal-store
credentials. Workers receive content-addressed references and verify the
snapshot before replacing those references with private, temporary paths.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .artifacts import ArtifactStore

INPUT_VERSION = 1
FORMATS = {"json", "jsonl", "csv", "parquet"}
REFERENCE = re.compile(r"^artifact://(input-[0-9]+-[a-f0-9]{64})$")
# Operator data may only be staged into an operator-controlled bucket, never
# a third-party repo such as the default HF artifact store.
OPERATOR_STORE_KINDS = {"s3"}


class RemoteInputError(ValueError):
    """Dataset preparation failed; messages deliberately omit local paths."""


def _sources(config: dict) -> list[dict]:
    dataset = config.get("dataset") or {}
    return (([dataset["source"]] if dataset.get("source") else [])
            + list(dataset.get("additional_sources") or [])
            + ([dataset["eval_source"]] if dataset.get("eval_source") else []))


def metadata_config(config: dict) -> dict:
    """Config for store provenance, without upload ids or local dataset paths."""
    cleaned = copy.deepcopy(config)
    for source in _sources(cleaned):
        for key in ("dataset_id", "file_path"):
            source.pop(key, None)
    return cleaned


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format(source: dict, filename: str) -> str:
    fmt = source.get("file_format") or Path(filename).suffix.lower().lstrip(".")
    if fmt not in FORMATS:
        raise RemoteInputError("Remote dataset format must be json, jsonl, csv or parquet.")
    return fmt


def require_operator_store(kind) -> None:
    """``kind`` is an :class:`config.ArtifactLocation` (or a legacy kind string)."""
    kind = getattr(kind, "kind", kind)
    if kind not in OPERATOR_STORE_KINDS:
        raise RemoteInputError("Uploaded, local and internal datasets can only be staged to an "
                               "operator-controlled S3 artifact store (ARTIFACT_STORE=external).")


def local_data_roots() -> list[Path]:
    from config import settings
    raw = settings.remote_local_data_roots or ""
    return [Path(part.strip()).expanduser().resolve()
            for part in re.split(f"[,{re.escape(os.pathsep)}]", raw) if part.strip()]


def allowed_local_file(file_path) -> Path:
    """Resolve a remote local_file source inside REMOTE_LOCAL_DATA_ROOTS."""
    from dataset_handlers.internal_store import is_private_name
    try:
        resolved = Path(file_path or "").resolve(strict=True)
    except (OSError, RuntimeError):
        raise RemoteInputError("Local dataset must be an existing regular file.") from None
    # Containment on resolved paths: symlinks and ".." cannot escape a root.
    if not any(resolved.is_relative_to(root) for root in local_data_roots()):
        raise RemoteInputError("Remote local_file datasets must be inside REMOTE_LOCAL_DATA_ROOTS.")
    if is_private_name(str(file_path)) or is_private_name(str(resolved)):
        raise RemoteInputError("Private local datasets cannot be exported to remote workers.")
    if not resolved.is_file():
        raise RemoteInputError("Local dataset must be an existing regular file.")
    return resolved


def check_upload_name(*names) -> None:
    from dataset_handlers.internal_store import is_private_name
    if any(isinstance(name, str) and is_private_name(name) for name in names):
        raise RemoteInputError("Private uploaded datasets cannot be exported to remote workers.")


def pinned_internal_uri(uri: str) -> tuple[str, str]:
    from dataset_handlers.internal_store import parse_swl_uri, InternalStoreError, is_private_name
    try:
        name, revision = parse_swl_uri(uri)
    except InternalStoreError:
        raise RemoteInputError("Remote internal datasets require a pinned swl:// revision.") from None
    if (revision == "latest" or not re.fullmatch(r"[A-Za-z0-9_.-]+", revision)
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in (".", "..")
                   for part in name.split("/")) or revision in (".", "..")):
        raise RemoteInputError("Remote internal datasets require a valid pinned swl:// revision.")
    if is_private_name(name):
        raise RemoteInputError("Private internal datasets cannot be exported to remote workers.")
    return name, revision


def _stage_internal(source: dict, work: Path, internal_store=None) -> tuple[Path, dict]:
    name, revision = pinned_internal_uri(source.get("uri") or "")
    if source.get("split", "train") != "train":
        raise RemoteInputError("Remote internal datasets currently support only the train split.")
    if internal_store is None:
        from dataset_handlers.factory import default_internal_store
        internal_store = default_internal_store()
    if internal_store is None:
        raise RemoteInputError("Internal dataset storage is not configured.")
    manifest = internal_store.get_manifest(name, revision)
    if manifest.is_private():
        raise RemoteInputError("Private internal datasets cannot be exported to remote workers.")
    if manifest.name != name or manifest.rev != revision or manifest.format != "parquet":
        raise RemoteInputError("Internal dataset revision or format does not match the request.")
    # Internal manifests represent one train dataset, potentially many shards.
    # Concatenate batches without loading an entire dataset into RAM. Workers
    # need only this job's artifact credentials, never the source bucket's key.
    import pyarrow.parquet as pq
    output = work / "data.parquet"
    writer = None
    try:
        for index, uri in enumerate(manifest.data_files):
            # A stored manifest is not authority to read another bucket/path.
            expected = internal_store.data_uri(name, revision, "")
            if not isinstance(uri, str) or not uri.startswith(expected):
                raise RemoteInputError("Internal dataset shard is outside its pinned revision.")
            tail = uri[len(expected):]
            if not tail or any(p in ("", ".", "..") for p in tail.split("/")) or "\\" in tail:
                raise RemoteInputError("Internal dataset shard path is invalid.")
            shard = work / f"shard-{index}.parquet"
            internal_store.download_file(uri, str(shard))
            parquet = pq.ParquetFile(shard)
            if writer is None:
                writer = pq.ParquetWriter(output, parquet.schema_arrow)
            for batch in parquet.iter_batches(batch_size=8192):
                writer.write_batch(batch)
            shard.unlink()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RemoteInputError("Internal dataset has no shards.")
    return output, {"source_type": "internal", "uri": f"swl://{name}@{revision}"}


def _mirror_for_remote(source: dict) -> bool:
    if source.get("streaming") or not source.get("repo_id"):
        return False
    try:
        from dataset_handlers.factory import default_prefer_internal
        return bool(source.get("prefer_internal")) or default_prefer_internal()
    except Exception:
        return False


def _mirror_hf_source(source: dict, config: dict, internal_store):
    """Rewrite an HF source in place into the internal mirror it maps to."""
    from dataset_handlers.mirror import ensure_mirror
    if internal_store is None:
        from dataset_handlers.factory import default_internal_store
        internal_store = default_internal_store()
    if internal_store is None:
        raise RemoteInputError("DATASET_MIRROR is on but the internal dataset store isn't configured.")
    try:
        name, rev = ensure_mirror(internal_store, source["repo_id"],
                                  split=source.get("split") or "train",
                                  config_name=source.get("config_name"),
                                  token=config.get("hf_token"))
    except Exception:
        raise RemoteInputError(f"Could not mirror HF dataset {source.get('repo_id')!r} for the remote run.") from None
    for key in ("repo_id", "config_name", "file_path", "dataset_id"):
        source.pop(key, None)
    source.update(source_type="internal", uri=f"swl://{name}@{rev}", split="train")
    return internal_store


def stage_remote_inputs(config: dict, uploads: dict, store: ArtifactStore,
                        *, internal_store=None) -> dict:
    """Return a copy with durable references; finish every upload before renting.

    The original filename, local path and internal-store manifest/credentials
    never enter the snapshot manifest. A digest in each artifact name pins the
    bytes even if submission is retried with a changed local file.
    """
    prepared = copy.deepcopy(config)
    mirrored = set()
    for index, source in enumerate(_sources(prepared)):
        kind = source.get("source_type", "huggingface")
        if kind == "huggingface" and _mirror_for_remote(source):
            # DATASET_MIRROR: snapshot (or reuse) the mirror on the control
            # plane and ship that, so the rented worker never calls the Hub.
            internal_store = _mirror_hf_source(source, config, internal_store)
            kind = "internal"
            mirrored.add(index)
        if kind == "huggingface":
            # Drop irrelevant source fields so arbitrary local paths cannot
            # hitch a ride alongside an otherwise valid HF source.
            for key in ("file_path", "dataset_id", "uri"):
                source.pop(key, None)
            continue
        if kind not in {"internal", "local_file", "upload"}:
            raise RemoteInputError("Unsupported remote dataset source.")
        require_operator_store(getattr(store, "kind", None))
        try:
            with tempfile.TemporaryDirectory(prefix="merlina-input-") as temporary:
                work = Path(temporary)
                provenance = {"source_type": kind}
                if kind == "internal":
                    data, provenance = _stage_internal(source, work, internal_store)
                    fmt = "parquet"
                elif kind == "local_file":
                    original = allowed_local_file(source.get("file_path"))
                    if Path(source.get("file_path")).is_symlink():
                        raise RemoteInputError("Local dataset must be an existing regular file.")
                    fmt = _format(source, str(original))
                    data = work / f"data.{fmt}"
                    shutil.copyfile(original, data)
                else:
                    upload = uploads.get(source.get("dataset_id"))
                    if not upload or not isinstance(upload.get("content"), bytes):
                        raise RemoteInputError("Uploaded dataset is unavailable; upload it again.")
                    check_upload_name(source.get("dataset_id"), upload.get("filename"))
                    fmt = _format(source, upload.get("filename") or "")
                    data = work / f"data.{fmt}"
                    data.write_bytes(upload["content"])
                digest = _digest(data)
                manifest = {"version": INPUT_VERSION, "file": data.name, "format": fmt,
                            "sha256": digest, "size_bytes": data.stat().st_size,
                            "provenance": provenance}
                manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
                name = f"input-{index}-{hashlib.sha256(manifest_bytes).hexdigest()}"
                (work / "input.json").write_bytes(manifest_bytes)
                store.push_dir(work, name)
                source.clear()
                # Preserve source-specific column mapping and split settings.
                original_source = _sources(config)[index]
                for key in ("column_mapping", "split"):
                    if key in original_source:
                        source[key] = copy.deepcopy(original_source[key])
                if index in mirrored:
                    source["split"] = "train"     # the mirror holds just that split
                source.update(source_type="local_file", file_format=fmt,
                              file_path=f"artifact://{name}")
        except RemoteInputError:
            raise
        except Exception:
            # Provider errors can contain credentials or local file paths.
            raise RemoteInputError(f"Could not stage remote dataset source {index}.") from None
    return prepared


def restore_remote_inputs(config: dict, store: ArtifactStore, root: Path) -> dict:
    """Verify all staged data before returning a worker-local runtime config."""
    restored = copy.deepcopy(config)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(_sources(restored)):
        kind = source.get("source_type", "huggingface")
        if kind == "huggingface":
            continue
        match = REFERENCE.fullmatch(source.get("file_path") or "")
        if kind != "local_file" or not match:
            raise RemoteInputError("Worker dataset requires a staged artifact reference.")
        try:
            destination = root / f"source-{index}"
            store.pull_dir(match[1], destination)
            manifest_path = destination / "input.json"
            if manifest_path.is_symlink():
                raise RemoteInputError("Invalid input manifest.")
            manifest_bytes = manifest_path.read_bytes()
            if not match[1].endswith("-" + hashlib.sha256(manifest_bytes).hexdigest()):
                raise RemoteInputError("Remote input manifest failed its integrity check.")
            manifest = json.loads(manifest_bytes)
            fmt = manifest.get("format")
            if (manifest.get("version") != INPUT_VERSION or fmt not in FORMATS
                    or manifest.get("file") != f"data.{fmt}"
                    or source.get("file_format") != fmt):
                raise RemoteInputError("Unsupported or invalid input manifest.")
            data = destination / manifest["file"]
            if (data.is_symlink() or not data.is_file()
                    or data.stat().st_size != manifest.get("size_bytes")
                    or _digest(data) != manifest.get("sha256")):
                raise RemoteInputError("Remote dataset failed its integrity check.")
            source["file_path"] = str(data.resolve())
        except RemoteInputError:
            raise
        except Exception:
            raise RemoteInputError(f"Could not restore remote dataset source {index}.") from None
    return restored
