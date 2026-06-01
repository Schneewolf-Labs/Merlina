"""Disk analysis + checkpoint cleanup for Merlina's ``results/`` tree.

Training runs accumulate full-size ``checkpoint-<step>`` directories fast — a
12B run is ~12 GiB per checkpoint, and with ``save_total_limit`` ≥ 2 every
finished job leaves several behind even though only the *last* one matters once
the run is over. This module is the shared engine behind both the
``scripts/cleanup_checkpoints.py`` CLI and the web UI's Cleanup section.

Two responsibilities:

* :func:`analyze_disk` — read-only breakdown of what's on disk (per-job sizes,
  per-checkpoint sizes, how much is reclaimable, model dir sizes, filesystem
  usage). Powers the analysis view.
* :func:`plan_cleanup` / :func:`run_cleanup` — decide what to prune and (only
  when asked) delete it.

**Safety invariant:** a job that is still active (training / running / queued /
loading) is *never* pruned — deleting checkpoints out from under a live trainer
breaks resume. Callers pass a ``{job_id: status}`` map (from Merlina's jobs DB);
any job whose status is in :data:`ACTIVE_STATUSES` is left strictly alone.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

# Statuses for which a job is NOT finished — its checkpoint dir is off limits.
ACTIVE_STATUSES = frozenset({
    "queued",
    "initializing",
    "loading_model",
    "loading_dataset",
    "training",
    "running",
})

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def human_size(num_bytes: int) -> str:
    """Render a byte count as a short human string (e.g. ``12.5 GB``)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def dir_size(path: Path) -> int:
    """Total size in bytes of all files under ``path`` (0 if missing)."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _checkpoints(job_dir: Path) -> list[Path]:
    """Checkpoint subdirs of ``job_dir``, sorted oldest→newest by step."""
    cks = [d for d in job_dir.iterdir() if d.is_dir() and _CKPT_RE.match(d.name)]
    return sorted(cks, key=lambda d: int(_CKPT_RE.match(d.name).group(1)))


def _step_of(ckpt: Path) -> int:
    return int(_CKPT_RE.match(ckpt.name).group(1))


def plan_cleanup(
    results_dir: Path,
    statuses: dict[str, str],
    keep: int = 1,
    purge_failed: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Build the deletion plan without touching anything.

    Returns ``(deletions, skipped)`` where each *deletion* is a dict
    ``{path, job_id, bytes, reason}`` and each *skipped* entry is
    ``{job_id, reason}``. Honors the active-job safety invariant.
    """
    deletions: list[dict] = []
    skipped: list[dict] = []
    if not results_dir.is_dir():
        return deletions, skipped

    for job_dir in sorted(results_dir.glob("job_*")):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        status = statuses.get(job_id, "unknown")

        if status in ACTIVE_STATUSES:
            skipped.append({"job_id": job_id, "reason": f"active ({status})"})
            continue

        if purge_failed and status == "failed":
            deletions.append({
                "path": str(job_dir), "job_id": job_id,
                "bytes": dir_size(job_dir), "reason": "failed job (whole dir)",
            })
            continue

        cks = _checkpoints(job_dir)
        if len(cks) <= keep:
            skipped.append({"job_id": job_id,
                            "reason": f"{len(cks)} checkpoint(s), keep={keep}"})
            continue

        kept = ", ".join(c.name for c in cks[-keep:])
        for ck in cks[:-keep]:  # everything but the newest `keep`
            deletions.append({
                "path": str(ck), "job_id": job_id,
                "bytes": dir_size(ck), "reason": f"old checkpoint [keeping {kept}]",
            })

    return deletions, skipped


def run_cleanup(
    results_dir: Path,
    statuses: dict[str, str],
    keep: int = 1,
    purge_failed: bool = False,
    apply: bool = False,
) -> dict:
    """Plan and (if ``apply``) execute cleanup. Returns a summary dict.

    ``{applied, keep, purge_failed, deletions[...], skipped[...],
       freed_bytes, freed_human, count}``. When ``apply`` is False this is a
    dry run — nothing is deleted but ``freed_*`` reflects what *would* be freed.
    """
    deletions, skipped = plan_cleanup(results_dir, statuses, keep, purge_failed)

    freed = 0
    for item in deletions:
        if apply:
            shutil.rmtree(item["path"], ignore_errors=True)
        item["deleted"] = apply
        item["human"] = human_size(item["bytes"])
        freed += item["bytes"]

    return {
        "applied": apply,
        "keep": keep,
        "purge_failed": purge_failed,
        "deletions": deletions,
        "skipped": skipped,
        "freed_bytes": freed,
        "freed_human": human_size(freed),
        "count": len(deletions),
    }


def _filesystem_usage(path: Path) -> dict | None:
    """Best-effort filesystem usage for the volume holding ``path``.

    Uses :mod:`shutil` (stdlib, always present) rather than psutil so it works
    in the mocked test server too. Returns None if it can't be determined.
    """
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            "total_bytes": total, "total_human": human_size(total),
            "used_bytes": used, "used_human": human_size(used),
            "free_bytes": free, "free_human": human_size(free),
            "percent": round(used / total * 100, 1) if total else 0.0,
        }
    except OSError:
        return None


def analyze_disk(
    results_dir: Path,
    models_dir: Path,
    statuses: dict[str, str],
    keep: int = 1,
) -> dict:
    """Read-only breakdown of ``results/`` and ``models/`` usage.

    Per job: status, total size, each checkpoint's size, whether it's active,
    and how many bytes are reclaimable at the given ``keep`` level. Also lists
    top-level model dirs and the host filesystem usage. Jobs are returned
    largest-first so the UI can surface the biggest offenders.
    """
    jobs: list[dict] = []
    results_total = 0
    results_reclaimable = 0

    if results_dir.is_dir():
        for job_dir in sorted(results_dir.glob("job_*")):
            if not job_dir.is_dir():
                continue
            job_id = job_dir.name
            status = statuses.get(job_id, "unknown")
            is_active = status in ACTIVE_STATUSES

            cks = _checkpoints(job_dir)
            ck_entries = []
            job_total = 0
            for ck in cks:
                sz = dir_size(ck)
                job_total += sz
                ck_entries.append({
                    "name": ck.name, "step": _step_of(ck),
                    "bytes": sz, "human": human_size(sz),
                })
            # Reclaimable = everything but the newest `keep`, and only if the
            # job is finished. Newest-first display, so reclaimable are the
            # tail of the oldest→newest list.
            reclaimable = 0
            if not is_active and len(cks) > keep:
                reclaimable = sum(e["bytes"] for e in ck_entries[:-keep])

            results_total += job_total
            results_reclaimable += reclaimable
            jobs.append({
                "job_id": job_id,
                "status": status,
                "active": is_active,
                "total_bytes": job_total,
                "total_human": human_size(job_total),
                "checkpoints": list(reversed(ck_entries)),  # newest first for UI
                "reclaimable_bytes": reclaimable,
                "reclaimable_human": human_size(reclaimable),
            })

    jobs.sort(key=lambda j: j["total_bytes"], reverse=True)

    model_items = []
    models_total = 0
    if models_dir.is_dir():
        for item in sorted(models_dir.iterdir()):
            if not item.is_dir():
                continue
            sz = dir_size(item)
            models_total += sz
            model_items.append({
                "name": item.name, "bytes": sz, "human": human_size(sz),
                "modified_date": _date_str(item.stat().st_mtime),
            })
        model_items.sort(key=lambda m: m["bytes"], reverse=True)

    return {
        "results": {
            "path": str(results_dir),
            "total_bytes": results_total,
            "total_human": human_size(results_total),
            "reclaimable_bytes": results_reclaimable,
            "reclaimable_human": human_size(results_reclaimable),
            "keep": keep,
            "job_count": len(jobs),
            "jobs": jobs,
        },
        "models": {
            "path": str(models_dir),
            "total_bytes": models_total,
            "total_human": human_size(models_total),
            "count": len(model_items),
            "items": model_items,
        },
        "filesystem": _filesystem_usage(results_dir if results_dir.exists() else Path(".")),
    }


def delete_models(models_dir: Path, names, apply: bool = False,
                  protected=frozenset()) -> dict:
    """Delete saved model dirs under ``models_dir`` by name. Dry-run unless apply.

    Unlike the HF cache, these are Merlina's own outputs and may NOT be
    re-downloadable — if a model was never pushed to the Hub, deletion is
    permanent. A ``protected`` set (models that are an active job's output or
    currently loaded for inference) is refused server-side as a backstop, even
    if the UI somehow offers them.
    """
    protected = set(protected)
    deleted, skipped = [], []
    freed = 0
    for name in names:
        # Guard against path traversal — only direct children of models_dir.
        if "/" in name or "\\" in name or name in (".", ".."):
            skipped.append({"name": name, "reason": "invalid name"})
            continue
        if name in protected:
            skipped.append({"name": name, "reason": "in use (active job or loaded)"})
            continue
        path = models_dir / name
        if not path.is_dir():
            skipped.append({"name": name, "reason": "not found"})
            continue
        sz = dir_size(path)
        if apply:
            shutil.rmtree(path, ignore_errors=True)
        freed += sz
        deleted.append({"name": name, "bytes": sz, "human": human_size(sz), "deleted": apply})

    return {
        "applied": apply,
        "deleted": deleted,
        "skipped": skipped,
        "freed_bytes": freed,
        "freed_human": human_size(freed),
        "count": len(deleted),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Derived artifacts Merlina leaves behind: GGUF exports + W&B run logs.
#
# GGUF files (models/<name>/gguf/*.gguf) are *derived* from the saved model via
# llama.cpp — always regenerable, often the single largest file in a model dir,
# so they're a high-value, low-risk reclaim. W&B run logs (wandb/run-*) are
# local mirrors of metrics already streamed to the W&B server.
# ─────────────────────────────────────────────────────────────────────────────

def _quant_of(gguf_name: str) -> str | None:
    """Extract the quant tag from the ``{model}.{quant}.gguf`` convention."""
    if not gguf_name.endswith(".gguf"):
        return None
    stem = gguf_name[: -len(".gguf")]
    return stem.rsplit(".", 1)[1] if "." in stem else None


def analyze_artifacts(models_dir: Path, wandb_dir: Path, loaded_gguf: str | None = None) -> dict:
    """Read-only breakdown of GGUF exports and W&B run logs.

    ``loaded_gguf`` is the absolute path of a GGUF currently loaded for
    inference (if any); the matching file is flagged ``loaded`` so the UI locks
    it. W&B reports total size + run count, excluding the active run.
    """
    loaded_resolved = str(Path(loaded_gguf).resolve()) if loaded_gguf else None

    gguf_files = []
    gguf_total = 0
    if models_dir.is_dir():
        for model_dir in sorted(models_dir.iterdir()):
            gdir = model_dir / "gguf"
            if not gdir.is_dir():
                continue
            for f in sorted(gdir.glob("*.gguf")):
                sz = f.stat().st_size
                gguf_total += sz
                gguf_files.append({
                    "model": model_dir.name,
                    "file": f.name,
                    "quant": _quant_of(f.name),
                    "bytes": sz,
                    "human": human_size(sz),
                    "modified_date": _date_str(f.stat().st_mtime),
                    "loaded": loaded_resolved is not None and str(f.resolve()) == loaded_resolved,
                })
    gguf_files.sort(key=lambda x: x["bytes"], reverse=True)

    # W&B: count run dirs and total size, excluding whatever ``latest-run``
    # points at (that's the live/active run — never offer to delete it).
    wandb_runs = []
    wandb_total = 0
    active_run = None
    if wandb_dir.is_dir():
        latest = wandb_dir / "latest-run"
        if latest.is_symlink() or latest.exists():
            try:
                active_run = (wandb_dir / os.readlink(latest)).name if latest.is_symlink() else latest.name
            except OSError:
                active_run = None
        for run in sorted(wandb_dir.glob("run-*")):
            if not run.is_dir():
                continue
            sz = dir_size(run)
            wandb_total += sz
            wandb_runs.append({"name": run.name, "bytes": sz,
                               "active": run.name == active_run})

    reclaimable_wandb = sum(r["bytes"] for r in wandb_runs if not r["active"])
    return {
        "gguf": {
            "total_bytes": gguf_total,
            "total_human": human_size(gguf_total),
            "count": len(gguf_files),
            "files": gguf_files,
        },
        "wandb": {
            "path": str(wandb_dir),
            "total_bytes": wandb_total,
            "total_human": human_size(wandb_total),
            "run_count": len(wandb_runs),
            "active_run": active_run,
            "reclaimable_bytes": reclaimable_wandb,
            "reclaimable_human": human_size(reclaimable_wandb),
        },
    }


def delete_gguf_files(models_dir: Path, items, apply: bool = False,
                      loaded_gguf: str | None = None) -> dict:
    """Delete GGUF files by {model, file}. Dry-run unless ``apply``.

    Refuses anything outside a ``models/<model>/gguf/`` dir, and a GGUF that is
    currently loaded for inference. Tidies an emptied ``gguf/`` dir.
    """
    loaded_resolved = str(Path(loaded_gguf).resolve()) if loaded_gguf else None
    deleted, skipped = [], []
    freed = 0
    for it in items:
        model, fname = it.get("model", ""), it.get("file", "")
        if not model or not fname or "/" in model or "\\" in model \
                or "/" in fname or "\\" in fname or fname in (".", ".."):
            skipped.append({"model": model, "file": fname, "reason": "invalid name"})
            continue
        if not fname.endswith(".gguf"):
            skipped.append({"model": model, "file": fname, "reason": "not a .gguf file"})
            continue
        path = models_dir / model / "gguf" / fname
        if not path.is_file():
            skipped.append({"model": model, "file": fname, "reason": "not found"})
            continue
        if loaded_resolved is not None and str(path.resolve()) == loaded_resolved:
            skipped.append({"model": model, "file": fname, "reason": "loaded for inference"})
            continue
        sz = path.stat().st_size
        if apply:
            path.unlink()
            gdir = path.parent
            if gdir.is_dir() and not any(gdir.iterdir()):
                gdir.rmdir()
        freed += sz
        deleted.append({"model": model, "file": fname, "bytes": sz,
                        "human": human_size(sz), "deleted": apply})

    return {"applied": apply, "deleted": deleted, "skipped": skipped,
            "freed_bytes": freed, "freed_human": human_size(freed), "count": len(deleted)}


def clear_wandb_runs(wandb_dir: Path, apply: bool = False) -> dict:
    """Delete W&B ``run-*`` dirs except the active one (``latest-run``).

    Dry-run unless ``apply``. Metrics already live on the W&B server; these are
    local mirrors.
    """
    if not wandb_dir.is_dir():
        return {"applied": apply, "deleted": [], "freed_bytes": 0,
                "freed_human": human_size(0), "count": 0}

    active_run = None
    latest = wandb_dir / "latest-run"
    if latest.is_symlink():
        try:
            active_run = (wandb_dir / os.readlink(latest)).name
        except OSError:
            active_run = None

    deleted = []
    freed = 0
    for run in sorted(wandb_dir.glob("run-*")):
        if not run.is_dir() or run.name == active_run:
            continue
        sz = dir_size(run)
        if apply:
            shutil.rmtree(run, ignore_errors=True)
        freed += sz
        deleted.append({"name": run.name, "bytes": sz})

    return {"applied": apply, "active_run": active_run, "deleted": deleted,
            "freed_bytes": freed, "freed_human": human_size(freed), "count": len(deleted)}


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace cache (~/.cache/huggingface/hub)
#
# Unlike results/, this cache is *shared* across every HF tool on the machine
# (ComfyUI, other trainers, …) and Merlina can't know what else relies on it —
# so the rule here is analysis-first, explicit selection, no heuristic pruning.
# Deletion goes through huggingface_hub's own ``delete_revisions`` so the
# blobs/refs/snapshots layout (symlinked, with shared blobs) is handled
# correctly and reclaimed space is computed accurately.
#
# ``scan_cache_dir`` / ``_scan`` are injectable so this is unit-testable
# without a real multi-hundred-GB cache on disk.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STALE_DAYS = 90


def _default_scan():
    """Import + call ``huggingface_hub.scan_cache_dir`` lazily.

    Kept out of module import so the checkpoint tooling works even where
    huggingface_hub isn't installed. Raises if the lib or cache is absent.
    """
    from huggingface_hub import scan_cache_dir
    return scan_cache_dir()


def analyze_hf_cache(stale_days: int = DEFAULT_STALE_DAYS, now: float | None = None,
                     _scan=None) -> dict:
    """Read-only breakdown of the HuggingFace cache, repos largest-first.

    Each repo: id, type (model/dataset), size, file count, last-accessed /
    last-modified dates, its revision commit hashes (needed for deletion), and
    a ``stale`` flag (not accessed within ``stale_days``). Returns
    ``{available: False, error}`` if the cache or huggingface_hub is missing —
    callers render that as an empty, non-fatal state.
    """
    import time as _time
    now = _time.time() if now is None else now
    scan = _scan or _default_scan

    try:
        info = scan()
    except Exception as exc:  # noqa: BLE001 — surfaced to UI, never fatal
        return {"available": False, "error": str(exc),
                "total_bytes": 0, "total_human": human_size(0),
                "stale_days": stale_days, "stale_reclaimable_bytes": 0,
                "stale_reclaimable_human": human_size(0), "repos": []}

    cutoff = stale_days * 86400
    repos = []
    stale_total = 0
    for r in info.repos:
        last_accessed = getattr(r, "last_accessed", None)
        is_stale = last_accessed is not None and (now - last_accessed) > cutoff
        size = r.size_on_disk
        if is_stale:
            stale_total += size
        repos.append({
            "repo_id": r.repo_id,
            "repo_type": r.repo_type,
            "bytes": size,
            "human": human_size(size),
            "nb_files": getattr(r, "nb_files", None),
            "last_accessed": last_accessed,
            "last_accessed_date": _date_str(last_accessed),
            "last_modified_date": _date_str(getattr(r, "last_modified", None)),
            "revisions": [rev.commit_hash for rev in r.revisions],
            "stale": is_stale,
        })

    repos.sort(key=lambda x: x["bytes"], reverse=True)
    return {
        "available": True,
        "total_bytes": info.size_on_disk,
        "total_human": human_size(info.size_on_disk),
        "repo_count": len(repos),
        "stale_days": stale_days,
        "stale_reclaimable_bytes": stale_total,
        "stale_reclaimable_human": human_size(stale_total),
        "repos": repos,
    }


def _date_str(epoch: float | None) -> str | None:
    if not epoch:
        return None
    import datetime
    return datetime.date.fromtimestamp(epoch).isoformat()


def delete_hf_repos(repos: list[dict], apply: bool = False, _scan=None) -> dict:
    """Delete whole HF cache repos by id+type. Dry-run unless ``apply``.

    ``repos`` is a list of ``{repo_id, repo_type}``. Gathers every revision of
    each matched repo and (only if ``apply``) calls
    ``delete_revisions(*hashes).execute()``. ``freed_bytes`` comes from HF's
    own ``expected_freed_size`` (which accounts for blobs shared between
    revisions), so it's accurate even in dry-run.
    """
    scan = _scan or _default_scan
    try:
        info = scan()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc), "applied": apply,
                "deleted": [], "freed_bytes": 0, "freed_human": human_size(0), "count": 0}

    wanted = {(r["repo_id"], r.get("repo_type", "model")) for r in repos}
    hashes: list[str] = []
    deleted = []
    for r in info.repos:
        if (r.repo_id, r.repo_type) in wanted:
            hashes.extend(rev.commit_hash for rev in r.revisions)
            deleted.append({"repo_id": r.repo_id, "repo_type": r.repo_type,
                            "bytes": r.size_on_disk, "human": human_size(r.size_on_disk)})

    freed = 0
    if hashes:
        strategy = info.delete_revisions(*hashes)
        freed = int(getattr(strategy, "expected_freed_size", 0))
        if apply:
            strategy.execute()

    return {
        "available": True,
        "applied": apply,
        "deleted": deleted,
        "freed_bytes": freed,
        "freed_human": human_size(freed),
        "count": len(deleted),
    }
