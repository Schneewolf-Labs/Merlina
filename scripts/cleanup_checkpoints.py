"""Reclaim disk by pruning old training checkpoints in ``results/``.

Grimoire (and the diffusion/VLM runners) save a fresh ``checkpoint-<step>``
directory every ``save_steps``. With ``save_total_limit`` ≥ 2 a finished job
leaves several full-size checkpoints behind even though, once a run is over,
only the *last* one is worth keeping. Across a few dozen jobs that adds up
fast — a single 12B run is ~12 GiB per checkpoint.

This is the CLI front-end for :mod:`src.disk_manager` (the same engine the web
UI's Cleanup section uses). It keeps the newest ``--keep`` checkpoints in each
``results/job_*`` directory and deletes the rest. **Safe by default:**

* Reads ``data/jobs.db`` and **never touches an active job** (training /
  running / queued / loading_*). Pruning a live run's checkpoints breaks resume.
* **Dry run unless you pass ``--apply``** — you always see what would go first.

Usage::

    python scripts/cleanup_checkpoints.py                       # preview
    python scripts/cleanup_checkpoints.py --apply               # prune to latest
    python scripts/cleanup_checkpoints.py --keep 2 --apply      # keep latest 2
    python scripts/cleanup_checkpoints.py --purge-failed --apply  # + nuke failed dirs
"""
import argparse
import sqlite3
import sys
from pathlib import Path

# Allow `python scripts/cleanup_checkpoints.py` from anywhere by putting the
# repo root (which contains the `src` package) on the path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.disk_manager import human_size, run_cleanup  # noqa: E402


def _load_job_statuses(db_path: Path) -> dict[str, str]:
    """Map ``job_id -> status`` from the Merlina jobs DB (empty if missing)."""
    if not db_path.exists():
        print(f"  ! no jobs DB at {db_path} — cannot detect active jobs", file=sys.stderr)
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT job_id, status FROM jobs").fetchall()
    finally:
        conn.close()
    return {job_id: status for job_id, status in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=_REPO_ROOT / "results",
                        help="Directory containing job_* checkpoint dirs (default: ./results)")
    parser.add_argument("--db", type=Path, default=_REPO_ROOT / "data" / "jobs.db",
                        help="Path to Merlina jobs DB (default: ./data/jobs.db)")
    parser.add_argument("--keep", type=int, default=1, metavar="N",
                        help="Keep the newest N checkpoints per job (default: 1)")
    parser.add_argument("--purge-failed", action="store_true",
                        help="Delete the entire results dir of FAILED jobs")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete. Without this flag it's a dry run.")
    args = parser.parse_args()

    if args.keep < 1:
        parser.error("--keep must be at least 1")
    if not args.results_dir.is_dir():
        print(f"No results dir at {args.results_dir} — nothing to do.")
        return 0

    statuses = _load_job_statuses(args.db)
    summary = run_cleanup(args.results_dir, statuses, keep=args.keep,
                          purge_failed=args.purge_failed, apply=args.apply)

    mode = "DELETING" if args.apply else "DRY RUN (pass --apply to delete)"
    print(f"=== Merlina checkpoint cleanup — {mode} ===")
    print(f"results: {args.results_dir}  |  keep latest {args.keep} per job\n")

    for s in summary["skipped"]:
        print(f"  skip  {s['job_id']}  ({s['reason']})")
    if summary["skipped"]:
        print()

    if not summary["deletions"]:
        print("Nothing to prune. ✨")
        return 0

    base = args.results_dir.parent
    for d in summary["deletions"]:
        rel = Path(d["path"]).relative_to(base)
        verb = "deleted" if args.apply else "would delete"
        print(f"  {verb}  {rel}  ({human_size(d['bytes'])})  — {d['reason']}")

    verb = "Freed" if args.apply else "Would free"
    print(f"\n=== {verb} {summary['freed_human']} across {summary['count']} item(s) ===")
    if not args.apply:
        print("Re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
