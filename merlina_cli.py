"""
Merlina console entry point.

Installed as the `merlina` command (see pyproject.toml). Lives at the top
level rather than in `src/` because `src/__init__.py` imports torch. Kept deliberately
import-light: the heavy ML stack (torch, transformers, ...) is only imported
once we actually start the server, so `merlina --version` works in a fresh
environment and a missing torch produces a friendly message instead of a
traceback.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

TORCH_INSTALL_HINT = """\
Merlina needs PyTorch, which is intentionally not installed automatically
(GPU environments ship CUDA-matched builds that pip would otherwise replace).

Install it first, matching your CUDA version, e.g.:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

then run `merlina serve` again.
"""


def _get_version() -> str:
    from version import __version__
    return __version__


def _resolve_store(which: str):
    """Build a named dataset-store backend from app settings, or exit with a
    helpful message when it isn't configured."""
    from dataset_handlers.factory import get_store

    store = get_store(which)
    if store is None:
        prefix = f"STORAGE_{which.upper()}_"
        print(
            f"The '{which}' dataset store is not configured.\n"
            f"Set {prefix}ENDPOINT / {prefix}BUCKET / {prefix}ACCESS_KEY / "
            f"{prefix}SECRET_KEY in the server's .env.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return store


def _parse_ref(ref: str):
    """Parse a dataset ref into ``(name, rev)``. Accepts an ``swl://`` URI or a
    bare ``name`` / ``name@rev``; rev defaults to ``latest``."""
    from dataset_handlers.internal_store import parse_swl_uri

    if ref.startswith("swl://"):
        return parse_swl_uri(ref)
    if "@" in ref:
        name, rev = ref.rsplit("@", 1)
        return name.strip("/"), rev
    return ref.strip("/"), "latest"


def _datasets_promote(args: argparse.Namespace) -> int:
    from dataset_handlers.internal_store import InternalStoreError
    from dataset_handlers.promote import promote

    name, rev = _parse_ref(args.name)

    src = _resolve_store(args.src)
    dst = _resolve_store(args.dst)
    if args.src == args.dst:
        print("--from and --to must differ.", file=sys.stderr)
        return 2

    try:
        plan = promote(
            src, dst, name, rev,
            overwrite=args.overwrite,
            allow_private=args.allow_private,
            dry_run=args.dry_run,
        )
    except InternalStoreError as e:
        print(f"Cannot promote: {e}", file=sys.stderr)
        return 1

    where = f"swl://{plan.name}@{plan.rev}"
    if plan.skipped:
        print(f"Already present in '{args.dst}': {where} — nothing to do "
              f"(use --overwrite to recopy).")
    elif args.dry_run:
        tag = " (overwrite)" if plan.already_present else ""
        print(f"DRY RUN — would copy {where} from '{args.src}' to '{args.dst}'{tag}:")
        print(f"  shards: {plan.num_shards}   size: {plan.human_bytes()}")
        print("No data was transferred. Re-run without --dry-run to promote.")
    else:
        print(f"Promoted {where} from '{args.src}' to '{args.dst}': "
              f"{plan.num_shards} shard(s), {plan.human_bytes()}.")
    return 0


def _datasets_put(args: argparse.Namespace) -> int:
    from dataset_handlers.internal_store import InternalStoreError
    from dataset_handlers.put import put_dataset

    from dataset_handlers.internal_store import is_private_name
    if args.store == "external" and (args.private or is_private_name(args.name)):
        print("error: private datasets never go to the external tier", file=sys.stderr)
        return 2
    store = _resolve_store(args.store)
    try:
        name, rev, created = put_dataset(store, args.file, args.name, private=args.private)
    except (InternalStoreError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"swl://{name}@{rev} ({'stored' if created else 'already stored'}; latest -> {rev})")
    return 0


def _datasets_info(args: argparse.Namespace) -> int:
    from dataset_handlers.internal_store import InternalStoreError

    name, rev = _parse_ref(args.name)
    store = _resolve_store(args.store)
    try:
        m = store.get_manifest(name, rev)
    except InternalStoreError as e:
        print(f"Not found: {e}", file=sys.stderr)
        return 1

    print(f"swl://{m.name}@{m.rev}")
    print(f"  store:    {args.store}")
    print(f"  format:   {m.format}")
    print(f"  rows:     {m.num_rows if m.num_rows is not None else 'unknown'}")
    print(f"  shards:   {len(m.data_files)}")
    print(f"  private:  {m.is_private()}")
    if m.source:
        src = m.source
        if src.get("hf_repo"):
            print(f"  source:   hf:{src['hf_repo']}@{src.get('hf_revision', '?')} "
                  f"(split={src.get('split')}, config={src.get('config_name')})")
        else:
            print(f"  source:   {src}")
    if m.created_at:
        print(f"  created:  {m.created_at}")
    return 0


# ── models: whole models in object storage, served without the Hub ──────────
def _model_store():
    from src.remote.artifacts import ArtifactStoreError, model_store
    try:
        return model_store()
    except ArtifactStoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return None


def _models_push(args: argparse.Namespace) -> int:
    from src.remote.artifacts import ArtifactStoreError
    from src.remote.inference import _STORE_MODEL_RE

    src = Path(args.dir)
    if not _STORE_MODEL_RE.match(args.name):
        print("error: name must be letters, digits, '.', '_' or '-', with up to three '/' (e.g. hf/org/repo)", file=sys.stderr)
        return 2
    if not src.is_dir() or src.is_symlink():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 2
    files = sorted(f for f in src.rglob("*") if f.is_file())
    links = [f for f in src.rglob("*") if f.is_symlink()]
    if links:
        print(f"error: {src} contains symlinks ({links[0].name}, ...); push a real copy "
              "(e.g. `kessel flatten` an overlay first)", file=sys.stderr)
        return 2
    if not ((src / "config.json").is_file() or (src / "adapter_config.json").is_file()
            or any(src.rglob("*.gguf"))):
        print(f"error: {src} has no config.json, adapter_config.json or .gguf files", file=sys.stderr)
        return 2
    store = _model_store()
    if store is None:
        return 1
    total = sum(f.stat().st_size for f in files)
    dest = f"s3://{store.bucket}/{store.prefix}/{args.name}"
    print(f"{src} -> {dest} ({store.tier} tier)")
    print(f"  files: {len(files)}  size: {total / 1024 ** 3:.1f} GiB")
    if args.dry_run:
        print("  dry run: nothing uploaded")
        return 0
    try:
        store.push_dir(src, args.name, snapshot=False, workers=args.workers)
    except ArtifactStoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"  pushed; serve with {{\"model\": {{\"store_model\": \"{args.name}\"}}}}")
    return 0


def _models_mirror(args: argparse.Namespace) -> int:
    from src.model_mirror import ModelMirrorError, ensure_model_mirror

    store = _model_store()
    if store is None:
        return 1
    try:
        ref = ensure_model_mirror(args.repo_id, store=store, revision=args.revision,
                                  token=os.environ.get("HF_TOKEN"), force=args.force,
                                  workers=args.workers)
    except ModelMirrorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    source = (store._manifest(ref[len("store:"):]).get("metadata") or {}).get("source") or {}
    print(f"{args.repo_id} -> {ref} (commit {str(source.get('hf_sha', '?'))[:12]}, {store.tier} tier)")
    print(f'  use it as {{"base_model": "{ref}"}}, or set MODEL_MIRROR=true to use mirrors automatically')
    return 0


def _models_info(args: argparse.Namespace) -> int:
    from src.remote.artifacts import ArtifactStoreError

    store = _model_store()
    if store is None:
        return 1
    try:
        manifest = store._manifest(args.name)
    except (FileNotFoundError, ArtifactStoreError) as e:
        print(f"Not found: {args.name} ({e or 'no manifest'})", file=sys.stderr)
        return 1
    files = manifest["files"]
    print(f"s3://{store.bucket}/{store.prefix}/{args.name}")
    print(f"  generation: {manifest['generation']}")
    print(f"  created:    {manifest.get('created_at')}")
    print(f"  files:      {len(files)}  size: {sum(f['size'] for f in files) / 1024 ** 3:.1f} GiB")
    return 0


# ── storage: tier browsing / sync ────────────────────────────────────────────
def _storage_tier(name: str, bucket=None):
    """Resolve a configured storage tier (optionally a view of another bucket
    on its endpoint) or exit with a helpful message."""
    from storage_tiers import StorageTierError, get_tier

    try:
        return get_tier(name).with_bucket(bucket)
    except StorageTierError as e:
        print(f"{e}. Configure it in the server's .env (see docs/storage-tiers.md).",
              file=sys.stderr)
        raise SystemExit(2)


def _storage_tiers(args: argparse.Namespace) -> int:
    from storage_tiers import du, get_tiers, human_bytes

    for t in get_tiers().values():
        d = t.describe()
        badge = "private" if d["private"] else "shared"
        if not d["configured"]:
            print(f"{d['name']:<10} (not configured)")
            continue
        line = f"{d['name']:<10} {badge:<8} {d['bucket']} @ {d['endpoint_host'] or 'default endpoint'}"
        if args.usage:
            try:
                u = du(t)
                line += f"   {u['objects']} objects, {human_bytes(u['bytes'])}"
            except Exception as e:  # noqa: BLE001
                line += f"   (usage unavailable: {type(e).__name__})"
        if d["hint_prefix"]:
            line += f"   [runs under {d['hint_prefix']}]"
        print(line)
    return 0


def _storage_ls(args: argparse.Namespace) -> int:
    from storage_tiers import human_bytes, ls

    t = _storage_tier(args.tier, args.bucket)
    out = ls(t, args.prefix or "", search=args.search or "", sort=args.sort,
             order="desc" if args.desc else "asc", limit=args.limit)
    where = f"{t.name}:{t.bucket}/{out['prefix']}"
    if not out["dirs"] and not out["objects"]:
        print(f"{where} is empty.")
        return 0
    print(where)
    for d in out["dirs"]:
        print(f"  {'':>10}  {'':<19}  {d['name']}")
    for o in out["objects"]:
        mtime = (o["mtime"] or "")[:19].replace("T", " ")
        print(f"  {human_bytes(o['size']):>10}  {mtime:<19}  {o['name']}")
    if out["truncated"]:
        print(f"  … {out['total'] - len(out['dirs']) - len(out['objects'])} more entr(y/ies) not shown "
              f"(use --limit)")
    return 0


def _storage_du(args: argparse.Namespace) -> int:
    from storage_tiers import du

    t = _storage_tier(args.tier, args.bucket)
    u = du(t, args.prefix or "")
    print(f"{t.name}:{t.bucket}/{u['prefix']}  {u['objects']} object(s), {u['human']}")
    return 0


def _storage_buckets(args: argparse.Namespace) -> int:
    from storage_tiers import du, list_buckets

    if args.create:  # kept for compatibility; prefer `merlina storage mb`
        args.name = args.create
        return _storage_mb(args)
    t = _storage_tier(args.tier)
    r = list_buckets(t)
    if not r["ok"]:
        print(r["message"], file=sys.stderr)
        if r["hint"]:
            print(f"hint: {r['hint']}", file=sys.stderr)
        print("Configured bucket(s):", file=sys.stderr)
    for b in r["buckets"]:
        line = f"{b['name']:<40}"
        line += "  * default" if b["default"] else ""
        if b["creation_date"]:
            line += f"  created {b['creation_date'][:10]}"
        if args.sizes:
            try:
                u = du(t.with_bucket(b["name"]))
                line += f"  {u['objects']} objects, {u['human']}"
            except Exception as e:  # noqa: BLE001
                line += f"  (size unavailable: {type(e).__name__})"
        print(line, file=sys.stdout if r["ok"] else sys.stderr)
    return 0 if r["ok"] else 1


def _storage_mb(args: argparse.Namespace) -> int:
    from storage_tiers import make_bucket

    t = _storage_tier(args.tier)
    r = make_bucket(t, args.name)
    print(r["message"], file=sys.stdout if r["ok"] else sys.stderr)
    if r.get("hint"):
        print(f"hint: {r['hint']}", file=sys.stderr)
    return 0 if r["ok"] else 1


def _storage_rb(args: argparse.Namespace) -> int:
    from storage_tiers import StorageTierError, delete_bucket

    t = _storage_tier(args.tier)
    try:
        plan = delete_bucket(t, args.name, dry_run=True)
        if not plan["ok"]:
            print(plan["message"], file=sys.stderr)
            if plan["hint"]:
                print(f"hint: {plan['hint']}", file=sys.stderr)
            return 1
        if not args.yes:
            print(f"Would delete bucket '{args.name}' on '{t.name}' "
                  f"({plan['objects']} object(s), {plan['human']}). "
                  f"Re-run with --yes{' --empty' if plan['objects'] else ''} to delete.")
            return 1
        r = delete_bucket(t, args.name, confirm_name=args.name, empty_first=args.empty)
    except StorageTierError as e:
        print(f"Refusing: {e}", file=sys.stderr)
        return 2
    print(r["message"], file=sys.stdout if r["ok"] else sys.stderr)
    if r.get("hint"):
        print(f"hint: {r['hint']}", file=sys.stderr)
    return 0 if r["ok"] else 1


def _storage_sync(args: argparse.Namespace) -> int:
    from storage_tiers import StorageTierError, human_bytes, sync

    src, dst = _storage_tier(args.src, args.src_bucket), _storage_tier(args.dst, args.dst_bucket)
    if src.name == dst.name and src.bucket == dst.bucket:
        print("Source and destination must differ (tier or bucket).", file=sys.stderr)
        return 2
    last = {"key": None}

    def progress(p):
        if p.get("current_key") and p["current_key"] != last["key"]:
            last["key"] = p["current_key"]
            print(f"  [{p['objects_done'] + 1}/{p['objects_total']}] {p['current_key']}  "
                  f"({human_bytes(p['bytes_done'])} / {human_bytes(p['bytes_total'])})",
                  flush=True)

    try:
        res = sync(src, dst, args.prefix or "", dry_run=args.dry_run, move=args.move,
                   verify=args.verify, overwrite=args.overwrite,
                   allow_private=args.allow_private,
                   progress_cb=None if args.dry_run else progress)
    except StorageTierError as e:
        print(f"Cannot sync: {e}", file=sys.stderr)
        return 1

    verb = "move" if args.move else "copy"
    head = f"{src.name}:{src.bucket}/{res.prefix} -> {dst.name}:{dst.bucket}"
    if args.dry_run:
        print(f"DRY RUN — {head}")
        print(f"  would {verb}: {len(res.to_copy)} object(s), {human_bytes(res.bytes_to_copy)}")
    else:
        print(f"{'Moved' if args.move else 'Synced'} {head}")
        print(f"  copied:  {len(res.copied)} object(s), {human_bytes(res.bytes_copied)}")
    print(f"  skipped: {len(res.skipped)} object(s), {human_bytes(res.bytes_skipped)} already present")
    if res.refused:
        print(f"  refused: {len(res.refused)} private object(s) — never copied to a shared tier:")
        for r in res.refused[:20]:
            print(f"    {r['key']}  ({r['reason']})")
        if len(res.refused) > 20:
            print(f"    … and {len(res.refused) - 20} more")
    if res.meta_missing:
        print(f"  note: {len(res.meta_missing)} object(s) too large for sha256 metadata "
              f"(compared by size next time; use --verify to hash)")
    if res.deleted:
        print(f"  deleted from {src.name}: {len(res.deleted)} object(s)")
    if res.move_aborted:
        print(f"  move aborted: {res.move_aborted}", file=sys.stderr)
    for e in res.errors[:20]:
        print(f"  error: {e['key']}: {e['error']}", file=sys.stderr)
    if args.dry_run:
        print("No data was transferred. Re-run without --dry-run to apply.")
    return 1 if res.errors else 0


def _storage_rm(args: argparse.Namespace) -> int:
    from storage_tiers import StorageTierError, delete

    t = _storage_tier(args.tier, args.bucket)
    try:
        if not args.yes:
            plan = delete(t, args.prefix, dry_run=True)
            print(f"Would delete {plan['objects']} object(s), {plan['human']} under "
                  f"{t.name}:{t.bucket}/{plan['prefix']}/. Re-run with --yes to delete.")
            return 1
        r = delete(t, args.prefix, confirm=True)
    except StorageTierError as e:
        print(f"Refusing: {e}", file=sys.stderr)
        return 2
    print(f"Deleted {r['objects']} object(s), {r['human']} under {t.name}:{t.bucket}/{r['prefix']}/.")
    return 0


def _config_migrate_env(args: argparse.Namespace) -> int:
    """Rename legacy storage variables in a .env (dry run unless --write).
    Prints key names only, never values."""
    from env_migration import format_plan, migrate_file

    path = args.path or ".env"
    if not os.path.isfile(path):
        print(f"No such file: {path}", file=sys.stderr)
        return 2
    plan, backup = migrate_file(path, write=args.write)
    print(format_plan(plan, write=args.write, backup=backup))
    return 1 if plan.conflicts else 0


def _serve(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("torch") is None:
        print(TORCH_INSTALL_HINT, file=sys.stderr)
        return 1

    # Settings are read from the environment when `config` is first imported,
    # so CLI flags are applied as env overrides before importing merlina.
    if args.host is not None:
        os.environ["HOST"] = args.host
    if args.port is not None:
        os.environ["PORT"] = str(args.port)

    import merlina
    merlina.main()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlina",
        description="Merlina - Magical Model Training. Run `merlina serve` and "
        "open the web UI to start fine-tuning.",
    )
    parser.add_argument(
        "--version", action="version", version=f"merlina {_get_version()}"
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve", help="Start the Merlina server (default command)"
    )
    serve_parser.add_argument(
        "--host", default=None, help="Bind address (default: 0.0.0.0, or HOST env var)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=None, help="Port (default: 8000, or PORT env var)"
    )

    # ── datasets: internal/external store management ─────────────────────────
    ds_parser = subparsers.add_parser(
        "datasets", help="Manage the internal/external dataset store"
    )
    ds_sub = ds_parser.add_subparsers(dest="ds_command")

    promote_p = ds_sub.add_parser(
        "promote",
        help="Copy a dataset revision between store tiers (e.g. MinIO -> R2)",
    )
    promote_p.add_argument("name", help="Dataset ref: swl://name[@rev], name, or name@rev")
    promote_p.add_argument(
        "--from", dest="src", default="internal", choices=["internal", "external"],
        help="Source tier (default: internal)",
    )
    promote_p.add_argument(
        "--to", dest="dst", default="external", choices=["internal", "external"],
        help="Destination tier (default: external)",
    )
    promote_p.add_argument(
        "--overwrite", action="store_true",
        help="Recopy even if the revision already exists in the destination",
    )
    promote_p.add_argument(
        "--allow-private", action="store_true",
        help="Override the private guard (danger: private data is meant to stay internal)",
    )
    promote_p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be copied (shards + size) without transferring anything",
    )

    put_p = ds_sub.add_parser("put", help="Store a local JSON/JSONL/CSV/Parquet file as swl://<name>@<rev>")
    put_p.add_argument("file", help="Dataset file")
    put_p.add_argument("name", help="Dataset name, e.g. b3/pr-review-dpo")
    put_p.add_argument("--private", action="store_true",
                       help="Mark the revision private (never promoted to the external tier)")
    put_p.add_argument("--store", default="internal", choices=["internal", "external"],
                       help="Which tier to write (default: internal)")

    info_p = ds_sub.add_parser("info", help="Show a stored dataset revision's manifest")
    info_p.add_argument("name", help="Dataset ref: swl://name[@rev], name, or name@rev")
    info_p.add_argument(
        "--store", default="internal", choices=["internal", "external"],
        help="Which tier to look in (default: internal)",
    )

    # ── models: whole models in object storage ──────────────────────────────
    md_parser = subparsers.add_parser(
        "models", help="Push whole models to object storage for remote serving (no Hub repo)"
    )
    md_sub = md_parser.add_subparsers(dest="md_command")
    push_p = md_sub.add_parser(
        "push", help="Upload a model directory to the artifact store as models/<name>")
    push_p.add_argument("dir", help="Model directory (config.json + safetensors)")
    push_p.add_argument("name", help="Store name, e.g. C0; serve with model.store_model")
    push_p.add_argument("--workers", type=int, default=8, help="Parallel uploads (default: 8)")
    push_p.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    mirror_p = md_sub.add_parser(
        "mirror", help="Copy a HuggingFace Hub model into the store as hf/<org>/<repo>")
    mirror_p.add_argument("repo_id", help="Hub model id, e.g. Qwen/Qwen3-8B")
    mirror_p.add_argument("--revision", default=None, help="Branch, tag or commit (default: main)")
    mirror_p.add_argument("--force", action="store_true",
                          help="Re-publish even if the mirror is already at the Hub's commit")
    mirror_p.add_argument("--workers", type=int, default=8, help="Parallel uploads (default: 8)")
    minfo_p = md_sub.add_parser("info", help="Show a stored model's manifest summary")
    minfo_p.add_argument("name")

    # ── storage: browse / sync / prune object-storage tiers ─────────────────
    st_parser = subparsers.add_parser(
        "storage", help="Browse, sync and prune object-storage tiers (internal/external/artifacts)"
    )
    st_sub = st_parser.add_subparsers(dest="st_command")
    tiers_p = st_sub.add_parser("tiers", help="List configured storage tiers")
    tiers_p.add_argument("--usage", action="store_true", help="Also total each tier (lists every object)")
    bucket_help = "Bucket on the tier's endpoint (default: the tier's configured bucket)"
    ls_p = st_sub.add_parser("ls", help="List one level of a tier")
    ls_p.add_argument("tier")
    ls_p.add_argument("prefix", nargs="?", default="")
    ls_p.add_argument("--bucket", default=None, help=bucket_help)
    ls_p.add_argument("--search", default="", help="Only names containing this text")
    ls_p.add_argument("--sort", default="name", choices=["name", "size", "mtime"])
    ls_p.add_argument("--desc", action="store_true", help="Reverse the sort order")
    ls_p.add_argument("--limit", type=int, default=1000, help="Max entries to show (default 1000)")
    du_p = st_sub.add_parser("du", help="Object count and bytes under a prefix")
    du_p.add_argument("tier")
    du_p.add_argument("prefix", nargs="?", default="")
    du_p.add_argument("--bucket", default=None, help=bucket_help)
    bk_p = st_sub.add_parser("buckets", help="List buckets on a tier's endpoint")
    bk_p.add_argument("tier")
    bk_p.add_argument("--sizes", action="store_true", help="Also total each bucket (lists every object)")
    bk_p.add_argument("--create", metavar="NAME", default=None, help=argparse.SUPPRESS)
    mb_p = st_sub.add_parser("mb", help="Make a bucket (uses the tier's admin key if configured)")
    mb_p.add_argument("tier")
    mb_p.add_argument("name")
    rb_p = st_sub.add_parser("rb", help="Remove a bucket (never a configured Merlina bucket)")
    rb_p.add_argument("tier")
    rb_p.add_argument("name")
    rb_p.add_argument("--empty", action="store_true", help="Delete its objects first if it is not empty")
    rb_p.add_argument("--yes", action="store_true", help="Actually delete (without it: preview)")
    sync_p = st_sub.add_parser("sync", help="Copy objects under a prefix from one tier to another")
    sync_p.add_argument("src")
    sync_p.add_argument("dst")
    sync_p.add_argument("prefix", nargs="?", default="")
    sync_p.add_argument("--src-bucket", default=None, help="Source bucket (default: the tier's)")
    sync_p.add_argument("--dst-bucket", default=None, help="Destination bucket (default: the tier's)")
    sync_p.add_argument("--dry-run", action="store_true", help="Plan only; transfer nothing")
    sync_p.add_argument("--move", action="store_true",
                        help="Delete source objects after every copy is verified (implies --verify)")
    sync_p.add_argument("--verify", action="store_true",
                        help="Hash objects to confirm identity (skip checks and post-write)")
    sync_p.add_argument("--overwrite", action="store_true", help="Recopy even identical objects")
    sync_p.add_argument("--allow-private", action="store_true",
                        help="Override the private guard (danger: private data is meant to stay internal)")
    rm_p = st_sub.add_parser("rm", help="Delete every object under a prefix")
    rm_p.add_argument("tier")
    rm_p.add_argument("prefix")
    rm_p.add_argument("--bucket", default=None, help=bucket_help)
    rm_p.add_argument("--yes", action="store_true", help="Actually delete (without it: preview)")

    # ── config: .env maintenance ──────────────────────────────────────────────
    cfg_parser = subparsers.add_parser("config", help="Maintain the server's .env")
    cfg_sub = cfg_parser.add_subparsers(dest="cfg_command")
    mig_p = cfg_sub.add_parser(
        "migrate-env",
        help="Rename legacy storage variables (S3_*, R2_*, MERLINA_ARTIFACT_*, ...) "
             "to STORAGE_*/DATASET_*/ARTIFACT_* (dry run unless --write)",
    )
    mig_p.add_argument("path", nargs="?", default=".env", help="The .env file (default: ./.env)")
    mig_p.add_argument("--write", action="store_true",
                       help="Apply the changes (writes a timestamped backup first)")

    args = parser.parse_args(argv)

    if args.command == "datasets":
        if args.ds_command == "promote":
            return _datasets_promote(args)
        if args.ds_command == "info":
            return _datasets_info(args)
        if args.ds_command == "put":
            return _datasets_put(args)
        ds_parser.print_help()
        return 2

    if args.command == "models":
        if args.md_command == "push":
            return _models_push(args)
        if args.md_command == "info":
            return _models_info(args)
        if args.md_command == "mirror":
            return _models_mirror(args)
        md_parser.print_help()
        return 2

    if args.command == "config":
        if args.cfg_command == "migrate-env":
            return _config_migrate_env(args)
        cfg_parser.print_help()
        return 2

    if args.command == "storage":
        handlers = {"tiers": _storage_tiers, "ls": _storage_ls, "du": _storage_du,
                    "buckets": _storage_buckets, "mb": _storage_mb, "rb": _storage_rb,
                    "sync": _storage_sync, "rm": _storage_rm}
        handler = handlers.get(args.st_command)
        if handler is None:
            st_parser.print_help()
            return 2
        return handler(args)

    # Bare `merlina` serves too, but without flags (use `merlina serve --port N`).
    if args.command is None:
        args.host = None
        args.port = None
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
