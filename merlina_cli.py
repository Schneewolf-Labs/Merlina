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
        prefix = "S3_" if which == "internal" else "R2_"
        print(
            f"The '{which}' dataset store is not configured.\n"
            f"Set {prefix}ENDPOINT_URL / {prefix}DATASET_BUCKET / {prefix}ACCESS_KEY / "
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

    info_p = ds_sub.add_parser("info", help="Show a stored dataset revision's manifest")
    info_p.add_argument("name", help="Dataset ref: swl://name[@rev], name, or name@rev")
    info_p.add_argument(
        "--store", default="internal", choices=["internal", "external"],
        help="Which tier to look in (default: internal)",
    )

    args = parser.parse_args(argv)

    if args.command == "datasets":
        if args.ds_command == "promote":
            return _datasets_promote(args)
        if args.ds_command == "info":
            return _datasets_info(args)
        ds_parser.print_help()
        return 2

    # Bare `merlina` serves too, but without flags (use `merlina serve --port N`).
    if args.command is None:
        args.host = None
        args.port = None
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
