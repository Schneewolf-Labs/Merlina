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

    args = parser.parse_args(argv)

    # Bare `merlina` serves too, but without flags (use `merlina serve --port N`).
    if args.command is None:
        args.host = None
        args.port = None
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
