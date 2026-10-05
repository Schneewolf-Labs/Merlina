"""A missing W&B key must cost metrics, not the whole run.

Preflight warns rather than errors when `use_wandb` is set with no key available, and the warning
says training will proceed without logging. The worker did not honour that: it set
`log_with="wandb"` regardless, accelerate initialised wandb, and the job died with
`UsageError: api_key not configured (no-tty)` — after the model and dataset were already loaded.

Observed on a real 9B run: validation passed with one warning, the worker started, and the job
failed minutes later. The fix is to check for a usable key (env var, ~/.netrc, or a prior
`wandb login`) and fall back to no tracker when there isn't one.
"""

import re
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "src" / "train_worker.py"


def wandb_block() -> str:
    src = WORKER.read_text()
    start = src.index("if config.use_wandb and is_main and wandb is not None:")
    return src[start:start + 2000]


def test_missing_key_does_not_enable_the_tracker():
    block = wandb_block()
    # The key check must happen before log_with is set, and a failure must be caught.
    assert "except Exception" in block, "no fallback path when W&B initialisation fails"
    assert "log_with = None" in block, "failure path does not disable the tracker"


def test_checks_more_than_the_env_var():
    """A user logged in via ~/.netrc has no WANDB_API_KEY but is perfectly able to log."""
    block = wandb_block()
    assert "api_key" in block, "does not consult wandb's own resolved credentials"
    assert not re.search(r'os\.getenv\(\s*["\']WANDB_API_KEY', block), \
        "gates on the env var alone, which disables W&B for netrc-authenticated users"


def test_preflight_still_only_warns():
    """The promise being honoured lives in preflight; it must stay a warning, not an error."""
    pre = (Path(__file__).resolve().parents[1] / "src" / "preflight_checks.py").read_text()
    i = pre.index("W&B logging enabled but no WANDB_API_KEY found")
    context = pre[max(0, i - 400):i]
    assert "self.warnings.append" in context, "missing W&B key should warn, not error"
