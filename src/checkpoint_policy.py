"""Checkpoint cadence policy.

Intermediate checkpoints are written with ``accelerator.save_state()``, which snapshots the
full model plus optimizer state — for a 27B LoRA run that is ~52 GB of model weights to save
the 637 MB of adapter that actually changed. On a unified-memory board (GPU and system RAM
share one pool) that write is enough to push free RAM below the memory guard's floor and abort
an otherwise healthy run.

Until the trainer learns to write adapter-only intermediate checkpoints, operators need a way
to turn the cadence down or off. ``save_steps`` does that:

    None (default)  → follow ``eval_steps``, i.e. the behaviour before this setting existed
    0               → no intermediate checkpoints; only the final model is saved
    <1              → ratio of total steps (0.5 = halfway and at the end)
    >=1             → absolute step count
"""

from typing import Optional

# Sentinel used to disable intermediate checkpointing: a save interval the run can never reach.
# The trainer takes a step count rather than a strategy flag, so "never" is expressed as
# "further away than the last step".
_NEVER = 10_000_000


def resolve_save_steps(
    save_steps: Optional[float],
    eval_steps: Optional[int],
    total_steps: Optional[int],
) -> Optional[int]:
    """Return the absolute ``save_steps`` to hand the trainer.

    Args:
        save_steps: the user's ``config.save_steps`` (None, 0, a ratio, or an absolute count).
        eval_steps: already-resolved absolute eval cadence, used when ``save_steps`` is None.
        total_steps: total optimizer steps for the run; required to expand a ratio.
    """
    if save_steps is None:
        return eval_steps

    if save_steps == 0:
        return _NEVER

    if save_steps < 1:
        if not total_steps:
            # Cannot expand a ratio without a step count — fall back to the eval cadence
            # rather than silently checkpointing every step.
            return eval_steps
        return max(1, int(total_steps * save_steps))

    return int(save_steps)


def describe(resolved: Optional[int]) -> str:
    """Human-readable summary for the training log."""
    if resolved is None:
        return "intermediate checkpoints: disabled (no eval cadence configured)"
    if resolved >= _NEVER:
        return "intermediate checkpoints: disabled (final model only)"
    return f"intermediate checkpoints: every {resolved} steps"
