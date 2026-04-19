"""Tests for build_grimoire_config — the shim that tolerates older grimoire
versions missing newer TrainingConfig fields.

Regression test for the case where Merlina was built against a grimoire
release that exposed torch_compile/use_liger/neftune_alpha/eval_on_start, but
the installed grimoire predates those fields. Previously the job crashed with
``TypeError: TrainingConfig.__init__() got an unexpected keyword argument
'torch_compile'``; now unknown defaults are dropped and explicitly-enabled
features raise a clear error.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.utils import build_grimoire_config


# ---------------------------------------------------------------------------
# Fake grimoire TrainingConfig dataclasses simulating different versions.
# ---------------------------------------------------------------------------


@dataclass
class OldTrainingConfig:
    """Simulates grimoire before Liger/torch_compile/NEFTune landed."""

    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    mixed_precision: str = "bf16"
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    seed: int = 42
    run_name: Optional[str] = None
    log_with: Optional[str] = None
    project_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None


@dataclass
class NewTrainingConfig(OldTrainingConfig):
    """Simulates current grimoire main with all opt-in fields."""

    torch_compile: bool = False
    use_liger: bool = False
    neftune_alpha: Optional[float] = None
    eval_on_start: bool = False


# ---------------------------------------------------------------------------
# Default-valued opt-ins: dropped silently on old grimoire, honored on new.
# ---------------------------------------------------------------------------


def test_old_grimoire_drops_unsupported_defaults():
    """Passing default torch_compile/use_liger to an old config should not crash."""
    cfg = build_grimoire_config(
        OldTrainingConfig,
        output_dir="./out",
        num_epochs=1,
        torch_compile=False,
        use_liger=False,
        neftune_alpha=None,
        eval_on_start=False,
    )
    assert cfg.output_dir == "./out"
    assert cfg.num_epochs == 1
    # Confirm the filtered kwargs never reached the dataclass.
    assert not hasattr(cfg, "torch_compile")


def test_new_grimoire_keeps_opt_in_fields():
    cfg = build_grimoire_config(
        NewTrainingConfig,
        output_dir="./out",
        torch_compile=True,
        use_liger=True,
        neftune_alpha=5.0,
        eval_on_start=True,
    )
    assert cfg.torch_compile is True
    assert cfg.use_liger is True
    assert cfg.neftune_alpha == 5.0
    assert cfg.eval_on_start is True


# ---------------------------------------------------------------------------
# Explicitly enabled but unsupported → must raise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwarg,value",
    [
        ("torch_compile", True),
        ("use_liger", True),
        ("neftune_alpha", 5.0),
        ("eval_on_start", True),
    ],
)
def test_enabled_opt_in_without_support_raises(kwarg, value):
    with pytest.raises(RuntimeError, match=kwarg):
        build_grimoire_config(OldTrainingConfig, output_dir="./out", **{kwarg: value})


# ---------------------------------------------------------------------------
# Non-opt-in unknown kwargs are always filtered, no matter the value.
# ---------------------------------------------------------------------------


def test_unknown_non_opt_in_kwargs_dropped_silently():
    """Adafactor-style pass-through kwargs should be dropped without raising."""
    cfg = build_grimoire_config(
        OldTrainingConfig,
        output_dir="./out",
        adafactor_relative_step=True,
        adafactor_scale_parameter=False,
        adafactor_decay_rate=-0.8,
    )
    assert cfg.output_dir == "./out"


def test_non_dataclass_raises():
    class NotADataclass:
        pass

    with pytest.raises(TypeError):
        build_grimoire_config(NotADataclass, foo=1)


# ---------------------------------------------------------------------------
# Regression: the exact kwarg list training_runner/train_worker pass.
# ---------------------------------------------------------------------------


def _full_merlina_kwargs():
    """Mirror the kwargs Merlina hands to grimoire's TrainingConfig."""
    return dict(
        output_dir="./results/job_x",
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        mixed_precision="bf16",
        optimizer="adamw",
        lr_scheduler="cosine",
        logging_steps=10,
        eval_steps=12,
        save_steps=12,
        save_total_limit=2,
        seed=42,
        run_name="merlina-run",
        log_with=None,
        project_name=None,
        wandb_tags=[],
        wandb_notes=None,
        # Features added in later grimoire releases:
        torch_compile=False,
        use_liger=False,
        neftune_alpha=None,
        eval_on_start=False,
        # Adafactor knobs — not on every grimoire:
        adafactor_relative_step=True,
        adafactor_scale_parameter=True,
        adafactor_warmup_init=True,
        adafactor_decay_rate=-0.8,
        adafactor_beta1=None,
        adafactor_clip_threshold=1.0,
    )


def test_regression_old_grimoire_does_not_crash():
    """The bug we hit: Merlina's full kwarg set must build on old grimoire."""
    cfg = build_grimoire_config(OldTrainingConfig, **_full_merlina_kwargs())
    assert cfg.output_dir == "./results/job_x"


def test_regression_new_grimoire_accepts_full_kwargs():
    cfg = build_grimoire_config(NewTrainingConfig, **_full_merlina_kwargs())
    assert cfg.torch_compile is False
    assert cfg.use_liger is False
