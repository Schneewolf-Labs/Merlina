"""Tests for the intermediate-checkpoint cadence policy."""

import pytest

from src.checkpoint_policy import _NEVER, describe, epoch_end_saves, resolve_save_steps


class TestResolveSaveSteps:
    def test_none_follows_eval_steps(self):
        """Default (None) preserves the pre-existing behaviour: save when we eval."""
        assert resolve_save_steps(None, 25, 100) == 25

    def test_none_with_no_eval_steps(self):
        assert resolve_save_steps(None, None, 100) is None

    def test_zero_disables_intermediate_checkpoints(self):
        """0 must produce an interval the run can never reach, not 'save every step'."""
        assert resolve_save_steps(0, 25, 100) >= _NEVER
        assert resolve_save_steps(0.0, 25, 100) >= _NEVER

    def test_ratio_expands_against_total_steps(self):
        assert resolve_save_steps(0.5, 25, 100) == 50
        assert resolve_save_steps(0.25, 25, 100) == 25

    def test_ratio_never_returns_zero(self):
        """A tiny ratio on a short run must not collapse to 0 (= every step)."""
        assert resolve_save_steps(0.01, 5, 10) == 1

    def test_ratio_without_total_steps_falls_back(self):
        """Without a step count we cannot expand a ratio; fall back rather than guess."""
        assert resolve_save_steps(0.5, 25, None) == 25

    def test_absolute_value_passes_through(self):
        assert resolve_save_steps(40, 25, 100) == 40
        assert resolve_save_steps(40.0, 25, 100) == 40

    @pytest.mark.parametrize("value", [1, 2, 999])
    def test_absolute_is_integer(self, value):
        assert isinstance(resolve_save_steps(value, 25, 100), int)


class TestDescribe:
    def test_disabled(self):
        assert "disabled" in describe(resolve_save_steps(0, 25, 100))

    def test_interval(self):
        assert "every 25 steps" in describe(25)

    def test_none(self):
        assert "disabled" in describe(None)


class TestEpochEndSaves:
    def test_disabled_only_when_save_steps_is_zero(self):
        """save_steps=0 means no intermediate checkpoints -- including at epoch boundaries.

        The end-of-epoch save is a separate path in the trainer writing the same full-model
        snapshot, so leaving it on defeats the setting.
        """
        assert epoch_end_saves(0) is False
        assert epoch_end_saves(0.0) is False

    def test_enabled_by_default(self):
        assert epoch_end_saves(None) is True

    @pytest.mark.parametrize("value", [0.5, 1, 25, 100])
    def test_enabled_for_any_real_cadence(self, value):
        assert epoch_end_saves(value) is True
