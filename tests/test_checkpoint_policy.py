"""Tests for the intermediate-checkpoint cadence policy."""

import pytest

from src.checkpoint_policy import _NEVER, describe, resolve_save_steps


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
