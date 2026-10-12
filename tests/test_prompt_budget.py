"""Prompt budgets: long prompts keep their end, the default budget follows max_length,
and heavy truncation is reported (and refused) before training starts."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from src import prompt_budget as pb
from src.prompt_budget import (
    BUDGET_COLUMN,
    PromptTruncationError,
    apply_prompt_budgets,
    budget_splits,
    build_report,
    check_truncation,
    count_prompt_tokens,
    encode_prompt_response_keep_end,
    record_reports,
    resolve_prompt_budget,
    tokenize_kto_row,
    tokenize_preference_row,
    tokenize_sft_row,
)


class CharTokenizer:
    """One token per character. Accepts a string or a batch (list of strings)."""

    def __init__(self, bos_token_id=None, eos_token_id=None, append_eos=False):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.append_eos = append_eos

    def _encode(self, text, add_special_tokens):
        ids = [ord(c) for c in text]
        if add_special_tokens:
            if self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
            if self.append_eos and self.eos_token_id is not None:
                ids = ids + [self.eos_token_id]
        return ids

    def __call__(self, text, add_special_tokens=True, **_):
        if isinstance(text, list):
            ids = [self._encode(t, add_special_tokens) for t in text]
            return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}
        ids = self._encode(text, add_special_tokens)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _decode(ids, bos=None):
    return "".join(chr(i) for i in ids if i != bos)


# --------------------------------------------------------------------------- #
# Budget resolution
# --------------------------------------------------------------------------- #

class TestResolvePromptBudget:
    def test_explicit_budget_is_used_as_is(self):
        assert resolve_prompt_budget(100, 8192, 1024) == 1024

    def test_auto_budget_is_max_length_minus_response(self):
        assert resolve_prompt_budget(200, 8192, None) == 7992

    def test_auto_budget_caps_response_claim_at_half(self):
        assert resolve_prompt_budget(10_000, 8192, None) == 4096

    def test_auto_budget_with_empty_response(self):
        assert resolve_prompt_budget(0, 2048, None) == 2048


# --------------------------------------------------------------------------- #
# Measurement + report
# --------------------------------------------------------------------------- #

class TestMeasurement:
    def test_prompt_count_includes_bos_and_drops_trailing_eos(self):
        tok = CharTokenizer(bos_token_id=1, eos_token_id=2, append_eos=True)
        # "abc" -> [BOS, a, b, c, EOS]; the encoder strips that EOS, so 4 tokens
        assert count_prompt_tokens(tok, ["abc"]) == [4]

    def test_preference_rows_reserve_the_longer_response(self):
        ds = Dataset.from_list([{"prompt": "p", "chosen": "aa", "rejected": "bbbbb"}])
        prompts, responses = pb.measure_dataset(ds, CharTokenizer(), ["chosen", "rejected"])
        assert prompts == [1]
        assert responses == [5]


class TestReport:
    def test_agent_transcripts_under_old_fixed_cap_are_all_flagged(self):
        """The voided run: 144 rows of ~6k-token prompts, max_prompt_length=1024."""
        prompts = [6000 + i for i in range(144)]
        responses = [150] * 144
        report = build_report(prompts, responses, max_length=8192, max_prompt_length=1024)

        assert report["prompts_truncated"] == 144
        assert report["prompts_truncated_fraction"] == 1.0
        assert report["prompt_tokens"]["p50"] == 6071
        assert report["prompt_tokens"]["max"] == 6143
        with pytest.raises(PromptTruncationError) as exc:
            check_truncation(report, 0.10)
        msg = str(exc.value)
        assert "144 of 144" in msg
        assert "max_prompt_length=1024" in msg
        assert "prompt_truncation_tolerance" in msg

    def test_same_rows_fit_under_the_auto_budget(self):
        prompts = [6000 + i for i in range(144)]
        report = build_report(prompts, [150] * 144, max_length=8192, max_prompt_length=None)

        assert report["budget_mode"] == "auto"
        assert report["prompts_truncated"] == 0
        assert report["responses_truncated"] == 0
        check_truncation(report, 0.10)  # no raise

    def test_percentiles_are_nearest_rank(self):
        report = build_report(list(range(1, 101)), [1] * 100, max_length=4096, max_prompt_length=None)
        assert report["prompt_tokens"] == {"p50": 50, "p90": 90, "p95": 95, "p99": 99, "max": 100}

    def test_response_truncation_counted(self):
        # explicit budget 100 of max_length 120: a 50-token response loses 30
        report = build_report([100], [50], max_length=120, max_prompt_length=100)
        assert report["responses_truncated"] == 1
        assert report["responses_with_no_tokens_left"] == 0

    def test_tokens_dropped_is_summed(self):
        report = build_report([300, 500, 100], [10, 10, 10], max_length=512, max_prompt_length=256)
        assert report["prompts_truncated"] == 2
        assert report["prompt_tokens_dropped"] == 44 + 244

    def test_tolerance_boundary(self):
        report = build_report([300] + [10] * 9, [5] * 10, max_length=512, max_prompt_length=256)
        assert report["prompts_truncated_fraction"] == pytest.approx(0.1)
        check_truncation(report, 0.10)  # exactly at tolerance is allowed
        with pytest.raises(PromptTruncationError):
            check_truncation(report, 0.05)

    def test_tolerance_one_never_fails(self):
        report = build_report([5000] * 10, [5] * 10, max_length=512, max_prompt_length=256)
        check_truncation(report, 1.0)

    def test_empty_dataset(self):
        report = build_report([], [], max_length=512, max_prompt_length=None)
        assert report["rows"] == 0
        check_truncation(report, 0.0)

    def test_auto_mode_error_suggests_a_max_length(self):
        report = build_report([1000] * 10, [100] * 10, max_length=512, max_prompt_length=None)
        with pytest.raises(PromptTruncationError, match="raise max_length"):
            check_truncation(report, 0.1)


# --------------------------------------------------------------------------- #
# Applying budgets to a dataset
# --------------------------------------------------------------------------- #

class TestApplyPromptBudgets:
    def test_adds_budget_column_and_returns_public_report(self):
        ds = Dataset.from_list([
            {"prompt": "x" * 900, "chosen": "y" * 50},
            {"prompt": "short", "chosen": "y" * 50},
        ])
        out, report = apply_prompt_budgets(ds, CharTokenizer(), training_mode="sft",
                                           max_length=512, max_prompt_length=None)
        assert out[BUDGET_COLUMN] == [462, 462]
        assert report["prompts_truncated"] == 1
        assert "budgets" not in report

    def test_enforce_raises_before_training(self):
        ds = Dataset.from_list([{"prompt": "x" * 900, "chosen": "y"}] * 4)
        with pytest.raises(PromptTruncationError):
            apply_prompt_budgets(ds, CharTokenizer(), training_mode="sft", max_length=512,
                                 max_prompt_length=256, tolerance=0.1, enforce=True)

    def test_logs_a_warning_when_rows_are_cut(self, caplog):
        ds = Dataset.from_list([{"prompt": "x" * 900, "chosen": "y"}])
        with caplog.at_level("WARNING", logger="src.prompt_budget"):
            apply_prompt_budgets(ds, CharTokenizer(), training_mode="sft", max_length=512,
                                 max_prompt_length=256)
        assert "Prompts truncated (keeping the end): 1/1" in caplog.text
        assert "p50/p90/p95/p99/max" in caplog.text

    def test_reapplying_replaces_existing_column(self):
        ds = Dataset.from_list([{"prompt": "p", "chosen": "c"}])
        once, _ = apply_prompt_budgets(ds, CharTokenizer(), training_mode="sft",
                                       max_length=512, max_prompt_length=None)
        twice, _ = apply_prompt_budgets(once, CharTokenizer(), training_mode="sft",
                                        max_length=512, max_prompt_length=300)
        assert twice[BUDGET_COLUMN] == [300]

    def test_budget_splits_enforces_train_and_reports_eval(self):
        config = SimpleNamespace(max_length=512, max_prompt_length=None,
                                 prompt_truncation_tolerance=0.1)
        train = Dataset.from_list([{"prompt": "p" * 10, "chosen": "c", "rejected": "r"}])
        evals = Dataset.from_list([{"prompt": "p" * 2000, "chosen": "c", "rejected": "r"}])
        train, evals, reports = budget_splits(train, evals, CharTokenizer(), config, "orpo")
        assert set(reports) == {"train", "eval"}
        assert reports["eval"]["prompts_truncated"] == 1  # eval is reported, not enforced
        assert BUDGET_COLUMN in evals.column_names

    def test_budget_splits_without_eval(self):
        config = SimpleNamespace(max_length=512, max_prompt_length=None)
        train = Dataset.from_list([{"prompt": "p", "response": "c", "label": True}])
        _, evals, reports = budget_splits(train, None, CharTokenizer(), config, "kto")
        assert evals is None
        assert set(reports) == {"train"}

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            pb.response_fields_for_mode("grpo")


# --------------------------------------------------------------------------- #
# Tokenization keeps the end of the prompt
# --------------------------------------------------------------------------- #

CHAT_PROMPT = "<sys>long rules and tool schemas</sys><user>what is 2+2?</user><tool>4</tool><asst>"
TAIL = "<user>what is 2+2?</user><tool>4</tool><asst>"


class TestKeepEndTokenization:
    @pytest.fixture(autouse=True)
    def _fallback(self, monkeypatch):
        # Under the test suite grimoire is mocked, so the local encoder is used.
        monkeypatch.setattr(pb, "GRIMOIRE_KEEPS_PROMPT_END", False)

    def test_sft_keeps_final_turns_and_generation_header(self):
        row = {"prompt": CHAT_PROMPT, "chosen": "four", BUDGET_COLUMN: len(TAIL)}
        out = tokenize_sft_row(row, CharTokenizer(), max_length=4096)
        assert _decode(out["input_ids"]) == TAIL + "four"
        assert out["labels"][:len(TAIL)] == [-100] * len(TAIL)
        assert _decode(out["labels"][len(TAIL):]) == "four"

    def test_bos_survives_truncation(self):
        row = {"prompt": "ABCDEF", "chosen": "XY", BUDGET_COLUMN: 3}
        out = tokenize_sft_row(row, CharTokenizer(bos_token_id=1), max_length=64)
        assert out["input_ids"] == [1, ord("E"), ord("F"), ord("X"), ord("Y")]
        assert out["labels"] == [-100, -100, -100, ord("X"), ord("Y")]

    def test_explicit_max_prompt_length_used_without_budget_column(self):
        out = tokenize_sft_row({"prompt": "ABCDEF", "chosen": "X"}, CharTokenizer(),
                               max_length=64, max_prompt_length=2)
        assert _decode(out["input_ids"]) == "EFX"

    def test_preference_pair_shares_one_prompt(self):
        row = {"prompt": "ABCDEFGH", "chosen": "c", "rejected": "rrrrr", BUDGET_COLUMN: 4}
        out = tokenize_preference_row(row, CharTokenizer(), max_length=64)
        assert _decode(out["chosen_input_ids"]) == "EFGHc"
        assert _decode(out["rejected_input_ids"]) == "EFGHrrrrr"
        assert out["chosen_labels"][:4] == out["rejected_labels"][:4] == [-100] * 4

    def test_kto_keeps_prompt_end(self):
        row = {"prompt": "ABCDEF", "response": "ok", "label": 0, BUDGET_COLUMN: 2}
        out = tokenize_kto_row(row, CharTokenizer(), max_length=64)
        assert _decode(out["input_ids"]) == "EFok"
        assert out["kto_label"] is False

    def test_trailing_prompt_eos_is_dropped(self):
        tok = CharTokenizer(eos_token_id=2, append_eos=True)
        out = encode_prompt_response_keep_end(tok, "AB", "CD")
        assert out["input_ids"] == [ord("A"), ord("B"), ord("C"), ord("D")]

    def test_end_to_end_auto_budget(self):
        """Measured budget + tokenization: the whole response and the prompt tail fit."""
        ds = Dataset.from_list([{"prompt": "S" * 600 + TAIL, "chosen": "four"}])
        ds, report = apply_prompt_budgets(ds, CharTokenizer(), training_mode="sft",
                                          max_length=256, max_prompt_length=None)
        out = tokenize_sft_row(ds[0], CharTokenizer(), max_length=256)
        assert len(out["input_ids"]) == 256
        assert _decode(out["input_ids"]).endswith(TAIL + "four")
        assert report["prompts_truncated"] == 1


class TestNativeGrimoirePath:
    """With a grimoire that supports prompt_truncation, Merlina delegates to it."""

    def test_delegates_with_keep_end_and_row_budget(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "grimoire.data", fake)
        monkeypatch.setattr(pb, "GRIMOIRE_KEEPS_PROMPT_END", True)
        tok = CharTokenizer()

        tokenize_sft_row({"prompt": "p", "chosen": "c", BUDGET_COLUMN: 77}, tok, max_length=128)
        kwargs = fake.tokenize_sft.call_args.kwargs
        assert kwargs["prompt_truncation"] == "keep_end"
        assert kwargs["max_prompt_length"] == 77
        assert kwargs["response_field"] == "chosen"

        tokenize_preference_row({"prompt": "p", "chosen": "c", "rejected": "r", BUDGET_COLUMN: 9},
                                tok, max_length=128)
        assert fake.tokenize_preference.call_args.kwargs["prompt_truncation"] == "keep_end"
        assert fake.tokenize_preference.call_args.kwargs["max_prompt_length"] == 9

        tokenize_kto_row({"prompt": "p", "response": "c", "label": True, BUDGET_COLUMN: 5},
                         tok, max_length=128)
        assert fake.tokenize_kto.call_args.kwargs["prompt_truncation"] == "keep_end"

    def test_feature_detection_false_when_grimoire_unavailable(self):
        # grimoire is a MagicMock in the test session: no real signature to inspect
        assert pb._grimoire_supports_keep_end() is False


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

class TestRecordReports:
    def test_stores_report_on_job_metrics_and_keeps_existing_keys(self, tmp_path):
        from src.job_manager import JobManager

        jm = JobManager(db_path=str(tmp_path / "jobs.db"))
        jm.create_job("job1", {"training_mode": "sft"})
        jm.update_job("job1", metrics={"other": 1})

        record_reports(jm, "job1", {"train": {"rows": 3, "prompts_truncated": 1}})

        metrics = jm.get_job("job1").metrics
        assert metrics["other"] == 1
        assert metrics["prompt_budget"]["train"]["prompts_truncated"] == 1

    def test_no_job_manager_is_a_noop(self):
        record_reports(None, "job1", {"train": {}})


# --------------------------------------------------------------------------- #
# Config + preflight
# --------------------------------------------------------------------------- #

class TestConfigDefaults:
    def _config(self, **overrides):
        from merlina import TrainingConfig
        base = {"base_model": "org/model", "output_name": "out"}
        base.update(overrides)
        return TrainingConfig(**base)

    def test_max_prompt_length_defaults_to_auto(self):
        config = self._config()
        assert config.max_prompt_length is None
        assert config.prompt_truncation_tolerance == pytest.approx(0.10)

    def test_explicit_values_still_accepted(self):
        config = self._config(max_length=8192, max_prompt_length=6144,
                              prompt_truncation_tolerance=1.0)
        assert config.max_prompt_length == 6144
        assert config.prompt_truncation_tolerance == 1.0

    def test_preflight_sequence_checks_accept_auto(self):
        from src.preflight_checks import PreflightValidator
        validator = PreflightValidator()
        result = validator._check_training_config(self._config())
        assert result["completion_length"] is None
        assert not any("max_prompt_length" in e for e in validator.errors)

    def test_preflight_still_rejects_prompt_budget_over_max_length(self):
        from src.preflight_checks import PreflightValidator
        validator = PreflightValidator()
        validator._check_training_config(self._config(max_length=1024, max_prompt_length=2048))
        assert any("must be less than" in e for e in validator.errors)
