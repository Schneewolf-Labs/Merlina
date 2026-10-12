"""
Prompt token budgets: how each row's prompt is fitted into ``max_length``.

Long agent transcripts are where silent prompt truncation does the most damage.
A 6k-token transcript cut to a fixed 1024-token cap keeping its *start* trains
the model on a system prompt plus tool schemas, followed by a reply to a
conversation it never saw. This module owns three things:

1. **Direction.** Over-long prompts keep their END (the latest turns, tool
   results and the assistant generation header the response follows), plus a
   leading BOS. Cutting into the system prompt loses some instructions, while
   cutting the tail loses the evidence the response is about. Newer grimoire
   does this natively (``prompt_truncation="keep_end"``). On an older grimoire
   we fall back to a local encoder with identical semantics, so the result does
   not depend on which grimoire version is installed.

2. **Budget.** ``max_prompt_length`` is optional. When it is unset ("auto"),
   each row's prompt gets ``max_length`` minus that row's response, but a
   response may claim at most half of ``max_length``. A row is cut only when
   prompt + response really does not fit, and then the prompt yields to the
   response only up to half the window. Preference pairs reserve room for the
   longer of chosen/rejected, so both sides keep an identical prompt.

3. **Visibility.** Before tokenization we measure every row and produce a
   report: prompt/response token percentiles and how many rows lose prompt or
   response tokens. Runners log it and store it on the job. If more than
   ``prompt_truncation_tolerance`` of the training rows would have their prompt
   cut, training fails before the first step with a message that says how to
   fix it.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Column added to the dataset holding each row's prompt token budget.
BUDGET_COLUMN = "_prompt_budget"

#: In auto mode a response may reserve at most this share of max_length.
AUTO_MAX_RESPONSE_SHARE = 0.5

#: Default fraction of training rows allowed to have their prompt cut.
DEFAULT_TRUNCATION_TOLERANCE = 0.10

PREFERENCE_MODES = {"orpo", "dpo", "simpo", "cpo", "ipo"}

_TOKENIZE_BATCH = 512


class PromptTruncationError(ValueError):
    """Too many rows would lose prompt tokens to truncation."""


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #

def response_fields_for_mode(training_mode: str) -> List[str]:
    """Columns holding the response(s) for a training mode's tokenizer input."""
    mode = training_mode.lower()
    if mode == "sft":
        return ["chosen"]
    if mode == "kto":
        return ["response"]
    if mode in PREFERENCE_MODES:
        return ["chosen", "rejected"]
    raise ValueError(f"Unknown training mode: {training_mode}")


def resolve_prompt_budget(response_tokens: int, max_length: int,
                          max_prompt_length: Optional[int]) -> int:
    """Token budget for one row's prompt.

    An explicit ``max_prompt_length`` is used as-is. In auto mode (``None``)
    the prompt gets whatever the response leaves of ``max_length``, with the
    response's claim capped at half the window.
    """
    if max_prompt_length:
        return int(max_prompt_length)
    reserve = min(max(int(response_tokens), 0), int(max_length * AUTO_MAX_RESPONSE_SHARE))
    return int(max_length) - reserve


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def _batched(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def count_prompt_tokens(tokenizer, prompts: Sequence[str]) -> List[int]:
    """Prompt token counts exactly as the encoder sees them.

    Prompts are tokenized with special tokens (BOS), and a trailing EOS that
    some tokenizers append is not counted. This matches
    ``encode_prompt_response``.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    counts: List[int] = []
    for chunk in _batched(list(prompts), _TOKENIZE_BATCH):
        for ids in tokenizer(list(chunk))["input_ids"]:
            n = len(ids)
            if eos_id is not None and n and ids[-1] == eos_id:
                n -= 1
            counts.append(n)
    return counts


def count_response_tokens(tokenizer, responses: Sequence[str]) -> List[int]:
    """Response token counts (no special tokens), matching the encoder."""
    counts: List[int] = []
    for chunk in _batched(list(responses), _TOKENIZE_BATCH):
        counts.extend(len(ids) for ids in tokenizer(list(chunk), add_special_tokens=False)["input_ids"])
    return counts


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def measure_dataset(dataset, tokenizer, response_fields: Sequence[str]) -> Tuple[List[int], List[int]]:
    """Return (prompt_tokens, response_tokens) per row.

    With several response fields (chosen/rejected), the row's response length is
    the longest of them, because that is what the prompt has to make room for.
    """
    columns = set(dataset.column_names)
    prompts = [_text(p) for p in dataset["prompt"]]
    prompt_tokens = count_prompt_tokens(tokenizer, prompts)

    response_tokens = [0] * len(prompts)
    for field in response_fields:
        if field not in columns:
            continue
        counts = count_response_tokens(tokenizer, [_text(r) for r in dataset[field]])
        response_tokens = [max(a, b) for a, b in zip(response_tokens, counts)]
    return prompt_tokens, response_tokens


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _percentiles(values: Sequence[int]) -> Dict[str, int]:
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)

    def pct(q: float) -> int:
        # nearest-rank percentile: no interpolation, always an observed value
        rank = max(1, math.ceil(q / 100.0 * len(ordered)))
        return int(ordered[rank - 1])

    return {"p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99), "max": int(ordered[-1])}


def build_report(prompt_tokens: Sequence[int], response_tokens: Sequence[int],
                 max_length: int, max_prompt_length: Optional[int]) -> Dict[str, Any]:
    """Summarize what fitting these rows into ``max_length`` will cut.

    Returns a JSON-serializable dict that includes the per-row ``budgets``
    (strip that key before persisting).
    """
    budgets: List[int] = []
    prompt_cut = response_cut = response_lost = 0
    prompt_tokens_dropped = 0
    for p, r in zip(prompt_tokens, response_tokens):
        budget = resolve_prompt_budget(r, max_length, max_prompt_length)
        budgets.append(budget)
        kept_prompt = min(p, budget)
        if p > budget:
            prompt_cut += 1
            prompt_tokens_dropped += p - budget
        room = max_length - kept_prompt
        if r > room:
            response_cut += 1
            if room <= 0 < r:
                response_lost += 1

    rows = len(budgets)
    return {
        "rows": rows,
        "max_length": int(max_length),
        "max_prompt_length": int(max_prompt_length) if max_prompt_length else None,
        "budget_mode": "explicit" if max_prompt_length else "auto",
        "prompt_tokens": _percentiles(prompt_tokens),
        "response_tokens": _percentiles(response_tokens),
        "prompts_truncated": prompt_cut,
        "prompts_truncated_fraction": (prompt_cut / rows) if rows else 0.0,
        "prompt_tokens_dropped": prompt_tokens_dropped,
        "responses_truncated": response_cut,
        "responses_with_no_tokens_left": response_lost,
        "truncation_side": "keep_end",
        "budgets": budgets,
    }


def public_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """The report without the per-row budgets (for logs and job storage)."""
    return {k: v for k, v in report.items() if k != "budgets"}


def format_report(report: Dict[str, Any], label: str = "train") -> str:
    pt, rt = report["prompt_tokens"], report["response_tokens"]
    budget = (f"max_prompt_length={report['max_prompt_length']}"
              if report["budget_mode"] == "explicit"
              else "max_prompt_length=auto (max_length minus each row's response, response capped at half)")
    return (
        f"Prompt budget [{label}]: {report['rows']} rows, max_length={report['max_length']}, {budget}. "
        f"Prompt tokens p50/p90/p95/p99/max = {pt['p50']}/{pt['p90']}/{pt['p95']}/{pt['p99']}/{pt['max']}; "
        f"response tokens p50/p95/max = {rt['p50']}/{rt['p95']}/{rt['max']}. "
        f"Prompts truncated (keeping the end): {report['prompts_truncated']}/{report['rows']} "
        f"({report['prompts_truncated_fraction']:.1%}), {report['prompt_tokens_dropped']} tokens dropped; "
        f"responses truncated: {report['responses_truncated']}"
        + (f", {report['responses_with_no_tokens_left']} with no response tokens left"
           if report["responses_with_no_tokens_left"] else "")
    )


def truncation_error_message(report: Dict[str, Any], tolerance: float) -> str:
    pt = report["prompt_tokens"]
    suggestion = pt["p99"] + report["response_tokens"]["p95"]
    if report["budget_mode"] == "explicit":
        fixes = [
            f"raise max_prompt_length (prompt p99 is {pt['p99']}, max {pt['max']} tokens) and "
            f"max_length with it",
            "clear max_prompt_length to use the automatic budget",
        ]
    else:
        fixes = [
            f"raise max_length (prompt p99 is {pt['p99']}, max {pt['max']} tokens; "
            f"about {suggestion} fits nearly every row)",
        ]
    fixes += [
        "shorten the prompts",
        "set prompt_truncation_tolerance=1.0 to train on the truncated rows anyway "
        "(truncation keeps the end of each prompt)",
    ]
    limits = f"max_length={report['max_length']}"
    if report["max_prompt_length"]:
        limits += f" / max_prompt_length={report['max_prompt_length']}"
    return (
        f"{report['prompts_truncated']} of {report['rows']} training rows "
        f"({report['prompts_truncated_fraction']:.0%}) would have their prompt truncated to fit "
        f"{limits}, above the allowed {tolerance:.0%}. To fix: " + "; or ".join(fixes) + "."
    )


def check_truncation(report: Dict[str, Any], tolerance: Optional[float]) -> None:
    """Raise PromptTruncationError if too many prompts would be cut."""
    if tolerance is None:
        tolerance = DEFAULT_TRUNCATION_TOLERANCE
    if report["rows"] and report["prompts_truncated_fraction"] > tolerance:
        raise PromptTruncationError(truncation_error_message(report, tolerance))


# --------------------------------------------------------------------------- #
# Runner entry point
# --------------------------------------------------------------------------- #

def apply_prompt_budgets(dataset, tokenizer, *, training_mode: str, max_length: int,
                         max_prompt_length: Optional[int], label: str = "train",
                         tolerance: Optional[float] = None, enforce: bool = False):
    """Measure ``dataset``, log the truncation report, and attach per-row budgets.

    Returns ``(dataset_with_budget_column, public_report)``. With
    ``enforce=True`` this raises :class:`PromptTruncationError` when the share
    of truncated prompts exceeds ``tolerance``. Runners enforce on the training
    split only.
    """
    fields = response_fields_for_mode(training_mode)
    prompt_tokens, response_tokens = measure_dataset(dataset, tokenizer, fields)
    report = build_report(prompt_tokens, response_tokens, max_length, max_prompt_length)

    message = format_report(report, label)
    if report["prompts_truncated"] or report["responses_with_no_tokens_left"]:
        logger.warning(message)
    else:
        logger.info(message)

    if enforce:
        check_truncation(report, tolerance)

    if BUDGET_COLUMN in dataset.column_names:
        dataset = dataset.remove_columns(BUDGET_COLUMN)
    dataset = dataset.add_column(BUDGET_COLUMN, report["budgets"])
    return dataset, public_report(report)


def budget_splits(train_dataset, eval_dataset, tokenizer, config, training_mode: str):
    """Apply prompt budgets to the train (enforced) and eval (reported) splits.

    Returns ``(train_dataset, eval_dataset, reports)``, where ``reports`` maps
    ``"train"``/``"eval"`` to public reports ready to store on the job.
    """
    max_prompt_length = getattr(config, "max_prompt_length", None)
    tolerance = getattr(config, "prompt_truncation_tolerance", None)
    reports: Dict[str, Any] = {}
    train_dataset, reports["train"] = apply_prompt_budgets(
        train_dataset, tokenizer, training_mode=training_mode, max_length=config.max_length,
        max_prompt_length=max_prompt_length, label="train", tolerance=tolerance, enforce=True,
    )
    if eval_dataset is not None and len(eval_dataset):
        eval_dataset, reports["eval"] = apply_prompt_budgets(
            eval_dataset, tokenizer, training_mode=training_mode, max_length=config.max_length,
            max_prompt_length=max_prompt_length, label="eval",
        )
    return train_dataset, eval_dataset, reports


def record_reports(job_manager, job_id: str, reports: Dict[str, Any]) -> None:
    """Store the truncation reports under ``job.metrics["prompt_budget"]``.

    Server stdout often goes nowhere, so the numbers live on the job, where
    ``GET /jobs/{id}`` returns them.
    """
    if job_manager is None or not reports:
        return
    try:
        job = job_manager.get_job(job_id)
        metrics = dict((job.metrics if job else None) or {})
        metrics["prompt_budget"] = reports
        job_manager.update_job(job_id, metrics=metrics)
    except Exception as e:  # reporting must never take down a run
        logger.debug(f"Could not store prompt budget report: {e}")


# --------------------------------------------------------------------------- #
# Tokenization (keep_end on any grimoire version)
# --------------------------------------------------------------------------- #

def _grimoire_supports_keep_end() -> bool:
    try:
        from grimoire.data.common import encode_prompt_response
        return "prompt_truncation" in inspect.signature(encode_prompt_response).parameters
    except Exception:  # grimoire missing or mocked out in tests
        return False


GRIMOIRE_KEEPS_PROMPT_END = _grimoire_supports_keep_end()


def _keep_end(prompt_ids: List[int], max_prompt_length: Optional[int],
              bos_token_id: Optional[int]) -> List[int]:
    if not max_prompt_length or len(prompt_ids) <= max_prompt_length:
        return prompt_ids
    head: List[int] = []
    if bos_token_id is not None and max_prompt_length > 1 and prompt_ids[0] == bos_token_id:
        head = prompt_ids[:1]
    keep = max_prompt_length - len(head)
    return head + prompt_ids[len(prompt_ids) - keep:]


def encode_prompt_response_keep_end(tokenizer, prompt, response, max_length=None,
                                    max_prompt_length=None, label_pad_token_id=-100):
    """Local twin of grimoire's ``encode_prompt_response(prompt_truncation="keep_end")``.

    Used only when the installed grimoire predates that option, so that prompt
    truncation never falls back to keeping the start.
    """
    prompt_ids = list(tokenizer(prompt)["input_ids"])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and prompt_ids and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]
    response_ids = list(tokenizer(response, add_special_tokens=False)["input_ids"])
    prompt_ids = _keep_end(prompt_ids, max_prompt_length, getattr(tokenizer, "bos_token_id", None))

    input_ids = prompt_ids + response_ids
    if max_length:
        input_ids = input_ids[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [label_pad_token_id] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _row_budget(example, max_prompt_length):
    budget = example.get(BUDGET_COLUMN) if hasattr(example, "get") else None
    return budget if budget else max_prompt_length


def tokenize_sft_row(example, tokenizer, max_length, max_prompt_length=None):
    """SFT tokenization of a prompt/chosen row, keeping the end of long prompts."""
    budget = _row_budget(example, max_prompt_length)
    if GRIMOIRE_KEEPS_PROMPT_END:
        from grimoire.data import tokenize_sft
        return tokenize_sft(example, tokenizer, max_length=max_length, max_prompt_length=budget,
                            prompt_field="prompt", response_field="chosen",
                            prompt_truncation="keep_end")
    return encode_prompt_response_keep_end(tokenizer, example["prompt"], example["chosen"],
                                           max_length=max_length, max_prompt_length=budget)


def tokenize_preference_row(example, tokenizer, max_length, max_prompt_length=None):
    """Preference tokenization. Chosen and rejected share one prompt budget."""
    budget = _row_budget(example, max_prompt_length)
    if GRIMOIRE_KEEPS_PROMPT_END:
        from grimoire.data import tokenize_preference
        return tokenize_preference(example, tokenizer, max_length=max_length,
                                   max_prompt_length=budget, prompt_truncation="keep_end")
    chosen = encode_prompt_response_keep_end(tokenizer, example["prompt"], example["chosen"],
                                             max_length=max_length, max_prompt_length=budget)
    rejected = encode_prompt_response_keep_end(tokenizer, example["prompt"], example["rejected"],
                                               max_length=max_length, max_prompt_length=budget)
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_labels": rejected["labels"],
    }


def tokenize_kto_row(example, tokenizer, max_length, max_prompt_length=None):
    """KTO tokenization of a prompt/response/label row."""
    budget = _row_budget(example, max_prompt_length)
    if GRIMOIRE_KEEPS_PROMPT_END:
        from grimoire.data import tokenize_kto
        return tokenize_kto(example, tokenizer, max_length=max_length, max_prompt_length=budget,
                            prompt_field="prompt", response_field="response", label_field="label",
                            prompt_truncation="keep_end")
    encoded = encode_prompt_response_keep_end(tokenizer, example["prompt"], example["response"],
                                              max_length=max_length, max_prompt_length=budget)
    encoded["kto_label"] = bool(example["label"])
    return encoded


if not GRIMOIRE_KEEPS_PROMPT_END:
    logger.info(
        "Installed grimoire has no prompt_truncation option; Merlina will use its own "
        "keep-end prompt encoder"
    )
