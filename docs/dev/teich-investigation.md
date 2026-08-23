# Teich Investigation

**Subject:** [TeichAI/teich](https://github.com/TeichAI/teich) — "Agent Data Infrastructure"
**Investigated:** 2026-08-23, against Teich v0.3.3 (alpha, Apache-2.0) and Merlina main @ `8215558`

## What Teich is

Teich is a Python package + CLI for preparing **agent sessions and chat datasets for SFT**.
It covers four things Merlina currently does not, and one thing Merlina does differently:

1. **Trace generation** (`teich generate`): runs Codex / Claude Code / Pi / Hermes / plain-chat
   agents against a `prompts.jsonl` and stages the resulting sessions as training traces
   (Docker only needed for this part).
2. **Session extraction** (`teich extract claude|codex|cursor|pi|hermes`): harvests *existing*
   local agent sessions (e.g. `~/.claude/projects`, `~/.codex/sessions`) into traces, anonymized
   by default.
3. **Normalization** (`teich convert`): everything lands in OpenAI-style JSONL rows —
   `{"prompt", "messages", "tools", "metadata", "system"?}` — with tool schemas, tool calls,
   tool results, and reasoning boundaries preserved as structure, not text.
4. **Audited preparation** (`prepare_data()`): renders through the *target tokenizer's* chat
   template, records typed supervision spans (reasoning / tool-call / final-answer), enforces
   `max_length` with explicit oversized policies (`drop` / `trim_followups` / `error`), and
   reports dropped, trimmed, and fully-masked rows.
5. **Response-only masking** (`mask_data(trainer, ...)`): converts the recorded spans into
   `-100` labels *after* trainer tokenization, with toggles for
   `train_on_reasoning` / `train_on_final_answers` / `train_on_tools`. Targets TRL `SFTTrainer`
   and Unsloth.

Their thesis, verbatim: *"Most SFT pipelines flatten agent data too early. That loses tool
schemas, tool results, reasoning boundaries, provenance, and the exact assistant spans you meant
to train on."*

Core dependencies are light (pydantic, pyyaml, typer, datasets, huggingface_hub, fastapi/uvicorn —
all already in or compatible with Merlina's stack; no torch/transformers in the core). 128 stars,
~200 commits, actively developed, explicitly alpha ("APIs may evolve").

## Where Merlina stands on the same problems

Merlina's tool-call handling (`dataset_handlers/tool_calls.py`) already got the hardest part
right, the same way Teich did: **never hardcode the tool-call dialect, render through the target
tokenizer's own template** (with the byte-for-byte `verify_roundtrip` guarantee). On that axis the
two projects independently converged.

The divergence is *when flattening happens*:

- `messages_converter.py:convert_messages_to_standard()` joins **all** user turns into one
  `prompt` and **all** assistant turns into one `chosen` with `\n\n`. `tool_calls.py:flatten_row()`
  does the same for tool conversations (tool results fold into the user side). For a multi-turn or
  agentic trace this **reorders the conversation**: the training sequence becomes
  `[every user/tool turn][every assistant turn]`, while inference interleaves them
  `user → assistant(tool_call) → tool_result → assistant → ...`.
- `grimoire.data.tokenize_sft` (used by `src/training_runner.py:1660`) then masks a **single
  prompt span** and supervises one contiguous completion. There is no way to express "supervise
  each assistant turn in place, mask the tool results between them."
- Consequence: Merlina is faithful *per piece* (each tool call renders in the model's native
  dialect) but structurally lossy *per conversation*. Single-turn rows — the common case for
  preference data — are unaffected. Multi-turn agent traces are exactly the case Teich exists for.
- Separately, CLAUDE.md already documents the truncation footgun ("the default 2048 truncated
  100% of one real tool-calling dataset ... trains on nothing"). Merlina has no per-row context
  check or truncation audit; Teich has `row_fits_context()`, oversized policies, and a
  fully-masked-row report as first-class features.

One more asymmetry worth naming: Teich is SFT-only. Merlina's preference modes
(ORPO/DPO/SimPO/CPO/IPO/KTO) are out of Teich's scope entirely, and the flatten-to-
(prompt, chosen, rejected) design exists *because* of those modes. Teich competes with nothing on
that side.

## Compatibility today (no code changes)

Teich's converted JSONL is already loadable by Merlina as-is:

- Rows carry a `messages` column → `has_messages_format()` triggers, and
  `convert_messages_dataset()` handles it (dropping `messages`/`tools`, emitting
  `system`/`prompt`/`chosen`).
- Rows with `tools` / `tool_calls` route through `flatten_row()`, which requires
  `format_type: "tokenizer"` + `model_name` — the documented requirement for tool data.
- Extraction traces from Claude Code / Codex sessions are, therefore, a viable *new dataset
  source* for Merlina users today, with the multi-turn flattening caveat above.

So the projects compose right now: **Teich as the upstream harvester/normalizer, Merlina as the
trainer** — with the caveat that Merlina's SFT collapses the turn structure Teich went to lengths
to preserve.

## Options

**A. Document the pairing (zero risk).** Add a section to `docs/user/dataset-guide.md`: use
`teich extract` / `teich generate` / `teich convert` to produce messages-format JSONL, upload to
Merlina, train with `format_type: "tokenizer"` + SFT. Note the multi-turn flattening caveat
honestly. No dependency taken.

**B. Borrow the audit surface (small, high value).** Add Teich-style checks to
`preflight_checks.py` / the pipeline, either by depending on `teich` for
`validate_tool_calls()` / `row_fits_context()` / `trace_is_complete()` or by implementing
equivalents: per-row token count vs `max_length` with a "N% of rows would truncate the
completion" **error** (not warning) when the supervised span is what gets cut. This directly
closes the footgun CLAUDE.md documents. If depending on alpha-status Teich feels risky, the
implementations are small enough to write natively — the value is the *checks*, not the code.

**C. Conversation-native SFT (the real fix, larger lift).** Add a path where multi-turn rows are
*not* flattened: render the full conversation once via the chat template and build labels that
mask everything except assistant spans (optionally: except reasoning, per Teich's toggles). Two
routes:
  - Use `prepare_data()` (returns rendered `text` + span metadata + optional `input_ids`) and
    convert its spans to grimoire-style labels. Note `mask_data()` itself is TRL/Unsloth-coupled
    and will **not** drop onto grimoire — only the span metadata transfers.
  - Or implement span-masked conversation tokenization in grimoire (a `tokenize_conversation`
    sibling to `tokenize_sft`), using Teich as design reference. This keeps Merlina
    dependency-free and fits grimoire's ownership of tokenization.

  Either way the UI change is small (a "keep conversation structure (multi-turn SFT)" toggle where
  the messages-format toggle already lives), and preference modes are untouched — they keep the
  flatten path, which is correct for paired data.

## Recommendation

Do **A** immediately, **B** soon (natively, no dependency — the checks are small and the
truncation footgun is already biting), and treat **C** as the roadmap item it deserves whenever
multi-turn agent-trace training becomes a real use case — likely via grimoire rather than a Teich
dependency, given Teich's alpha API and TRL coupling. Revisit Teich as a direct dependency once
its API stabilizes; license (Apache-2.0) and dependency footprint pose no obstacle.
