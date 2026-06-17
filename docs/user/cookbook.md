# Merlina Cookbook

Battle-tested, end-to-end configs for the common training shapes — plus the
non-obvious settings that aren't discoverable from the schema alone. The
[`examples/`](../../examples/) folder has the *minimal* starting points; this
collects fuller, real-world recipes and the gotchas that cost a debugging
session if you don't know them.

## Running a config

A config is a single `TrainingConfig` object. Two ways to launch it:

**HTTP** (server running):

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d @my_config.json
```

**Programmatically** (no server) — handy for scripted/queued runs:

```python
from pydantic import TypeAdapter
from datetime import datetime
import merlina
from src.training_runner import run_training_sync

config = TypeAdapter(merlina.TrainingConfig).validate_python(CFG)   # validates first
job_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"
merlina.job_manager.create_job(job_id, config.model_dump())
run_training_sync(job_id, config, merlina.job_manager, {}, None)
```

> **Tip:** gate the launcher behind `if os.environ.get("DRY_RUN") == "1": return`
> *after* `validate_python` + `create_job`. Running with `DRY_RUN=1` validates
> the whole config and surfaces schema errors before you pay for a model load.

---

## Recipe 1 — Text SFT (prompt → completion)

Train on good completions only. Local Parquet source, the model's **native chat
template** via `format_type: tokenizer`, LoRA, and a long `max_length` so long
targets aren't truncated.

```json
{
  "base_model": "google/gemma-4-12B-it",
  "output_name": "my-sft-model",
  "training_mode": "sft",

  "use_lora": true,
  "lora_r": 64,
  "lora_alpha": 32,
  "lora_dropout": 0.05,

  "num_epochs": 3,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-5,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.05,
  "optimizer_type": "paged_adamw_8bit",
  "gradient_checkpointing": true,
  "max_length": 8192,

  "merge_lora_before_upload": false,
  "push_to_hub": false,

  "dataset": {
    "source": { "source_type": "local_file", "file_path": "data/train.parquet", "file_format": "parquet" },
    "format": { "format_type": "tokenizer" },
    "training_mode": "sft",
    "test_size": 0.05
  }
}
```

Your rows need `prompt` + `chosen` columns (`chosen` is the target completion).
Map other column names with `dataset.column_mapping`, e.g. `{"input": "prompt", "output": "chosen"}`.

---

## Recipe 2 — Preference tuning (ORPO → merged model)

ORPO needs no reference model, so it fits large bases in one pass. Trains a LoRA
and **merges it to full weights** for the upload. Rows need `prompt` / `chosen`
/ `rejected` (`system` optional).

```json
{
  "base_model": "google/gemma-4-31B-it",
  "output_name": "my-orpo-model",
  "training_mode": "orpo",
  "beta": 0.1,

  "use_lora": true,
  "lora_r": 64,
  "lora_alpha": 32,
  "lora_dropout": 0.05,

  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 32,
  "learning_rate": 5e-5,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.05,
  "max_grad_norm": 0.5,
  "optimizer_type": "paged_adamw_8bit",
  "gradient_checkpointing": true,
  "max_length": 2048,

  "merge_lora_before_upload": true,
  "push_to_hub": false,

  "dataset": {
    "source": { "source_type": "huggingface", "repo_id": "schneewolflabs/Athanorlite-DPO", "split": "train" },
    "format": { "format_type": "tokenizer" },
    "training_mode": "orpo",
    "test_size": 0.02
  }
}
```

---

## Recipe 3 — Multimodal VLM Stage-2 FFT

Full fine-tune of a grafted vision-language model (frozen ViT + projector +
text decoder). Reloads a Stage-1 checkpoint and trains the decoder + projector.
Dataset rows are `{messages, image}` in the unified ArtemisMix schema.

```json
{
  "base_model": "/path/to/A3-stage1",
  "output_name": "my-vlm-model",
  "training_mode": "vlm_stage2",
  "stage": "stage2",
  "image_token_id": 22,

  "use_lora": false,
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 16,
  "learning_rate": 1e-5,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.03,
  "optimizer_type": "paged_adamw_8bit",
  "gradient_checkpointing": true,
  "max_length": 2048,
  "max_pixels": 1638400,

  "dataset": {
    "source": { "source_type": "huggingface", "repo_id": "/path/to/local/ArtemisMix", "split": "train" },
    "test_size": 0.001,
    "training_mode": "sft"
  }
}
```

---

## Non-obvious settings (the gotchas)

**`format.format_type` — use `tokenizer` for non-ChatML models.** `chatml` (the
default) emits literal `<|im_start|>` markup. For Gemma / Llama-3 / Mistral /
Qwen, set `format_type: "tokenizer"` so rows render with the *model's own* chat
template. Mismatched template = the model learns the wrong control tokens.

**LoRA on a multimodal base — scope `target_modules` to the text decoder.**
Unified-multimodal models (e.g. Gemma 4 12B/31B) have vision/audio towers whose
linear layers share names like `q_proj` but aren't standard `nn.Linear`, so PEFT
errors trying to wrap them. Target only the language-model layers:

```python
target_modules = [
    f"language_model.layers.{i}.{p}"
    for i in range(NUM_TEXT_LAYERS)            # e.g. 60 for gemma-4-31B
    for p in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
              "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
]
```

**Local files: prefer Parquet, and know how the split works.** `local_file`
loads JSON/JSONL/CSV/Parquet. Parquet preserves row order, which matters when
you care about *which* rows land in train vs eval. By default the eval split is
a **random** `test_size` fraction (`shuffle_dataset` controls the shuffle).

**Explicit / held-out eval set — `dataset.eval_source`.** To evaluate on a
specific dataset (e.g. train on one distribution, eval on another) instead of a
random split, set `dataset.eval_source` to a full `DatasetSource`. It's used as
the eval split verbatim and the random `test_size` split is skipped.

**Fitting a big model on one node.** The reliable combo: `use_lora` (or full FT
with) `optimizer_type: paged_adamw_8bit` + `gradient_checkpointing: true` +
`batch_size: 1` + a larger `gradient_accumulation_steps` for the effective batch.
Grad-checkpointing trades ~33% compute for a large activation-memory saving.

**`max_length` — measure before you set it.** It truncates the tokenized
prompt+target. If targets are long (code, multi-turn, documents), check the
token-length distribution first; truncating the target teaches the model to
emit incomplete output. A too-small cap is a silent quality bug, not an error.

**Eval cadence costs wall-clock.** `eval_steps` (a fraction of total steps, e.g.
`0.1`) evaluates the **whole** eval set each time, synchronously. On a large
eval set + large model this can dominate the run. Keep the eval set modest, or
raise `eval_steps`, if you don't need a fine-grained curve.

**`paged_adamw_8bit` / `merge_lora_before_upload`.** Merging happens only on the
upload path; with `push_to_hub: false` the run saves the **standalone LoRA
adapter** (handy — it ports to other same-architecture bases). Set
`merge_lora_before_upload: true` to ship full merged weights.
