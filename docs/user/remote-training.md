# Remote Training on Rented Compute ☁️🧙‍♀️

Train models bigger than your GPUs. Remote mode turns your local Merlina
into a control plane: give it your RunPod API key and it provisions a GPU
instance on **your** account, runs the training there, streams progress
back into the normal Merlina UI/API, pulls the trained adapter home, and
terminates the instance. You pay the provider directly — Merlina just
orchestrates.

> This is different from [running Merlina *on* a pod](runpod.md). Remote
> mode keeps Merlina (and your job history) local and rents machines
> per-job.

## Setup

1. Create a RunPod API key: https://www.runpod.io/console/user/settings
2. Add it to your `.env`:
   ```bash
   RUNPOD_API_KEY=rpa_...
   HF_TOKEN=hf_...   # needed: artifacts travel via a private HF repo
   ```
3. Restart Merlina.

## Running a remote job

Add a `remote` block to your training config and submit as usual:

```json
POST /train
{
  "base_model": "mistralai/Mistral-Nemo-Instruct-2407",
  "output_name": "you/my-finetune",
  "training_mode": "sft",
  "use_4bit": true,
  "dataset": {
    "source": {"source_type": "huggingface", "repo_id": "you/my-dataset"}
  },
  "remote": {
    "enabled": true,
    "max_runtime_hours": 12,
    "max_cost_per_hr": 5.0
  }
}
```

Leave the GPU choice out and the **sizing advisor** picks the cheapest
instance that fits: it reads the model's real `config.json` (MoE-aware —
a 1T-parameter Kimi-K2 with 32B active params sizes correctly), estimates
VRAM and disk, and selects GPU type/count from live RunPod offers. Or
pick explicitly with `"gpu_type_id": "NVIDIA H200", "gpu_count": 8`.

Preview the plan first — no money is spent:

```
POST /remote/plan      → sizing, instance pick, est. $/hr, stage layout
GET  /remote/offers    → rentable GPU types with current prices
```

Progress streams into the same job status / WebSocket / metrics endpoints
as local runs. Stop a remote job the normal way (`POST /jobs/{id}/stop`);
the worker checkpoints gracefully and the instance is terminated.

## The piecemeal pipeline

A remote run is a plan of **stages** that hand artifacts to each other
through a durable store (a private HF repo per run, e.g.
`you/merlina-run-job_x`). Because stages only communicate through the
store, each one can run on a different machine:

| Stage | Runs on | Needs | Output |
|---|---|---|---|
| `train` | provisioned GPU instance | VRAM | `adapter` artifact (small) |
| `merge` | control plane, separate instance, or skipped | RAM + disk, **no GPU** | `merged` weights |

`remote.merge_strategy` controls the merge:

- `auto` (default) — merge locally for small bases; **skip for very large
  ones** (>~80 GB native weights), producing an adapter-only run
- `local` — merge on your machine after the adapter is pulled
- `remote` — provision a second, cheaper big-disk instance to merge
  (this is how you merge a model whose weights don't fit your home rig)
- `skip` — adapter-only, always

For 1T-class models (Kimi-K2 and friends) adapter-only is the practical
choice: the trained adapter is a few GB, while merged weights would be
~2 TB to move around. Inference stacks can apply the adapter directly.

## Safety rails

- Instances are **always terminated** when the run ends, fails, or is
  stopped (set `remote.keep_instance_on_failure: true` to keep a failed
  instance for debugging — it keeps billing until *you* kill it).
- `max_runtime_hours` (default 48) hard-caps the run.
- `max_cost_per_hr` rejects instance picks above your budget.
- The worker's status endpoint is token-authenticated; provider proxy
  URLs alone aren't enough to read or control your run.

## Current limitations

- Datasets must be on the HuggingFace Hub (local files / uploads don't
  travel to workers yet); same for the base model.
- Text-LLM training modes only (no VLM / diffusion remote runs yet).
- Spot/interruptible instances aren't used yet (no mid-run resume).
- Configure via API/config JSON — the form UI is a follow-up.

## How it works (one paragraph)

The control plane builds the plan, creates the instance with the job
description in its environment, and polls a small authenticated HTTP
endpoint the worker exposes through the provider's port proxy — so your
machine never needs to be reachable from the internet. On the instance,
the worker runs the exact same training pipeline as local Merlina
(including multi-GPU model-parallel sharding via `device_map=auto` for
models that don't fit one GPU), pushes the adapter to the artifact repo,
and reports done; the control plane pulls the adapter into `./models/`,
runs or skips the merge per plan, and tears everything down.
