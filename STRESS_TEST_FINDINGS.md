# Stress-test findings — 2026-07-23 (single-GPU DGX Spark / GB10, remote API driving)

Driving Merlina programmatically from another host (single-GPU A2 ORPO LoRA
runs) surfaced four issues. Bug 1 fixed in this branch/PR; bug 2 mitigated;
bugs 3–4 documented for runtime debugging (not blind-patched — would risk
regressions in an engine I can't test here).

## 1. FIXED — single-GPU training blocked the entire API
`_make_training_callback` gated the subprocess path on `num_gpus > 1`, so
1-GPU jobs ran `run_training_sync` in a thread-pool executor. The GIL-holding
model load + train loop starved uvicorn; `/health`, `/status`, `/jobs/*/stop`
all hung for the whole run. Fix: route single-GPU through
`run_training_distributed` (accelerate `--num_processes 1`, no `--multi_gpu`);
`strategy="single"` keeps the in-thread path. Verified: API responsive during
training, stop endpoint reachable. (+ `PYTHONUNBUFFERED=1` for live logs.)

## 2. MITIGATED — graceful stop could leave a run unstoppable
Observed: a running job ignored `/stop` for ~2h. The Merlina wiring looks
correct (`/stop` → `job_queue.cancel` → `job_manager.request_stop` sets DB
`stop_requested`; `_monitor_distributed_progress` SIGTERMs the process group;
worker `FileProgressCallback.on_step_end` checks the flag and calls
`trainer.request_stop()`). So the break is downstream — either grimoire's
`request_stop()` not halting the loop, or the monitor loop stalling under
memory pressure (bug 4), or SIGTERM being swallowed by the worker's handler
without the loop honoring the flag.
Mitigation in this branch: `_monitor_distributed_progress` escalates SIGTERM
→ SIGKILL after 180s, so a stop is always honored.
TODO (runtime): confirm grimoire `GrimoireTrainer.request_stop()` actually
sets `should_training_stop` and the train loop checks it each step.

## 3. UNCONFIRMED — possible stale/wrong dataset for a run
Observed: a job whose config named `schneewolflabs/a4-base-lean` (verified 1500
rows on HF) reported `total_steps=917`, which maps to neither 1500 nor the
full 8931 at effective batch 16 (would be ~94 or ~558). Loss tracked a prior
full-data run — but the lean set is a *subset* of it, so that's not proof.
NOT patched: I could not locate a caching bug in `DatasetPipeline`/loader, and
917 doesn't cleanly map to any known dataset, so a "fix" would be a guess.
TODO (runtime): log `len(train_dataset)` right after `pipeline.prepare()` and
before `steps_per_epoch`; confirm the loader honors `dataset_name` and isn't
serving an HF-cache or `max_samples` artifact from a previous job.

## 4. DOCUMENTED — unified-memory API starvation during training
On the GB10 (GPU+system RAM one pool), the training subprocess consumed enough
memory that the *main* Merlina process AND an unrelated Jupyter server both
went HTTP-unresponsive (TCP still accepted; ping fine) — i.e. the box was
thrashing, not crashed. Distinct from bug 1 (that was GIL, this is memory).
Also: a 12B "4-bit" (bitsandbytes) run used ~96GB and loaded at 8.3s/shard
(~50min); bf16+grad-checkpointing trained at the same ~90s/step. On this HW,
4-bit buys no memory and loads ~25× slower — prefer bf16/fp8 + checkpointing.
TODO: cap the training subprocess's memory headroom (or lower the API
process's footprint) so the control plane survives training memory spikes.
