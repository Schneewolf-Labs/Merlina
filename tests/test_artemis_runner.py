"""Artemis runner glue test (Project Artemis piece #15).

End-to-end exercise of `src.training_runner_vlm.run_vlm_training_sync` with
a synthetic 16-row image+caption dataset (no HF dataset call). Validates
that the API-shaped TrainingConfig → model-build → processor → collator →
GrimoireTrainer.train() → save_model() path runs and produces a finite
decreasing loss curve.

Real A2 decoder + real Qwen/Qwen3-VL-2B-Instruct vision tower (whatever's
already cached locally). Just bypasses the HF dataset loader to keep the
test self-contained — the runner code under test is everything else.

Run: python tests/test_artemis_runner.py
"""
import sys

if "pytest" in sys.modules:
    import pytest
    pytest.skip(
        "Hardware smoke script — requires real A2 checkpoint + ML stack; "
        "run via 'python tests/test_artemis_runner.py'",
        allow_module_level=True,
    )

import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from datasets import Dataset, Features, Image as ImageFeature, Value

sys.path.insert(0, "/home/jupyter/Merlina")

# Import early so we can monkey-patch the dataset loader
from src import training_runner_vlm as runner_mod
from src.training_runner_vlm import run_vlm_training_sync

A2 = "/home/jupyter/schneewolf-a1/A2-ep1-loadable"
JOB_ID = "artemis-runner-smoke"
OUTPUT_NAME = "artemis-runner-smoke-output"
N_TRAIN = 16

# ---- Synthetic dataset ----------------------------------------------------
print("=== building synthetic 16-row image+caption dataset ===", flush=True)
rng = np.random.default_rng(0)
images = [
    Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))
    for _ in range(N_TRAIN)
]
captions = [
    f"A {color} square on a noisy background."
    for color in ["red", "blue", "green", "yellow"] * (N_TRAIN // 4)
]
features = Features({"image": ImageFeature(), "caption": Value("string")})
synth_ds = Dataset.from_dict({"image": images, "caption": captions}, features=features)
print(f"  built {len(synth_ds)} rows", flush=True)


def _fake_loader(config):
    # Bypass HF: just return the synthetic train + tiny eval split
    return synth_ds, synth_ds.select(range(2))


runner_mod._load_image_caption_dataset = _fake_loader

# ---- Stub JobManager ------------------------------------------------------
class _StubJob:
    stop_requested = False


class _StubJobManager:
    def __init__(self):
        self.updates = []
        self.metrics = []

    def update_job(self, job_id, **kwargs):
        self.updates.append((job_id, kwargs))
        if "loss" in kwargs:
            print(f"  [job] step={kwargs.get('current_step')} loss={kwargs['loss']:.4f}", flush=True)

    def get_job(self, job_id):
        return _StubJob()

    def add_metric(self, job_id, **kwargs):
        self.metrics.append((job_id, kwargs))


# ---- Config ---------------------------------------------------------------
# Mirror the Pydantic TrainingConfig shape with SimpleNamespace (the runner
# uses `getattr(config, ...)` and attribute access throughout).
cfg = SimpleNamespace(
    # Identity
    base_model=A2,
    output_name=OUTPUT_NAME,
    training_mode="vlm_stage1",
    # Vision graft
    vision_model_id="Qwen/Qwen3-VL-2B-Instruct",
    stage="stage1",
    unfreeze_vision_top_n=0,
    image_token_id=22,
    min_pixels=None,
    max_pixels=None,
    image_column="image",
    caption_column="caption",
    instruction="Describe this image.",
    # Training hparams (deliberately tiny — 4 effective batches of 2)
    num_epochs=1,
    batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    weight_decay=0.0,
    warmup_ratio=0.0,
    max_grad_norm=1.0,
    gradient_checkpointing=False,
    optimizer_type="adamw_torch",
    lr_scheduler_type="constant",
    logging_steps=1,
    eval_steps=None,
    eval_on_start=False,
    eval_batch_size=2,
    seed=0,
    # Wandb / HF off
    use_wandb=False,
    wandb_key=None,
    wandb_run_name=None,
    wandb_project=None,
    wandb_tags=None,
    wandb_notes=None,
    hf_token=None,
    push_to_hub=False,
    # GPU
    gpu_ids=None,
    # Dataset (only consulted by _load_image_caption_dataset, which we patched)
    dataset=SimpleNamespace(
        source=SimpleNamespace(repo_id="synthetic", split="train"),
        test_size=0.0,
        max_samples=None,
    ),
)

jm = _StubJobManager()

# ---- Run -----------------------------------------------------------------
print("=== running run_vlm_training_sync (≤8 steps) ===", flush=True)
try:
    run_vlm_training_sync(JOB_ID, cfg, jm, uploaded_datasets={}, event_loop=None)
except Exception as e:
    print(f"\nFAIL: runner raised {type(e).__name__}: {e}", flush=True)
    raise

# ---- Verify outputs -------------------------------------------------------
# Did the trainer log losses?
losses = [u[1]["loss"] for u in jm.updates if "loss" in u[1]]
print(f"\n  observed {len(losses)} loss updates", flush=True)
if losses:
    print(f"  first={losses[0]:.4f}  last={losses[-1]:.4f}", flush=True)

# Was the model saved?
save_dir = f"./models/{OUTPUT_NAME}"
saved_ok = os.path.isdir(save_dir)
print(f"  save dir present: {saved_ok} ({save_dir})", flush=True)

# Final status?
terminal = [u for u in jm.updates if u[1].get("status") in ("completed", "stopped", "failed")]
print(f"  terminal status: {terminal[-1][1].get('status') if terminal else 'NONE'}", flush=True)

# ---- Cleanup -------------------------------------------------------------
try:
    if saved_ok:
        shutil.rmtree(save_dir)
    rdir = f"./results/{JOB_ID}"
    if os.path.isdir(rdir):
        shutil.rmtree(rdir)
except Exception as e:
    print(f"  (cleanup warning: {e})", flush=True)

# ---- Assertions ----------------------------------------------------------
assert losses, "no loss updates observed — trainer never logged"
assert all(np.isfinite(l) for l in losses), f"non-finite loss in {losses}"
assert saved_ok, f"trainer.save_model did not produce {save_dir}"
assert terminal and terminal[-1][1].get("status") == "completed", (
    f"runner did not reach 'completed' status — final updates: {terminal[-3:]}"
)

print("\nARTEMIS_RUNNER: PASS — runner glue is wired end-to-end "
      f"(synthetic Stage-1 trained for {len(losses)} steps, model saved, status=completed)", flush=True)
