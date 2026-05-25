"""Atelier diffusion training runner.

Parallel narrow path to ``run_training_sync`` for diffusion-model LoRA
training (Qwen-Image text-to-image, Qwen-Image-Edit image-to-image,
SDXL). Mirrors the shape of ``training_runner_vlm.py`` but the body
swaps Grimoire for Atelier:

- Loads an Atelier ``ModelAdapter`` (QwenImageAdapter / QwenEditAdapter
  / SDXLAdapter) instead of an HF causal LM
- Builds an ``EditingDataset`` from uploaded images + captions
- Pre-computes embeddings via ``atelier.data.cache_embeddings`` then
  hands the cache to the dataset (frees encoder VRAM before the
  transformer is moved to GPU)
- Drives ``atelier.AtelierTrainer`` with ``FlowMatchingLoss`` + LoRA
- Reuses ``WebSocketCallback`` from the text runner — Atelier's
  ``_fire`` is duck-typed so the same callback object Just Works
  across engines.

Dispatched by ``_resolve_sibling_runner`` in ``src/training_runner.py``
when ``config.model_type == 'diffusion'`` (or ``training_mode`` starts
with ``diffusion_``).
"""
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
import wandb

from src.job_manager import JobManager
from src.training_runner import (
    WebSocketCallback,
    _cleanup_training_resources,
    send_websocket_update,
)
from src.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


# ── Adapter registry ────────────────────────────────────────────────
# Map training_mode → (atelier_adapter_class_name, loss_class_name,
# default_target_modules). Loss is currently always flow_matching but
# keeping it pluggable for future preference-mode diffusion training.
_DIFFUSION_MODES = {
    "diffusion_qwen_image":  ("QwenImageAdapter", "FlowMatchingLoss",
                              ["to_k", "to_q", "to_v", "to_out.0"]),
    "diffusion_qwen_edit":   ("QwenEditAdapter",  "FlowMatchingLoss",
                              ["to_k", "to_q", "to_v", "to_out.0"]),
    "diffusion_sdxl":        ("SDXLAdapter",      "FlowMatchingLoss",
                              ["to_k", "to_q", "to_v", "to_out.0"]),
}


def _resolve_adapter_and_loss(training_mode: str):
    """Look up the (adapter_cls, loss_cls, default_targets) tuple for a mode."""
    spec = _DIFFUSION_MODES.get(training_mode.lower())
    if spec is None:
        valid = ", ".join(sorted(_DIFFUSION_MODES))
        raise ValueError(
            f"Unknown diffusion training_mode '{training_mode}'. Valid: {valid}"
        )
    from atelier import adapters, losses
    adapter_cls = getattr(adapters, spec[0])
    loss_cls = getattr(losses, spec[1])
    return adapter_cls, loss_cls, spec[2]


def _adapter_family_short_name(adapter_class_name: str) -> str:
    """QwenImageAdapter → qwen_image, QwenEditAdapter → qwen_edit, etc."""
    if adapter_class_name == "QwenImageAdapter":
        return "qwen_image"
    if adapter_class_name == "QwenEditAdapter":
        return "qwen_edit"
    if adapter_class_name == "SDXLAdapter":
        return "sdxl"
    return "qwen_image"  # safe fallback


def _generate_post_training_samples(
    *,
    job_id: str,
    config: Any,
    lora_dir: str,
    base_model: str,
    adapter_family: str,
    job_manager: JobManager,
    event_loop: Optional[Any] = None,
) -> None:
    """Run scripts/generate_diffusion_samples.py as a subprocess.

    Loads a fresh diffusers pipeline (the training process having released
    its 38 GiB transformer means there's room for sampling now), generates
    a small batch of preview images against a default prompt set, writes
    them to ``<lora_dir>/samples/`` with a manifest. UI fetches them via
    ``/jobs/{job_id}/samples``.
    """
    import subprocess
    import sys

    samples_dir = Path(lora_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    job_manager.update_job(job_id, status="generating_samples", progress=0.97)
    send_websocket_update(
        websocket_manager.send_status_update(
            job_id=job_id, status="generating_samples", progress=0.97,
            message="Rendering sample images from the freshly-trained LoRA",
        ),
        event_loop,
    )

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_diffusion_samples.py"
    if not script_path.exists():
        logger.warning(f"sample generator script not found at {script_path}; skipping")
        return

    cmd = [
        sys.executable, str(script_path),
        "--base-model", base_model,
        "--lora-dir", str(lora_dir),
        "--out-dir", str(samples_dir),
        "--adapter", _adapter_family_short_name(adapter_family),
        "--num-steps", str(int(getattr(config, "sample_num_steps", None) or 25)),
        "--width", str(int(getattr(config, "image_resolution", None) or 1024)),
        "--height", str(int(getattr(config, "image_resolution", None) or 1024)),
    ]
    # Optional prompt override (Pydantic field added in 2.0 for power users)
    sample_prompts = getattr(config, "sample_prompts", None)
    if sample_prompts:
        cmd += ["--prompts", json.dumps(sample_prompts)]

    logger.info(f"[samples] launching subprocess: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30 * 60)
    if res.returncode != 0:
        logger.warning(f"[samples] subprocess exit {res.returncode}\n"
                       f"stdout: {res.stdout[-500:]}\nstderr: {res.stderr[-500:]}")
        return
    logger.info(f"[samples] generated {samples_dir}: {res.stdout.strip().splitlines()[-1] if res.stdout else 'ok'}")


def _materialize_image_dataset(config: Any, uploaded_datasets: dict, job_id: str):
    """Build an HF Dataset of (prompt, image) rows.

    Three sources, in order of precedence:
      1. ``config.dataset_jsonl_path`` — local JSONL with prompt + image keys
      2. ``uploaded_datasets[id]`` — Merlina-side upload (typically a
         multipart-form JSONL the user dropped on the page)
      3. ``config.dataset_name`` — HF hub repo id
    """
    from datasets import Dataset, load_dataset
    from datasets import Image as DSImage

    rows = None

    if getattr(config, "dataset_jsonl_path", None):
        rows = []
        jsonl_path = Path(os.path.expanduser(config.dataset_jsonl_path)).resolve()
        # Resolve relative `image` / `chosen` / `rejected` paths against the
        # JSONL's parent directory, not Merlina's cwd. Lets the training
        # manifest stay portable (drop the whole dataset dir + JSONL
        # anywhere, training reads it from there).
        jsonl_dir = jsonl_path.parent
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                for key in ("image", "chosen", "rejected"):
                    if key in row and isinstance(row[key], str):
                        p = Path(row[key])
                        if not p.is_absolute():
                            row[key] = str((jsonl_dir / p).resolve())
                rows.append(row)
    elif uploaded_datasets:
        # Materialize whichever upload is the dataset (mirrors text path).
        first_id = next(iter(uploaded_datasets))
        upload_dir = Path(f"./uploads/{job_id}")
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Uploaded payload is expected to be a JSONL string with
        # absolute image paths (Merlina's upload UI writes file paths,
        # not raw bytes, for image datasets — see frontend/js/dataset.js).
        rows = []
        for line in uploaded_datasets[first_id].splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    elif getattr(config, "dataset_name", None):
        ds = load_dataset(config.dataset_name, split=getattr(config, "dataset_split", "train") or "train")
        # Diffusion datasets on the hub usually have 'image' + 'prompt' or 'caption'
        if "chosen" not in ds.column_names:
            if "image" in ds.column_names:
                ds = ds.rename_column("image", "chosen")
        if "prompt" not in ds.column_names and "caption" in ds.column_names:
            ds = ds.rename_column("caption", "prompt")
        return ds

    if rows is None or len(rows) == 0:
        raise ValueError(
            "No dataset rows resolved for diffusion training. Provide one of: "
            "dataset_jsonl_path, an uploaded JSONL dataset, or dataset_name."
        )

    # Normalize the 'image' / 'chosen' key
    for r in rows:
        if "chosen" not in r and "image" in r:
            r["chosen"] = r["image"]

    ds = Dataset.from_list(rows)
    if "chosen" in ds.column_names:
        ds = ds.cast_column("chosen", DSImage())
    if "rejected" in ds.column_names:
        ds = ds.cast_column("rejected", DSImage())
    return ds


def run_diffusion_training_sync(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
    event_loop: Optional[Any] = None,
) -> None:
    """Run a diffusion LoRA training job via Atelier.

    Contract matches ``run_training_sync`` and ``run_vlm_training_sync``
    (job lifecycle, GPU cleanup in all exit paths). The body is a narrow
    diffusion path: no chat template, no tokenizer, no preference-mode
    rendering.
    """
    adapter = None
    trainer = None

    try:
        job_manager.update_job(job_id, status="initializing", progress=0.0)
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="initializing", progress=0.0,
            ),
            event_loop,
        )

        # GPU pinning (mirrors text + VLM paths)
        if config.gpu_ids is not None:
            from src.gpu_utils import get_gpu_manager
            get_gpu_manager().set_visible_devices(config.gpu_ids)
            logger.info(f"Using GPUs: {config.gpu_ids}")

        # Wandb (atelier's trainer drives init via accelerator.init_trackers)
        wandb_run_name = None
        wandb_project = None
        if config.use_wandb:
            if config.wandb_key:
                wandb.login(key=config.wandb_key)
            wandb_run_name = config.wandb_run_name or config.output_name or job_id
            wandb_project = config.wandb_project or "merlina-training"
            logger.info(f"W&B Project: {wandb_project}, Run: {wandb_run_name}")

        if config.hf_token:
            os.environ["HF_TOKEN"] = config.hf_token

        # Resolve adapter + loss from training_mode
        adapter_cls, loss_cls, default_targets = _resolve_adapter_and_loss(
            config.training_mode
        )

        # ── Build dataset ────────────────────────────────────────
        job_manager.update_job(job_id, status="loading_dataset", progress=0.05)
        raw_dataset = _materialize_image_dataset(config, uploaded_datasets, job_id)
        logger.info(f"Diffusion dataset: {len(raw_dataset)} rows")

        # ── Build adapter ────────────────────────────────────────
        job_manager.update_job(job_id, status="loading_model", progress=0.10)
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="loading_model", progress=0.10,
                message=f"Loading {adapter_cls.__name__} from {config.model_name}",
            ),
            event_loop,
        )

        adapter_kwargs = {}
        if adapter_cls.__name__ in ("QwenImageAdapter", "QwenEditAdapter"):
            # Stage the loads so encoders + transformer don't have to
            # coexist in VRAM during the pre-compute phase.
            adapter_kwargs["defer_transformer"] = True
        # model_name overrides base_model when set; otherwise fall back so
        # diffusion jobs can reuse the existing single base_model field.
        model_path = getattr(config, "model_name", None) or config.base_model
        adapter = adapter_cls(model_path, **adapter_kwargs)

        # ── Pre-compute embeddings ───────────────────────────────
        job_manager.update_job(job_id, status="caching_embeddings", progress=0.15)
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="caching_embeddings", progress=0.15,
                message="Pre-computing text + image embeddings",
            ),
            event_loop,
        )

        from atelier.data import EditingDataset, cache_embeddings
        cache_dir = f"./results/{job_id}/cache"
        text_emb, target_emb, control_emb = cache_embeddings(
            raw_dataset, adapter, cache_dir=cache_dir,
            target_area=int(getattr(config, "image_resolution", 1024)) ** 2,
        )

        # Free encoders + bring the transformer onto the GPU
        if hasattr(adapter, "free_encoders"):
            adapter.free_encoders()
        if hasattr(adapter, "move_transformer_to_device"):
            adapter.move_transformer_to_device()

        train_dataset = EditingDataset(
            raw_dataset,
            cached_text_embeddings=text_emb or None,
            cached_target_embeddings=target_emb or None,
            cached_control_embeddings=control_emb or None,
        )

        # ── Build LoRA config ────────────────────────────────────
        # lora_rank wins over the shared lora_r; lora_target_modules wins
        # over both adapter defaults and the LLM-shaped target_modules
        # field (whose 'q_proj' etc. names don't exist in DiT/UNet).
        from peft import LoraConfig
        lora_r = getattr(config, "lora_rank", None) or getattr(config, "lora_r", None) or 32
        targets = getattr(config, "lora_target_modules", None) or default_targets
        peft_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(getattr(config, "lora_alpha", 64)),
            target_modules=targets,
            lora_dropout=float(getattr(config, "lora_dropout", 0.05)),
            bias="none",
            init_lora_weights="gaussian",
        )

        # ── Build Atelier TrainingConfig ─────────────────────────
        from atelier import TrainingConfig as AtelierTrainingConfig
        output_dir = f"./results/{job_id}"
        atelier_config = AtelierTrainingConfig(
            output_dir=output_dir,
            num_epochs=int(config.num_epochs),
            batch_size=int(config.batch_size),
            gradient_accumulation_steps=int(config.gradient_accumulation_steps),
            learning_rate=float(config.learning_rate),
            weight_decay=float(getattr(config, "weight_decay", 0.0)),
            warmup_ratio=float(getattr(config, "warmup_ratio", 0.05)),
            max_grad_norm=float(getattr(config, "max_grad_norm", 1.0)),
            mixed_precision=("bf16" if torch.cuda.is_available()
                             and torch.cuda.get_device_capability()[0] >= 8
                             else "fp16"),
            gradient_checkpointing=bool(getattr(config, "gradient_checkpointing", True)),
            optimizer=getattr(config, "optimizer_type", "adafactor"),
            lr_scheduler=getattr(config, "lr_scheduler_type", "cosine"),
            logging_steps=int(getattr(config, "logging_steps", 10)),
            save_steps=int(getattr(config, "save_steps", 500) or 500),
            save_total_limit=2,
            save_on_epoch_end=True,
            seed=int(getattr(config, "seed", 42)),
            run_name=wandb_run_name if config.use_wandb else config.output_name,
            log_with="wandb" if config.use_wandb else None,
            project_name=wandb_project,
            wandb_tags=getattr(config, "wandb_tags", None) or [],
            wandb_notes=getattr(config, "wandb_notes", None),
        )

        # ── Build trainer + train ────────────────────────────────
        from atelier import AtelierTrainer
        trainer = AtelierTrainer(
            adapter=adapter,
            config=atelier_config,
            loss_fn=loss_cls(),
            train_dataset=train_dataset,
            peft_config=peft_config,
            callbacks=[WebSocketCallback(job_id, job_manager, event_loop)],
        )

        # Capture W&B URL after accelerator init
        if config.use_wandb and wandb.run is not None:
            wandb_url = wandb.run.get_url()
            logger.info(f"W&B run URL: {wandb_url}")
            job_manager.update_job(job_id, wandb_url=wandb_url)

        logger.info(f"Starting Atelier {config.training_mode} training")
        trainer.train()
        was_stopped = getattr(trainer, "stopped_early", False)

        # ── Save final LoRA ──────────────────────────────────────
        job_manager.update_job(
            job_id,
            status="saving_stopped" if was_stopped else "saving",
            progress=0.95,
        )
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="saving", progress=0.95,
            ),
            event_loop,
        )

        final_output_dir = f"./models/{config.output_name}"
        trainer.save_model(final_output_dir)
        logger.info(f"LoRA saved to {final_output_dir}")

        final_step = trainer.global_step
        final_max_steps = trainer.max_steps

        # Free VRAM before any post-training upload work
        logger.info("🧹 Cleaning up diffusion training resources...")
        del trainer, adapter
        adapter = None
        trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ VRAM freed")

        # Post-training sample generation — fresh process, fresh pipeline.
        # Non-fatal: the LoRA itself is the deliverable, samples are sugar.
        if not was_stopped:
            try:
                _generate_post_training_samples(
                    job_id=job_id,
                    config=config,
                    lora_dir=final_output_dir,
                    base_model=model_path,
                    adapter_family=adapter_cls.__name__,
                    job_manager=job_manager,
                    event_loop=event_loop,
                )
            except Exception as e:
                logger.warning(f"Sample generation failed (non-fatal): {e}")

        job_manager.update_job(
            job_id,
            status="stopped" if was_stopped else "completed",
            progress=1.0,
            current_step=final_step,
            total_steps=final_max_steps,
        )
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="stopped" if was_stopped else "completed",
                progress=1.0,
                current_step=final_step,
                total_steps=final_max_steps,
            ),
            event_loop,
        )

    except Exception as e:
        logger.exception(f"Diffusion training failed: {e}")
        job_manager.update_job(job_id, status="failed", error=str(e))
        # progress is a required positional on send_status_update; without
        # it the failure-path itself raises and masks the original error.
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="failed", progress=0.0, message=str(e),
            ),
            event_loop,
        )
        raise
    finally:
        # _cleanup_training_resources expects (model, trainer); pass the
        # adapter in the model slot — its parameters are what holds VRAM.
        _cleanup_training_resources(adapter, trainer)
