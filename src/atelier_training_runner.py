"""
Atelier Training Runner — diffusion model training for Merlina.

This module provides the core training functionality for diffusion models,
including:
- Adapter-based model loading (QwenEdit, SDXL)
- Image dataset handling with embedding caching
- Multiple training modes: flow_matching, epsilon, diffusion_dpo/orpo/simpo/cpo/ipo/kto
- WebSocket progress updates
- Background HuggingFace Hub uploads
- Weights & Biases integration

Mirrors the structure of training_runner.py (LLM training) but uses
Atelier instead of Grimoire.
"""

import os
import gc
import math
import torch
import wandb
import logging
import asyncio
import threading
from typing import Optional, Any

from peft import LoraConfig
from huggingface_hub import HfApi

from src.job_manager import JobManager
from src.websocket_manager import websocket_manager
from src.model_card import generate_wandb_run_name
from src.utils import get_num_gpus
from src.training_runner import send_websocket_update, _cleanup_training_resources

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atelier WebSocket Callback
# ---------------------------------------------------------------------------
# We cannot reuse WebSocketCallback from training_runner because it inherits
# from grimoire.TrainerCallback.  Atelier has its own TrainerCallback base
# class with the same method signatures, so we create a parallel version.

class AtelierWebSocketCallback:
    """
    Custom callback to send training metrics via WebSocket and handle stop
    requests.  Uses Atelier's TrainerCallback interface:
        on_step_end(trainer, step, loss, metrics)
        on_log(trainer, metrics)
        on_evaluate(trainer, metrics)
    """

    def __init__(self, job_id: str, job_manager: JobManager, event_loop=None):
        self.job_id = job_id
        self.job_manager = job_manager
        self.event_loop = event_loop

    # -- lifecycle (no-ops, present for completeness) ----------------------
    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch):
        pass

    def on_save(self, trainer, path):
        pass

    # -- active callbacks --------------------------------------------------

    def on_step_end(self, trainer, step, loss, metrics):
        """Called at the end of each training step to check for stop requests."""
        job = self.job_manager.get_job(self.job_id)
        if job and job.stop_requested:
            logger.info(f"Stop requested for job {self.job_id} - gracefully stopping training")
            trainer.request_stop()

            send_websocket_update(
                websocket_manager.send_status_update(
                    job_id=self.job_id,
                    status="stopping",
                    progress=step / trainer.max_steps if trainer.max_steps else 0.5,
                    current_step=step,
                    total_steps=trainer.max_steps,
                    message="Stop requested - saving checkpoint..."
                ),
                self.event_loop
            )

    def on_log(self, trainer, metrics):
        """Called when trainer logs metrics."""
        update_data = {}

        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)

        if "train/loss" in metrics:
            update_data["loss"] = float(metrics["train/loss"])

        if "train/learning_rate" in metrics:
            update_data["learning_rate"] = float(metrics["train/learning_rate"])

        if trainer.global_step:
            update_data["current_step"] = trainer.global_step

        if trainer.max_steps:
            update_data["total_steps"] = trainer.max_steps
            progress = 0.3 + (0.6 * (trainer.global_step / trainer.max_steps))
            update_data["progress"] = min(progress, 0.9)

        if update_data:
            self.job_manager.update_job(self.job_id, **update_data)

            if trainer.global_step and "loss" in update_data:
                self.job_manager.add_metric(
                    self.job_id,
                    step=trainer.global_step,
                    loss=update_data.get("loss"),
                    eval_loss=update_data.get("eval_loss"),
                    learning_rate=update_data.get("learning_rate"),
                    gpu_memory_used=gpu_memory
                )

            send_websocket_update(
                websocket_manager.send_status_update(
                    job_id=self.job_id,
                    status="training",
                    progress=update_data.get("progress", 0.5),
                    current_step=update_data.get("current_step"),
                    total_steps=update_data.get("total_steps"),
                    loss=update_data.get("loss"),
                    eval_loss=update_data.get("eval_loss"),
                    learning_rate=update_data.get("learning_rate"),
                    gpu_memory=gpu_memory
                ),
                self.event_loop
            )

    def on_evaluate(self, trainer, metrics):
        """Called after evaluation completes."""
        update_data = {}
        if "eval/loss" in metrics:
            update_data["eval_loss"] = float(metrics["eval/loss"])
        if update_data:
            self.job_manager.update_job(self.job_id, **update_data)


# ---------------------------------------------------------------------------
# Background HuggingFace Hub upload
# ---------------------------------------------------------------------------

def _run_background_atelier_upload(
    config: Any,
    final_output_dir: str,
    training_mode: str,
    job_id: str,
    job_manager: JobManager,
    event_loop=None,
) -> None:
    """
    Run HuggingFace Hub upload for diffusion models in a background thread.

    For diffusion models we always use upload_folder() — no LoRA merging
    (merging diffusion LoRA is architecture-specific and not needed initially).

    Args:
        config: Training configuration
        final_output_dir: Path to saved model
        training_mode: Training method name (flow_matching, diffusion_dpo, etc.)
        job_id: Job identifier
        job_manager: JobManager for status updates
        event_loop: Event loop for WebSocket updates
    """
    try:
        logger.info(f"Starting background upload for atelier job {job_id}")
        job_manager.update_job(job_id, status="uploading")

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="uploading",
                progress=0.95
            ),
            event_loop
        )

        api = HfApi()

        # Create or get the repository
        repo_url = api.create_repo(
            repo_id=config.output_name,
            token=config.hf_token,
            private=getattr(config, "hf_hub_private", True),
            exist_ok=True
        )
        logger.info(f"Repository ready: {repo_url}")

        # Upload the saved model directory directly
        if getattr(config, "use_lora", False):
            commit_msg = f"Upload LoRA adapter trained with Merlina Atelier ({training_mode})"
        else:
            commit_msg = f"Upload model trained with Merlina Atelier ({training_mode})"

        logger.info(f"Uploading model to HuggingFace Hub: {config.output_name}")

        api.upload_folder(
            folder_path=final_output_dir,
            repo_id=config.output_name,
            token=config.hf_token,
            commit_message=commit_msg
        )

        logger.info(f"Model uploaded successfully to {config.output_name}")

        # Update job status after successful upload
        job_manager.update_job(job_id, status="completed", progress=1.0, output_dir=final_output_dir)

        send_websocket_update(
            websocket_manager.send_completion(
                job_id=job_id,
                output_dir=final_output_dir
            ),
            event_loop
        )

    except Exception as upload_error:
        logger.error(f"Background upload failed for atelier job {job_id}: {str(upload_error)}", exc_info=True)
        logger.warning("Training completed successfully, but upload failed")
        logger.info(f"Model saved locally at: {final_output_dir}")
        # Mark as completed anyway since training succeeded
        job_manager.update_job(job_id, status="completed", progress=1.0, output_dir=final_output_dir)

        send_websocket_update(
            websocket_manager.send_completion(
                job_id=job_id,
                output_dir=final_output_dir
            ),
            event_loop
        )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

async def run_atelier_training_async(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
) -> None:
    """
    Async wrapper for atelier training to enable WebSocket updates.

    Runs the synchronous training function in a thread pool executor
    to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, run_atelier_training_sync,
        job_id, config, job_manager, uploaded_datasets, loop
    )


def run_atelier_training_sync(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    """
    Run a diffusion model training job with Atelier.

    Supports adapter-based model loading (QwenEdit, SDXL), embedding caching,
    multiple loss functions, LoRA fine-tuning, and WebSocket progress updates.

    Args:
        job_id: Unique job identifier for tracking
        config: Training configuration containing all hyperparameters
        job_manager: JobManager instance for persistence and status updates
        uploaded_datasets: Dictionary mapping dataset IDs to uploaded content
        event_loop: Event loop for WebSocket updates (optional, for async context)

    Note:
        This function handles cleanup of GPU resources in all cases (success,
        failure, or interruption) to prevent OOM errors in subsequent jobs.
    """
    # Initialize resources to None for proper cleanup tracking
    adapter = None
    trainer = None

    try:
        # Late import — atelier may not be installed
        try:
            from atelier import AtelierTrainer, TrainingConfig as AtelierConfig
            from atelier.adapters import QwenEditAdapter, SDXLAdapter
            from atelier.losses import (
                FlowMatchingLoss, EpsilonLoss,
                DiffusionDPOLoss, DiffusionORPOLoss, DiffusionSimPOLoss,
                DiffusionCPOLoss, DiffusionIPOLoss, DiffusionKTOLoss,
            )
            from atelier.data import (
                cache_embeddings, EditingDataset, GenerationDataset,
            )
        except ImportError as e:
            raise ImportError(
                f"Atelier is not installed. Install it with: "
                f"pip install -e /path/to/atelier  -- {e}"
            ) from e

        import datasets as hf_datasets

        # -----------------------------------------------------------------
        # 1. Initialize
        # -----------------------------------------------------------------
        job_manager.update_job(job_id, status="initializing", progress=0.0)

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="initializing",
                progress=0.0
            ),
            event_loop
        )

        # Set visible GPUs if specified
        if getattr(config, "gpu_ids", None) is not None:
            from src.gpu_utils import get_gpu_manager
            gpu_manager = get_gpu_manager()
            gpu_manager.set_visible_devices(config.gpu_ids)
            logger.info(f"Using GPUs: {config.gpu_ids}")

        # Configure Weights & Biases
        wandb_run_name = None
        wandb_project = None

        if getattr(config, "use_wandb", False):
            wandb_key = getattr(config, "wandb_key", None)
            if wandb_key:
                wandb.login(key=wandb_key)

            wandb_run_name = getattr(config, "wandb_run_name", None)
            if not wandb_run_name:
                wandb_run_name = generate_wandb_run_name(config)
                logger.info(f"Auto-generated W&B run name: {wandb_run_name}")

            wandb_project = getattr(config, "wandb_project", None) or "merlina-atelier"
            logger.info(f"W&B Project: {wandb_project}, Run: {wandb_run_name}")

        # Set HF token if provided
        hf_token = getattr(config, "hf_token", None)
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Determine dtype based on GPU capability
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # -----------------------------------------------------------------
        # 2. Create adapter
        # -----------------------------------------------------------------
        job_manager.update_job(job_id, status="loading_model", progress=0.1)

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="loading_model",
                progress=0.1,
                message=f"Loading {config.adapter_type} adapter"
            ),
            event_loop
        )

        adapter_type = config.adapter_type.lower()

        if adapter_type == "qwen_edit":
            logger.info(f"Creating QwenEditAdapter from {config.base_model}")
            adapter = QwenEditAdapter(config.base_model, dtype=torch_dtype)
        elif adapter_type == "sdxl":
            weights_path = getattr(config, "model_weights_path", None)
            logger.info(f"Creating SDXLAdapter from {config.base_model} (weights={weights_path})")
            adapter = SDXLAdapter(config.base_model, weights=weights_path, dtype=torch_dtype)
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. "
                f"Supported: qwen_edit, sdxl"
            )

        # -----------------------------------------------------------------
        # 3. Load dataset
        # -----------------------------------------------------------------
        job_manager.update_job(job_id, status="loading_dataset", progress=0.2)

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="loading_dataset",
                progress=0.2
            ),
            event_loop
        )

        # Determine dataset source from Merlina's DatasetConfig structure
        dataset_cfg = getattr(config, "dataset", None)
        if dataset_cfg is not None:
            if hasattr(dataset_cfg, "source"):
                source = dataset_cfg.source
            elif isinstance(dataset_cfg, dict):
                source = dataset_cfg.get("source")
            else:
                source = None
            if source is not None:
                if hasattr(source, "repo_id"):
                    dataset_name = source.repo_id
                    dataset_split = source.split or "train"
                elif isinstance(source, dict):
                    dataset_name = source.get("repo_id")
                    dataset_split = source.get("split", "train")
                else:
                    dataset_name = None
                    dataset_split = "train"
            else:
                dataset_name = None
                dataset_split = "train"
            max_samples = getattr(dataset_cfg, "max_samples", None)
            if max_samples is None and isinstance(dataset_cfg, dict):
                max_samples = dataset_cfg.get("max_samples")
        else:
            # Fallback to flat config attributes
            dataset_name = getattr(config, "dataset_name", None)
            dataset_split = getattr(config, "dataset_split", "train")
            max_samples = getattr(config, "max_samples", None)

        if not dataset_name:
            raise ValueError("Dataset repository ID is required for atelier training")

        logger.info(f"Loading dataset: {dataset_name} (split={dataset_split})")

        dataset = hf_datasets.load_dataset(
            dataset_name,
            split=dataset_split,
            token=hf_token,
        )

        # Limit samples if configured
        if max_samples and max_samples < len(dataset):
            dataset = dataset.shuffle(seed=getattr(config, "seed", 42)).select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")

        logger.info(f"Dataset loaded: {len(dataset)} samples, columns: {dataset.column_names}")

        # -----------------------------------------------------------------
        # 4. Optionally cache embeddings (qwen_edit only)
        # -----------------------------------------------------------------
        cached_text = None
        cached_targets = None
        cached_controls = None

        if getattr(config, "cache_embeddings", False) and adapter_type == "qwen_edit":
            image_size = getattr(config, "image_size", 1024)
            cache_dir = getattr(config, "cache_dir", None) or f"./cache/{job_id}"

            logger.info(f"Caching embeddings to {cache_dir} (target_area={image_size**2})")

            send_websocket_update(
                websocket_manager.send_status_update(
                    job_id=job_id,
                    status="caching_embeddings",
                    progress=0.25,
                    message="Pre-computing embeddings..."
                ),
                event_loop
            )

            cached_text, cached_targets, cached_controls = cache_embeddings(
                dataset, adapter, cache_dir=cache_dir, target_area=image_size**2
            )

            logger.info(f"Cached {len(cached_text)} embeddings")

        # -----------------------------------------------------------------
        # 5. Create dataset wrapper
        # -----------------------------------------------------------------
        image_size = getattr(config, "image_size", 1024)

        if adapter_type == "qwen_edit":
            train_dataset = EditingDataset(
                dataset,
                cached_text_embeddings=cached_text,
                cached_target_embeddings=cached_targets,
                cached_control_embeddings=cached_controls,
                max_samples=max_samples,
            )
            logger.info(f"Created EditingDataset with {len(train_dataset)} samples")
        elif adapter_type == "sdxl":
            train_dataset = GenerationDataset(
                dataset,
                tokenizer=adapter.tokenizer,
                tokenizer_2=adapter.tokenizer_2,
                image_size=image_size,
                max_samples=max_samples,
            )
            logger.info(f"Created GenerationDataset with {len(train_dataset)} samples")

        # -----------------------------------------------------------------
        # 6. Create loss function
        # -----------------------------------------------------------------
        training_mode = config.training_mode.lower()
        logger.info(f"Using training mode: {training_mode}")

        if training_mode == "flow_matching":
            loss_fn = FlowMatchingLoss()
        elif training_mode == "epsilon":
            loss_fn = EpsilonLoss()
        elif training_mode == "diffusion_dpo":
            loss_fn = DiffusionDPOLoss(
                beta=getattr(config, "beta", 0.4),
                sft_weight=getattr(config, "sft_weight", 0.3),
            )
        elif training_mode == "diffusion_orpo":
            loss_fn = DiffusionORPOLoss(
                beta=getattr(config, "beta", 0.1),
            )
        elif training_mode == "diffusion_simpo":
            loss_fn = DiffusionSimPOLoss(
                beta=getattr(config, "beta", 2.0),
                gamma=getattr(config, "gamma", 0.5),
            )
        elif training_mode == "diffusion_cpo":
            loss_fn = DiffusionCPOLoss(
                beta=getattr(config, "beta", 0.1),
                label_smoothing=getattr(config, "label_smoothing", 0.0),
            )
        elif training_mode == "diffusion_ipo":
            loss_fn = DiffusionIPOLoss(
                beta=getattr(config, "beta", 0.1),
            )
        elif training_mode == "diffusion_kto":
            loss_fn = DiffusionKTOLoss(
                beta=getattr(config, "beta", 0.1),
            )
        else:
            raise ValueError(
                f"Unknown training mode: {training_mode}. Supported: "
                f"flow_matching, epsilon, diffusion_dpo, diffusion_orpo, "
                f"diffusion_simpo, diffusion_cpo, diffusion_ipo, diffusion_kto"
            )

        # -----------------------------------------------------------------
        # 7. Setup LoRA (if enabled)
        # -----------------------------------------------------------------
        peft_config = None
        if getattr(config, "use_lora", True):
            peft_config = LoraConfig(
                r=getattr(config, "lora_r", 64),
                lora_alpha=getattr(config, "lora_alpha", 128),
                lora_dropout=getattr(config, "lora_dropout", 0.0),
                target_modules=getattr(config, "target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
            )
            logger.info(f"LoRA enabled with rank={peft_config.r}, alpha={peft_config.lora_alpha}")
        else:
            logger.info("LoRA disabled - training full model")

        # -----------------------------------------------------------------
        # 8. Training config and trainer
        # -----------------------------------------------------------------
        job_manager.update_job(job_id, status="training", progress=0.3)

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="training",
                progress=0.3
            ),
            event_loop
        )

        output_dir = f"./results/{job_id}"

        # Determine mixed precision setting
        if torch_dtype == torch.bfloat16:
            mixed_precision = "bf16"
        elif torch_dtype == torch.float16:
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"

        # Convert eval_steps: <1 means ratio, >=1 means absolute steps
        eval_steps_cfg = getattr(config, "eval_steps", None)
        eval_steps = None
        if eval_steps_cfg:
            if eval_steps_cfg < 1:
                num_gpus = get_num_gpus()
                batch_size = getattr(config, "batch_size", 1)
                grad_accum = getattr(config, "gradient_accumulation_steps", 1)
                effective_batch = batch_size * grad_accum * num_gpus
                steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
                num_epochs = getattr(config, "num_epochs", 1)
                total_steps = steps_per_epoch * num_epochs
                eval_steps = max(1, int(total_steps * eval_steps_cfg))
                logger.info(
                    f"Eval steps: {eval_steps} (ratio={eval_steps_cfg}, "
                    f"dataset={len(train_dataset)}, batch={batch_size}, "
                    f"grad_accum={grad_accum}, num_gpus={num_gpus}, "
                    f"epochs={num_epochs}, total_steps={total_steps})"
                )
            else:
                eval_steps = int(eval_steps_cfg)

        atelier_config = AtelierConfig(
            output_dir=output_dir,
            num_epochs=getattr(config, "num_epochs", 1),
            batch_size=getattr(config, "batch_size", 1),
            gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
            learning_rate=getattr(config, "learning_rate", 1e-4),
            weight_decay=getattr(config, "weight_decay", 0.01),
            warmup_ratio=getattr(config, "warmup_ratio", 0.1),
            max_grad_norm=getattr(config, "max_grad_norm", 1.0),
            mixed_precision=mixed_precision,
            gradient_checkpointing=getattr(config, "gradient_checkpointing", True),
            optimizer=getattr(config, "optimizer_type", "adamw"),
            lr_scheduler=getattr(config, "lr_scheduler_type", "cosine"),
            logging_steps=getattr(config, "logging_steps", 10),
            eval_steps=eval_steps,
            save_steps=eval_steps,
            save_total_limit=2,
            seed=getattr(config, "seed", 42),
            run_name=wandb_run_name if getattr(config, "use_wandb", False) else getattr(config, "output_name", job_id),
            log_with="wandb" if getattr(config, "use_wandb", False) else None,
            project_name=wandb_project,
        )

        trainer = AtelierTrainer(
            adapter=adapter,
            config=atelier_config,
            loss_fn=loss_fn,
            train_dataset=train_dataset,
            peft_config=peft_config,
            callbacks=[AtelierWebSocketCallback(job_id, job_manager, event_loop)],
        )

        # Capture W&B run URL after trainer init (accelerator owns the wandb run)
        if getattr(config, "use_wandb", False) and wandb.run is not None:
            wandb_url = wandb.run.get_url()
            logger.info(f"W&B run URL: {wandb_url}")
            job_manager.update_job(job_id, wandb_url=wandb_url)

        # -----------------------------------------------------------------
        # 9. Train
        # -----------------------------------------------------------------
        logger.info("Starting atelier training")
        trainer.train()

        # Check if training was stopped early
        was_stopped = trainer.stopped_early

        # -----------------------------------------------------------------
        # 10. Save model
        # -----------------------------------------------------------------
        save_status = "saving_stopped" if was_stopped else "saving"
        job_manager.update_job(job_id, status=save_status, progress=0.9)

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="saving",
                progress=0.9
            ),
            event_loop
        )

        final_output_dir = f"./models/{getattr(config, 'output_name', job_id)}"
        trainer.save_model(final_output_dir)

        # Capture step info before cleanup
        final_step = trainer.global_step
        final_max_steps = trainer.max_steps

        # -----------------------------------------------------------------
        # 11. Clean up GPU resources
        # -----------------------------------------------------------------
        logger.info("Cleaning up training resources to free VRAM...")
        del trainer, adapter
        trainer = None
        adapter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VRAM freed successfully")

        # -----------------------------------------------------------------
        # 12. Optionally upload to HuggingFace Hub
        # -----------------------------------------------------------------
        upload_thread_started = False
        if getattr(config, "push_to_hub", False):
            is_distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, "is_initialized") else False
            should_upload = (not is_distributed) or (torch.distributed.get_rank() == 0)

            if should_upload:
                if not hf_token:
                    logger.warning("push_to_hub enabled but no HF token provided - skipping upload")
                else:
                    logger.info("Starting background upload thread...")
                    upload_thread = threading.Thread(
                        target=_run_background_atelier_upload,
                        args=(config, final_output_dir, training_mode, job_id, job_manager, event_loop),
                        name=f"AtelierUploadThread-{job_id}",
                        daemon=False,
                    )
                    upload_thread.start()
                    upload_thread_started = True
                    logger.info(f"Background upload thread started for job {job_id}")
            else:
                logger.info("Skipping HuggingFace upload (not main process in distributed training)")

        # Mark as completed or stopped
        if upload_thread_started:
            logger.info(f"Training finished, upload running in background for job {job_id}")
        else:
            final_status = "stopped" if was_stopped else "completed"
            final_progress = 1.0 if not was_stopped else (
                final_step / final_max_steps if final_max_steps else 0.9
            )

            job_manager.update_job(
                job_id,
                status=final_status,
                progress=final_progress,
                output_dir=final_output_dir
            )

            send_websocket_update(
                websocket_manager.send_completion(
                    job_id=job_id,
                    output_dir=final_output_dir
                ),
                event_loop
            )

            if was_stopped:
                logger.info(f"Training stopped early for job {job_id} at step {final_step}/{final_max_steps}")
            else:
                logger.info(f"Atelier training completed for job {job_id}")

        # Finish wandb run
        if getattr(config, "use_wandb", False) and wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished successfully")

    except Exception as e:
        logger.error(f"Atelier training failed for job {job_id}: {str(e)}", exc_info=True)
        job_manager.update_job(job_id, status="failed", error=str(e))

        send_websocket_update(
            websocket_manager.send_error(
                job_id=job_id,
                error=str(e)
            ),
            event_loop
        )

        # Finish wandb run even on failure
        try:
            if getattr(config, "use_wandb", False) and wandb.run is not None:
                wandb.finish(exit_code=1)
                logger.info("W&B run finished (marked as failed)")
        except Exception as wandb_error:
            logger.debug(f"Could not finish W&B run on failure: {wandb_error}")

    finally:
        # Ensure GPU memory is freed even on error
        # _cleanup_training_resources expects (model, trainer) but for atelier
        # the adapter holds the model.  Pass adapter as the "model" argument.
        _cleanup_training_resources(adapter, trainer)
