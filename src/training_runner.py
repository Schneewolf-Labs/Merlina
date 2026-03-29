"""
Enhanced Training Runner with WebSocket updates and better error handling.

This module provides the core training functionality for Merlina, including:
- Model loading with quantization support
- Dataset preparation and formatting
- Multiple training modes: SFT, ORPO, DPO, SimPO, CPO, IPO, KTO
- WebSocket progress updates
- Background HuggingFace Hub uploads
- Weights & Biases integration
"""

import os
import gc
import sys
import json
import time
import shutil
import signal
import tempfile
import subprocess
import torch
import wandb
import logging
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Callable

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    # transformers v5+ renamed AutoModelForVision2Seq to AutoModelForImageTextToText
    from transformers import AutoModelForImageTextToText as AutoModelForVLM
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    except ImportError:
        AutoModelForVLM = None
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM
from grimoire import GrimoireTrainer, TrainingConfig, TrainerCallback
from grimoire.losses import SFTLoss, ORPOLoss, DPOLoss, SimPOLoss, CPOLoss, IPOLoss, KTOLoss
from grimoire.data import tokenize_sft, tokenize_preference, tokenize_kto

from huggingface_hub import HfApi
from src.model_card import generate_model_readme, upload_model_readme, generate_wandb_run_name

from dataset_handlers import (
    DatasetPipeline,
    HuggingFaceLoader,
    LocalFileLoader,
    UploadedDatasetLoader,
    get_formatter,
    get_chat_template_for_format,
    create_loader_from_config
)
from src.job_manager import JobManager
from src.websocket_manager import websocket_manager
from src.preflight_checks import is_local_model_path
from src.utils import get_num_gpus, calculate_effective_batch_size, get_torch_dtype

logger = logging.getLogger(__name__)

# VLM architecture detection patterns
_VLM_ARCH_PATTERNS = ['ForConditionalGeneration', 'ForVision', 'ImageText']


def _detect_is_vlm(model_name: str) -> bool:
    """Detect if a model is a VLM by checking its config for vision components."""
    try:
        auto_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Check for vision config (most reliable indicator)
        if hasattr(auto_config, 'vision_config') or hasattr(auto_config, 'visual'):
            return True
        # Check architecture name patterns
        architectures = getattr(auto_config, 'architectures', []) or []
        if architectures:
            arch = architectures[0]
            if any(p in arch for p in _VLM_ARCH_PATTERNS):
                return True
    except Exception as e:
        logger.debug(f"Could not auto-detect model type for {model_name}: {e}")
    return False


def _get_auto_model_class(model_name: str, model_type: str = "auto"):
    """Return the appropriate AutoModel class for loading.

    Args:
        model_name: HuggingFace model ID or local path.
        model_type: "auto" (detect), "causal_lm", or "vlm".

    Returns:
        (auto_model_class, is_vlm) tuple.
    """
    if model_type == "vlm":
        is_vlm = True
    elif model_type == "causal_lm":
        is_vlm = False
    else:
        # Auto-detect
        is_vlm = _detect_is_vlm(model_name)

    if is_vlm:
        if AutoModelForVLM is None:
            raise ImportError(
                "No VLM auto-model class found in your transformers version. "
                "Upgrade transformers or use model_type='causal_lm'."
            )
        logger.info("Model type: VLM — using %s", getattr(AutoModelForVLM, '__name__', 'AutoModelForVLM'))
        return AutoModelForVLM, True
    else:
        logger.info("Model type: Causal LM — using AutoModelForCausalLM")
        return AutoModelForCausalLM, False


def _run_background_upload(
    config: Any,
    final_output_dir: str,
    training_mode: str,
    job_id: str,
    job_manager: JobManager,
    event_loop=None
) -> None:
    """
    Run HuggingFace Hub upload in a background thread.

    This allows the job queue worker to continue processing other jobs
    while the upload (which can take a long time for large models) runs.

    Uses upload_folder() to upload files directly without loading the model
    into GPU memory, avoiding OOM errors when another training job is running.

    Args:
        config: Training configuration
        final_output_dir: Path to saved model
        training_mode: Training method name (sft, orpo, dpo, simpo, cpo, ipo)
        job_id: Job identifier
        job_manager: JobManager for status updates
        event_loop: Event loop for WebSocket updates
    """
    try:
        logger.info(f"🚀 Starting background upload for job {job_id}")
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
        repo_visibility = "private" if config.hf_hub_private else "public"

        # Create or get the repository
        repo_url = api.create_repo(
            repo_id=config.output_name,
            token=config.hf_token,
            private=config.hf_hub_private,
            exist_ok=True
        )
        logger.info(f"📦 Repository ready: {repo_url}")

        # Handle upload based on whether LoRA was used and merge preference
        if config.use_lora and config.merge_lora_before_upload:
            # For LoRA merge, we need to load the model - try CPU-only to avoid GPU conflicts
            logger.info(f"🔄 Merging LoRA adapter with base model for upload (using CPU)...")

            try:
                # Force CPU-only loading to avoid GPU OOM when another job is training
                model_type = getattr(config, 'model_type', 'auto')
                reload_cls, _ = _get_auto_model_class(config.base_model, model_type)
                base_model_reload = reload_cls.from_pretrained(
                    config.base_model,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",  # Force CPU to avoid GPU conflicts
                    trust_remote_code=True,
                )

                model_merged = PeftModel.from_pretrained(
                    base_model_reload,
                    final_output_dir,
                    device_map="cpu"
                )
                model_merged = model_merged.merge_and_unload()

                logger.info(f"📤 Pushing merged model to HuggingFace Hub as {repo_visibility} repository...")
                logger.info(f"   Repository: {config.output_name}")

                model_merged.push_to_hub(
                    config.output_name,
                    token=config.hf_token,
                    private=config.hf_hub_private
                )
                logger.info(f"✅ Merged model uploaded successfully!")

                # Upload tokenizer separately
                tokenizer = AutoTokenizer.from_pretrained(final_output_dir)
                tokenizer.push_to_hub(
                    config.output_name,
                    token=config.hf_token,
                    private=config.hf_hub_private
                )
                logger.info(f"✅ Tokenizer uploaded successfully!")

                # Clean up
                del model_merged, base_model_reload
                gc.collect()

            except Exception as merge_error:
                # If CPU merge fails (e.g., not enough RAM), fall back to uploading adapter only
                logger.warning(f"⚠️ Could not merge LoRA on CPU: {merge_error}")
                logger.info("📤 Falling back to uploading LoRA adapter only...")

                # Upload the saved model directory directly (contains adapter files)
                api.upload_folder(
                    folder_path=final_output_dir,
                    repo_id=config.output_name,
                    token=config.hf_token,
                    commit_message=f"Upload LoRA adapter trained with Merlina ({training_mode})"
                )
                logger.info(f"✅ LoRA adapter uploaded successfully!")
                logger.info(f"💡 To use: PeftModel.from_pretrained('{config.base_model}', '{config.output_name}')")

        else:
            # For adapter-only or full model uploads, use upload_folder() directly
            # This doesn't require loading the model into memory at all
            if config.use_lora:
                logger.info(f"📤 Uploading LoRA adapter to HuggingFace Hub as {repo_visibility} repository...")
                commit_msg = f"Upload LoRA adapter trained with Merlina ({training_mode})"
            else:
                logger.info(f"📤 Uploading full model to HuggingFace Hub as {repo_visibility} repository...")
                commit_msg = f"Upload model trained with Merlina ({training_mode})"

            logger.info(f"   Repository: {config.output_name}")

            # Upload the entire model directory directly - no model loading needed!
            api.upload_folder(
                folder_path=final_output_dir,
                repo_id=config.output_name,
                token=config.hf_token,
                commit_message=commit_msg
            )

            if config.use_lora:
                logger.info(f"✅ LoRA adapter uploaded successfully!")
                logger.info(f"💡 To use: PeftModel.from_pretrained('{config.base_model}', '{config.output_name}')")
            else:
                logger.info(f"✅ Model uploaded successfully!")

        # Generate and upload README with model card
        hub_url = f"https://huggingface.co/{config.output_name}"
        logger.info(f"🎉 Model published at: {hub_url}")
        logger.info("📝 Generating model card README...")
        readme_content = generate_model_readme(config, training_mode)
        upload_model_readme(config.output_name, readme_content, config.hf_token)

        # Update job status after successful upload
        logger.info(f"✅ Background upload completed for job {job_id}")
        job_manager.update_job(job_id, status="completed", progress=1.0, output_dir=final_output_dir)

        send_websocket_update(
            websocket_manager.send_completion(
                job_id=job_id,
                output_dir=final_output_dir
            ),
            event_loop
        )

    except Exception as upload_error:
        logger.error(f"❌ Background upload failed for job {job_id}: {str(upload_error)}", exc_info=True)
        logger.warning("⚠️ Training completed successfully, but upload failed")
        logger.info(f"💾 Model saved locally at: {final_output_dir}")
        # Mark as completed anyway since training succeeded - upload is optional
        job_manager.update_job(job_id, status="completed", progress=1.0, output_dir=final_output_dir)

        send_websocket_update(
            websocket_manager.send_completion(
                job_id=job_id,
                output_dir=final_output_dir
            ),
            event_loop
        )


def send_websocket_update(coro, loop=None):
    """
    Helper to properly schedule WebSocket updates from sync context.

    Args:
        coro: Coroutine to execute
        loop: Event loop to schedule in (optional, uses global if not provided)
    """
    try:
        if loop is None:
            # No loop provided, just close the coroutine to avoid warnings
            coro.close()
            return

        # Schedule coroutine in the provided event loop
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as e:
        logger.debug(f"Could not send WebSocket update: {e}")


class WebSocketCallback(TrainerCallback):
    """
    Custom callback to send training metrics via WebSocket and handle stop requests.

    Uses Grimoire's TrainerCallback interface:
        on_step_end(trainer, step, loss, metrics)
        on_log(trainer, metrics)
        on_evaluate(trainer, metrics)
    """

    def __init__(self, job_id: str, job_manager: JobManager, event_loop=None):
        self.job_id = job_id
        self.job_manager = job_manager
        self.event_loop = event_loop

    def on_step_end(self, trainer, step, loss, metrics):
        """Called at the end of each training step to check for stop requests"""
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
        """Called when trainer logs metrics"""
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
        """Called after evaluation completes"""
        update_data = {}
        if "eval/loss" in metrics:
            update_data["eval_loss"] = float(metrics["eval/loss"])
        if update_data:
            self.job_manager.update_job(self.job_id, **update_data)


def _cleanup_training_resources(model: Optional[Any], trainer: Optional[Any]) -> None:
    """
    Clean up training resources to free GPU memory.

    This function safely deletes model and trainer objects and clears
    CUDA cache to prevent OOM errors in subsequent training jobs.

    Args:
        model: The model object to clean up (can be None)
        trainer: The trainer object to clean up (can be None)
    """
    try:
        # Delete trainer first as it may hold references to model
        if trainer is not None:
            del trainer
            logger.debug("Trainer cleaned up")

        # Then delete model
        if model is not None:
            del model
            logger.debug("Model cleaned up")

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cleaned up")

    except Exception as cleanup_error:
        logger.debug(f"Error during cleanup (non-fatal): {cleanup_error}")


def _get_distributed_gpu_count(config: Any) -> int:
    """
    Determine how many GPUs should participate in distributed training.

    Returns the count based on explicit gpu_ids or CUDA device count.
    Returns 0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0
    if config.gpu_ids is not None:
        return len(config.gpu_ids)
    return torch.cuda.device_count()


def _serialize_uploaded_datasets(uploaded_datasets: dict, tmp_dir: str) -> Optional[str]:
    """
    Save in-memory uploaded datasets to a temp directory so the subprocess can access them.

    Returns the directory path, or None if there are no uploaded datasets.
    """
    if not uploaded_datasets:
        return None

    ds_dir = os.path.join(tmp_dir, "uploaded_datasets")
    os.makedirs(ds_dir, exist_ok=True)

    meta = {}
    for dataset_id, entry in uploaded_datasets.items():
        filename = entry.get("filename", f"{dataset_id}.json")
        file_path = os.path.join(ds_dir, filename)
        content = entry.get("content", b"")
        if isinstance(content, bytes):
            with open(file_path, "wb") as f:
                f.write(content)
        else:
            with open(file_path, "w") as f:
                f.write(str(content))
        meta[dataset_id] = {"filename": filename}

    with open(os.path.join(ds_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    return ds_dir


def _monitor_distributed_progress(
    proc: subprocess.Popen,
    progress_file: str,
    job_id: str,
    job_manager: JobManager,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    """
    Monitor a distributed training subprocess by tailing its progress file.

    Forwards updates to the WebSocket manager and job manager.
    Blocks until the subprocess exits.
    """
    last_pos = 0

    while proc.poll() is None:
        # Check for stop requests and forward to subprocess
        job = job_manager.get_job(job_id)
        if job and job.stop_requested:
            logger.info(f"Forwarding stop request to distributed subprocess for job {job_id}")
            try:
                # Send SIGTERM to the entire process group
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Read new progress lines
        try:
            if os.path.exists(progress_file):
                with open(progress_file, "r") as f:
                    f.seek(last_pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            _handle_progress_record(record, job_id, job_manager, event_loop)
                        except json.JSONDecodeError:
                            pass
                    last_pos = f.tell()
        except Exception as e:
            logger.debug(f"Error reading progress file: {e}")

        time.sleep(1.0)

    # Process exited — read remaining lines
    try:
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        _handle_progress_record(record, job_id, job_manager, event_loop)
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass

    # Check exit code
    if proc.returncode != 0:
        # Check if the job was already marked as failed/completed by the worker
        job = job_manager.get_job(job_id)
        if job and job.status not in ("completed", "stopped", "failed", "uploading"):
            error_msg = f"Distributed training subprocess exited with code {proc.returncode}"
            logger.error(error_msg)
            job_manager.update_job(job_id, status="failed", error=error_msg)
            send_websocket_update(
                websocket_manager.send_error(job_id=job_id, error=error_msg),
                event_loop,
            )


def _handle_progress_record(
    record: dict,
    job_id: str,
    job_manager: JobManager,
    event_loop: Optional[asyncio.AbstractEventLoop],
) -> None:
    """Handle a single progress record from the distributed worker."""
    rec_type = record.get("type", "")

    if rec_type == "metrics":
        update_data = {}
        if "loss" in record:
            update_data["loss"] = record["loss"]
        if "learning_rate" in record:
            update_data["learning_rate"] = record["learning_rate"]
        if "current_step" in record:
            update_data["current_step"] = record["current_step"]
        if "total_steps" in record:
            update_data["total_steps"] = record["total_steps"]
        if "progress" in record:
            update_data["progress"] = record["progress"]

        # Send WebSocket update
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="training",
                progress=record.get("progress", 0.5),
                current_step=record.get("current_step"),
                total_steps=record.get("total_steps"),
                loss=record.get("loss"),
                learning_rate=record.get("learning_rate"),
                gpu_memory=record.get("gpu_memory"),
            ),
            event_loop,
        )

    elif rec_type == "eval":
        if "eval_loss" in record:
            send_websocket_update(
                websocket_manager.send_status_update(
                    job_id=job_id,
                    status="training",
                    eval_loss=record["eval_loss"],
                ),
                event_loop,
            )

    elif rec_type == "completed":
        output_dir = record.get("output_dir", "")
        send_websocket_update(
            websocket_manager.send_completion(job_id=job_id, output_dir=output_dir),
            event_loop,
        )

    elif rec_type == "error":
        send_websocket_update(
            websocket_manager.send_error(job_id=job_id, error=record.get("error", "Unknown error")),
            event_loop,
        )

    elif rec_type == "status":
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status=record.get("status", "training"),
                message=record.get("message"),
            ),
            event_loop,
        )


def run_training_distributed(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    """
    Launch training as a subprocess via ``accelerate launch`` for multi-GPU DDP.

    This is the multi-GPU alternative to ``run_training_sync``.  It serialises
    the training configuration and uploaded datasets to temporary files, spawns
    ``accelerate launch --multi_gpu --num_processes N src/train_worker.py``,
    and monitors a JSONL progress file to relay updates back to the WebSocket
    and job manager.

    Args:
        job_id: Unique job identifier.
        config: Training configuration (Pydantic model).
        job_manager: JobManager for persistence.
        uploaded_datasets: In-memory uploaded dataset dict.
        event_loop: Event loop for WebSocket updates.
    """
    num_gpus = _get_distributed_gpu_count(config)
    logger.info(f"Launching distributed training for job {job_id} with {num_gpus} GPUs")

    tmp_dir = tempfile.mkdtemp(prefix=f"merlina_{job_id}_")
    config_path = os.path.join(tmp_dir, "config.json")
    progress_file = os.path.join(tmp_dir, "progress.jsonl")

    try:
        job_manager.update_job(job_id, status="initializing", progress=0.0)
        send_websocket_update(
            websocket_manager.send_status_update(job_id=job_id, status="initializing", progress=0.0),
            event_loop,
        )

        # Serialize config
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f)

        # Serialize uploaded datasets to disk
        ds_dir = _serialize_uploaded_datasets(uploaded_datasets, tmp_dir)

        # Create empty progress file
        open(progress_file, "w").close()

        # Determine mixed precision for accelerate flag
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            accel_mp = "bf16"
        else:
            accel_mp = "fp16"

        # Build accelerate launch command
        worker_script = os.path.join(os.path.dirname(__file__), "train_worker.py")
        db_path = str(job_manager.db_path)

        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--num_processes", str(num_gpus),
            "--mixed_precision", accel_mp,
            "--multi_gpu",
            worker_script,
            "--config-path", config_path,
            "--job-id", job_id,
            "--progress-file", progress_file,
            "--db-path", db_path,
        ]
        if ds_dir:
            cmd.extend(["--uploaded-datasets-dir", ds_dir])

        # Set up environment
        env = os.environ.copy()
        if config.gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpu_ids)

        logger.info(f"Launching: {' '.join(cmd)}")

        # Start subprocess in its own process group for clean signal handling
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Start a thread to log subprocess stdout
        def _log_stdout():
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.info(f"[worker] {line}")

        log_thread = threading.Thread(target=_log_stdout, daemon=True)
        log_thread.start()

        # Monitor progress until subprocess exits
        _monitor_distributed_progress(proc, progress_file, job_id, job_manager, event_loop)

        log_thread.join(timeout=5)

    except Exception as e:
        logger.error(f"Distributed training failed for job {job_id}: {e}", exc_info=True)
        job_manager.update_job(job_id, status="failed", error=str(e))
        send_websocket_update(
            websocket_manager.send_error(job_id=job_id, error=str(e)),
            event_loop,
        )

    finally:
        # Clean up temp files
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


async def run_training_async(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict
) -> None:
    """
    Async wrapper for training to enable WebSocket updates.

    Runs the synchronous training function in a thread pool executor
    to avoid blocking the event loop.

    Args:
        job_id: Unique job identifier
        config: Training configuration
        job_manager: JobManager instance
        uploaded_datasets: Dictionary of uploaded datasets
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_training_sync, job_id, config, job_manager, uploaded_datasets)


def run_training_sync(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
    event_loop: Optional[asyncio.AbstractEventLoop] = None
) -> None:
    """
    Run training job with enhanced monitoring and WebSocket updates.

    Supports multiple training modes (SFT, ORPO, DPO, SimPO, CPO, IPO, KTO)
    with optional LoRA adapters.

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
    model = None
    trainer = None

    try:
        # Update job status
        job_manager.update_job(job_id, status="initializing", progress=0.0)

        # Send WebSocket update
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="initializing",
                progress=0.0
            ),
            event_loop
        )

        # Set visible GPUs if specified
        if config.gpu_ids is not None:
            from src.gpu_utils import get_gpu_manager
            gpu_manager = get_gpu_manager()
            gpu_manager.set_visible_devices(config.gpu_ids)
            logger.info(f"Using GPUs: {config.gpu_ids}")

        # NOTE: Do NOT create a bare Accelerator() here — it initializes a global
        # AcceleratorState singleton that conflicts with GrimoireTrainer's Accelerator
        # (which needs mixed_precision). The trainer creates its own accelerator.

        # Configure Weights & Biases
        wandb_run_name = None
        wandb_project = None

        if config.use_wandb:
            # Login to wandb if key provided
            if config.wandb_key:
                wandb.login(key=config.wandb_key)

            # Generate W&B run name if not provided
            wandb_run_name = config.wandb_run_name
            if not wandb_run_name:
                wandb_run_name = generate_wandb_run_name(config)
                logger.info(f"Auto-generated W&B run name: {wandb_run_name}")

            # Set W&B project name (default if not specified)
            wandb_project = config.wandb_project or "merlina-training"
            logger.info(f"W&B Project: {wandb_project}, Run: {wandb_run_name}")

            # NOTE: Do NOT call wandb.init() here — Grimoire's Accelerator handles
            # the full wandb lifecycle via init_trackers(). Calling it here creates
            # a conflict where accelerate gets a stale run handle and metrics don't log.
            # The wandb URL is captured after trainer creation below.

        # Set HF token if provided
        if config.hf_token:
            os.environ['HF_TOKEN'] = config.hf_token

        # Determine dtype based on GPU capability
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # Determine attention implementation with smart fallback
        attn_implementation = config.attn_implementation

        if attn_implementation == "auto":
            # Auto-select best available implementation
            if torch.cuda.is_available():
                compute_cap = torch.cuda.get_device_capability()[0]

                # Try flash_attention_2 for Ampere+ GPUs (compute capability >= 8)
                if compute_cap >= 8:
                    try:
                        import flash_attn
                        attn_implementation = "flash_attention_2"
                        logger.info("Auto-selected flash_attention_2 (Ampere+ GPU detected)")
                    except ImportError:
                        logger.warning("flash_attn not available, trying sdpa")
                        attn_implementation = "sdpa"
                else:
                    # For older GPUs, use sdpa or eager
                    attn_implementation = "sdpa"
                    logger.info(f"Auto-selected sdpa (GPU compute capability {compute_cap} < 8)")
            else:
                # CPU fallback
                attn_implementation = "eager"
                logger.info("Auto-selected eager (CPU mode)")
        else:
            # User explicitly selected an implementation
            logger.info(f"Using user-selected attention: {attn_implementation}")

        logger.info(f"Using {torch_dtype} with {attn_implementation}")

        # Setup quantization
        bnb_config = None
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # Load model and tokenizer
        job_manager.update_job(job_id, status="loading_model", progress=0.1)

        # Determine if loading from local path or HuggingFace
        is_local = is_local_model_path(config.base_model)
        model_source = "local directory" if is_local else "HuggingFace Hub"
        logger.info(f"Loading model from {model_source}: {config.base_model}")

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="loading_model",
                progress=0.1,
                message=f"Loading model from {model_source}"
            ),
            event_loop
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_type = getattr(config, 'model_type', 'auto')
        auto_model_cls, is_vlm = _get_auto_model_class(config.base_model, model_type)

        model = auto_model_cls.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype if not bnb_config else None,
            device_map="auto",
            trust_remote_code=True,
        )

        # Note: prepare_model_for_kbit_training is handled by GrimoireTrainer
        # when peft_config is provided, so we don't call it here.

        # Setup LoRA (only if enabled)
        peft_config = None
        if config.use_lora:
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=config.target_modules,
                modules_to_save=config.modules_to_save if config.modules_to_save else None
            )
            logger.info(f"LoRA enabled with rank={config.lora_r}, alpha={config.lora_alpha}")
        else:
            logger.info("LoRA disabled - training full model")

        # Load and prepare dataset
        job_manager.update_job(job_id, status="loading_dataset", progress=0.2)

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="loading_dataset",
                progress=0.2
            ),
            event_loop
        )

        logger.info(f"Loading dataset with config: {config.dataset.source.source_type}")

        # Create primary loader using factory
        loader = create_loader_from_config(
            source_config=config.dataset.source,
            uploaded_datasets=uploaded_datasets,
            hf_token=config.hf_token
        )

        # Create additional loaders if configured
        additional_loaders = []
        if config.dataset.additional_sources:
            logger.info(f"Creating loaders for {len(config.dataset.additional_sources)} additional dataset(s)")
            for extra_source in config.dataset.additional_sources:
                extra_loader = create_loader_from_config(
                    source_config=extra_source,
                    uploaded_datasets=uploaded_datasets,
                    hf_token=config.hf_token
                )
                extra_mapping = extra_source.column_mapping if hasattr(extra_source, 'column_mapping') else None
                extra_convert = config.dataset.convert_messages_format
                additional_loaders.append((extra_loader, extra_mapping, extra_convert))

        # Get formatter
        formatter = get_formatter(
            format_type=config.dataset.format.format_type,
            custom_templates=config.dataset.format.custom_templates,
            tokenizer=tokenizer if config.dataset.format.format_type == 'tokenizer' else None
        )

        # Create pipeline and prepare dataset
        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            column_mapping=config.dataset.column_mapping,
            test_size=config.dataset.test_size,
            max_samples=config.dataset.max_samples,
            seed=config.seed,
            shuffle=config.shuffle_dataset,
            training_mode=config.training_mode,
            convert_messages_format=config.dataset.convert_messages_format,
            additional_loaders=additional_loaders if additional_loaders else None,
            deduplicate=config.dataset.deduplicate,
            dedupe_strategy=config.dataset.dedupe_strategy,
        )

        train_dataset, eval_dataset = pipeline.prepare()

        # Setup training arguments
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

        # Choose training mode and tokenize dataset accordingly
        training_mode = config.training_mode.lower()
        logger.info(f"Using training mode: {training_mode}")

        # Preference-based methods that all use tokenize_preference
        PREFERENCE_MODES = {"orpo", "dpo", "simpo", "cpo", "ipo"}

        if training_mode == "sft":
            logger.info("Configuring SFT (Supervised Fine-Tuning)")
            loss_fn = SFTLoss()

            # Tokenize: concatenate prompt+chosen, mask prompt tokens in labels
            train_dataset = train_dataset.map(
                lambda x: tokenize_sft(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                    prompt_field="prompt", response_field="chosen",
                ),
                remove_columns=train_dataset.column_names,
            )
            eval_dataset = eval_dataset.map(
                lambda x: tokenize_sft(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                    prompt_field="prompt", response_field="chosen",
                ),
                remove_columns=eval_dataset.column_names,
            )
        elif training_mode == "kto":
            logger.info("Configuring KTO (Kahneman-Tversky Optimization)")
            loss_fn = KTOLoss(beta=config.beta)

            # KTO uses unpaired binary feedback: each example has a response + label (True/False)
            # Split chosen/rejected pairs into separate examples with binary labels
            from datasets import Dataset as HFDataset, concatenate_datasets

            def _split_to_kto(dataset):
                """Split paired chosen/rejected into unpaired KTO examples."""
                positive_rows = []
                negative_rows = []

                for row in dataset:
                    prompt = row["prompt"]
                    # Chosen → positive example
                    positive_rows.append({
                        "prompt": prompt,
                        "response": row["chosen"],
                        "label": True,
                    })
                    # Rejected → negative example (if present and non-empty)
                    rejected = row.get("rejected", "")
                    if rejected and str(rejected).strip():
                        negative_rows.append({
                            "prompt": prompt,
                            "response": rejected,
                            "label": False,
                        })

                parts = [HFDataset.from_list(positive_rows)]
                if negative_rows:
                    parts.append(HFDataset.from_list(negative_rows))
                    logger.info(f"KTO split: {len(positive_rows)} positive + {len(negative_rows)} negative examples")
                else:
                    logger.info(f"KTO: {len(positive_rows)} positive examples (no rejected responses)")

                return concatenate_datasets(parts).shuffle(seed=config.seed)

            train_dataset = _split_to_kto(train_dataset)
            eval_dataset = _split_to_kto(eval_dataset)

            # Tokenize: prompt+response with binary label
            train_dataset = train_dataset.map(
                lambda x: tokenize_kto(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                    prompt_field="prompt", response_field="response",
                    label_field="label",
                ),
                remove_columns=train_dataset.column_names,
            )
            eval_dataset = eval_dataset.map(
                lambda x: tokenize_kto(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                    prompt_field="prompt", response_field="response",
                    label_field="label",
                ),
                remove_columns=eval_dataset.column_names,
            )
        elif training_mode in PREFERENCE_MODES:
            # Create appropriate loss function for each preference method
            if training_mode == "orpo":
                logger.info("Configuring ORPO (Odds Ratio Preference Optimization)")
                loss_fn = ORPOLoss(beta=config.beta)
            elif training_mode == "dpo":
                logger.info("Configuring DPO (Direct Preference Optimization)")
                loss_fn = DPOLoss(beta=config.beta, label_smoothing=config.label_smoothing)
            elif training_mode == "simpo":
                logger.info("Configuring SimPO (Simple Preference Optimization)")
                loss_fn = SimPOLoss(beta=config.beta, gamma=config.gamma)
            elif training_mode == "cpo":
                logger.info("Configuring CPO (Contrastive Preference Optimization)")
                loss_fn = CPOLoss(beta=config.beta, label_smoothing=config.label_smoothing)
            elif training_mode == "ipo":
                logger.info("Configuring IPO (Identity Preference Optimization)")
                loss_fn = IPOLoss(beta=config.beta)

            # Tokenize: chosen and rejected pairs with prompt masking
            train_dataset = train_dataset.map(
                lambda x: tokenize_preference(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                ),
                remove_columns=train_dataset.column_names,
            )
            eval_dataset = eval_dataset.map(
                lambda x: tokenize_preference(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                ),
                remove_columns=eval_dataset.column_names,
            )
        else:
            raise ValueError(f"Unknown training mode: {training_mode}. Supported: sft, orpo, dpo, simpo, cpo, ipo, kto")

        # Determine mixed precision setting
        if torch_dtype == torch.bfloat16:
            mixed_precision = "bf16"
        elif torch_dtype == torch.float16:
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"

        # Convert eval_steps: <1 means ratio (e.g. 0.2 = every 20%), >=1 means absolute steps
        eval_steps = None
        if config.eval_steps:
            if config.eval_steps < 1:
                import math
                num_gpus = get_num_gpus()
                effective_batch = config.batch_size * config.gradient_accumulation_steps * num_gpus
                steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
                total_steps = steps_per_epoch * config.num_epochs
                eval_steps = max(1, int(total_steps * config.eval_steps))
                logger.info(
                    f"Eval steps: {eval_steps} (ratio={config.eval_steps}, "
                    f"dataset={len(train_dataset)}, batch={config.batch_size}, "
                    f"grad_accum={config.gradient_accumulation_steps}, "
                    f"num_gpus={num_gpus}, "
                    f"epochs={config.num_epochs}, total_steps={total_steps})"
                )
            else:
                eval_steps = int(config.eval_steps)

        grimoire_config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            max_length=config.max_length,
            mixed_precision=mixed_precision,
            gradient_checkpointing=config.gradient_checkpointing,
            optimizer=config.optimizer_type,
            lr_scheduler=config.lr_scheduler_type,
            disable_dropout=(training_mode in PREFERENCE_MODES or training_mode == "kto"),
            logging_steps=config.logging_steps,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            save_total_limit=2,
            seed=config.seed,
            run_name=wandb_run_name if config.use_wandb else config.output_name,
            log_with="wandb" if config.use_wandb else None,
            project_name=wandb_project,
            wandb_tags=config.wandb_tags or [],
            wandb_notes=config.wandb_notes,
        )

        trainer = GrimoireTrainer(
            model=model,
            tokenizer=tokenizer,
            config=grimoire_config,
            loss_fn=loss_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            callbacks=[WebSocketCallback(job_id, job_manager, event_loop)],
        )

        # Capture W&B run URL after trainer init (accelerator owns the wandb run)
        if config.use_wandb and wandb.run is not None:
            wandb_url = wandb.run.get_url()
            logger.info(f"W&B run URL: {wandb_url}")
            job_manager.update_job(job_id, wandb_url=wandb_url)

        # Train
        logger.info("Starting training")
        trainer.train()

        # Check if training was stopped early
        was_stopped = trainer.stopped_early

        # Save model
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

        # If the model doesn't have a chat template and we used a known format,
        # embed that format's chat template into the tokenizer so the saved/uploaded
        # model carries it for inference.
        format_type = config.dataset.format.format_type
        has_existing_template = (
            hasattr(tokenizer, 'chat_template')
            and tokenizer.chat_template is not None
        )
        if not has_existing_template and format_type != 'tokenizer':
            chat_template = get_chat_template_for_format(format_type)
            if chat_template:
                tokenizer.chat_template = chat_template
                logger.info(
                    f"Applied '{format_type}' chat template to tokenizer "
                    f"(base model had no chat template)"
                )

        final_output_dir = f"./models/{config.output_name}"
        trainer.save_model(final_output_dir)

        # Capture step info before cleanup
        final_step = trainer.global_step
        final_max_steps = trainer.max_steps

        # Clean up trainer and model to free VRAM before potential merge/upload
        # This prevents OOM when loading the model again for merging
        logger.info("🧹 Cleaning up training resources to free VRAM...")
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("✅ VRAM freed successfully")

        # Optional: merge and upload (runs in background thread to not block queue)
        upload_thread_started = False
        if config.push_to_hub:
            # Check if we should upload (main process in distributed, or always in single GPU)
            is_distributed = torch.distributed.is_initialized()
            should_upload = (not is_distributed) or (torch.distributed.get_rank() == 0)

            if should_upload:
                # Validate HF token is provided
                if not config.hf_token:
                    logger.warning("⚠️ push_to_hub enabled but no HF token provided - skipping upload")
                    logger.info("💡 Provide HF_TOKEN in .env or via hf_token parameter to enable uploads")
                else:
                    # Start upload in a background thread
                    # This allows the job queue worker to continue processing other jobs
                    # while the upload (which can take a long time) runs in the background
                    # Note: Using daemon=False so uploads can complete even during shutdown
                    # The thread handles its own error cases gracefully
                    logger.info("📤 Starting background upload thread...")
                    upload_thread = threading.Thread(
                        target=_run_background_upload,
                        args=(config, final_output_dir, training_mode, job_id, job_manager, event_loop),
                        name=f"UploadThread-{job_id}",
                        daemon=False  # Non-daemon to allow upload completion
                    )
                    upload_thread.start()
                    upload_thread_started = True
                    logger.info(f"📤 Background upload thread started for job {job_id}")
            else:
                logger.info("⏭️ Skipping HuggingFace upload (not main process in distributed training)")

        # Mark as completed or stopped (unless upload thread is running - it will handle final status)
        if upload_thread_started:
            # Upload thread is running in background - it will mark job as completed when done
            # For now, mark as "uploading" so the UI shows correct status
            logger.info(f"📤 Training finished, upload running in background for job {job_id}")
            # Note: _run_background_upload already sets status to "uploading"
        else:
            # No upload - mark as completed/stopped immediately
            final_status = "stopped" if was_stopped else "completed"
            final_progress = 1.0 if not was_stopped else (final_step / final_max_steps if final_max_steps else 0.9)

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
                logger.info(f"Training completed for job {job_id}")

        # Finish wandb run to mark it as complete
        if config.use_wandb and wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished successfully")

    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {str(e)}", exc_info=True)
        job_manager.update_job(job_id, status="failed", error=str(e))

        send_websocket_update(
            websocket_manager.send_error(
                job_id=job_id,
                error=str(e)
            ),
            event_loop
        )

        # Finish wandb run even on failure to mark it as crashed
        try:
            if config.use_wandb and wandb.run is not None:
                wandb.finish(exit_code=1)
                logger.info("W&B run finished (marked as failed)")
        except Exception as wandb_error:
            logger.debug(f"Could not finish W&B run on failure: {wandb_error}")

    finally:
        # Ensure GPU memory is freed even on error
        # This prevents OOM errors in subsequent training jobs
        _cleanup_training_resources(model, trainer)
