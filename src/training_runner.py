"""
Enhanced Training Runner with WebSocket updates and better error handling
"""

import os
import gc
import torch
import wandb
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import ORPOConfig, ORPOTrainer
from accelerate import Accelerator

from dataset_handlers import (
    DatasetPipeline,
    HuggingFaceLoader,
    LocalFileLoader,
    UploadedDatasetLoader,
    get_formatter
)
from src.job_manager import JobManager
from src.websocket_manager import websocket_manager
from src.preflight_checks import is_local_model_path

logger = logging.getLogger(__name__)


def generate_wandb_run_name(config: Any) -> str:
    """
    Generate a descriptive W&B run name from training configuration.

    Format: [model_name]-[lr]-[batch]-[epochs]ep-[optimizer]-[attention]
    Example: llama3-8b-5e-6LR-256B-2ep-adamw8bit-flash2

    Args:
        config: Training configuration object

    Returns:
        Generated run name string
    """
    # Extract model name from path (e.g., "meta-llama/Llama-3-8B" -> "llama3-8b")
    model_name = config.base_model.split('/')[-1].lower()
    # Simplify common model names
    model_name = model_name.replace('-instruct', '').replace('-base', '').replace('meta-', '')

    # Format learning rate (e.g., 0.000005 -> "5e-6")
    lr = config.learning_rate
    if lr >= 1e-3:
        lr_str = f"{lr:.0e}".replace('e-0', 'e-')
    else:
        lr_str = f"{lr:.0e}".replace('e-0', 'e-')

    # Calculate effective batch size
    effective_batch = config.batch_size * config.gradient_accumulation_steps

    # Simplify optimizer name
    opt = config.optimizer_type.replace('paged_', '').replace('_', '')

    # Simplify attention
    attn_map = {
        'flash_attention_2': 'flash2',
        'sdpa': 'sdpa',
        'eager': 'eager',
        'auto': 'auto'
    }
    attn = attn_map.get(config.attn_implementation, config.attn_implementation)

    # Build run name
    parts = [
        model_name,
        f"{lr_str}LR",
        f"{effective_batch}B",
        f"{config.num_epochs}ep",
        opt,
        attn
    ]

    run_name = "-".join(parts)

    # Add optional suffix for special settings
    suffixes = []
    if config.use_4bit:
        suffixes.append("4bit")
    if config.gradient_checkpointing:
        suffixes.append("gc")
    if config.beta != 0.1:  # Non-default ORPO beta
        suffixes.append(f"beta{config.beta}")

    if suffixes:
        run_name += f"-{'-'.join(suffixes)}"

    return run_name


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
    Custom callback to send training metrics via WebSocket
    """

    def __init__(self, job_id: str, job_manager: JobManager, event_loop=None):
        self.job_id = job_id
        self.job_manager = job_manager
        self.event_loop = event_loop

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs:
            # Update job with latest metrics
            update_data = {}

            # Initialize gpu_memory at the start to avoid UnboundLocalError
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)

            if "loss" in logs:
                update_data["loss"] = float(logs["loss"])

            if "eval_loss" in logs:
                update_data["eval_loss"] = float(logs["eval_loss"])

            if "learning_rate" in logs:
                update_data["learning_rate"] = float(logs["learning_rate"])

            if state.global_step:
                update_data["current_step"] = state.global_step

            if state.max_steps:
                update_data["total_steps"] = state.max_steps
                # Calculate progress based on steps
                progress = 0.3 + (0.6 * (state.global_step / state.max_steps))
                update_data["progress"] = min(progress, 0.9)

            # Update database
            if update_data:
                self.job_manager.update_job(self.job_id, **update_data)

                # Add to metrics table
                if state.global_step and "loss" in logs:
                    self.job_manager.add_metric(
                        self.job_id,
                        step=state.global_step,
                        loss=update_data.get("loss"),
                        eval_loss=update_data.get("eval_loss"),
                        learning_rate=update_data.get("learning_rate"),
                        gpu_memory_used=gpu_memory
                    )

                # Send WebSocket update (run in event loop)
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


async def run_training_async(job_id: str, config: Any, job_manager: JobManager, uploaded_datasets: dict):
    """Async wrapper for training to enable WebSocket updates"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_training_sync, job_id, config, job_manager, uploaded_datasets)


def run_training_sync(job_id: str, config: Any, job_manager: JobManager, uploaded_datasets: dict, event_loop=None):
    """
    Run ORPO training job with enhanced monitoring.

    Args:
        job_id: Unique job identifier
        config: Training configuration
        job_manager: Job manager instance
        uploaded_datasets: Dictionary of uploaded datasets
        event_loop: Event loop for WebSocket updates (optional)
    """
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

        # Setup accelerator
        accelerator = Accelerator()

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

            # Initialize wandb with project, name, and tags
            wandb_config = {
                "model": config.base_model,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
                "epochs": config.num_epochs,
                "optimizer": config.optimizer_type,
                "attention": config.attn_implementation,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "beta": config.beta,
                "seed": config.seed
            }

            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                tags=config.wandb_tags or [],
                notes=config.wandb_notes,
                config=wandb_config
            )

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

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype if not bnb_config else None,
            device_map="auto",
            trust_remote_code=True,
        )

        if bnb_config:
            model = prepare_model_for_kbit_training(model)

        # Setup LoRA
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.target_modules
        )

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

        # Create appropriate loader based on source type
        if config.dataset.source.source_type == "huggingface":
            loader = HuggingFaceLoader(
                repo_id=config.dataset.source.repo_id,
                split=config.dataset.source.split,
                token=config.hf_token
            )
        elif config.dataset.source.source_type == "local_file":
            loader = LocalFileLoader(
                file_path=config.dataset.source.file_path,
                file_format=config.dataset.source.file_format
            )
        elif config.dataset.source.source_type == "upload":
            dataset_id = config.dataset.source.dataset_id
            if dataset_id not in uploaded_datasets:
                raise ValueError(f"Uploaded dataset '{dataset_id}' not found")

            file_content, filename = uploaded_datasets[dataset_id]
            loader = UploadedDatasetLoader(
                file_content=file_content,
                filename=filename,
                file_format=config.dataset.source.file_format
            )
        else:
            raise ValueError(f"Invalid source_type: {config.dataset.source.source_type}")

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
            seed=config.seed,  # Use config seed instead of hardcoded 42
            shuffle=config.shuffle_dataset  # Use config shuffle setting
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

        orpo_args = ORPOConfig(
            run_name=wandb_run_name if config.use_wandb else config.output_name,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,  # Use config scheduler type
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_length - config.max_prompt_length,
            beta=config.beta,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim=config.optimizer_type,  # Use config optimizer type
            num_train_epochs=config.num_epochs,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            logging_steps=config.logging_steps,  # Use config logging steps
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,  # Use config max grad norm
            weight_decay=config.weight_decay,  # Use config weight decay
            adam_beta1=config.adam_beta1,  # Use config adam beta1
            adam_beta2=config.adam_beta2,  # Use config adam beta2
            adam_epsilon=config.adam_epsilon,  # Use config adam epsilon
            report_to=["wandb"] if config.use_wandb else [],
            output_dir=output_dir,
            bf16=torch_dtype == torch.bfloat16,
            fp16=torch_dtype == torch.float16,
            save_strategy="steps",
            save_steps=config.eval_steps,
            save_total_limit=2,
            seed=config.seed,  # Set seed for training
            gradient_checkpointing=config.gradient_checkpointing,  # Enable gradient checkpointing if requested
        )

        # Create trainer with WebSocket callback
        # Note: ORPOTrainer in newer TRL versions requires processing_class (tokenizer)
        trainer = ORPOTrainer(
            model=model,
            args=orpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            callbacks=[WebSocketCallback(job_id, job_manager, event_loop)]
        )

        # Train
        logger.info("Starting training")
        train_result = trainer.train()

        # Save model
        job_manager.update_job(job_id, status="saving", progress=0.9)

        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="saving",
                progress=0.9
            ),
            event_loop
        )

        final_output_dir = f"./models/{config.output_name}"
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        # Optional: merge and upload
        if config.push_to_hub and accelerator.is_main_process:
            job_manager.update_job(job_id, status="uploading")

            send_websocket_update(
                websocket_manager.send_status_update(
                    job_id=job_id,
                    status="uploading",
                    progress=0.95
                ),
                event_loop
            )

            # Reload for merging
            base_model_reload = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            model_merged = PeftModel.from_pretrained(base_model_reload, final_output_dir)
            model_merged = model_merged.merge_and_unload()

            # Push to hub
            logger.info(f"Pushing to HuggingFace Hub (private={config.hf_hub_private})")
            model_merged.push_to_hub(
                config.output_name,
                token=config.hf_token,
                private=config.hf_hub_private
            )
            tokenizer.push_to_hub(
                config.output_name,
                token=config.hf_token,
                private=config.hf_hub_private
            )

        # Cleanup
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        # Mark as completed
        job_manager.update_job(
            job_id,
            status="completed",
            progress=1.0,
            output_dir=final_output_dir
        )

        send_websocket_update(
            websocket_manager.send_completion(
                job_id=job_id,
                output_dir=final_output_dir
            ),
            event_loop
        )

        logger.info(f"Training completed for job {job_id}")

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
