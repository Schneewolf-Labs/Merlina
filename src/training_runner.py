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
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from src.orpo_standalone import ORPOConfig, MerlinaORPOTrainer as ORPOTrainer
from trl import SFTTrainer, SFTConfig
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

    Format: [model_name]-[lora_r]-[lr]-[batch]-[epochs]ep-[optimizer]-[attention]
    Example: llama3-8b-r64-5e-6LR-256B-2ep-adamw8bit-flash2
    Example (no LoRA): llama3-8b-full-5e-6LR-256B-2ep-adamw8bit-flash2

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

    # Build run name - include LoRA rank or "full" if not using LoRA
    parts = [
        model_name,
        f"r{config.lora_r}" if config.use_lora else "full",
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
    # Only add beta for ORPO mode (SFT doesn't use beta)
    if config.beta != 0.1 and config.training_mode == "orpo":
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
    Custom callback to send training metrics via WebSocket and handle stop requests
    """

    def __init__(self, job_id: str, job_manager: JobManager, event_loop=None):
        self.job_id = job_id
        self.job_manager = job_manager
        self.event_loop = event_loop

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step to check for stop requests"""
        # Check if stop was requested
        job = self.job_manager.get_job(self.job_id)
        if job and job.stop_requested:
            logger.info(f"Stop requested for job {self.job_id} - gracefully stopping training")
            control.should_training_stop = True

            # Send WebSocket notification
            send_websocket_update(
                websocket_manager.send_status_update(
                    job_id=self.job_id,
                    status="stopping",
                    progress=state.global_step / state.max_steps if state.max_steps else 0.5,
                    current_step=state.global_step,
                    total_steps=state.max_steps,
                    message="Stop requested - saving checkpoint..."
                ),
                self.event_loop
            )

        return control

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

        # Set visible GPUs if specified
        if config.gpu_ids is not None:
            from src.gpu_utils import get_gpu_manager
            gpu_manager = get_gpu_manager()
            gpu_manager.set_visible_devices(config.gpu_ids)
            logger.info(f"Using GPUs: {config.gpu_ids}")

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
                "use_lora": config.use_lora,
                "lora_r": config.lora_r if config.use_lora else None,
                "lora_alpha": config.lora_alpha if config.use_lora else None,
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

            # Capture W&B run URL and save to job
            if wandb.run is not None:
                wandb_url = wandb.run.get_url()
                logger.info(f"W&B run URL: {wandb_url}")
                job_manager.update_job(job_id, wandb_url=wandb_url)

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

        # Setup LoRA (only if enabled)
        peft_config = None
        if config.use_lora:
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=config.target_modules
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

        # Choose trainer based on training mode
        training_mode = config.training_mode.lower()
        logger.info(f"Using training mode: {training_mode}")

        if training_mode == "sft":
            # SFT Configuration
            logger.info("Configuring SFT (Supervised Fine-Tuning) trainer")

            # For SFT, we need to format the dataset as text
            # Convert the dataset to have a 'text' field with prompt + chosen
            def format_for_sft(example):
                """Format dataset entry for SFT training"""
                return {
                    'text': example['prompt'] + example['chosen']
                }

            train_dataset = train_dataset.map(format_for_sft, remove_columns=train_dataset.column_names)
            eval_dataset = eval_dataset.map(format_for_sft, remove_columns=eval_dataset.column_names)

            sft_args = SFTConfig(
                run_name=wandb_run_name if config.use_wandb else config.output_name,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                optim=config.optimizer_type,
                num_train_epochs=config.num_epochs,
                eval_strategy="steps",
                eval_steps=config.eval_steps,
                logging_steps=config.logging_steps,
                warmup_ratio=config.warmup_ratio,
                max_grad_norm=config.max_grad_norm,
                weight_decay=config.weight_decay,
                adam_beta1=config.adam_beta1,
                adam_beta2=config.adam_beta2,
                adam_epsilon=config.adam_epsilon,
                report_to=["wandb"] if config.use_wandb else [],
                output_dir=output_dir,
                bf16=torch_dtype == torch.bfloat16,
                fp16=torch_dtype == torch.float16,
                save_strategy="steps",
                save_steps=config.eval_steps,
                save_total_limit=2,
                seed=config.seed,
                gradient_checkpointing=config.gradient_checkpointing,
                # SFT-specific parameters (use max_length, not max_seq_length)
                max_length=config.max_length,
                dataset_text_field="text",
                packing=False,
            )

            # Create SFT trainer
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                processing_class=tokenizer,
                callbacks=[WebSocketCallback(job_id, job_manager, event_loop)]
            )
        else:
            # ORPO Configuration (default)
            logger.info("Configuring ORPO (Odds Ratio Preference Optimization) trainer")

            orpo_args = ORPOConfig(
                run_name=wandb_run_name if config.use_wandb else config.output_name,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                max_length=config.max_length,
                max_prompt_length=config.max_prompt_length,
                max_completion_length=config.max_length - config.max_prompt_length,
                beta=config.beta,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                optim=config.optimizer_type,
                num_train_epochs=config.num_epochs,
                eval_strategy="steps",
                eval_steps=config.eval_steps,
                logging_steps=config.logging_steps,
                warmup_ratio=config.warmup_ratio,
                max_grad_norm=config.max_grad_norm,
                weight_decay=config.weight_decay,
                adam_beta1=config.adam_beta1,
                adam_beta2=config.adam_beta2,
                adam_epsilon=config.adam_epsilon,
                report_to=["wandb"] if config.use_wandb else [],
                output_dir=output_dir,
                bf16=torch_dtype == torch.bfloat16,
                fp16=torch_dtype == torch.float16,
                save_strategy="steps",
                save_steps=config.eval_steps,
                save_total_limit=2,
                seed=config.seed,
                gradient_checkpointing=config.gradient_checkpointing,
            )

            # Create ORPO trainer
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

        # Check if training was stopped early
        job = job_manager.get_job(job_id)
        was_stopped = job and job.stop_requested

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

        final_output_dir = f"./models/{config.output_name}"
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        # Optional: merge and upload
        if config.push_to_hub:
            # Check if we should upload (main process in distributed, or always in single GPU)
            should_upload = accelerator.is_main_process if accelerator.num_processes > 1 else True

            if should_upload:
                try:
                    # Validate HF token is provided
                    if not config.hf_token:
                        logger.warning("⚠️ push_to_hub enabled but no HF token provided - skipping upload")
                        logger.info("💡 Provide HF_TOKEN in .env or via hf_token parameter to enable uploads")
                    else:
                        job_manager.update_job(job_id, status="uploading")

                        send_websocket_update(
                            websocket_manager.send_status_update(
                                job_id=job_id,
                                status="uploading",
                                progress=0.95
                            ),
                            event_loop
                        )

                        # Handle upload based on whether LoRA was used
                        if config.use_lora:
                            # Check if we should merge LoRA or upload adapter only
                            if config.merge_lora_before_upload:
                                logger.info(f"🔄 Merging LoRA adapter with base model for upload...")

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
                                repo_visibility = "private" if config.hf_hub_private else "public"
                                logger.info(f"📤 Pushing merged model to HuggingFace Hub as {repo_visibility} repository...")
                                logger.info(f"   Repository: {config.output_name}")

                                # Upload model
                                model_url = model_merged.push_to_hub(
                                    config.output_name,
                                    token=config.hf_token,
                                    private=config.hf_hub_private
                                )
                                logger.info(f"✅ Merged model uploaded successfully!")

                                # Upload tokenizer
                                tokenizer_url = tokenizer.push_to_hub(
                                    config.output_name,
                                    token=config.hf_token,
                                    private=config.hf_hub_private
                                )
                                logger.info(f"✅ Tokenizer uploaded successfully!")

                                # Construct the hub URL
                                hub_url = f"https://huggingface.co/{config.output_name}"
                                logger.info(f"🎉 Merged model published at: {hub_url}")

                                # Clean up merged model to free memory
                                del model_merged, base_model_reload
                                gc.collect()
                                torch.cuda.empty_cache()
                            else:
                                # Upload LoRA adapter only (much smaller!)
                                logger.info(f"📤 Uploading LoRA adapter only (not merged)...")

                                # Push to hub
                                repo_visibility = "private" if config.hf_hub_private else "public"
                                logger.info(f"📤 Pushing LoRA adapter to HuggingFace Hub as {repo_visibility} repository...")
                                logger.info(f"   Repository: {config.output_name}")
                                logger.info(f"   Base model: {config.base_model}")
                                logger.info(f"   LoRA rank: {config.lora_r}")

                                # Load the adapter model
                                adapter_model = AutoPeftModelForCausalLM.from_pretrained(
                                    final_output_dir,
                                    low_cpu_mem_usage=True,
                                    torch_dtype=torch.bfloat16,
                                    device_map="auto",
                                )

                                # Upload adapter
                                adapter_model.push_to_hub(
                                    config.output_name,
                                    token=config.hf_token,
                                    private=config.hf_hub_private
                                )
                                logger.info(f"✅ LoRA adapter uploaded successfully!")

                                # Upload tokenizer
                                tokenizer.push_to_hub(
                                    config.output_name,
                                    token=config.hf_token,
                                    private=config.hf_hub_private
                                )
                                logger.info(f"✅ Tokenizer uploaded successfully!")

                                # Construct the hub URL
                                hub_url = f"https://huggingface.co/{config.output_name}"
                                logger.info(f"🎉 LoRA adapter published at: {hub_url}")
                                logger.info(f"💡 To use: PeftModel.from_pretrained('{config.base_model}', '{config.output_name}')")

                                # Clean up
                                del adapter_model
                                gc.collect()
                                torch.cuda.empty_cache()
                        else:
                            # For full model training (no LoRA), just push the saved model directly
                            logger.info(f"📤 Uploading full model to HuggingFace Hub...")

                            # Reload model for upload
                            upload_model = AutoModelForCausalLM.from_pretrained(
                                final_output_dir,
                                low_cpu_mem_usage=True,
                                torch_dtype=torch.bfloat16,
                                device_map="auto",
                            )

                            # Push to hub
                            repo_visibility = "private" if config.hf_hub_private else "public"
                            logger.info(f"📤 Pushing to HuggingFace Hub as {repo_visibility} repository...")
                            logger.info(f"   Repository: {config.output_name}")

                            # Upload model
                            model_url = upload_model.push_to_hub(
                                config.output_name,
                                token=config.hf_token,
                                private=config.hf_hub_private
                            )
                            logger.info(f"✅ Model uploaded successfully!")

                            # Upload tokenizer
                            tokenizer_url = tokenizer.push_to_hub(
                                config.output_name,
                                token=config.hf_token,
                                private=config.hf_hub_private
                            )
                            logger.info(f"✅ Tokenizer uploaded successfully!")

                            # Construct the hub URL
                            hub_url = f"https://huggingface.co/{config.output_name}"
                            logger.info(f"🎉 Model published at: {hub_url}")

                            # Clean up
                            del upload_model
                            gc.collect()
                            torch.cuda.empty_cache()

                except Exception as upload_error:
                    # Log upload failure but don't fail the entire job
                    logger.error(f"❌ Failed to upload to HuggingFace Hub: {str(upload_error)}", exc_info=True)
                    logger.warning("⚠️ Training completed successfully, but upload failed")
                    logger.info(f"💾 Model saved locally at: {final_output_dir}")
                    # Continue to mark training as completed (upload is optional)
            else:
                logger.info("⏭️ Skipping HuggingFace upload (not main process in distributed training)")

        # Cleanup
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        # Mark as completed or stopped
        final_status = "stopped" if was_stopped else "completed"
        final_progress = 1.0 if not was_stopped else (job.current_step / job.total_steps if job.total_steps else 0.9)

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
            logger.info(f"Training stopped early for job {job_id} at step {job.current_step}/{job.total_steps}")
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
        except Exception:
            pass  # Ignore errors when finishing wandb on failure
