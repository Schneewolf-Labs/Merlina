#!/usr/bin/env python
"""
Standalone training worker for distributed (multi-GPU) training.

Launched via:
    accelerate launch --num_processes N --mixed_precision bf16 \
        src/train_worker.py --config-path /tmp/config.json --job-id job_123 \
        --progress-file /tmp/progress.jsonl --db-path ./data/jobs.db

This script is the entry point for each DDP process spawned by accelerate.
Only rank 0 writes progress updates to the progress file and SQLite DB.
"""

import argparse
import json
import os
import sys
import gc
import signal
import logging
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

try:
    import wandb
except ImportError:
    wandb = None

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVLM
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM
    except ImportError:
        AutoModelForVLM = None

from peft import LoraConfig
from grimoire import GrimoireTrainer, TrainingConfig as GrimoireTrainingConfig, TrainerCallback
from src.muon_support import MuonGrimoireTrainer
from grimoire.losses import SFTLoss, ORPOLoss, DPOLoss, SimPOLoss, CPOLoss, IPOLoss, KTOLoss
from grimoire.data import tokenize_sft, tokenize_preference, tokenize_kto

from dataset_handlers import (
    DatasetPipeline,
    HuggingFaceLoader,
    LocalFileLoader,
    UploadedDatasetLoader,
    get_formatter,
)
from dataset_handlers.formatters import get_chat_template_for_format
from dataset_handlers.factory import create_loader_from_config

from src.job_manager import JobManager
from src.model_card import generate_wandb_run_name

logger = logging.getLogger(__name__)

# VLM architecture detection patterns
_VLM_ARCH_PATTERNS = ['ForConditionalGeneration', 'ForVision', 'ImageText']

# Flag set by SIGTERM handler for graceful stop
_stop_requested = False


def _detect_is_vlm(model_name: str) -> bool:
    """Detect if a model is a VLM by checking its config for vision components."""
    try:
        auto_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(auto_config, 'vision_config') or hasattr(auto_config, 'visual'):
            return True
        architectures = getattr(auto_config, 'architectures', []) or []
        if architectures:
            arch = architectures[0]
            if any(p in arch for p in _VLM_ARCH_PATTERNS):
                return True
    except Exception as e:
        logger.debug(f"Could not auto-detect model type for {model_name}: {e}")
    return False


def _get_auto_model_class(model_name: str, model_type: str = "auto"):
    """Return the appropriate AutoModel class for loading."""
    if model_type == "vlm":
        is_vlm = True
    elif model_type == "causal_lm":
        is_vlm = False
    else:
        is_vlm = _detect_is_vlm(model_name)

    if is_vlm:
        if AutoModelForVLM is None:
            raise ImportError("No VLM auto-model class found in your transformers version.")
        logger.info("Model type: VLM — using %s", getattr(AutoModelForVLM, '__name__', 'AutoModelForVLM'))
        return AutoModelForVLM, True
    else:
        logger.info("Model type: Causal LM — using AutoModelForCausalLM")
        return AutoModelForCausalLM, False


def _cleanup_training_resources(model, trainer):
    """Clean up training resources to free GPU memory."""
    try:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _do_hub_upload(config, final_output_dir: str, training_mode: str, job_id: str, job_manager, was_stopped: bool = False):
    """Upload model to HuggingFace Hub (subprocess version, no WebSocket)."""
    from huggingface_hub import HfApi
    from peft import PeftModel
    from src.model_card import generate_model_readme, upload_model_readme

    api = HfApi()
    api.create_repo(
        repo_id=config.output_name,
        token=config.hf_token,
        private=config.hf_hub_private,
        exist_ok=True,
    )

    if config.use_lora and config.merge_lora_before_upload:
        try:
            logger.info("Merging LoRA adapter with base model on CPU...")
            auto_cls, _ = _get_auto_model_class(config.base_model, getattr(config, "model_type", "auto"))
            base = auto_cls.from_pretrained(
                config.base_model,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            merged = PeftModel.from_pretrained(base, final_output_dir, device_map="cpu")
            merged = merged.merge_and_unload()
            merged.push_to_hub(config.output_name, token=config.hf_token, private=config.hf_hub_private)

            tokenizer = AutoTokenizer.from_pretrained(final_output_dir)
            tokenizer.push_to_hub(config.output_name, token=config.hf_token, private=config.hf_hub_private)
            del merged, base
            gc.collect()
        except Exception as e:
            logger.warning(f"CPU merge failed ({e}), uploading adapter only")
            api.upload_folder(
                folder_path=final_output_dir,
                repo_id=config.output_name,
                token=config.hf_token,
                commit_message=f"Upload LoRA adapter ({training_mode})",
            )
    else:
        api.upload_folder(
            folder_path=final_output_dir,
            repo_id=config.output_name,
            token=config.hf_token,
            commit_message=f"Upload model ({training_mode})",
        )

    readme_content = generate_model_readme(config, training_mode)
    upload_model_readme(config.output_name, readme_content, config.hf_token)

    logger.info(f"Model published at: https://huggingface.co/{config.output_name}")
    if job_manager:
        final_status = "stopped" if was_stopped else "completed"
        job_manager.update_job(job_id, status=final_status, progress=1.0, output_dir=final_output_dir)


def _sigterm_handler(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    global _stop_requested
    _stop_requested = True
    logger.info("SIGTERM received — will stop after current step")


class FileProgressCallback(TrainerCallback):
    """
    Callback that writes training progress to a JSONL file.

    Only rank 0 should instantiate this callback so that a single
    process writes to the shared progress file.
    """

    def __init__(self, job_id: str, progress_file: str, job_manager: JobManager = None):
        self.job_id = job_id
        self.progress_file = progress_file
        self.job_manager = job_manager

    def _write(self, record: dict):
        """Append a JSON line to the progress file."""
        try:
            with open(self.progress_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write progress: {e}")

    def on_step_end(self, trainer, step, loss, metrics):
        """Check for stop requests."""
        global _stop_requested
        if _stop_requested:
            logger.info(f"Stop requested for job {self.job_id} — stopping training")
            trainer.request_stop()
            self._write({
                "type": "status",
                "status": "stopping",
                "message": "Stop requested — saving checkpoint...",
            })

        # Also check DB for stop_requested (in case parent updated it)
        if self.job_manager:
            job = self.job_manager.get_job(self.job_id)
            if job and job.stop_requested:
                logger.info(f"Stop requested (via DB) for job {self.job_id}")
                trainer.request_stop()

    def on_log(self, trainer, metrics):
        """Write training metrics."""
        record = {"type": "metrics"}

        if "train/loss" in metrics:
            record["loss"] = float(metrics["train/loss"])
        if "train/learning_rate" in metrics:
            record["learning_rate"] = float(metrics["train/learning_rate"])
        if trainer.global_step:
            record["current_step"] = trainer.global_step
        if trainer.max_steps:
            record["total_steps"] = trainer.max_steps
            record["progress"] = min(0.3 + 0.6 * (trainer.global_step / trainer.max_steps), 0.9)

        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            record["gpu_memory"] = gpu_memory

        self._write(record)

        # Also update DB
        if self.job_manager and trainer.global_step:
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
            if update_data:
                self.job_manager.update_job(self.job_id, **update_data)
            if "loss" in record:
                self.job_manager.add_metric(
                    self.job_id,
                    step=trainer.global_step,
                    loss=record.get("loss"),
                    learning_rate=record.get("learning_rate"),
                    gpu_memory_used=gpu_memory,
                )

    def on_evaluate(self, trainer, metrics):
        """Write eval metrics."""
        record = {"type": "eval"}
        if "eval/loss" in metrics:
            record["eval_loss"] = float(metrics["eval/loss"])
            self._write(record)
            if self.job_manager:
                self.job_manager.update_job(self.job_id, eval_loss=record["eval_loss"])


def run_worker(args):
    """Main training logic executed by each DDP process."""
    # Install SIGTERM handler
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Determine rank early (set by accelerate launch)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    is_main = global_rank == 0

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load config — wrap in try/except so failures are reported to the
    # progress file (the parent monitor relies on it).
    try:
        with open(args.config_path, "r") as f:
            config_dict = json.load(f)

        # Import TrainingConfig from merlina. This triggers module-level
        # initialization (FastAPI app, DB, etc.) which is harmless in the
        # subprocess context. An alternative is a shared models module, but
        # this keeps the Pydantic model as the single source of truth.
        from merlina import TrainingConfig
        from pydantic import TypeAdapter
        config = TypeAdapter(TrainingConfig).validate_python(config_dict)
    except Exception as e:
        logger.error(f"Failed to load training config: {e}", exc_info=True)
        if is_main:
            with open(args.progress_file, "a") as f:
                f.write(json.dumps({"type": "error", "error": f"Config load failed: {e}"}) + "\n")
            try:
                jm = JobManager(db_path=args.db_path)
                jm.update_job(args.job_id, status="failed", error=str(e))
            except Exception:
                pass
        return

    logger.info(f"Worker started: rank={global_rank}, local_rank={local_rank}")

    # Only rank 0 updates the DB and progress file
    job_manager = None
    if is_main and args.db_path:
        job_manager = JobManager(db_path=args.db_path)
        job_manager.update_job(args.job_id, status="initializing", progress=0.0)

    # Load uploaded datasets from temp directory (if any)
    uploaded_datasets = {}
    if args.uploaded_datasets_dir and os.path.isdir(args.uploaded_datasets_dir):
        meta_path = os.path.join(args.uploaded_datasets_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for dataset_id, info in meta.items():
                file_path = os.path.join(args.uploaded_datasets_dir, info["filename"])
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        content = f.read()
                    uploaded_datasets[dataset_id] = {
                        "content": content,
                        "filename": info["filename"],
                        "size": len(content),
                    }

    model = None
    trainer = None

    try:
        # ---- Model loading ----
        if is_main and job_manager:
            job_manager.update_job(args.job_id, status="loading_model", progress=0.1)

        if config.hf_token:
            os.environ["HF_TOKEN"] = config.hf_token

        # Determine dtype
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # Attention implementation
        attn_implementation = config.attn_implementation
        if attn_implementation == "auto":
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability()
                if cc[0] >= 8:
                    try:
                        import flash_attn  # noqa: F401
                        attn_implementation = "flash_attention_2"
                    except ImportError:
                        attn_implementation = "sdpa"
                else:
                    attn_implementation = "sdpa"
            else:
                attn_implementation = "eager"
        logger.info(f"Attention implementation: {attn_implementation}")

        # Quantization config
        bnb_config = None
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_type = getattr(config, "model_type", "auto")
        auto_model_cls, is_vlm = _get_auto_model_class(config.base_model, model_type)

        # KEY FIX: In DDP mode, each process loads the full model on its own GPU.
        # device_map="auto" would spread across GPUs, breaking DDP.
        device_map = {"": local_rank}
        logger.info(f"Loading model on device cuda:{local_rank} (DDP mode)")

        model = auto_model_cls.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype if not bnb_config else None,
            device_map=device_map,
            trust_remote_code=True,
        )

        # LoRA setup
        peft_config = None
        if config.use_lora:
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=config.target_modules or None,
                modules_to_save=config.modules_to_save or None,
            )

        # ---- Dataset loading ----
        if is_main and job_manager:
            job_manager.update_job(args.job_id, status="loading_dataset", progress=0.2)

        loader = create_loader_from_config(
            source_config=config.dataset.source,
            uploaded_datasets=uploaded_datasets,
            hf_token=config.hf_token,
        )

        additional_loaders = []
        if config.dataset.additional_sources:
            for extra_source in config.dataset.additional_sources:
                extra_loader = create_loader_from_config(
                    source_config=extra_source,
                    uploaded_datasets=uploaded_datasets,
                    hf_token=config.hf_token,
                )
                extra_mapping = extra_source.column_mapping if hasattr(extra_source, "column_mapping") else None
                extra_convert = config.dataset.convert_messages_format
                additional_loaders.append((extra_loader, extra_mapping, extra_convert))

        formatter = get_formatter(
            format_type=config.dataset.format.format_type,
            custom_templates=config.dataset.format.custom_templates,
            tokenizer=tokenizer if config.dataset.format.format_type == "tokenizer" else None,
        )

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

        if is_main and job_manager:
            job_manager.update_job(args.job_id, status="training", progress=0.3)

        # ---- Tokenize based on training mode ----
        training_mode = config.training_mode.lower()
        PREFERENCE_MODES = {"orpo", "dpo", "simpo", "cpo", "ipo"}

        if training_mode == "sft":
            loss_fn = SFTLoss()
            train_dataset = train_dataset.map(
                lambda x: tokenize_sft(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                    prompt_field="prompt", response_field="chosen",
                ),
                remove_columns=train_dataset.column_names,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    lambda x: tokenize_sft(
                        x, tokenizer, max_length=config.max_length,
                        max_prompt_length=config.max_prompt_length,
                        prompt_field="prompt", response_field="chosen",
                    ),
                    remove_columns=eval_dataset.column_names,
                )
        elif training_mode == "kto":
            loss_fn = KTOLoss(beta=config.beta)
            from datasets import Dataset as HFDataset, concatenate_datasets

            def _split_to_kto(dataset):
                positive_rows, negative_rows = [], []
                for row in dataset:
                    positive_rows.append({"prompt": row["prompt"], "response": row["chosen"], "label": True})
                    rejected = row.get("rejected", "")
                    if rejected and str(rejected).strip():
                        negative_rows.append({"prompt": row["prompt"], "response": rejected, "label": False})
                parts = [HFDataset.from_list(positive_rows)]
                if negative_rows:
                    parts.append(HFDataset.from_list(negative_rows))
                return concatenate_datasets(parts).shuffle(seed=config.seed)

            train_dataset = _split_to_kto(train_dataset)
            if eval_dataset is not None:
                eval_dataset = _split_to_kto(eval_dataset)
            train_dataset = train_dataset.map(
                lambda x: tokenize_kto(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                    prompt_field="prompt", response_field="response", label_field="label",
                ),
                remove_columns=train_dataset.column_names,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    lambda x: tokenize_kto(
                        x, tokenizer, max_length=config.max_length,
                        max_prompt_length=config.max_prompt_length,
                        prompt_field="prompt", response_field="response", label_field="label",
                    ),
                    remove_columns=eval_dataset.column_names,
                )
        elif training_mode in PREFERENCE_MODES:
            if training_mode == "orpo":
                loss_fn = ORPOLoss(beta=config.beta)
            elif training_mode == "dpo":
                loss_fn = DPOLoss(beta=config.beta, label_smoothing=config.label_smoothing)
            elif training_mode == "simpo":
                loss_fn = SimPOLoss(beta=config.beta, gamma=config.gamma)
            elif training_mode == "cpo":
                loss_fn = CPOLoss(beta=config.beta, label_smoothing=config.label_smoothing)
            elif training_mode == "ipo":
                loss_fn = IPOLoss(beta=config.beta)
            train_dataset = train_dataset.map(
                lambda x: tokenize_preference(
                    x, tokenizer, max_length=config.max_length,
                    max_prompt_length=config.max_prompt_length,
                ),
                remove_columns=train_dataset.column_names,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    lambda x: tokenize_preference(
                        x, tokenizer, max_length=config.max_length,
                        max_prompt_length=config.max_prompt_length,
                    ),
                    remove_columns=eval_dataset.column_names,
                )
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        # ---- Build trainer ----
        if torch_dtype == torch.bfloat16:
            mixed_precision = "bf16"
        elif torch_dtype == torch.float16:
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"

        # W&B config (only rank 0)
        wandb_run_name = None
        wandb_project = None
        log_with = None
        if config.use_wandb and is_main and wandb is not None:
            if config.wandb_key:
                wandb.login(key=config.wandb_key)
            wandb_run_name = config.wandb_run_name or generate_wandb_run_name(config)
            wandb_project = config.wandb_project or "merlina-training"
            log_with = "wandb"

        output_dir = f"./results/{args.job_id}"

        # Compute eval_steps
        eval_steps = None
        if config.eval_steps:
            if config.eval_steps < 1:
                import math
                num_gpus = int(os.environ.get("WORLD_SIZE", 1))
                effective_batch = config.batch_size * config.gradient_accumulation_steps * num_gpus
                steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
                total_steps = steps_per_epoch * config.num_epochs
                eval_steps = max(1, int(total_steps * config.eval_steps))
            else:
                eval_steps = int(config.eval_steps)

        grimoire_config = GrimoireTrainingConfig(
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
            log_with=log_with,
            project_name=wandb_project,
            wandb_tags=config.wandb_tags or [],
            wandb_notes=config.wandb_notes,
        )

        callbacks = []
        if is_main:
            callbacks.append(
                FileProgressCallback(args.job_id, args.progress_file, job_manager)
            )

        trainer_cls = GrimoireTrainer
        trainer_kwargs = {}
        if config.optimizer_type == "muon":
            trainer_cls = MuonGrimoireTrainer
            trainer_kwargs["muon_momentum"] = config.muon_momentum

        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            config=grimoire_config,
            loss_fn=loss_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            callbacks=callbacks,
            **trainer_kwargs,
        )

        # Capture W&B URL
        if is_main and config.use_wandb and wandb is not None and wandb.run is not None:
            wandb_url = wandb.run.get_url()
            logger.info(f"W&B run URL: {wandb_url}")
            if job_manager:
                job_manager.update_job(args.job_id, wandb_url=wandb_url)

        # ---- Train ----
        logger.info("Starting training")
        trainer.train()

        was_stopped = trainer.stopped_early

        # ---- Save model (rank 0 only) ----
        if is_main:
            if job_manager:
                job_manager.update_job(args.job_id, status="saving", progress=0.9)

            # Embed chat template if needed
            format_type = config.dataset.format.format_type
            has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
            if not has_template and format_type != "tokenizer":
                chat_template = get_chat_template_for_format(format_type)
                if chat_template:
                    tokenizer.chat_template = chat_template

            final_output_dir = f"./models/{config.output_name}"
            trainer.save_model(final_output_dir)

            final_step = trainer.global_step
            final_max_steps = trainer.max_steps

            # Cleanup before potential upload
            del trainer, model
            trainer = None
            model = None
            gc.collect()
            torch.cuda.empty_cache()

            # Write completion to progress file
            with open(args.progress_file, "a") as f:
                f.write(json.dumps({
                    "type": "completed",
                    "was_stopped": was_stopped,
                    "output_dir": final_output_dir,
                    "final_step": final_step,
                    "final_max_steps": final_max_steps,
                }) + "\n")

            # Handle Hub upload (inline — no websocket needed in subprocess)
            if config.push_to_hub and config.hf_token:
                if job_manager:
                    job_manager.update_job(args.job_id, status="uploading", progress=0.95)
                try:
                    _do_hub_upload(config, final_output_dir, training_mode, args.job_id, job_manager, was_stopped)
                except Exception as upload_err:
                    logger.error(f"Upload failed: {upload_err}", exc_info=True)
                    logger.info(f"Model saved locally at: {final_output_dir}")
                    if job_manager:
                        job_manager.update_job(args.job_id, status="completed", progress=1.0, output_dir=final_output_dir)
            else:
                if job_manager:
                    final_status = "stopped" if was_stopped else "completed"
                    final_progress = 1.0 if not was_stopped else (
                        final_step / final_max_steps if final_max_steps else 0.9
                    )
                    job_manager.update_job(
                        args.job_id,
                        status=final_status,
                        progress=final_progress,
                        output_dir=final_output_dir,
                    )

            # Finish wandb
            if config.use_wandb and wandb is not None and wandb.run is not None:
                wandb.finish()

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if is_main:
            if job_manager:
                job_manager.update_job(args.job_id, status="failed", error=str(e))
            # Write error to progress file
            with open(args.progress_file, "a") as f:
                f.write(json.dumps({"type": "error", "error": str(e)}) + "\n")
            if config.use_wandb and wandb is not None and wandb.run is not None:
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass

    finally:
        _cleanup_training_resources(model, trainer)


def main():
    parser = argparse.ArgumentParser(description="Merlina distributed training worker")
    parser.add_argument("--config-path", required=True, help="Path to training config JSON")
    parser.add_argument("--job-id", required=True, help="Job identifier")
    parser.add_argument("--progress-file", required=True, help="Path to write progress JSONL")
    parser.add_argument("--db-path", default="./data/jobs.db", help="Path to SQLite database")
    parser.add_argument("--uploaded-datasets-dir", default=None, help="Path to uploaded datasets temp dir")
    args = parser.parse_args()

    run_worker(args)


if __name__ == "__main__":
    main()
