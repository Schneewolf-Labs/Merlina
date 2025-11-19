# merlina.py
"""
Merlina - Magical Model Training Backend
ORPO training for LLMs with a delightful interface
"""

import os
import gc
import json
import torch
import wandb
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import ORPOConfig, ORPOTrainer
from accelerate import Accelerator

# Import our custom dataset handling module
from dataset_handlers import (
    DatasetPipeline,
    HuggingFaceLoader,
    LocalFileLoader,
    UploadedDatasetLoader,
    get_formatter
)

# Import new modules for persistence, WebSockets, and validation
from src.job_manager import JobManager
from src.websocket_manager import websocket_manager
from src.preflight_checks import validate_config
from src.config_manager import ConfigManager
from src.job_queue import JobQueue, JobPriority
from src.gpu_utils import get_gpu_manager

# Import data editor modules
from src.data_editor import EditorRow, EditorSession, ValidationResult, TransformationConfig
from src.data_editor.import_engine import ImportEngine
from src.data_editor.session_manager import SessionManager
from src.data_editor.validation import ValidationEngine
from src.data_editor.transformations import TransformationEngine

# Import configuration
from config import settings

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Train LLMs with ORPO, powered by magic ✨",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize job manager with persistent storage
job_manager = JobManager(db_path=settings.database_path)

# Initialize job queue with configurable concurrency
job_queue = JobQueue(max_concurrent_jobs=settings.max_concurrent_jobs, job_manager=job_manager)

# Initialize config manager
config_manager = ConfigManager(config_dir=str(settings.data_dir / "configs"))

# Global storage for uploaded datasets (still in-memory, could be persisted later)
uploaded_datasets = {}  # dataset_id -> bytes content

# Global cache for preloaded tokenizers (for preview functionality)
tokenizer_cache = {}  # model_name -> tokenizer instance

# Keep backwards compatibility - jobs dict now proxies to job_manager
class JobsProxy:
    """Proxy dict-like access to job_manager for backwards compatibility"""
    def __getitem__(self, job_id: str):
        job = job_manager.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        return {
            "status": job.status,
            "progress": job.progress,
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "loss": job.loss,
            "error": job.error,
            "wandb_url": job.wandb_url
        }

    def __setitem__(self, job_id: str, value: dict):
        # Update job in database
        job_manager.update_job(job_id, **value)

    def __contains__(self, job_id: str):
        return job_manager.get_job(job_id) is not None

    def items(self):
        jobs_list = job_manager.list_jobs()
        for job in jobs_list:
            yield job.job_id, {
                "status": job.status,
                "progress": job.progress
            }

jobs = JobsProxy()

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Check if frontend files exist in configured directory
FRONTEND_DIR = settings.frontend_dir if settings.frontend_dir.is_absolute() else SCRIPT_DIR / settings.frontend_dir
if not FRONTEND_DIR.exists():
    # Also check current directory
    if (SCRIPT_DIR / "index.html").exists():
        FRONTEND_DIR = SCRIPT_DIR
    else:
        logger.warning(f"Frontend files not found. Please create a 'frontend' directory with index.html, styles.css, and script.js")

# Pydantic models for dataset configuration
class DatasetSource(BaseModel):
    """Configuration for dataset source"""
    source_type: str = Field(..., description="Type of dataset source: huggingface, local_file, upload")

    # For HuggingFace datasets
    repo_id: Optional[str] = Field(None, description="HuggingFace repository ID")
    split: str = Field("train", description="Dataset split to use")

    # For local file datasets
    file_path: Optional[str] = Field(None, description="Path to local dataset file")
    file_format: Optional[str] = Field(None, description="File format: json, jsonl, csv, parquet")

    # For uploaded datasets (handled separately via upload endpoint)
    dataset_id: Optional[str] = Field(None, description="ID of previously uploaded dataset")


class DatasetFormat(BaseModel):
    """Configuration for dataset formatting"""
    format_type: str = Field("chatml", description="Format type: chatml, llama3, mistral, qwen3, tokenizer, custom")

    # For custom format
    custom_templates: Optional[dict] = Field(
        None,
        description="Custom templates (required if format_type is 'custom')"
    )

    # For qwen3 format
    enable_thinking: Optional[bool] = Field(
        True,
        description="Enable thinking mode for Qwen3 format (default: True)"
    )


class DatasetConfig(BaseModel):
    """Complete dataset configuration"""
    source: DatasetSource = Field(
        default=DatasetSource(
            source_type="huggingface",
            repo_id="schneewolflabs/Athanor-DPO",
            split="train"
        ),
        description="Dataset source configuration"
    )

    format: DatasetFormat = Field(
        default=DatasetFormat(format_type="chatml"),
        description="Dataset format configuration"
    )

    # Optional: model name for tokenizer-based formatting in preview
    model_name: Optional[str] = Field(None, description="Model name (required for tokenizer format preview)")

    # Column mapping (if dataset uses different column names)
    column_mapping: Optional[dict] = Field(
        None,
        description="Map dataset columns to expected names (system, prompt, chosen, rejected)"
    )

    # Additional options
    test_size: float = Field(0.01, ge=0.001, le=0.5, description="Fraction of data for evaluation")
    max_samples: Optional[int] = Field(None, description="Limit dataset size (for testing)")


# Pydantic models
class TrainingConfig(BaseModel):
    # Model settings
    base_model: str = Field(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        description="Base model to fine-tune (HuggingFace model ID or local directory path)"
    )
    output_name: str = Field(..., description="Name for the output model")
    
    # LoRA settings
    use_lora: bool = Field(True, description="Enable LoRA (Low-Rank Adaptation)")
    lora_r: int = Field(64, ge=8, le=256)
    lora_alpha: int = Field(32, ge=8, le=256)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    
    # Training hyperparameters
    learning_rate: float = Field(5e-6, ge=1e-8, le=1e-3)
    num_epochs: int = Field(2, ge=1, le=10)
    batch_size: int = Field(1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(16, ge=1, le=128)
    max_length: int = Field(2048, ge=512, le=8192)
    max_prompt_length: int = Field(1024, ge=256, le=4096)

    # Training mode
    training_mode: str = Field("orpo", description="Training mode: 'sft' or 'orpo'")

    # ORPO specific
    beta: float = Field(0.1, ge=0.01, le=1.0, description="ORPO beta parameter (only used when training_mode='orpo')")

    # Dataset configuration
    dataset: DatasetConfig = Field(
        default_factory=lambda: DatasetConfig(),
        description="Dataset configuration"
    )

    # Priority 1 settings (main UI)
    seed: int = Field(42, ge=0, le=99999, description="Random seed for reproducibility")
    max_grad_norm: float = Field(0.3, ge=0.1, le=5.0, description="Gradient clipping threshold")

    # Optional settings
    warmup_ratio: float = Field(0.05, ge=0.0, le=0.5)
    eval_steps: float = Field(0.2, ge=0.1, le=1.0)
    use_4bit: bool = Field(True, description="Use 4-bit quantization")
    use_wandb: bool = Field(True, description="Log to Weights & Biases")
    push_to_hub: bool = Field(False, description="Push to HuggingFace Hub")
    merge_lora_before_upload: bool = Field(True, description="Merge LoRA with base model before uploading (if False, uploads LoRA adapter only)")
    hf_hub_private: bool = Field(True, description="Make HuggingFace Hub repository private")
    hf_token: Optional[str] = Field(None, description="HuggingFace token for pushing")
    wandb_key: Optional[str] = Field(None, description="Weights & Biases API key")

    # Priority 2 settings (advanced)
    shuffle_dataset: bool = Field(True, description="Shuffle training data")
    weight_decay: float = Field(0.01, ge=0.0, le=0.5, description="L2 regularization strength")
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler type")
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save VRAM")
    logging_steps: int = Field(1, ge=1, le=100, description="Log metrics every N steps")

    # Optimizer settings
    optimizer_type: str = Field(
        "paged_adamw_8bit",
        description="Optimizer type (paged_adamw_8bit, paged_adamw_32bit, adamw_8bit, adamw_torch, adamw_hf, adafactor, sgd)"
    )
    adam_beta1: float = Field(0.9, ge=0.8, le=0.99, description="Adam beta1 parameter (first moment decay)")
    adam_beta2: float = Field(0.999, ge=0.9, le=0.9999, description="Adam beta2 parameter (second moment decay)")
    adam_epsilon: float = Field(1e-8, ge=1e-9, le=1e-6, description="Adam epsilon for numerical stability")

    # Attention settings
    attn_implementation: str = Field(
        "auto",
        description="Attention implementation (auto, flash_attention_2, sdpa, eager)"
    )

    # GPU settings
    gpu_ids: Optional[list[int]] = Field(
        None,
        description="List of GPU indices to use for training (e.g., [0] or [0, 1]). If None, uses all available GPUs or CUDA_VISIBLE_DEVICES"
    )

    # Weights & Biases settings
    wandb_project: Optional[str] = Field(None, description="W&B project name (default: 'merlina-training')")
    wandb_run_name: Optional[str] = Field(None, description="W&B run name (auto-generated if not provided)")
    wandb_tags: Optional[list[str]] = Field(None, description="W&B tags for organizing runs")
    wandb_notes: Optional[str] = Field(None, description="Notes about this training run")

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    error: Optional[str] = None
    wandb_url: Optional[str] = None
    queue_position: Optional[int] = None
    queue_state: Optional[str] = None

# Old training function - kept for reference, now using training_runner.py
# def run_training(job_id: str, config: TrainingConfig):
#     """Run ORPO training job - DEPRECATED, see training_runner.py"""
#     pass

# Placeholder to keep line numbers similar (will be removed)
def _old_run_training_deprecated(job_id: str, config: TrainingConfig):
    """DEPRECATED: Old training function. Using training_runner.py instead"""
    try:
        # Update job status
        jobs[job_id] = {"status": "initializing", "progress": 0.0}
        
        # Setup accelerator
        accelerator = Accelerator()
        
        # Login to wandb if configured
        if config.use_wandb and config.wandb_key:
            wandb.login(key=config.wandb_key)
            
        # Set HF token if provided
        if config.hf_token:
            os.environ['HF_TOKEN'] = config.hf_token
            
        # Determine dtype and attention
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"
            
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
        jobs[job_id]["status"] = "loading_model"
        jobs[job_id]["progress"] = 0.1
        
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
        
        # Load and prepare dataset using new modular system
        jobs[job_id]["status"] = "loading_dataset"
        jobs[job_id]["progress"] = 0.2

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
            # Get uploaded dataset from storage
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

        # Get formatter (pass tokenizer if using tokenizer format)
        formatter = get_formatter(
            format_type=config.dataset.format.format_type,
            custom_templates=config.dataset.format.custom_templates,
            tokenizer=tokenizer if config.dataset.format.format_type == 'tokenizer' else None,
            enable_thinking=config.dataset.format.enable_thinking
        )

        # Create pipeline and prepare dataset
        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            column_mapping=config.dataset.column_mapping,
            test_size=config.dataset.test_size,
            max_samples=config.dataset.max_samples,
            seed=42
        )

        train_dataset, eval_dataset = pipeline.prepare()

        # Create dict with train/test for compatibility
        dataset = {"train": train_dataset, "test": eval_dataset}
        
        # Setup training arguments
        jobs[job_id]["status"] = "training"
        jobs[job_id]["progress"] = 0.3
        
        output_dir = f"./results/{job_id}"
        
        orpo_args = ORPOConfig(
            run_name=config.output_name,
            learning_rate=config.learning_rate,
            lr_scheduler_type="cosine",
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_length - config.max_prompt_length,
            beta=config.beta,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim="paged_adamw_8bit",
            num_train_epochs=config.num_epochs,
            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
            logging_steps=1,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=0.3,
            report_to=["wandb"] if config.use_wandb else [],
            output_dir=output_dir,
            bf16=torch_dtype == torch.bfloat16,
            fp16=torch_dtype == torch.float16,
            save_strategy="steps",
            save_steps=config.eval_steps,
            save_total_limit=2,
        )
        
        # Create trainer
        trainer = ORPOTrainer(
            model=model,
            args=orpo_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=peft_config,
            tokenizer=tokenizer,
        )
        
        # Train
        logger.info("Starting training")
        train_result = trainer.train()
        
        # Save model
        jobs[job_id]["status"] = "saving"
        jobs[job_id]["progress"] = 0.9
        
        final_output_dir = f"./models/{config.output_name}"
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        # Optional: merge and upload
        if config.push_to_hub and accelerator.is_main_process:
            jobs[job_id]["status"] = "uploading"
            
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
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        logger.info(f"Training completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# Mount static files for frontend
if FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_frontend():
        """Serve the main HTML page"""
        return FileResponse(FRONTEND_DIR / "index.html")
    
    # Mount the frontend directory to serve static files
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    
    # Also serve CSS, JS modules, and images from root for simplicity
    @app.get("/styles.css")
    async def serve_css():
        return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")

    @app.get("/js/{file_path:path}")
    async def serve_js_modules(file_path: str):
        """Serve JavaScript modules from js/ directory"""
        js_file = FRONTEND_DIR / "js" / file_path
        if js_file.exists() and js_file.is_file():
            return FileResponse(js_file, media_type="application/javascript")
        return {"error": "File not found"}

    @app.get("/merlina.png")
    async def serve_logo():
        return FileResponse(FRONTEND_DIR / "merlina.png", media_type="image/png")
else:
    @app.get("/")
    async def root():
        return {
            "name": "Merlina",
            "version": "1.0.0",
            "description": "Magical Model Training Backend",
            "message": "Frontend not found. Please add index.html, styles.css, and script.js to the 'frontend' directory"
        }

# API Endpoints
@app.get("/api")
async def api_info():
    return {
        "name": "Merlina",
        "version": "1.0.0",
        "description": "Magical Model Training Backend",
        "endpoints": {
            "POST /train": "Start a new training job",
            "GET /status/{job_id}": "Get job status",
            "GET /jobs": "List all jobs",
            "GET /api/docs": "API documentation"
        }
    }

@app.post("/validate", response_model=dict)
async def validate_training_config(config: TrainingConfig):
    """
    Validate training configuration before starting.
    Checks GPU, VRAM, disk space, model access, etc.
    """
    try:
        is_valid, results = validate_config(config)
        return {
            "valid": is_valid,
            "results": results
        }
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train", response_model=JobResponse)
async def create_training_job(config: TrainingConfig, priority: Optional[str] = "normal"):
    """
    Create and queue a training job.
    Runs pre-flight validation before queueing.

    Args:
        config: Training configuration
        priority: Job priority (low, normal, high) - default: normal
    """
    # Run pre-flight validation
    is_valid, validation_results = validate_config(config)

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Training configuration validation failed",
                "errors": validation_results.get("errors", []),
                "warnings": validation_results.get("warnings", [])
            }
        )

    # Parse priority
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH
    }
    job_priority = priority_map.get(priority.lower(), JobPriority.NORMAL)

    # Create job in database
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_manager.create_job(job_id, config.model_dump())

    # Import training runner
    from src.training_runner import run_training_sync
    import asyncio

    # Get the current event loop for WebSocket updates
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # Define callback that will be executed by worker
    def training_callback(job_id: str, config_dict: dict):
        """Wrapper to convert config dict back to TrainingConfig"""
        from pydantic import TypeAdapter
        config_obj = TypeAdapter(TrainingConfig).validate_python(config_dict)
        run_training_sync(job_id, config_obj, job_manager, uploaded_datasets, loop)

    # Submit job to queue
    position = job_queue.submit(
        job_id=job_id,
        config=config.model_dump(),
        callback=training_callback,
        priority=job_priority
    )

    # Return response with warnings if any
    message = f"Training spell cast! Job {job_id} queued at position {position}."
    if validation_results.get("warnings"):
        message += f" Note: {len(validation_results['warnings'])} warning(s) detected."

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=message
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a training job with queue information"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Get queue status
    queue_status = job_queue.get_status(job_id)

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        current_step=job.get("current_step"),
        total_steps=job.get("total_steps"),
        loss=job.get("loss"),
        error=job.get("error"),
        wandb_url=job.get("wandb_url"),
        queue_position=queue_status.get("position"),
        queue_state=queue_status.get("state")
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        job_id: {
            "status": job["status"],
            "progress": job.get("progress", 0.0)
        }
        for job_id, job in jobs.items()
    }


@app.get("/jobs/history")
async def get_job_history(limit: int = 50, offset: int = 0, status: Optional[str] = None):
    """
    Get job history with pagination.
    Now persisted across server restarts!
    """
    jobs_list = job_manager.list_jobs(status=status, limit=limit, offset=offset)
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "output_dir": job.output_dir,
                "error": job.error
            }
            for job in jobs_list
        ],
        "count": len(jobs_list)
    }


@app.get("/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Get detailed metrics for a job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    metrics = job_manager.get_metrics(job_id)
    return {
        "job_id": job_id,
        "metrics": metrics,
        "count": len(metrics)
    }


@app.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """
    Cancel or stop a job.
    - For queued jobs: Removes from queue immediately
    - For running jobs: Graceful stop after current step
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check queue status
    queue_status = job_queue.get_status(job_id)

    # If job is queued, cancel it from the queue
    if queue_status.get("state") == "queued":
        success = job_queue.cancel(job_id)
        if success:
            logger.info(f"Job {job_id} cancelled from queue")
            return {
                "status": "success",
                "message": f"Job {job_id} removed from queue",
                "job_id": job_id,
                "was_queued": True
            }

    # If job is running, request graceful stop
    if queue_status.get("state") == "running" or job.status in ["training", "loading_model", "loading_dataset", "initializing"]:
        success = job_queue.cancel(job_id)  # This will set stop_requested flag
        if success:
            logger.info(f"Stop requested for running job {job_id}")
            return {
                "status": "success",
                "message": f"Stop request sent to job {job_id}. Training will stop after current step.",
                "job_id": job_id,
                "was_queued": False
            }

    # Job is not in a stoppable state
    return {
        "status": "warning",
        "message": f"Job {job_id} is in state '{job.status}' and cannot be stopped",
        "job_id": job_id
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a specific job and its metrics"""
    success = job_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": "success",
        "message": f"Job {job_id} deleted successfully",
        "job_id": job_id
    }


@app.delete("/jobs")
async def clear_all_jobs():
    """Delete all jobs and metrics"""
    count = job_manager.clear_all_jobs()
    return {
        "status": "success",
        "message": f"Cleared all jobs ({count} jobs deleted)",
        "deleted_count": count
    }


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time training updates.
    Connect to receive live status updates, metrics, and progress.
    """
    await websocket_manager.connect(websocket, job_id)
    try:
        # Send initial status
        job = job_manager.get_job(job_id)
        if job:
            await websocket.send_json({
                "type": "status_update",
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "current_step": job.current_step,
                "total_steps": job.total_steps,
                "loss": job.loss,
                "eval_loss": job.eval_loss
            })

        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for any messages from client (heartbeat, etc.)
                data = await websocket.receive_text()
                # Echo back as heartbeat acknowledgment
                await websocket.send_json({"type": "heartbeat", "status": "ok"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        websocket_manager.disconnect(websocket, job_id)


@app.get("/stats")
async def get_stats():
    """Get database and system statistics"""
    return {
        "database": job_manager.get_stats(),
        "websockets": {
            "total_connections": websocket_manager.get_connection_count()
        },
        "queue": job_queue.get_queue_stats()
    }


@app.get("/queue/status")
async def get_queue_status():
    """
    Get overall queue status and statistics.

    Returns information about:
    - Number of queued jobs
    - Number of running jobs
    - Available worker slots
    - Queue configuration
    """
    stats = job_queue.get_queue_stats()
    queued_jobs = job_queue.list_queued_jobs()
    running_jobs = job_queue.list_running_jobs()

    return {
        "stats": stats,
        "queued_jobs": queued_jobs,
        "running_jobs": running_jobs
    }


@app.get("/queue/jobs")
async def list_queue_jobs():
    """
    List all jobs in the queue (queued and running).

    Returns detailed information about queue position and status.
    """
    return {
        "queued": job_queue.list_queued_jobs(),
        "running": job_queue.list_running_jobs()
    }


# GPU management endpoints
@app.get("/gpu/list")
async def list_gpus():
    """
    List all available GPUs with detailed information.

    Returns information about each GPU including:
    - Index
    - Name
    - Memory (total, free, used)
    - Utilization
    - Temperature
    - Power usage
    - Compute capability
    """
    try:
        gpu_manager = get_gpu_manager()

        if not gpu_manager.is_cuda_available():
            return {
                "status": "no_cuda",
                "message": "CUDA is not available on this system",
                "gpus": []
            }

        gpus = gpu_manager.list_gpus()

        return {
            "status": "success",
            "gpus": [gpu.to_dict() for gpu in gpus],
            "count": len(gpus)
        }

    except Exception as e:
        logger.error(f"Failed to list GPUs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/{index}")
async def get_gpu_info(index: int):
    """
    Get detailed information about a specific GPU.

    Args:
        index: GPU index (0-based)
    """
    try:
        gpu_manager = get_gpu_manager()

        if not gpu_manager.is_cuda_available():
            raise HTTPException(status_code=404, detail="CUDA is not available")

        gpu = gpu_manager.get_gpu_info(index)

        if not gpu:
            raise HTTPException(status_code=404, detail=f"GPU {index} not found")

        return {
            "status": "success",
            "gpu": gpu.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/available")
async def get_available_gpus(min_free_memory_mb: int = 4000):
    """
    Get list of available GPUs with sufficient free memory.

    Args:
        min_free_memory_mb: Minimum free memory in MB (default: 4000)

    Returns:
        List of GPU indices that meet the memory requirement
    """
    try:
        gpu_manager = get_gpu_manager()

        if not gpu_manager.is_cuda_available():
            return {
                "status": "no_cuda",
                "available_gpus": [],
                "message": "CUDA is not available"
            }

        available = gpu_manager.get_available_gpus(min_free_memory_mb)

        return {
            "status": "success",
            "available_gpus": available,
            "count": len(available),
            "min_free_memory_mb": min_free_memory_mb
        }

    except Exception as e:
        logger.error(f"Failed to get available GPUs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/recommended")
async def get_recommended_gpu():
    """
    Get the recommended GPU for training.
    Returns the GPU with the most free memory.
    """
    try:
        gpu_manager = get_gpu_manager()

        if not gpu_manager.is_cuda_available():
            raise HTTPException(status_code=404, detail="CUDA is not available")

        recommended = gpu_manager.get_recommended_gpu()

        if recommended is None:
            raise HTTPException(status_code=404, detail="No GPUs available")

        gpu = gpu_manager.get_gpu_info(recommended)

        return {
            "status": "success",
            "recommended_index": recommended,
            "gpu": gpu.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommended GPU: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Dataset management endpoints
@app.post("/dataset/preview")
async def preview_dataset(config: DatasetConfig):
    """Preview dataset without formatting"""
    try:
        # Create loader
        if config.source.source_type == "huggingface":
            loader = HuggingFaceLoader(
                repo_id=config.source.repo_id,
                split=config.source.split
            )
        elif config.source.source_type == "local_file":
            loader = LocalFileLoader(
                file_path=config.source.file_path,
                file_format=config.source.file_format
            )
        elif config.source.source_type == "upload":
            dataset_id = config.source.dataset_id
            if dataset_id not in uploaded_datasets:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

            file_content, filename = uploaded_datasets[dataset_id]
            loader = UploadedDatasetLoader(
                file_content=file_content,
                filename=filename,
                file_format=config.source.file_format
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {config.source.source_type}")

        # Get formatter (just for pipeline, we'll preview raw)
        # For tokenizer format, we can't preview without a model, so use chatml as default
        formatter_type = config.format.format_type
        if formatter_type == 'tokenizer':
            logger.warning("Cannot preview with 'tokenizer' format without loading the model. Using 'chatml' for preview.")
            formatter_type = 'chatml'

        formatter = get_formatter(
            format_type=formatter_type,
            custom_templates=config.format.custom_templates,
            enable_thinking=config.format.enable_thinking
        )

        # Create pipeline
        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            column_mapping=config.column_mapping,
            test_size=config.test_size,
            max_samples=config.max_samples
        )

        # Preview raw data
        preview = pipeline.preview(num_samples=10)

        return {
            "status": "success",
            "samples": preview,
            "num_samples": len(preview)
        }

    except Exception as e:
        logger.error(f"Dataset preview failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dataset/preview-formatted")
async def preview_formatted_dataset(config: DatasetConfig):
    """Preview dataset with formatting applied"""
    try:
        # Create loader
        if config.source.source_type == "huggingface":
            loader = HuggingFaceLoader(
                repo_id=config.source.repo_id,
                split=config.source.split
            )
        elif config.source.source_type == "local_file":
            loader = LocalFileLoader(
                file_path=config.source.file_path,
                file_format=config.source.file_format
            )
        elif config.source.source_type == "upload":
            dataset_id = config.source.dataset_id
            if dataset_id not in uploaded_datasets:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

            file_content, filename = uploaded_datasets[dataset_id]
            loader = UploadedDatasetLoader(
                file_content=file_content,
                filename=filename,
                file_format=config.source.file_format
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {config.source.source_type}")

        # Get formatter
        formatter_type = config.format.format_type

        # For tokenizer format, check if we have a cached tokenizer
        if formatter_type == 'tokenizer':
            if config.model_name and config.model_name in tokenizer_cache:
                # Use the cached tokenizer
                logger.info(f"Using cached tokenizer for preview: {config.model_name}")
                formatter = get_formatter(
                    format_type='tokenizer',
                    tokenizer=tokenizer_cache[config.model_name]
                )
            else:
                # Fall back to chatml
                logger.warning("Cannot preview with 'tokenizer' format without preloading the model. Using 'chatml' for preview.")
                logger.info("Tip: Use the 'Validate & Preload Model' button to enable tokenizer format preview.")
                formatter_type = 'chatml'
                formatter = get_formatter(
                    format_type=formatter_type,
                    custom_templates=config.format.custom_templates,
                    enable_thinking=config.format.enable_thinking
                )
        else:
            formatter = get_formatter(
                format_type=formatter_type,
                custom_templates=config.format.custom_templates,
                enable_thinking=config.format.enable_thinking
            )

        # Create pipeline
        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            column_mapping=config.column_mapping,
            test_size=config.test_size,
            max_samples=config.max_samples
        )

        # Preview formatted data
        preview = pipeline.preview_formatted(num_samples=5)

        return {
            "status": "success",
            "samples": preview,
            "num_samples": len(preview)
        }

    except Exception as e:
        logger.error(f"Dataset preview failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dataset/columns")
async def get_dataset_columns(config: DatasetConfig):
    """
    Get column names and sample data from dataset for mapping.
    Returns available columns and a few sample rows.
    """
    try:
        # Create loader
        if config.source.source_type == "huggingface":
            loader = HuggingFaceLoader(
                repo_id=config.source.repo_id,
                split=config.source.split
            )
        elif config.source.source_type == "local_file":
            loader = LocalFileLoader(
                file_path=config.source.file_path,
                file_format=config.source.file_format
            )
        elif config.source.source_type == "upload":
            dataset_id = config.source.dataset_id
            if dataset_id not in uploaded_datasets:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

            file_content, filename = uploaded_datasets[dataset_id]
            loader = UploadedDatasetLoader(
                file_content=file_content,
                filename=filename,
                file_format=config.source.file_format
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {config.source.source_type}")

        # Load dataset
        logger.info("Loading dataset to inspect columns...")
        dataset = loader.load()

        # Get column names
        columns = dataset.column_names

        # Get a few sample rows
        num_samples = min(3, len(dataset))
        samples = [dict(row) for row in dataset.select(range(num_samples))]

        return {
            "status": "success",
            "columns": columns,
            "samples": samples,
            "total_rows": len(dataset)
        }

    except Exception as e:
        logger.error(f"Failed to get dataset columns: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dataset/upload")
async def upload_dataset(file: bytes = None, filename: str = None):
    """Upload a dataset file"""
    from fastapi import File, Form, UploadFile

    # This endpoint needs to be called with multipart/form-data
    # We'll create a proper version
    pass


# Proper upload endpoint with FastAPI's UploadFile
@app.post("/dataset/upload-file")
async def upload_dataset_file(file: UploadFile = File(...)):
    """Upload a dataset file and return dataset ID"""
    try:
        # Read file content
        content = await file.read()

        # Generate dataset ID
        import hashlib
        from datetime import datetime

        dataset_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(content).hexdigest()[:8]}"

        # Store in memory (in production, save to disk/S3)
        uploaded_datasets[dataset_id] = (content, file.filename)

        logger.info(f"Uploaded dataset: {file.filename} -> {dataset_id}")

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "filename": file.filename,
            "size": len(content)
        }

    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dataset/uploads")
async def list_uploaded_datasets():
    """List all uploaded datasets"""
    return {
        "datasets": [
            {
                "dataset_id": dataset_id,
                "filename": filename,
                "size": len(content)
            }
            for dataset_id, (content, filename) in uploaded_datasets.items()
        ]
    }


# ===== Data Editor Endpoints =====

# Initialize data editor components
editor_session_manager = SessionManager(db_path=str(settings.data_dir / "editor_sessions.db"))
editor_import_engine = ImportEngine()
editor_validation_engine = ValidationEngine()
editor_transformation_engine = TransformationEngine()


# Pydantic models for data editor
class EditorImportRequest(BaseModel):
    """Request to import a file into the editor"""
    source_type: str = Field(..., description="Source type: upload, local_file")
    dataset_id: Optional[str] = Field(None, description="For uploaded datasets")
    file_path: Optional[str] = Field(None, description="For local files")
    session_name: str = Field(..., description="Name for the editing session")


class EditorRowUpdate(BaseModel):
    """Request to update a row"""
    prompt: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    system: Optional[str] = None
    reasoning: Optional[str] = None


class EditorRowCreate(BaseModel):
    """Request to create a new row"""
    prompt: str
    chosen: str
    rejected: Optional[str] = None  # Optional for SFT mode
    system: Optional[str] = None
    reasoning: Optional[str] = None


class EditorTransformRequest(BaseModel):
    """Request to transform data"""
    session_id: str
    column_mapping: Dict[str, Any]
    generate_rejected: bool = False
    rejected_strategy: Optional[str] = None
    add_system_message: Optional[str] = None


class EditorExportRequest(BaseModel):
    """Request to export session data"""
    session_id: str
    format: str = Field("json", description="Export format: json, jsonl, csv")
    only_valid: bool = Field(True, description="Export only valid rows")
    direct_upload: bool = Field(False, description="Upload directly for training")


@app.post("/editor/import")
async def editor_import_file(
    file: UploadFile = File(...),
    session_name: str = Form(...),
    training_mode: str = Form("orpo")
):
    """
    Import a file into the data editor

    Args:
        file: Dataset file to import
        session_name: Name for the editing session
        training_mode: Training mode ("orpo" or "sft"), defaults to "orpo"

    Returns: session_id and import metadata
    """
    try:
        # Validate training mode
        training_mode = training_mode.lower()
        if training_mode not in ["orpo", "sft"]:
            raise HTTPException(status_code=400, detail="training_mode must be 'orpo' or 'sft'")

        # Read file content
        content = await file.read()

        # Import file
        rows_data, metadata = editor_import_engine.import_file(file.filename, content)

        # Detect schema and suggest column mapping
        suggested_mapping = editor_import_engine.suggest_column_mapping(rows_data)

        # Create editor session with training mode
        session_id = editor_session_manager.create_session(
            name=session_name,
            source_file=file.filename,
            training_mode=training_mode
        )

        # Convert raw rows to EditorRow objects (without transformation yet)
        editor_rows = []
        for idx, row_data in enumerate(rows_data):
            editor_row = EditorRow(
                idx=idx,
                metadata={'original': row_data}
            )
            editor_rows.append(editor_row)

        # Add rows to session
        editor_session_manager.add_rows(session_id, editor_rows, record_operation=False)

        # Update session with metadata
        editor_session_manager.update_session(
            session_id,
            statistics=metadata
        )

        logger.info(f"Imported {len(rows_data)} rows into editor session {session_id} (mode: {training_mode})")

        return {
            "status": "success",
            "session_id": session_id,
            "training_mode": training_mode,
            "num_rows": len(rows_data),
            "metadata": metadata,
            "suggested_mapping": suggested_mapping,
            "schema_type": editor_import_engine.detect_schema_type(rows_data)
        }

    except Exception as e:
        logger.error(f"Editor import failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/session/create")
async def editor_create_session(
    name: str,
    source_file: Optional[str] = None,
    training_mode: str = "orpo"
):
    """
    Create a new empty editing session

    Args:
        name: Name for the session
        source_file: Optional source filename
        training_mode: Training mode ("orpo" or "sft"), defaults to "orpo"
    """
    try:
        # Validate training mode
        training_mode = training_mode.lower()
        if training_mode not in ["orpo", "sft"]:
            raise HTTPException(status_code=400, detail="training_mode must be 'orpo' or 'sft'")

        session_id = editor_session_manager.create_session(name, source_file, training_mode)

        return {
            "status": "success",
            "session_id": session_id,
            "training_mode": training_mode
        }

    except Exception as e:
        logger.error(f"Failed to create editor session: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/editor/session/{session_id}")
async def editor_get_session(session_id: str, limit: int = 100, offset: int = 0):
    """Get an editing session with pagination"""
    try:
        session = editor_session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Paginate rows
        total_rows = len(session.rows)
        paginated_rows = session.rows[offset:offset + limit]

        return {
            "status": "success",
            "session": {
                "session_id": session.session_id,
                "name": session.name,
                "source_file": session.source_file,
                "column_mapping": session.column_mapping,
                "training_mode": session.training_mode,
                "statistics": session.statistics,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "total_rows": total_rows
            },
            "rows": [row.to_dict() for row in paginated_rows],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_rows,
                "has_more": offset + limit < total_rows
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get editor session: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/editor/sessions")
async def editor_list_sessions(limit: int = 50, offset: int = 0):
    """List all editing sessions"""
    try:
        sessions = editor_session_manager.list_sessions(limit, offset)

        return {
            "status": "success",
            "sessions": sessions,
            "count": len(sessions)
        }

    except Exception as e:
        logger.error(f"Failed to list editor sessions: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/editor/session/{session_id}")
async def editor_delete_session(session_id: str):
    """Delete an editing session"""
    try:
        success = editor_session_manager.delete_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return {
            "status": "success",
            "message": f"Session {session_id} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete editor session: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/session/{session_id}/row")
async def editor_add_row(session_id: str, row: EditorRowCreate):
    """Add a new row to the session"""
    try:
        session = editor_session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Create new row with next index
        new_idx = max([r.idx for r in session.rows], default=-1) + 1

        editor_row = EditorRow(
            idx=new_idx,
            prompt=row.prompt,
            chosen=row.chosen,
            rejected=row.rejected,
            system=row.system,
            reasoning=row.reasoning
        )

        # Validate the row using session's training mode
        validation_result = editor_validation_engine.validate_row(editor_row, training_mode=session.training_mode)
        editor_row.validation_errors = validation_result.errors
        editor_row.validation_warnings = validation_result.warnings

        # Add row
        editor_session_manager.add_rows(session_id, [editor_row], record_operation=True)

        return {
            "status": "success",
            "row": editor_row.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add row: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/editor/session/{session_id}/row/{idx}")
async def editor_update_row(session_id: str, idx: int, updates: EditorRowUpdate):
    """Update a row in the session"""
    try:
        # Get update dict
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}

        if not update_dict:
            raise HTTPException(status_code=400, detail="No updates provided")

        # Update row
        success = editor_session_manager.update_row(session_id, idx, update_dict, record_operation=True)

        if not success:
            raise HTTPException(status_code=404, detail=f"Row {idx} not found in session {session_id}")

        # Re-validate the row
        session = editor_session_manager.get_session(session_id)
        row = session.get_row(idx)

        if row:
            validation_result = editor_validation_engine.validate_row(row, training_mode=session.training_mode)
            editor_session_manager.update_row(
                session_id, idx,
                {
                    "validation_errors": validation_result.errors,
                    "validation_warnings": validation_result.warnings
                },
                record_operation=False
            )

        return {
            "status": "success",
            "message": f"Row {idx} updated"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update row: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/editor/session/{session_id}/row/{idx}")
async def editor_delete_row(session_id: str, idx: int):
    """Delete a row from the session"""
    try:
        success = editor_session_manager.delete_row(session_id, idx, record_operation=True)

        if not success:
            raise HTTPException(status_code=404, detail=f"Row {idx} not found in session {session_id}")

        return {
            "status": "success",
            "message": f"Row {idx} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete row: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/transform")
async def editor_transform_data(request: EditorTransformRequest):
    """Apply transformations to session data"""
    try:
        session = editor_session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

        # Create transformation config
        transform_config = TransformationConfig(
            column_mapping=request.column_mapping,
            generate_rejected=request.generate_rejected,
            rejected_strategy=request.rejected_strategy,
            add_system_message=request.add_system_message
        )

        # Apply transformations to each row
        transformed_count = 0
        for row in session.rows:
            if row.metadata and 'original' in row.metadata:
                # Transform the original data
                transformed_row = editor_transformation_engine.transform_row(
                    row.metadata['original'],
                    transform_config
                )

                # Update the row
                editor_session_manager.update_row(
                    request.session_id,
                    row.idx,
                    {
                        'prompt': transformed_row.prompt,
                        'chosen': transformed_row.chosen,
                        'rejected': transformed_row.rejected,
                        'system': transformed_row.system,
                        'reasoning': transformed_row.reasoning
                    },
                    record_operation=False
                )
                transformed_count += 1

        # Update session column mapping
        editor_session_manager.update_session(
            request.session_id,
            column_mapping=request.column_mapping
        )

        # Validate all rows using session's training mode
        session = editor_session_manager.get_session(request.session_id)
        validation_result = editor_validation_engine.validate_session(session, training_mode=session.training_mode)

        logger.info(f"Transformed {transformed_count} rows in session {request.session_id}")

        return {
            "status": "success",
            "transformed_rows": transformed_count,
            "validation": validation_result.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/validate/{session_id}")
async def editor_validate_session(session_id: str):
    """Validate all rows in a session"""
    try:
        session = editor_session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Run validation using session's training mode
        validation_result = editor_validation_engine.validate_session(session, training_mode=session.training_mode)

        # Update validation results for each row
        for row in session.rows:
            row_result = editor_validation_engine.validate_row(row, training_mode=session.training_mode)
            editor_session_manager.update_row(
                session_id,
                row.idx,
                {
                    "validation_errors": row_result.errors,
                    "validation_warnings": row_result.warnings
                },
                record_operation=False
            )

        # Update session statistics
        editor_session_manager.update_session(
            session_id,
            statistics=validation_result.statistics
        )

        return {
            "status": "success",
            "validation": validation_result.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/session/{session_id}/undo")
async def editor_undo(session_id: str):
    """Undo the last operation"""
    try:
        success = editor_session_manager.undo(session_id)

        if not success:
            raise HTTPException(status_code=400, detail="Nothing to undo")

        return {
            "status": "success",
            "message": "Undo successful"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Undo failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/session/{session_id}/redo")
async def editor_redo(session_id: str):
    """Redo the next operation"""
    try:
        success = editor_session_manager.redo(session_id)

        if not success:
            raise HTTPException(status_code=400, detail="Nothing to redo")

        return {
            "status": "success",
            "message": "Redo successful"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Redo failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/export")
async def editor_export_data(request: EditorExportRequest):
    """Export session data"""
    try:
        session = editor_session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

        # Get rows to export
        if request.only_valid:
            rows_to_export = session.get_valid_rows()
        else:
            rows_to_export = session.rows

        if not rows_to_export:
            raise HTTPException(status_code=400, detail="No rows to export")

        # Convert to export format
        export_data = []
        for row in rows_to_export:
            row_dict = row.get_all_fields()
            # Remove None values
            row_dict = {k: v for k, v in row_dict.items() if v is not None}
            export_data.append(row_dict)

        # If direct upload, add to uploaded_datasets
        if request.direct_upload:
            import hashlib
            from datetime import datetime

            # Convert to JSON
            content_str = json.dumps(export_data, indent=2)
            content_bytes = content_str.encode('utf-8')

            # Generate dataset ID
            dataset_id = f"editor_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(content_bytes).hexdigest()[:8]}"
            filename = f"{session.name.replace(' ', '_')}.json"

            # Store in uploaded_datasets
            uploaded_datasets[dataset_id] = (content_bytes, filename)

            logger.info(f"Exported editor session {request.session_id} to dataset {dataset_id}")

            return {
                "status": "success",
                "dataset_id": dataset_id,
                "filename": filename,
                "num_rows": len(export_data)
            }
        else:
            # Return data for download
            return {
                "status": "success",
                "data": export_data,
                "num_rows": len(export_data),
                "format": request.format
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/editor/generate-pairs/{session_id}")
async def editor_generate_preference_pairs(session_id: str, strategy: str = "truncate_50"):
    """Generate rejected responses for rows that only have chosen"""
    try:
        session = editor_session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Generate rejected responses
        generated_count = 0
        for row in session.rows:
            if row.chosen and not row.rejected:
                rejected = editor_transformation_engine.generate_rejected_response(
                    row.chosen,
                    strategy
                )
                editor_session_manager.update_row(
                    session_id,
                    row.idx,
                    {"rejected": rejected},
                    record_operation=False
                )
                generated_count += 1

        logger.info(f"Generated {generated_count} rejected responses in session {session_id}")

        return {
            "status": "success",
            "generated_count": generated_count,
            "strategy": strategy
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preference pair generation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ===== Config Management Endpoints =====

class SaveConfigRequest(BaseModel):
    """Request to save a training configuration"""
    name: str = Field(..., description="Name for the configuration")
    config: dict = Field(..., description="Training configuration to save")
    description: str = Field("", description="Optional description of the configuration")
    tags: list[str] = Field(default_factory=list, description="Optional tags for categorization")


@app.post("/configs/save")
async def save_config(request: SaveConfigRequest):
    """
    Save a training configuration for later reuse.

    The configuration is saved as a JSON file in data/configs/ directory.
    """
    try:
        filepath = config_manager.save_config(
            name=request.name,
            config=request.config,
            description=request.description,
            tags=request.tags
        )

        logger.info(f"Saved configuration: {request.name}")

        return {
            "status": "success",
            "message": f"Configuration '{request.name}' saved successfully",
            "filepath": filepath
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@app.get("/configs/list")
async def list_configs(tag: Optional[str] = None):
    """
    List all saved training configurations.

    Args:
        tag: Optional tag to filter configurations
    """
    try:
        configs = config_manager.list_configs(tag=tag)
        return {
            "status": "success",
            "configs": configs,
            "count": len(configs)
        }

    except Exception as e:
        logger.error(f"Failed to list configurations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list configurations: {str(e)}")


@app.get("/configs/{name}")
async def get_config(name: str, include_metadata: bool = False):
    """
    Load a saved training configuration.

    Args:
        name: Name of the configuration to load
        include_metadata: If True, include metadata in response (default: False)
    """
    try:
        if include_metadata:
            config = config_manager.load_config(name)
        else:
            config = config_manager.get_config_without_metadata(name)

        return {
            "status": "success",
            "config": config
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Configuration '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")


@app.delete("/configs/{name}")
async def delete_config(name: str):
    """
    Delete a saved training configuration.

    Args:
        name: Name of the configuration to delete
    """
    try:
        deleted = config_manager.delete_config(name)

        if deleted:
            logger.info(f"Deleted configuration: {name}")
            return {
                "status": "success",
                "message": f"Configuration '{name}' deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Configuration '{name}' not found")

    except Exception as e:
        logger.error(f"Failed to delete configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete configuration: {str(e)}")


class ExportConfigRequest(BaseModel):
    """Request to export a configuration"""
    name: str = Field(..., description="Name of the configuration to export")
    output_path: str = Field(..., description="Path to export the configuration to")


@app.post("/configs/export")
async def export_config(request: ExportConfigRequest):
    """
    Export a configuration to a specific file path.
    """
    try:
        filepath = config_manager.export_config(request.name, request.output_path)

        logger.info(f"Exported configuration '{request.name}' to {filepath}")

        return {
            "status": "success",
            "message": f"Configuration exported to {filepath}",
            "filepath": filepath
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Configuration '{request.name}' not found")
    except Exception as e:
        logger.error(f"Failed to export configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export configuration: {str(e)}")


class ImportConfigRequest(BaseModel):
    """Request to import a configuration"""
    filepath: str = Field(..., description="Path to the configuration file to import")
    name: Optional[str] = Field(None, description="Optional name for the imported config")


@app.post("/configs/import")
async def import_config(request: ImportConfigRequest):
    """
    Import a configuration from an external file.
    """
    try:
        if not Path(request.filepath).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.filepath}")

        saved_path = config_manager.import_config(request.filepath, request.name)

        logger.info(f"Imported configuration from {request.filepath}")

        return {
            "status": "success",
            "message": "Configuration imported successfully",
            "filepath": saved_path
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to import configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import configuration: {str(e)}")


# ===== Model Preload Endpoints =====

class ModelPreloadRequest(BaseModel):
    """Request to preload a model's tokenizer"""
    model_name: str = Field(..., description="HuggingFace model ID")
    hf_token: Optional[str] = Field(None, description="HuggingFace API token (for gated models)")


@app.post("/model/preload")
async def preload_model_tokenizer(request: ModelPreloadRequest):
    """
    Preload a model's tokenizer for dataset preview.
    This downloads the model files and caches the tokenizer for fast preview.
    """
    try:
        model_name = request.model_name

        # Check if already cached
        if model_name in tokenizer_cache:
            tokenizer = tokenizer_cache[model_name]
            logger.info(f"Using cached tokenizer for {model_name}")
        else:
            logger.info(f"Loading tokenizer for {model_name}...")

            # Set HF token if provided
            if request.hf_token:
                os.environ["HF_TOKEN"] = request.hf_token

            # Load tokenizer (this will download model files if needed)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=request.hf_token
            )

            # Cache the tokenizer
            tokenizer_cache[model_name] = tokenizer
            logger.info(f"Tokenizer loaded and cached for {model_name}")

        # Extract useful info about the tokenizer
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

        return {
            "status": "success",
            "model_name": model_name,
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "has_chat_template": has_chat_template,
            "chat_template_preview": str(tokenizer.chat_template)[:200] + "..." if has_chat_template else None,
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token,
        }

    except Exception as e:
        logger.error(f"Failed to preload model tokenizer: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load tokenizer: {str(e)}")


@app.get("/model/cached")
async def list_cached_models():
    """List currently cached model tokenizers"""
    return {
        "cached_models": list(tokenizer_cache.keys()),
        "count": len(tokenizer_cache)
    }


if __name__ == "__main__":
    import uvicorn

    # Determine the URL for display
    display_url = settings.domain or f"http://localhost:{settings.port}"

    print("\n✨ Starting Merlina - Magical Model Training ✨")
    print(f"🧙‍♀️ Visit {display_url} to access the interface")
    print(f"📚 API documentation: {display_url}/api/docs")
    print(f"📁 Frontend directory: {FRONTEND_DIR}")
    print(f"💾 Database: {settings.database_path}")
    print(f"📊 Log level: {settings.log_level}")
    if settings.cuda_visible_devices:
        print(f"🎮 GPUs: {settings.cuda_visible_devices}")
    print("=" * 50 + "\n")

    uvicorn.run(app, host=settings.host, port=settings.port)