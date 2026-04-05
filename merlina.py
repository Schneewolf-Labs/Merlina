# merlina.py
"""
Merlina - Magical Model Training Backend
ORPO training for LLMs with a delightful interface
"""

import os
import gc
import asyncio
import torch
import wandb
import logging
import threading
from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
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
from accelerate import Accelerator

# Import our custom dataset handling module
from dataset_handlers import (
    DatasetPipeline,
    HuggingFaceLoader,
    LocalFileLoader,
    UploadedDatasetLoader,
    get_formatter,
    create_loader_from_config,
    LoaderCreationError
)

# Import new modules for persistence, WebSockets, and validation
from src.job_manager import JobManager
from src.websocket_manager import websocket_manager
from src.preflight_checks import validate_config
from src.config_manager import ConfigManager
from src.job_queue import JobQueue, JobPriority
from src.gpu_utils import get_gpu_manager
from src.presets import get_preset, get_all_presets

# Import configuration
from config import settings

# Import version information
from version import __version__, get_version_info, get_version_string

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Train LLMs with ORPO, powered by magic ✨",
    version=__version__,
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
# Thread-safe access via lock
# Structure: dataset_id -> {"content": bytes, "filename": str, "uploaded_at": datetime, "size": int}
uploaded_datasets = {}
_uploaded_datasets_lock = threading.Lock()

# Default TTL for uploaded datasets (24 hours)
UPLOAD_TTL_HOURS = 24

# Global cache for preloaded tokenizers (for preview functionality)
# Thread-safe access via lock
tokenizer_cache = {}  # model_name -> tokenizer instance
_tokenizer_cache_lock = threading.Lock()

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

    # Per-source column mapping (overrides DatasetConfig.column_mapping for this source)
    column_mapping: Optional[dict] = Field(
        None,
        description="Map this dataset's columns to expected names (system, prompt, chosen, rejected)"
    )


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
            repo_id="schneewolflabs/Athanorlite-DPO",
            split="train"
        ),
        description="Primary dataset source configuration"
    )

    additional_sources: list[DatasetSource] = Field(
        default_factory=list,
        description="Additional dataset sources to concatenate with the primary source"
    )

    format: DatasetFormat = Field(
        default=DatasetFormat(format_type="chatml"),
        description="Dataset format configuration"
    )

    # Optional: model name for tokenizer-based formatting in preview
    model_name: Optional[str] = Field(None, description="Model name (required for tokenizer format preview)")

    # Column mapping for the primary source (if dataset uses different column names)
    column_mapping: Optional[dict] = Field(
        None,
        description="Map primary dataset columns to expected names (system, prompt, chosen, rejected). "
                    "Additional sources use their own column_mapping field."
    )

    # Messages format conversion
    convert_messages_format: bool = Field(
        True,
        description="Automatically detect and convert messages format to standard format"
    )

    # Deduplication
    deduplicate: bool = Field(False, description="Remove duplicate samples before training")
    dedupe_strategy: str = Field(
        "prompt_chosen",
        description="Deduplication strategy: 'prompt', 'chosen', 'prompt_chosen', or 'exact'"
    )

    # Additional options
    test_size: float = Field(0.01, ge=0.001, le=0.5, description="Fraction of data for evaluation")
    max_samples: Optional[int] = Field(None, description="Limit dataset size (for testing)")

    # Training mode (affects schema validation)
    training_mode: str = Field("orpo", description="Training mode: 'sft', 'orpo', 'dpo', 'simpo', 'cpo', 'ipo', or 'kto'. For SFT/KTO, rejected column is optional.")


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
    modules_to_save: list[str] = Field(
        default=[],
        description="Modules to fully fine-tune (e.g., embed_tokens, lm_head). Not LoRA-adapted, but saved with adapter."
    )
    lora_task_type: str = Field(
        "CAUSAL_LM",
        description="PEFT task type for LoRA: 'CAUSAL_LM' (text generation, default), 'SEQ_2_SEQ_LM' (encoder-decoder), or 'FEATURE_EXTRACTION' (embeddings only)"
    )

    # Training hyperparameters
    learning_rate: float = Field(5e-6, ge=1e-8, le=1e-3)
    num_epochs: int = Field(2, ge=1, le=10)
    batch_size: int = Field(1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(16, ge=1, le=128)
    max_length: int = Field(2048, ge=512, le=8192)
    max_prompt_length: int = Field(1024, ge=256, le=4096)

    # Model type
    model_type: str = Field(
        "auto",
        description="Model type: 'auto' (detect from config), 'causal_lm' (text-only LLM), or 'vlm' (vision-language model)"
    )

    # Training mode
    training_mode: str = Field(
        "orpo",
        description="Training mode: 'sft', 'orpo', 'dpo', 'simpo', 'cpo', 'ipo', or 'kto'"
    )

    # Preference optimization parameters
    beta: float = Field(0.1, ge=0.01, le=10.0, description="Beta parameter for preference optimization (ORPO, DPO, SimPO, CPO, IPO, KTO)")
    label_smoothing: float = Field(0.0, ge=0.0, le=0.5, description="Label smoothing for DPO/CPO loss")
    gamma: float = Field(0.5, ge=0.0, le=5.0, description="SimPO reward margin between chosen and rejected")

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
    eval_steps: float = Field(0.2, gt=0, description="<1 = ratio of total steps, >=1 = absolute step count")
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
        description="Optimizer type (paged_adamw_8bit, paged_adamw_32bit, adamw_8bit, adamw_torch, adamw_hf, adafactor, sgd, muon)"
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
    multi_gpu_strategy: Literal["auto", "ddp", "single"] = Field(
        "auto",
        description=(
            "Multi-GPU strategy: "
            "'auto' (DDP when >1 GPU, single-process otherwise), "
            "'ddp' (force Distributed Data Parallel via accelerate launch), "
            "'single' (single-process with device_map=auto, for model parallelism)"
        )
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


# === Inference Models ===

class InferenceLoadRequest(BaseModel):
    """Request to load a model for inference"""
    model_name: str = Field(..., description="HuggingFace model ID, local path, or name of a trained model in models/ directory")
    use_4bit: bool = Field(True, description="Load with 4-bit quantization")
    hf_token: Optional[str] = Field(None, description="HuggingFace token for gated models")

class InferenceChatRequest(BaseModel):
    """Request to chat with the loaded model"""
    messages: list[dict] = Field(..., description="Chat messages in OpenAI format [{role, content}]")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=200)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    do_sample: bool = Field(True, description="Use sampling (False = greedy decoding)")


def _get_inference_device(model):
    """Get the correct input device for a model loaded with device_map='auto'."""
    if hasattr(model, 'hf_device_map'):
        # Pick the device of the first module
        first_device = next(iter(model.hf_device_map.values()))
        return torch.device(first_device)
    return model.device


# Global inference model state (one model at a time)
_inference_state = {
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "is_lora": False,
    "base_model_name": None,
    "use_4bit": False,
}
_inference_lock = threading.Lock()

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

    @app.get("/css/{file_path:path}")
    async def serve_css_modules(file_path: str):
        """Serve CSS files from css/ directory"""
        css_file = FRONTEND_DIR / "css" / file_path
        if css_file.exists() and css_file.is_file():
            return FileResponse(css_file, media_type="text/css")
        raise HTTPException(status_code=404, detail="CSS file not found")

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
@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        - status: "healthy" or "degraded"
        - cuda_available: Whether CUDA/GPU is available
        - database: Database connection status
        - queue: Job queue status
        - uptime_seconds: Approximate uptime (if tracked)
    """
    health_status = "healthy"
    checks = {}

    # Check CUDA availability
    try:
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        checks["cuda"] = {
            "available": cuda_available,
            "gpu_count": gpu_count
        }
    except Exception as e:
        checks["cuda"] = {"available": False, "error": str(e)}
        health_status = "degraded"

    # Check database connectivity
    try:
        db_stats = job_manager.get_stats()
        checks["database"] = {
            "connected": True,
            "total_jobs": db_stats.get("total_jobs", 0)
        }
    except Exception as e:
        checks["database"] = {"connected": False, "error": str(e)}
        health_status = "degraded"

    # Check job queue
    try:
        queue_stats = job_queue.get_queue_stats()
        checks["queue"] = {
            "running": True,
            "queued_jobs": queue_stats.get("queued", 0),
            "running_jobs": queue_stats.get("running", 0)
        }
    except Exception as e:
        checks["queue"] = {"running": False, "error": str(e)}
        health_status = "degraded"

    # Check disk space (warn if low)
    try:
        import psutil
        disk = psutil.disk_usage(str(settings.models_dir.parent))
        free_gb = disk.free / (1024**3)
        checks["disk"] = {
            "free_gb": round(free_gb, 2),
            "warning": free_gb < 10
        }
        if free_gb < 5:
            health_status = "degraded"
    except Exception as e:
        checks["disk"] = {"error": str(e)}

    return {
        "status": health_status,
        "version": __version__,
        "checks": checks
    }


@app.get("/api")
async def api_info():
    return {
        "name": "Merlina",
        "version": __version__,
        "description": "Magical Model Training Backend",
        "endpoints": {
            "POST /train": "Start a new training job",
            "GET /status/{job_id}": "Get job status",
            "GET /jobs": "List all jobs",
            "GET /version": "Get version information",
            "GET /health": "Health check for monitoring",
            "GET /api/docs": "API documentation"
        }
    }

@app.get("/version")
async def get_version():
    """Get detailed version information"""
    return get_version_info()


@app.get("/presets")
async def list_presets():
    """List recommended presets for all training modes."""
    return get_all_presets()


@app.get("/presets/{training_mode}")
async def get_training_preset(training_mode: str):
    """Get recommended hyperparameters for a training mode.

    Returns paper-backed defaults for learning_rate, beta, epochs, etc.
    """
    preset = get_preset(training_mode)
    if preset is None:
        raise HTTPException(
            status_code=404,
            detail=f"No preset for training mode '{training_mode}'. "
                   f"Available: sft, orpo, dpo, simpo, cpo, ipo, kto"
        )
    return preset


@app.post("/validate", response_model=dict)
async def validate_training_config(config: TrainingConfig):
    """
    Validate training configuration before starting.
    Checks GPU, VRAM, disk space, model access, etc.
    """
    try:
        # Run validation in thread pool to avoid blocking when GPU is busy
        is_valid, results = await asyncio.wait_for(
            asyncio.to_thread(validate_config, config),
            timeout=30.0  # 30 second timeout
        )
        return {
            "valid": is_valid,
            "results": results
        }
    except asyncio.TimeoutError:
        logger.warning("Validation timed out - GPU may be busy with training")
        return {
            "valid": True,
            "results": {
                "valid": True,
                "warnings": ["Validation timed out - GPU busy with training. Some checks were skipped."],
                "errors": [],
                "checks": {"note": "Validation skipped due to timeout"}
            }
        }
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


def _make_training_callback(event_loop):
    """Create a training callback that auto-selects single-process or DDP.

    Returns a callback suitable for JobQueue.submit().
    """
    from src.training_runner import run_training_sync, run_training_distributed, _get_distributed_gpu_count

    def training_callback(job_id: str, config_dict: dict):
        from pydantic import TypeAdapter
        config_obj = TypeAdapter(TrainingConfig).validate_python(config_dict)

        strategy = config_obj.multi_gpu_strategy
        num_gpus = _get_distributed_gpu_count(config_obj)
        use_distributed = strategy != "single" and num_gpus > 1

        if use_distributed:
            logger.info(
                f"Using distributed DDP training with {num_gpus} GPUs "
                f"(strategy={strategy})"
            )
            run_training_distributed(
                job_id, config_obj, job_manager, uploaded_datasets, event_loop
            )
        else:
            run_training_sync(
                job_id, config_obj, job_manager, uploaded_datasets, event_loop
            )

    return training_callback


@app.post("/train", response_model=JobResponse)
async def create_training_job(config: TrainingConfig, priority: Optional[str] = "normal"):
    """
    Create and queue a training job.
    Runs pre-flight validation before queueing.

    Args:
        config: Training configuration
        priority: Job priority (low, normal, high) - default: normal
    """
    # Run pre-flight validation in a thread pool to avoid blocking when GPU is busy
    # This prevents timeouts when queuing jobs while training is running
    validation_skipped = False
    try:
        is_valid, validation_results = await asyncio.wait_for(
            asyncio.to_thread(validate_config, config),
            timeout=30.0  # 30 second timeout for validation
        )
    except asyncio.TimeoutError:
        # Validation timed out (likely GPU busy with training)
        # Allow queuing with a warning instead of blocking
        logger.warning("Validation timed out - GPU may be busy. Queueing job without full validation.")
        is_valid = True
        validation_skipped = True
        validation_results = {
            "valid": True,
            "warnings": ["Pre-flight validation skipped (timeout) - GPU busy with training"],
            "errors": []
        }

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

    # Get the current event loop for WebSocket updates
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    training_callback = _make_training_callback(loop)

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
                "error": job.error,
                "config_summary": {
                    "base_model": job.config.get("base_model", ""),
                    "output_name": job.config.get("output_name", ""),
                    "training_mode": job.config.get("training_mode", ""),
                } if job.config else None
            }
            for job in jobs_list
        ],
        "count": len(jobs_list)
    }


@app.get("/jobs/{job_id}/config")
async def get_job_config(job_id: str):
    """
    Get the training configuration used for a specific job.
    Useful for reusing a previous job's config as a starting point for a new training run.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": "success",
        "job_id": job_id,
        "config": job.config
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


@app.post("/jobs/{job_id}/retry", response_model=JobResponse)
async def retry_job(job_id: str, priority: Optional[str] = "normal"):
    """
    Retry a failed or stopped job with the same configuration.
    Creates a new job using the original job's config.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ("failed", "stopped"):
        raise HTTPException(
            status_code=400,
            detail=f"Only failed or stopped jobs can be retried. Job is '{job.status}'."
        )

    if not job.config:
        raise HTTPException(
            status_code=400,
            detail="Job has no stored configuration to retry."
        )

    # Re-submit using the stored config via the existing train endpoint logic
    from pydantic import TypeAdapter
    config = TypeAdapter(TrainingConfig).validate_python(job.config)

    # Parse priority
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH
    }
    job_priority = priority_map.get(priority.lower(), JobPriority.NORMAL)

    # Create new job
    new_job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_manager.create_job(new_job_id, config.model_dump())

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    training_callback = _make_training_callback(loop)

    position = job_queue.submit(
        job_id=new_job_id,
        config=config.model_dump(),
        callback=training_callback,
        priority=job_priority
    )

    return JobResponse(
        job_id=new_job_id,
        status="queued",
        message=f"Retrying spell! New job {new_job_id} queued at position {position} (retry of {job_id})."
    )


class UploadJobRequest(BaseModel):
    hf_token: str = Field(..., description="HuggingFace API token for authentication")
    output_name: Optional[str] = Field(None, description="Override repository name (defaults to original output_name)")
    merge_lora_before_upload: bool = Field(True, description="Merge LoRA with base model before uploading")
    hf_hub_private: bool = Field(True, description="Make HuggingFace Hub repository private")


@app.post("/jobs/{job_id}/upload", response_model=JobResponse)
async def upload_job(job_id: str, request: UploadJobRequest):
    """
    Upload or re-upload a completed/stopped job's model to HuggingFace Hub.
    The job must have saved model artifacts (output_dir must exist).
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ("completed", "stopped"):
        raise HTTPException(
            status_code=400,
            detail=f"Only completed or stopped jobs can be uploaded. Job is '{job.status}'."
        )

    if not job.output_dir:
        raise HTTPException(
            status_code=400,
            detail="Job has no saved model artifacts (output_dir is missing)."
        )

    output_dir = job.output_dir
    if not os.path.isdir(output_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Model directory not found: {output_dir}. The model files may have been deleted."
        )

    if not job.config:
        raise HTTPException(
            status_code=400,
            detail="Job has no stored configuration."
        )

    # Build a config object with upload-relevant fields from the original config,
    # overridden by the request parameters
    from pydantic import TypeAdapter
    config = TypeAdapter(TrainingConfig).validate_python(job.config)

    # Apply upload overrides
    config.hf_token = request.hf_token
    config.merge_lora_before_upload = request.merge_lora_before_upload
    config.hf_hub_private = request.hf_hub_private
    if request.output_name:
        config.output_name = request.output_name

    training_mode = config.training_mode

    # Get event loop for WebSocket updates
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    # Start upload in background thread
    from src.training_runner import _run_background_upload

    upload_thread = threading.Thread(
        target=_run_background_upload,
        args=(config, output_dir, training_mode, job_id, job_manager, event_loop),
        name=f"UploadThread-{job_id}",
        daemon=False
    )
    upload_thread.start()

    logger.info(f"📤 Background upload started for job {job_id} -> {config.output_name}")

    return JobResponse(
        job_id=job_id,
        status="uploading",
        message=f"Upload started for {config.output_name}. Model is being pushed to HuggingFace Hub."
    )


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
        await websocket_manager.disconnect(websocket, job_id)


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


# Dataset management endpoints
@app.post("/dataset/preview")
async def preview_dataset(config: DatasetConfig, offset: int = 0, limit: int = 10):
    """Preview dataset without formatting"""
    try:
        # Create loader using factory
        try:
            loader = create_loader_from_config(
                source_config=config.source,
                uploaded_datasets=uploaded_datasets
            )
        except LoaderCreationError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))

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
            max_samples=config.max_samples,
            training_mode=config.training_mode,
            convert_messages_format=config.convert_messages_format
        )

        # Preview raw data with offset and limit
        samples, total_count = pipeline.preview(num_samples=limit, offset=offset)

        return {
            "status": "success",
            "samples": samples,
            "num_samples": len(samples),
            "total_count": total_count,
            "offset": offset,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Dataset preview failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dataset/preview-formatted")
async def preview_formatted_dataset(config: DatasetConfig, offset: int = 0, limit: int = 5):
    """Preview dataset with formatting applied"""
    try:
        # Create loader using factory
        try:
            loader = create_loader_from_config(
                source_config=config.source,
                uploaded_datasets=uploaded_datasets
            )
        except LoaderCreationError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))

        # Get formatter
        formatter_type = config.format.format_type

        # For tokenizer format, check if we have a cached tokenizer (thread-safe)
        if formatter_type == 'tokenizer':
            with _tokenizer_cache_lock:
                cached_tokenizer = tokenizer_cache.get(config.model_name) if config.model_name else None

            if cached_tokenizer is not None:
                # Use the cached tokenizer
                logger.info(f"Using cached tokenizer for preview: {config.model_name}")
                formatter = get_formatter(
                    format_type='tokenizer',
                    tokenizer=cached_tokenizer
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
            max_samples=config.max_samples,
            training_mode=config.training_mode,
            convert_messages_format=config.convert_messages_format
        )

        # Preview formatted data with offset and limit
        samples, total_count = pipeline.preview_formatted(num_samples=limit, offset=offset)

        return {
            "status": "success",
            "samples": samples,
            "num_samples": len(samples),
            "total_count": total_count,
            "offset": offset,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Dataset preview failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dataset/stats")
async def get_dataset_stats(config: DatasetConfig):
    """Compute dataset statistics: row count, avg lengths, token estimates, class balance."""
    try:
        try:
            loader = create_loader_from_config(
                source_config=config.source,
                uploaded_datasets=uploaded_datasets
            )
        except LoaderCreationError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))

        formatter_type = config.format.format_type
        if formatter_type == 'tokenizer':
            formatter_type = 'chatml'

        formatter = get_formatter(
            format_type=formatter_type,
            custom_templates=config.format.custom_templates,
            enable_thinking=config.format.enable_thinking
        )

        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            column_mapping=config.column_mapping,
            test_size=config.test_size,
            max_samples=config.max_samples,
            training_mode=config.training_mode,
            convert_messages_format=config.convert_messages_format
        )

        stats = pipeline.compute_stats()
        return {"status": "success", **stats}

    except Exception as e:
        logger.error(f"Dataset stats failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/dataset/columns")
async def get_dataset_columns(config: DatasetConfig):
    """
    Get column names and sample data from dataset for mapping.
    Returns available columns and a few sample rows.
    """
    try:
        # Create loader using factory
        try:
            loader = create_loader_from_config(
                source_config=config.source,
                uploaded_datasets=uploaded_datasets
            )
        except LoaderCreationError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))

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
from fastapi import File, UploadFile as FastAPIUploadFile

@app.post("/dataset/upload-file")
async def upload_dataset_file(file: FastAPIUploadFile = File(...)):
    """Upload a dataset file and return dataset ID"""
    try:
        # Read file content
        content = await file.read()

        # Generate dataset ID
        import hashlib

        dataset_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(content).hexdigest()[:8]}"

        # Store in memory with metadata (in production, save to disk/S3)
        with _uploaded_datasets_lock:
            uploaded_datasets[dataset_id] = {
                "content": content,
                "filename": file.filename,
                "uploaded_at": datetime.now(),
                "size": len(content)
            }

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
    """List all uploaded datasets with TTL information"""
    from datetime import timedelta

    now = datetime.now()
    datasets_list = []

    with _uploaded_datasets_lock:
        for dataset_id, entry in uploaded_datasets.items():
            # Handle both old and new format
            if isinstance(entry, dict):
                uploaded_at = entry.get("uploaded_at", now)
                expires_at = uploaded_at + timedelta(hours=UPLOAD_TTL_HOURS)
                ttl_remaining = max(0, (expires_at - now).total_seconds())

                datasets_list.append({
                    "dataset_id": dataset_id,
                    "filename": entry["filename"],
                    "size": entry["size"],
                    "uploaded_at": uploaded_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "ttl_remaining_seconds": int(ttl_remaining)
                })
            else:
                # Legacy tuple format
                content, filename = entry
                datasets_list.append({
                    "dataset_id": dataset_id,
                    "filename": filename,
                    "size": len(content),
                    "uploaded_at": None,
                    "expires_at": None,
                    "ttl_remaining_seconds": None
                })

    return {"datasets": datasets_list}


@app.post("/dataset/cleanup")
async def cleanup_expired_uploads():
    """
    Clean up expired uploaded datasets based on TTL.
    Removes datasets older than UPLOAD_TTL_HOURS (default: 24 hours).
    """
    from datetime import timedelta

    now = datetime.now()
    expired_ids = []
    freed_bytes = 0

    with _uploaded_datasets_lock:
        for dataset_id, entry in list(uploaded_datasets.items()):
            if isinstance(entry, dict):
                uploaded_at = entry.get("uploaded_at")
                if uploaded_at:
                    age = now - uploaded_at
                    if age > timedelta(hours=UPLOAD_TTL_HOURS):
                        expired_ids.append(dataset_id)
                        freed_bytes += entry.get("size", 0)

        # Remove expired datasets
        for dataset_id in expired_ids:
            del uploaded_datasets[dataset_id]
            logger.info(f"Cleaned up expired upload: {dataset_id}")

    return {
        "status": "success",
        "cleaned_count": len(expired_ids),
        "cleaned_ids": expired_ids,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2)
    }


@app.delete("/dataset/uploads/{dataset_id}")
async def delete_uploaded_dataset(dataset_id: str):
    """Delete a specific uploaded dataset"""
    with _uploaded_datasets_lock:
        if dataset_id not in uploaded_datasets:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        entry = uploaded_datasets[dataset_id]
        size = entry.get("size", 0) if isinstance(entry, dict) else len(entry[0])
        del uploaded_datasets[dataset_id]

    logger.info(f"Deleted uploaded dataset: {dataset_id}")
    return {
        "status": "success",
        "message": f"Dataset '{dataset_id}' deleted",
        "freed_bytes": size
    }


@app.post("/cleanup/artifacts")
async def cleanup_failed_job_artifacts(max_age_hours: int = 72):
    """
    Clean up training artifacts from failed jobs.

    Args:
        max_age_hours: Delete artifacts from failed jobs older than this (default: 72 hours)

    This removes:
    - Results directories for failed/stopped jobs
    - Incomplete model checkpoints
    """
    import shutil
    from datetime import timedelta

    cleaned_dirs = []
    freed_bytes = 0
    errors = []

    # Get failed/stopped jobs older than max_age
    cutoff = datetime.now() - timedelta(hours=max_age_hours)

    failed_jobs = job_manager.list_jobs(status="failed", limit=1000)
    stopped_jobs = job_manager.list_jobs(status="stopped", limit=1000)

    for job in failed_jobs + stopped_jobs:
        try:
            # Parse job creation time
            if job.created_at:
                job_time = datetime.fromisoformat(job.created_at.replace('Z', '+00:00').replace('+00:00', ''))
                if job_time > cutoff:
                    continue  # Skip recent jobs

            # Check for results directory
            results_dir = Path(f"./results/{job.job_id}")
            if results_dir.exists():
                dir_size = sum(f.stat().st_size for f in results_dir.rglob('*') if f.is_file())
                shutil.rmtree(results_dir)
                cleaned_dirs.append(str(results_dir))
                freed_bytes += dir_size
                logger.info(f"Cleaned up results directory: {results_dir}")

        except Exception as e:
            errors.append(f"Error cleaning {job.job_id}: {str(e)}")
            logger.warning(f"Failed to clean up job {job.job_id}: {e}")

    return {
        "status": "success",
        "cleaned_directories": cleaned_dirs,
        "cleaned_count": len(cleaned_dirs),
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "errors": errors if errors else None
    }


@app.get("/cleanup/status")
async def get_cleanup_status():
    """
    Get information about cleanable resources.
    Shows expired uploads and failed job artifacts that can be cleaned.
    """
    from datetime import timedelta

    now = datetime.now()

    # Check expired uploads
    expired_uploads = []
    with _uploaded_datasets_lock:
        for dataset_id, entry in uploaded_datasets.items():
            if isinstance(entry, dict):
                uploaded_at = entry.get("uploaded_at")
                if uploaded_at:
                    age = now - uploaded_at
                    if age > timedelta(hours=UPLOAD_TTL_HOURS):
                        expired_uploads.append({
                            "dataset_id": dataset_id,
                            "filename": entry["filename"],
                            "size": entry["size"],
                            "age_hours": round(age.total_seconds() / 3600, 1)
                        })

    # Check failed job artifacts
    failed_jobs = job_manager.list_jobs(status="failed", limit=100)
    stopped_jobs = job_manager.list_jobs(status="stopped", limit=100)

    cleanable_artifacts = []
    for job in failed_jobs + stopped_jobs:
        results_dir = Path(f"./results/{job.job_id}")
        if results_dir.exists():
            try:
                dir_size = sum(f.stat().st_size for f in results_dir.rglob('*') if f.is_file())
                cleanable_artifacts.append({
                    "job_id": job.job_id,
                    "status": job.status,
                    "path": str(results_dir),
                    "size_mb": round(dir_size / (1024 * 1024), 2)
                })
            except Exception:
                pass

    return {
        "expired_uploads": {
            "count": len(expired_uploads),
            "items": expired_uploads,
            "total_bytes": sum(u["size"] for u in expired_uploads)
        },
        "failed_job_artifacts": {
            "count": len(cleanable_artifacts),
            "items": cleanable_artifacts,
            "total_mb": sum(a["size_mb"] for a in cleanable_artifacts)
        }
    }


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

    except HTTPException:
        raise
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

    except HTTPException:
        raise
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

        # Check if already cached (thread-safe)
        with _tokenizer_cache_lock:
            cached_tokenizer = tokenizer_cache.get(model_name)

        if cached_tokenizer is not None:
            tokenizer = cached_tokenizer
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

            # Cache the tokenizer (thread-safe)
            with _tokenizer_cache_lock:
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


class ModelLayersRequest(BaseModel):
    """Request to detect LoRA-compatible layers in a model"""
    model_name: str = Field(..., description="HuggingFace model ID or local path")
    hf_token: Optional[str] = Field(None, description="HuggingFace API token (for gated models)")


# Cache for model layer detection results
_layer_cache: Dict[str, dict] = {}
_layer_cache_lock = threading.Lock()


@app.post("/model/layers")
async def detect_model_layers(request: ModelLayersRequest):
    """
    Detect LoRA-compatible layers in a model.
    Returns all Linear layer names that can be targeted by LoRA.
    """
    import torch.nn as nn

    try:
        model_name = request.model_name

        # Check cache first
        with _layer_cache_lock:
            if model_name in _layer_cache:
                logger.info(f"Using cached layer info for {model_name}")
                return _layer_cache[model_name]

        logger.info(f"Detecting layers for {model_name}...")

        # Set HF token if provided
        if request.hf_token:
            os.environ["HF_TOKEN"] = request.hf_token

        # Load model with minimal memory footprint for layer detection
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load on CPU to minimize GPU memory
            trust_remote_code=True,
            token=request.hf_token,
            low_cpu_mem_usage=True,
        )

        # Find all unique layer name patterns (Linear and Embedding)
        linear_layer_names = set()
        embedding_layer_names = set()
        layer_details = []

        for name, module in model.named_modules():
            # Check for nn.Linear or any linear-like layer (quantized, custom, etc.)
            # Linear layers have in_features and out_features attributes
            is_linear = isinstance(module, nn.Linear) or (
                hasattr(module, 'in_features') and
                hasattr(module, 'out_features') and
                hasattr(module, 'weight')
            )

            # Check for nn.Embedding layers
            is_embedding = isinstance(module, nn.Embedding) or (
                hasattr(module, 'num_embeddings') and
                hasattr(module, 'embedding_dim') and
                hasattr(module, 'weight')
            )

            if is_linear:
                # Extract the layer type name (e.g., "q_proj" from "model.layers.0.self_attn.q_proj")
                layer_type = name.split(".")[-1]
                linear_layer_names.add(layer_type)

                # Collect detailed info for first occurrence of each type
                if not any(d["name"] == layer_type for d in layer_details):
                    layer_details.append({
                        "name": layer_type,
                        "full_path_example": name,
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "layer_class": "linear",
                    })

            elif is_embedding:
                layer_type = name.split(".")[-1]
                embedding_layer_names.add(layer_type)

                if not any(d["name"] == layer_type for d in layer_details):
                    layer_details.append({
                        "name": layer_type,
                        "full_path_example": name,
                        "in_features": module.num_embeddings,
                        "out_features": module.embedding_dim,
                        "layer_class": "embedding",
                    })

        # Combine all trainable layers
        all_layer_names = linear_layer_names | embedding_layer_names
        sorted_layers = sorted(all_layer_names)

        # Categorize layers into common groups
        attention_layers = [l for l in sorted_layers if any(x in l for x in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "attn"])]
        mlp_layers = [l for l in sorted_layers if any(x in l for x in ["up_proj", "down_proj", "gate_proj", "fc1", "fc2", "mlp", "dense"])]
        # Embedding category: actual nn.Embedding layers + lm_head (output projection)
        embedding_layers = sorted(embedding_layer_names | {l for l in linear_layer_names if "lm_head" in l})
        other_layers = [l for l in sorted_layers if l not in attention_layers + mlp_layers + embedding_layers]

        # Default recommended layers (common LoRA targets - excluding embeddings)
        default_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        recommended = [l for l in sorted_layers if l in default_targets]

        # If no standard layers found, recommend attention-like layers
        if not recommended:
            recommended = attention_layers[:4] if attention_layers else sorted_layers[:4]

        result = {
            "status": "success",
            "model_name": model_name,
            "layers": sorted_layers,
            "layer_details": sorted(layer_details, key=lambda x: x["name"]),
            "categories": {
                "attention": attention_layers,
                "mlp": mlp_layers,
                "embedding": embedding_layers,
                "other": other_layers,
            },
            "recommended": recommended,
            "total_layers": len(sorted_layers),
            "total_linear": len(linear_layer_names),
            "total_embedding": len(embedding_layer_names),
        }

        # Cache the result
        with _layer_cache_lock:
            _layer_cache[model_name] = result

        # Clean up model to free memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Detected {len(linear_layer_names)} linear + {len(embedding_layer_names)} embedding layers in {model_name}")
        return result

    except Exception as e:
        logger.error(f"Failed to detect model layers: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to detect layers: {str(e)}")


# ==================== Inference Endpoints ====================

@app.get("/inference/models")
async def list_inference_models():
    """List locally trained models available for inference"""
    if settings.models_dir.is_absolute():
        models_dir = settings.models_dir
    else:
        models_dir = SCRIPT_DIR / settings.models_dir
    local_models = []

    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            adapter_config_path = model_dir / "adapter_config.json"
            model_info = {
                "name": model_dir.name,
                "path": str(model_dir),
            }

            if adapter_config_path.exists():
                import json as _json
                try:
                    with open(adapter_config_path) as f:
                        adapter_cfg = _json.load(f)
                    model_info["is_lora"] = True
                    model_info["base_model"] = adapter_cfg.get(
                        "base_model_name_or_path", "unknown"
                    )
                    model_info["lora_rank"] = adapter_cfg.get(
                        "r", adapter_cfg.get("lora_alpha")
                    )
                except Exception:
                    model_info["is_lora"] = True
                    model_info["base_model"] = "unknown"
            else:
                # Could be a full model
                model_info["is_lora"] = False
                model_info["base_model"] = None

            local_models.append(model_info)

    return {"models": local_models}


@app.post("/inference/load")
async def load_inference_model(request: InferenceLoadRequest):
    """Load a model for inference."""

    with _inference_lock:
        # Unload existing model first
        if _inference_state["model"] is not None:
            del _inference_state["model"]
            _inference_state["model"] = None
            _inference_state["tokenizer"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model_name = request.model_name
    is_lora = False
    base_model_name = None

    # Check if it's a trained model from models/ directory
    if settings.models_dir.is_absolute():
        models_dir = settings.models_dir
    else:
        models_dir = SCRIPT_DIR / settings.models_dir
    local_adapter_path = models_dir / model_name

    has_adapter = (
        local_adapter_path.exists()
        and (local_adapter_path / "adapter_config.json").exists()
    )
    if has_adapter:
        # It's a LoRA adapter from our models/ directory
        import json as _json
        adapter_cfg_path = local_adapter_path / "adapter_config.json"
        with open(adapter_cfg_path) as f:
            adapter_cfg = _json.load(f)
        base_model_name = adapter_cfg.get(
            "base_model_name_or_path"
        )
        if not base_model_name:
            raise HTTPException(
                status_code=400,
                detail="adapter_config.json missing "
                       "base_model_name_or_path",
            )
        is_lora = True
        logger.info(
            f"Loading LoRA adapter '{model_name}' "
            f"on base model '{base_model_name}'"
        )
    else:
        # It's either a HuggingFace model ID or a local full model path
        base_model_name = model_name
        logger.info(f"Loading model: {model_name}")

    try:
        # Determine dtype
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # Quantization config
        bnb_config = None
        if request.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        has_local_tokenizer = (
            is_lora
            and (local_adapter_path / "tokenizer_config.json").exists()
        )
        tokenizer_source = (
            str(local_adapter_path) if has_local_tokenizer
            else base_model_name
        )
        tokenizer_kwargs = {"trust_remote_code": True}
        if request.hf_token:
            tokenizer_kwargs["token"] = request.hf_token

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model (use AutoModelForCausalLM for inference)
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        if request.hf_token:
            model_kwargs["token"] = request.hf_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, **model_kwargs
        )

        # Load LoRA adapter on top
        if is_lora:
            model = PeftModel.from_pretrained(model, str(local_adapter_path))
            logger.info(f"LoRA adapter loaded from {local_adapter_path}")

        model.eval()

        with _inference_lock:
            _inference_state["model"] = model
            _inference_state["tokenizer"] = tokenizer
            _inference_state["model_name"] = model_name
            _inference_state["is_lora"] = is_lora
            _inference_state["base_model_name"] = base_model_name
            _inference_state["use_4bit"] = request.use_4bit

        # Get memory usage
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB"

        return {
            "status": "loaded",
            "model_name": model_name,
            "is_lora": is_lora,
            "base_model": base_model_name,
            "use_4bit": request.use_4bit,
            "gpu_memory": gpu_mem,
        }

    except Exception as e:
        logger.error(f"Failed to load inference model: {e}")
        # Clean up on failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load model: {str(e)}",
        )


@app.post("/inference/unload")
async def unload_inference_model():
    """Unload the current inference model to free VRAM"""
    with _inference_lock:
        if _inference_state["model"] is None:
            return {"status": "no_model", "message": "No model is currently loaded"}

        model_name = _inference_state["model_name"]
        del _inference_state["model"]
        _inference_state["model"] = None
        _inference_state["tokenizer"] = None
        _inference_state["model_name"] = None
        _inference_state["is_lora"] = False
        _inference_state["base_model_name"] = None
        _inference_state["use_4bit"] = False

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "unloaded",
        "message": f"Model '{model_name}' unloaded",
    }


@app.get("/inference/status")
async def inference_status():
    """Get the current inference model status"""
    with _inference_lock:
        if _inference_state["model"] is None:
            return {"loaded": False}

        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB"

        return {
            "loaded": True,
            "model_name": _inference_state["model_name"],
            "is_lora": _inference_state["is_lora"],
            "base_model": _inference_state["base_model_name"],
            "use_4bit": _inference_state["use_4bit"],
            "gpu_memory": gpu_mem,
        }


@app.post("/inference/chat")
async def inference_chat(request: InferenceChatRequest):
    """Generate a response from the loaded model"""
    with _inference_lock:
        model = _inference_state["model"]
        tokenizer = _inference_state["tokenizer"]

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Use /inference/load first",
        )

    def _generate():
        """Run generation in a thread to avoid blocking."""
        text = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            text, return_tensors="pt"
        ).to(_get_inference_device(model))

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=(
                    request.temperature if request.do_sample
                    else 1.0
                ),
                top_p=(
                    request.top_p if request.do_sample
                    else 1.0
                ),
                top_k=(
                    request.top_k if request.do_sample
                    else 0
                ),
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ), len(new_tokens)

    try:
        response_text, n_tokens = await asyncio.to_thread(
            _generate
        )

        return {
            "response": response_text,
            "tokens_generated": n_tokens,
            "model_name": _inference_state["model_name"],
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )


@app.websocket("/ws-inference")
async def inference_stream(websocket: WebSocket):
    """Stream inference tokens via WebSocket"""
    from transformers import TextIteratorStreamer

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            with _inference_lock:
                model = _inference_state["model"]
                tokenizer = _inference_state["tokenizer"]

            if model is None or tokenizer is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "No model loaded",
                })
                continue

            messages = data.get("messages", [])
            max_new_tokens = data.get("max_new_tokens", 512)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 50)
            repetition_penalty = data.get("repetition_penalty", 1.1)
            do_sample = data.get("do_sample", True)

            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer(
                    text, return_tensors="pt"
                ).to(_get_inference_device(model))

                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                    "top_k": top_k if do_sample else 0,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "streamer": streamer,
                }

                # Use an asyncio queue to bridge the
                # blocking streamer and the async websocket
                token_queue = asyncio.Queue()

                def _generate_and_collect():
                    """Generate tokens and push to queue."""
                    try:
                        gen_thread = threading.Thread(
                            target=lambda: model.generate(
                                **generation_kwargs
                            )
                        )
                        gen_thread.start()
                        for text in streamer:
                            asyncio.run_coroutine_threadsafe(
                                token_queue.put(
                                    ("token", text)
                                ),
                                loop,
                            )
                        gen_thread.join()
                    except Exception as exc:
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put(("error", str(exc))),
                            loop,
                        )
                    finally:
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put(("done", None)),
                            loop,
                        )

                loop = asyncio.get_event_loop()
                bg = threading.Thread(
                    target=_generate_and_collect
                )
                bg.start()

                token_count = 0
                while True:
                    msg_type, payload = await token_queue.get()
                    if msg_type == "token":
                        token_count += 1
                        await websocket.send_json({
                            "type": "token",
                            "text": payload,
                        })
                    elif msg_type == "error":
                        await websocket.send_json({
                            "type": "error",
                            "message": payload,
                        })
                        break
                    else:  # done
                        break

                bg.join()
                await websocket.send_json({
                    "type": "done",
                    "tokens_generated": token_count,
                })

            except Exception as e:
                logger.error(f"Streaming inference failed: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        logger.info("Inference WebSocket disconnected")
    except Exception as e:
        logger.error(f"Inference WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    from config import validate_cuda_visible_devices

    # Determine the URL for display
    display_url = settings.domain or f"http://localhost:{settings.port}"

    logger.info("Starting Merlina - Magical Model Training")
    logger.info(f"Version: {__version__}")
    logger.info(f"Visit {display_url} to access the interface")
    logger.info(f"API documentation: {display_url}/api/docs")
    logger.info(f"Health check: {display_url}/health")
    logger.info(f"Frontend directory: {FRONTEND_DIR}")
    logger.info(f"Database: {settings.database_path}")
    logger.info(f"Log level: {settings.log_level}")

    # Check torch version compatibility
    torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if torch_version < (2, 5):
        logger.error(
            f"PyTorch {torch.__version__} is too old (need >= 2.5.0). "
            "Run: pip install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu128"
        )

    # GPU startup check
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        logger.info(f"CUDA available: {gpu_count} GPU(s) detected")
        for i, name in enumerate(gpu_names):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")

        if settings.cuda_visible_devices:
            logger.info(f"CUDA_VISIBLE_DEVICES: {settings.cuda_visible_devices}")
    else:
        logger.warning("CUDA not available - GPU training will not work")

    uvicorn.run(app, host=settings.host, port=settings.port)