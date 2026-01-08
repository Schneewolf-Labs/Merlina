# merlina.py
"""
Merlina - Magical Model Training Backend
ORPO training for LLMs with a delightful interface
"""

import os
import gc
import torch
import wandb
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any
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
from trl import ORPOConfig, ORPOTrainer
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
uploaded_datasets = {}  # dataset_id -> (bytes content, filename)
_uploaded_datasets_lock = threading.Lock()

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

    # Training mode (affects schema validation)
    training_mode: str = Field("orpo", description="Training mode: 'sft' or 'orpo'. For SFT, rejected column is optional.")


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
            "GET /api/docs": "API documentation"
        }
    }

@app.get("/version")
async def get_version():
    """Get detailed version information"""
    return get_version_info()

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
async def preview_dataset(config: DatasetConfig):
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
            training_mode=config.training_mode
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
            training_mode=config.training_mode
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
        from datetime import datetime

        dataset_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(content).hexdigest()[:8]}"

        # Store in memory (in production, save to disk/S3)
        with _uploaded_datasets_lock:
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


if __name__ == "__main__":
    import uvicorn

    # Determine the URL for display
    display_url = settings.domain or f"http://localhost:{settings.port}"

    logger.info("Starting Merlina - Magical Model Training")
    logger.info(f"Visit {display_url} to access the interface")
    logger.info(f"API documentation: {display_url}/api/docs")
    logger.info(f"Frontend directory: {FRONTEND_DIR}")
    logger.info(f"Database: {settings.database_path}")
    logger.info(f"Log level: {settings.log_level}")
    if settings.cuda_visible_devices:
        logger.info(f"GPUs: {settings.cuda_visible_devices}")

    uvicorn.run(app, host=settings.host, port=settings.port)