# Merlina API Documentation

**Version:** 1.1.0
**Base URL:** `http://localhost:8000` (configurable via `.env`)

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL & Configuration](#base-url--configuration)
4. [Common Concepts](#common-concepts)
5. [API Endpoints](#api-endpoints)
   - [General](#general)
   - [Training Job Management](#training-job-management)
   - [Queue Management](#queue-management)
   - [GPU Management](#gpu-management)
   - [Dataset Management](#dataset-management)
   - [Configuration Management](#configuration-management)
   - [Model Management](#model-management)
6. [WebSocket API](#websocket-api)
7. [Request/Response Schemas](#requestresponse-schemas)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

---

## Overview

The Merlina API provides a comprehensive interface for training language models using ORPO (Odds Ratio Preference Optimization) with LoRA adapters. The API supports:

- **Training Job Management:** Create, monitor, and control training jobs
- **Job Queue:** Priority-based job queue with configurable concurrency
- **Real-time Updates:** WebSocket connections for live training progress
- **Dataset Management:** Load from HuggingFace, local files, or upload custom datasets
- **GPU Management:** Monitor and select GPUs for training
- **Configuration Management:** Save and reuse training configurations
- **Persistent Storage:** All jobs persisted in SQLite database

### Interactive API Documentation

FastAPI provides interactive API documentation:

- **Swagger UI:** `/api/docs` - Interactive API explorer
- **ReDoc:** `/api/redoc` - Alternative documentation view

---

## Authentication

**Current Version:** No authentication required (v1.1.0)

For production deployments, it is recommended to:
- Use reverse proxy with authentication (nginx, Caddy, etc.)
- Restrict `CORS_ORIGINS` in `.env` file
- Implement API key authentication if needed

---

## Base URL & Configuration

The API base URL is configurable via environment variables in `.env`:

```bash
HOST=0.0.0.0              # Server bind address
PORT=8000                 # Server port
DOMAIN=                   # Public domain (e.g., https://merlina.example.com)
```

**Default Base URL:** `http://localhost:8000`

---

## Common Concepts

### Job States

Jobs progress through the following states:

| State | Description |
|-------|-------------|
| `queued` | Job waiting in queue for execution |
| `initializing` | Setting up training environment |
| `loading_model` | Loading base model and tokenizer |
| `loading_dataset` | Loading and formatting dataset |
| `training` | Active training in progress |
| `saving` | Saving trained model |
| `uploading` | Uploading to HuggingFace Hub |
| `completed` | Training finished successfully |
| `failed` | Training failed with error |
| `stopped` | Job stopped by user request |
| `cancelled` | Job cancelled while in queue |

### Job Priority

Jobs can be submitted with priority levels:

- `low` - Runs after normal and high priority jobs
- `normal` (default) - Standard priority
- `high` - Runs before normal and low priority jobs

### Dataset Sources

Three types of dataset sources are supported:

1. **HuggingFace Hub:** Load datasets from huggingface.co
2. **Local Files:** Load from local JSON/JSONL/CSV/Parquet files
3. **Uploaded Files:** Upload files via API

### Dataset Formats

Supported chat template formats:

- `chatml` - ChatML format (default)
- `llama3` - Llama 3 format
- `mistral` - Mistral format
- `qwen3` - Qwen 3 format (supports thinking mode)
- `tokenizer` - Uses model's native chat template
- `custom` - User-defined custom templates

---

## API Endpoints

### General

#### Get API Information

```http
GET /api
```

Returns basic API information and available endpoints.

**Response:**
```json
{
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
```

---

#### Get System Statistics

```http
GET /stats
```

Returns database, WebSocket, and queue statistics.

**Response:**
```json
{
  "database": {
    "total_jobs": 42,
    "total_metrics": 15234,
    "db_size_mb": 2.5
  },
  "websockets": {
    "total_connections": 3
  },
  "queue": {
    "queued_count": 2,
    "running_count": 1,
    "max_concurrent": 1,
    "available_slots": 0
  }
}
```

---

### Training Job Management

#### Validate Training Configuration

```http
POST /validate
```

Validates training configuration before starting. Checks GPU availability, VRAM, disk space, model access, and more.

**Request Body:**
```json
{
  "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "output_name": "my-llama-ft",
  "lora_r": 64,
  "lora_alpha": 32,
  "learning_rate": 5e-6,
  "num_epochs": 2,
  "batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_length": 2048,
  "max_prompt_length": 1024,
  "beta": 0.1,
  "use_4bit": true,
  "dataset": {
    "source": {
      "source_type": "huggingface",
      "repo_id": "schneewolflabs/Athanor-DPO",
      "split": "train"
    },
    "format": {
      "format_type": "chatml"
    }
  }
}
```

**Response:**
```json
{
  "valid": true,
  "results": {
    "errors": [],
    "warnings": [
      "Flash Attention 2 not available, falling back to eager attention"
    ],
    "checks": {
      "gpu_available": true,
      "vram_sufficient": true,
      "disk_space_ok": true,
      "model_accessible": true,
      "dataset_valid": true
    }
  }
}
```

**Status Codes:**
- `200 OK` - Validation completed
- `400 Bad Request` - Invalid configuration

---

#### Create Training Job

```http
POST /train?priority=normal
```

Creates and queues a new training job with automatic validation.

**Query Parameters:**
- `priority` (optional) - Job priority: `low`, `normal` (default), `high`

**Request Body:** Same as `/validate` (see [TrainingConfig Schema](#trainingconfig))

**Response:**
```json
{
  "job_id": "job_20250116_143022",
  "status": "queued",
  "message": "Training spell cast! Job job_20250116_143022 queued at position 1."
}
```

**Status Codes:**
- `200 OK` - Job created successfully
- `400 Bad Request` - Validation failed or invalid configuration

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/train?priority=high" \
  -H "Content-Type: application/json" \
  -d @training_config.json
```

---

#### Get Job Status

```http
GET /status/{job_id}
```

Returns current status and progress of a training job.

**Path Parameters:**
- `job_id` - Job identifier (e.g., `job_20250116_143022`)

**Response:**
```json
{
  "job_id": "job_20250116_143022",
  "status": "training",
  "progress": 0.45,
  "current_step": 234,
  "total_steps": 520,
  "loss": 0.234,
  "error": null,
  "wandb_url": "https://wandb.ai/user/project/runs/abc123",
  "queue_position": null,
  "queue_state": "running"
}
```

**Status Codes:**
- `200 OK` - Job found
- `404 Not Found` - Job not found

---

#### List All Jobs

```http
GET /jobs
```

Returns a summary of all jobs.

**Response:**
```json
{
  "job_20250116_143022": {
    "status": "training",
    "progress": 0.45
  },
  "job_20250116_120000": {
    "status": "completed",
    "progress": 1.0
  }
}
```

---

#### Get Job History

```http
GET /jobs/history?status=completed&limit=50&offset=0
```

Returns paginated job history with detailed information. Jobs persist across server restarts.

**Query Parameters:**
- `status` (optional) - Filter by status (e.g., `completed`, `failed`, `training`)
- `limit` (optional, default: 50) - Maximum number of jobs to return
- `offset` (optional, default: 0) - Number of jobs to skip

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_20250116_143022",
      "status": "completed",
      "progress": 1.0,
      "created_at": "2025-01-16T14:30:22",
      "updated_at": "2025-01-16T15:45:10",
      "output_dir": "./models/my-llama-ft",
      "error": null
    }
  ],
  "count": 1
}
```

---

#### Get Job Metrics

```http
GET /jobs/{job_id}/metrics
```

Returns detailed time-series training metrics for a job.

**Path Parameters:**
- `job_id` - Job identifier

**Response:**
```json
{
  "job_id": "job_20250116_143022",
  "metrics": [
    {
      "step": 1,
      "loss": 0.845,
      "eval_loss": 0.923,
      "learning_rate": 5e-6,
      "timestamp": "2025-01-16T14:35:00"
    },
    {
      "step": 10,
      "loss": 0.654,
      "eval_loss": 0.712,
      "learning_rate": 4.8e-6,
      "timestamp": "2025-01-16T14:37:00"
    }
  ],
  "count": 520
}
```

**Status Codes:**
- `200 OK` - Job found
- `404 Not Found` - Job not found

---

#### Stop/Cancel Job

```http
POST /jobs/{job_id}/stop
```

Cancels a queued job or requests graceful stop of a running job.

**Path Parameters:**
- `job_id` - Job identifier

**Response (Queued Job):**
```json
{
  "status": "success",
  "message": "Job job_20250116_143022 removed from queue",
  "job_id": "job_20250116_143022",
  "was_queued": true
}
```

**Response (Running Job):**
```json
{
  "status": "success",
  "message": "Stop request sent to job job_20250116_143022. Training will stop after current step.",
  "job_id": "job_20250116_143022",
  "was_queued": false
}
```

**Status Codes:**
- `200 OK` - Stop request processed
- `404 Not Found` - Job not found

---

#### Delete Job

```http
DELETE /jobs/{job_id}
```

Deletes a specific job and all associated metrics from the database.

**Path Parameters:**
- `job_id` - Job identifier

**Response:**
```json
{
  "status": "success",
  "message": "Job job_20250116_143022 deleted successfully",
  "job_id": "job_20250116_143022"
}
```

**Status Codes:**
- `200 OK` - Job deleted
- `404 Not Found` - Job not found

---

#### Clear All Jobs

```http
DELETE /jobs
```

Deletes all jobs and metrics from the database.

**Response:**
```json
{
  "status": "success",
  "message": "Cleared all jobs (42 jobs deleted)",
  "deleted_count": 42
}
```

---

### Queue Management

#### Get Queue Status

```http
GET /queue/status
```

Returns overall queue statistics and lists of queued and running jobs.

**Response:**
```json
{
  "stats": {
    "queued_count": 2,
    "running_count": 1,
    "max_concurrent": 1,
    "available_slots": 0
  },
  "queued_jobs": [
    {
      "job_id": "job_20250116_150000",
      "priority": 5,
      "position": 1
    },
    {
      "job_id": "job_20250116_151000",
      "priority": 3,
      "position": 2
    }
  ],
  "running_jobs": [
    {
      "job_id": "job_20250116_143022",
      "priority": 5
    }
  ]
}
```

---

#### List Queue Jobs

```http
GET /queue/jobs
```

Returns detailed information about queued and running jobs.

**Response:**
```json
{
  "queued": [
    {
      "job_id": "job_20250116_150000",
      "priority": 5,
      "position": 1
    }
  ],
  "running": [
    {
      "job_id": "job_20250116_143022",
      "priority": 5
    }
  ]
}
```

---

### GPU Management

#### List All GPUs

```http
GET /gpu/list
```

Lists all available GPUs with detailed information.

**Response:**
```json
{
  "status": "success",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 23120,
      "memory_used_mb": 1444,
      "utilization_percent": 5,
      "temperature_c": 42,
      "power_usage_w": 45.2,
      "compute_capability": "8.9"
    }
  ],
  "count": 1
}
```

**Response (No CUDA):**
```json
{
  "status": "no_cuda",
  "message": "CUDA is not available on this system",
  "gpus": []
}
```

---

#### Get GPU Information

```http
GET /gpu/{index}
```

Returns detailed information about a specific GPU.

**Path Parameters:**
- `index` - GPU index (0-based)

**Response:**
```json
{
  "status": "success",
  "gpu": {
    "index": 0,
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_mb": 24564,
    "memory_free_mb": 23120,
    "memory_used_mb": 1444,
    "utilization_percent": 5,
    "temperature_c": 42,
    "power_usage_w": 45.2,
    "compute_capability": "8.9"
  }
}
```

**Status Codes:**
- `200 OK` - GPU found
- `404 Not Found` - GPU not found or CUDA unavailable

---

#### Get Available GPUs

```http
GET /gpu/available?min_free_memory_mb=4000
```

Returns list of GPU indices with sufficient free memory.

**Query Parameters:**
- `min_free_memory_mb` (optional, default: 4000) - Minimum free memory in MB

**Response:**
```json
{
  "status": "success",
  "available_gpus": [0, 1],
  "count": 2,
  "min_free_memory_mb": 4000
}
```

---

#### Get Recommended GPU

```http
GET /gpu/recommended
```

Returns the GPU with the most free memory.

**Response:**
```json
{
  "status": "success",
  "recommended_index": 0,
  "gpu": {
    "index": 0,
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_mb": 24564,
    "memory_free_mb": 23120,
    "memory_used_mb": 1444,
    "utilization_percent": 5,
    "temperature_c": 42,
    "power_usage_w": 45.2,
    "compute_capability": "8.9"
  }
}
```

**Status Codes:**
- `200 OK` - GPU found
- `404 Not Found` - No GPUs available

---

### Dataset Management

#### Preview Dataset (Raw)

```http
POST /dataset/preview
```

Previews raw dataset without formatting (up to 10 samples).

**Request Body:**
```json
{
  "source": {
    "source_type": "huggingface",
    "repo_id": "schneewolflabs/Athanor-DPO",
    "split": "train"
  },
  "format": {
    "format_type": "chatml"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "samples": [
    {
      "prompt": "What is the capital of France?",
      "chosen": "The capital of France is Paris.",
      "rejected": "The capital of France is London.",
      "system": "You are a helpful assistant."
    }
  ],
  "num_samples": 10
}
```

**Status Codes:**
- `200 OK` - Preview successful
- `400 Bad Request` - Invalid dataset configuration

---

#### Preview Dataset (Formatted)

```http
POST /dataset/preview-formatted
```

Previews dataset with formatting applied (up to 5 samples).

**Request Body:** Same as `/dataset/preview` (see [DatasetConfig Schema](#datasetconfig))

**Response:**
```json
{
  "status": "success",
  "samples": [
    {
      "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
      "chosen": "The capital of France is Paris.<|im_end|>",
      "rejected": "The capital of France is London.<|im_end|>"
    }
  ],
  "num_samples": 5
}
```

---

#### Get Dataset Columns

```http
POST /dataset/columns
```

Returns column names and sample data for column mapping.

**Request Body:**
```json
{
  "source": {
    "source_type": "huggingface",
    "repo_id": "schneewolflabs/Athanor-DPO",
    "split": "train"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "columns": ["system", "prompt", "chosen", "rejected"],
  "samples": [
    {
      "system": "You are a helpful assistant.",
      "prompt": "What is the capital of France?",
      "chosen": "The capital of France is Paris.",
      "rejected": "The capital of France is London."
    }
  ],
  "total_rows": 10000
}
```

---

#### Upload Dataset File

```http
POST /dataset/upload-file
```

Uploads a dataset file and returns a dataset ID for later use.

**Request:** Multipart form data with file

**Response:**
```json
{
  "status": "success",
  "dataset_id": "upload_20250116_143022_a1b2c3d4",
  "filename": "my_dataset.jsonl",
  "size": 1048576
}
```

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/dataset/upload-file" \
  -F "file=@my_dataset.jsonl"
```

**Status Codes:**
- `200 OK` - Upload successful
- `400 Bad Request` - Upload failed

---

#### List Uploaded Datasets

```http
GET /dataset/uploads
```

Lists all uploaded datasets.

**Response:**
```json
{
  "datasets": [
    {
      "dataset_id": "upload_20250116_143022_a1b2c3d4",
      "filename": "my_dataset.jsonl",
      "size": 1048576
    }
  ]
}
```

---

### Configuration Management

#### Save Configuration

```http
POST /configs/save
```

Saves a training configuration for later reuse.

**Request Body:**
```json
{
  "name": "llama3-8b-default",
  "config": {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "lora_r": 64,
    "lora_alpha": 32,
    "learning_rate": 5e-6
  },
  "description": "Default configuration for Llama 3 8B fine-tuning",
  "tags": ["llama3", "8b", "default"]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration 'llama3-8b-default' saved successfully",
  "filepath": "/path/to/data/configs/llama3-8b-default.json"
}
```

**Status Codes:**
- `200 OK` - Configuration saved
- `400 Bad Request` - Invalid configuration

---

#### List Configurations

```http
GET /configs/list?tag=llama3
```

Lists all saved configurations with optional tag filtering.

**Query Parameters:**
- `tag` (optional) - Filter by tag

**Response:**
```json
{
  "status": "success",
  "configs": [
    {
      "name": "llama3-8b-default",
      "description": "Default configuration for Llama 3 8B fine-tuning",
      "tags": ["llama3", "8b", "default"],
      "created_at": "2025-01-16T14:30:22",
      "updated_at": "2025-01-16T14:30:22"
    }
  ],
  "count": 1
}
```

---

#### Get Configuration

```http
GET /configs/{name}?include_metadata=false
```

Loads a saved configuration.

**Path Parameters:**
- `name` - Configuration name

**Query Parameters:**
- `include_metadata` (optional, default: false) - Include metadata in response

**Response (without metadata):**
```json
{
  "status": "success",
  "config": {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "lora_r": 64,
    "lora_alpha": 32,
    "learning_rate": 5e-6
  }
}
```

**Response (with metadata):**
```json
{
  "status": "success",
  "config": {
    "name": "llama3-8b-default",
    "description": "Default configuration for Llama 3 8B fine-tuning",
    "tags": ["llama3", "8b", "default"],
    "created_at": "2025-01-16T14:30:22",
    "updated_at": "2025-01-16T14:30:22",
    "config": {
      "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
      "lora_r": 64,
      "lora_alpha": 32,
      "learning_rate": 5e-6
    }
  }
}
```

**Status Codes:**
- `200 OK` - Configuration found
- `404 Not Found` - Configuration not found

---

#### Delete Configuration

```http
DELETE /configs/{name}
```

Deletes a saved configuration.

**Path Parameters:**
- `name` - Configuration name

**Response:**
```json
{
  "status": "success",
  "message": "Configuration 'llama3-8b-default' deleted successfully"
}
```

**Status Codes:**
- `200 OK` - Configuration deleted
- `404 Not Found` - Configuration not found

---

#### Export Configuration

```http
POST /configs/export
```

Exports a configuration to a specific file path.

**Request Body:**
```json
{
  "name": "llama3-8b-default",
  "output_path": "/path/to/export/config.json"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration exported to /path/to/export/config.json",
  "filepath": "/path/to/export/config.json"
}
```

**Status Codes:**
- `200 OK` - Configuration exported
- `404 Not Found` - Configuration not found

---

#### Import Configuration

```http
POST /configs/import
```

Imports a configuration from an external file.

**Request Body:**
```json
{
  "filepath": "/path/to/import/config.json",
  "name": "imported-config"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration imported successfully",
  "filepath": "/path/to/data/configs/imported-config.json"
}
```

**Status Codes:**
- `200 OK` - Configuration imported
- `404 Not Found` - File not found

---

### Model Management

#### Preload Model Tokenizer

```http
POST /model/preload
```

Preloads a model's tokenizer for dataset preview. Downloads model files and caches the tokenizer.

**Request Body:**
```json
{
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "hf_token": "hf_..."
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "vocab_size": 128256,
  "model_max_length": 8192,
  "has_chat_template": true,
  "chat_template_preview": "{% set loop_messages = messages %}...",
  "pad_token": "<|end_of_text|>",
  "eos_token": "<|end_of_text|>",
  "bos_token": "<|begin_of_text|>"
}
```

**Status Codes:**
- `200 OK` - Tokenizer loaded
- `400 Bad Request` - Failed to load tokenizer

---

#### List Cached Models

```http
GET /model/cached
```

Lists currently cached model tokenizers.

**Response:**
```json
{
  "cached_models": [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2"
  ],
  "count": 2
}
```

---

## WebSocket API

### Connect to Job Updates

```
WebSocket /ws/{job_id}
```

Establishes a WebSocket connection for real-time training updates.

**Path Parameters:**
- `job_id` - Job identifier

**Connection:**
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${job_id}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket closed');
};
```

### Message Types

#### Status Update

Sent periodically during training with progress and metrics.

```json
{
  "type": "status_update",
  "job_id": "job_20250116_143022",
  "status": "training",
  "progress": 0.45,
  "current_step": 234,
  "total_steps": 520,
  "loss": 0.234,
  "eval_loss": 0.256,
  "learning_rate": 4.8e-6,
  "gpu_memory": 12.5,
  "timestamp": 1705413022.123
}
```

#### Metrics Update

Sent at specific training steps with detailed metrics.

```json
{
  "type": "metrics",
  "job_id": "job_20250116_143022",
  "step": 100,
  "metrics": {
    "loss": 0.234,
    "eval_loss": 0.256,
    "learning_rate": 4.8e-6,
    "grad_norm": 0.15
  },
  "timestamp": 1705413022.123
}
```

#### Error

Sent when training fails.

```json
{
  "type": "error",
  "job_id": "job_20250116_143022",
  "error": "CUDA out of memory",
  "timestamp": 1705413022.123
}
```

#### Completion

Sent when training completes successfully.

```json
{
  "type": "completed",
  "job_id": "job_20250116_143022",
  "output_dir": "./models/my-llama-ft",
  "final_metrics": {
    "final_loss": 0.123,
    "final_eval_loss": 0.145
  },
  "timestamp": 1705413022.123
}
```

#### Heartbeat

Sent in response to client messages to maintain connection.

```json
{
  "type": "heartbeat",
  "status": "ok"
}
```

---

## Request/Response Schemas

### TrainingConfig

Complete training configuration schema.

```json
{
  "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "output_name": "my-llama-ft",

  "use_lora": true,
  "lora_r": 64,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"],

  "learning_rate": 5e-6,
  "num_epochs": 2,
  "batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_length": 2048,
  "max_prompt_length": 1024,
  "beta": 0.1,

  "seed": 42,
  "max_grad_norm": 0.3,
  "warmup_ratio": 0.05,
  "eval_steps": 0.2,
  "use_4bit": true,
  "use_wandb": true,
  "push_to_hub": false,
  "merge_lora_before_upload": true,
  "hf_hub_private": true,
  "hf_token": "hf_...",
  "wandb_key": "...",

  "shuffle_dataset": true,
  "weight_decay": 0.01,
  "lr_scheduler_type": "cosine",
  "gradient_checkpointing": false,
  "logging_steps": 1,

  "optimizer_type": "paged_adamw_8bit",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8,

  "attn_implementation": "auto",

  "gpu_ids": [0],

  "wandb_project": "merlina-training",
  "wandb_run_name": "llama3-ft-v1",
  "wandb_tags": ["llama3", "orpo"],
  "wandb_notes": "Fine-tuning Llama 3 with ORPO",

  "dataset": {
    "source": {
      "source_type": "huggingface",
      "repo_id": "schneewolflabs/Athanor-DPO",
      "split": "train"
    },
    "format": {
      "format_type": "chatml",
      "custom_templates": null,
      "enable_thinking": true
    },
    "model_name": null,
    "column_mapping": null,
    "test_size": 0.01,
    "max_samples": null
  }
}
```

### DatasetConfig

Dataset configuration schema.

```json
{
  "source": {
    "source_type": "huggingface",
    "repo_id": "schneewolflabs/Athanor-DPO",
    "split": "train",
    "file_path": null,
    "file_format": null,
    "dataset_id": null
  },
  "format": {
    "format_type": "chatml",
    "custom_templates": null,
    "enable_thinking": true
  },
  "model_name": null,
  "column_mapping": null,
  "test_size": 0.01,
  "max_samples": null
}
```

**Dataset Source Types:**

1. **HuggingFace:**
```json
{
  "source_type": "huggingface",
  "repo_id": "schneewolflabs/Athanor-DPO",
  "split": "train"
}
```

2. **Local File:**
```json
{
  "source_type": "local_file",
  "file_path": "/path/to/dataset.jsonl",
  "file_format": "jsonl"
}
```

3. **Uploaded:**
```json
{
  "source_type": "upload",
  "dataset_id": "upload_20250116_143022_a1b2c3d4",
  "file_format": "jsonl"
}
```

**Custom Format Example:**
```json
{
  "format_type": "custom",
  "custom_templates": {
    "system_prefix": "[SYSTEM] ",
    "system_suffix": "\n",
    "user_prefix": "[USER] ",
    "user_suffix": "\n",
    "assistant_prefix": "[ASSISTANT] ",
    "assistant_suffix": "\n"
  }
}
```

---

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "detail": "Error message or details object"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request parameters or validation failed |
| 404 | Not Found | Resource not found (job, config, GPU, etc.) |
| 500 | Internal Server Error | Server error occurred |

### Common Error Scenarios

#### Validation Failed (400)

```json
{
  "detail": {
    "message": "Training configuration validation failed",
    "errors": [
      "GPU not available",
      "Insufficient VRAM (required: 16GB, available: 8GB)"
    ],
    "warnings": [
      "Flash Attention 2 not available"
    ]
  }
}
```

#### Job Not Found (404)

```json
{
  "detail": "Job not found"
}
```

#### Configuration Not Found (404)

```json
{
  "detail": "Configuration 'my-config' not found"
}
```

#### GPU Not Available (404)

```json
{
  "detail": "CUDA is not available"
}
```

---

## Examples

### Complete Training Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Validate configuration
config = {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "output_name": "my-llama-ft",
    "lora_r": 64,
    "lora_alpha": 32,
    "learning_rate": 5e-6,
    "num_epochs": 2,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_length": 2048,
    "max_prompt_length": 1024,
    "beta": 0.1,
    "use_4bit": True,
    "dataset": {
        "source": {
            "source_type": "huggingface",
            "repo_id": "schneewolflabs/Athanor-DPO",
            "split": "train"
        },
        "format": {
            "format_type": "chatml"
        }
    }
}

# Validate
response = requests.post(f"{BASE_URL}/validate", json=config)
validation = response.json()

if not validation["valid"]:
    print("Validation failed:", validation["results"]["errors"])
    exit(1)

print("Validation passed!")

# 2. Create training job
response = requests.post(f"{BASE_URL}/train?priority=high", json=config)
job = response.json()
job_id = job["job_id"]

print(f"Job created: {job_id}")

# 3. Monitor progress
while True:
    response = requests.get(f"{BASE_URL}/status/{job_id}")
    status = response.json()

    print(f"Status: {status['status']}, Progress: {status['progress']:.2%}")

    if status["status"] in ["completed", "failed", "stopped"]:
        break

    time.sleep(5)

# 4. Get final metrics
if status["status"] == "completed":
    response = requests.get(f"{BASE_URL}/jobs/{job_id}/metrics")
    metrics = response.json()
    print(f"Training completed with {metrics['count']} metric points")
else:
    print(f"Training {status['status']}: {status.get('error', 'Unknown')}")
```

### WebSocket Monitoring

```python
import asyncio
import websockets
import json

async def monitor_job(job_id):
    uri = f"ws://localhost:8000/ws/{job_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to job {job_id}")

        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "status_update":
                print(f"Progress: {data['progress']:.2%}, Loss: {data.get('loss', 'N/A')}")

            elif data["type"] == "metrics":
                print(f"Step {data['step']}: {data['metrics']}")

            elif data["type"] == "completed":
                print(f"Training completed! Model saved to: {data['output_dir']}")
                break

            elif data["type"] == "error":
                print(f"Error: {data['error']}")
                break

# Run
asyncio.run(monitor_job("job_20250116_143022"))
```

### Upload and Train with Custom Dataset

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Upload dataset file
with open("my_dataset.jsonl", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/dataset/upload-file", files=files)
    upload = response.json()
    dataset_id = upload["dataset_id"]

print(f"Dataset uploaded: {dataset_id}")

# 2. Preview the dataset
preview_config = {
    "source": {
        "source_type": "upload",
        "dataset_id": dataset_id,
        "file_format": "jsonl"
    },
    "format": {
        "format_type": "chatml"
    }
}

response = requests.post(f"{BASE_URL}/dataset/preview", json=preview_config)
preview = response.json()
print(f"Dataset has {preview['num_samples']} samples")

# 3. Train with uploaded dataset
config = {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "output_name": "custom-ft",
    # ... other training parameters ...
    "dataset": preview_config
}

response = requests.post(f"{BASE_URL}/train", json=config)
job = response.json()
print(f"Training started: {job['job_id']}")
```

### GPU Selection

```python
import requests

BASE_URL = "http://localhost:8000"

# Get recommended GPU
response = requests.get(f"{BASE_URL}/gpu/recommended")
recommended = response.json()

gpu_index = recommended["recommended_index"]
print(f"Using GPU {gpu_index}: {recommended['gpu']['name']}")

# Configure training to use specific GPU
config = {
    # ... other parameters ...
    "gpu_ids": [gpu_index]
}

response = requests.post(f"{BASE_URL}/train", json=config)
```

### Configuration Management

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Save configuration
save_request = {
    "name": "llama3-default",
    "config": {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "lora_r": 64,
        "lora_alpha": 32,
        # ... other parameters ...
    },
    "description": "Default Llama 3 configuration",
    "tags": ["llama3", "default"]
}

response = requests.post(f"{BASE_URL}/configs/save", json=save_request)
print(response.json())

# 2. List configurations
response = requests.get(f"{BASE_URL}/configs/list?tag=llama3")
configs = response.json()
print(f"Found {configs['count']} configurations")

# 3. Load configuration
response = requests.get(f"{BASE_URL}/configs/llama3-default")
config = response.json()["config"]

# 4. Use loaded configuration for training
response = requests.post(f"{BASE_URL}/train", json=config)
```

---

## Rate Limits & Quotas

**Current Version:** No rate limits implemented (v1.1.0)

For production deployments, consider implementing:
- Request rate limiting
- Concurrent job limits per user
- Dataset upload size limits (configurable via `MAX_UPLOAD_SIZE_MB` in `.env`)

---

## Changelog

### Version 1.1.0
- Added persistent job storage with SQLite
- Added WebSocket real-time updates
- Added pre-flight validation
- Added job queue with priority support
- Added GPU management endpoints
- Added configuration management
- Added model tokenizer preloading

### Version 1.0.0
- Initial API release
- Basic training job management
- Dataset loading and formatting
- HuggingFace Hub integration

---

## Support

For issues, questions, or feature requests:
- **GitHub:** https://github.com/Schneewolf-Labs/Merlina
- **Documentation:** See `CLAUDE.md` for developer documentation
- **API Docs:** Visit `/api/docs` for interactive documentation
