# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Merlina is a magical LLM training system with support for both ORPO (Odds Ratio Preference Optimization) and SFT (Supervised Fine-Tuning). It provides a delightful wizard-themed web interface for fine-tuning language models with LoRA adapters. The system supports flexible dataset loading from multiple sources and automatic chat template formatting.

**New in v1.2:**
- **SFT Mode**: Train with only chosen responses (rejected field not required)
- Dynamic UI that adapts based on selected training mode

**New in v1.1:**
- Persistent job storage with SQLite database
- Real-time WebSocket updates during training
- Pre-flight validation to catch errors before training
- Detailed metrics tracking and history
- Support for both HuggingFace model IDs and local model directories
- Private/public repository control for HuggingFace Hub uploads

## Development Commands

### Running the Application
```bash
python merlina.py
# Access UI at http://localhost:8000
# API docs at http://localhost:8000/api/docs
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run individual test files
python tests/test_tokenizer_formatter.py
python tests/test_dataset_loaders.py
python tests/test_pipeline.py
python tests/test_api_endpoints.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Code Architecture

### Core Components

**Backend (merlina.py)**
- FastAPI application serving REST API and static frontend
- Main entry point for the application
- Integrates all modules from `src/` and `dataset_handlers/`
- Background task execution using FastAPI BackgroundTasks

**Source Modules (src/)**
- `job_manager.py` - SQLite persistence for jobs and metrics
- `websocket_manager.py` - Real-time WebSocket connections
- `preflight_checks.py` - Configuration validation
- `training_runner.py` - Enhanced training with callbacks
- `job_queue.py` - Job queue with priority and concurrency control

**Dataset Handling Module (dataset_handlers/)**

The dataset system uses a modular strategy pattern with three main abstractions:

1. **DatasetLoader** (base.py, loaders.py)
   - Abstract base class for loading strategies
   - Implementations: `HuggingFaceLoader`, `LocalFileLoader`, `UploadedDatasetLoader`
   - Each loader handles a specific source type (HF Hub, local files, uploaded content)

2. **DatasetFormatter** (base.py, formatters.py)
   - Abstract base class for formatting strategies
   - Implementations: `ChatMLFormatter`, `Llama3Formatter`, `MistralFormatter`, `CustomFormatter`, `TokenizerFormatter`
   - `TokenizerFormatter` is special - it uses the model's native chat template from tokenizer_config.json
   - Factory function `get_formatter()` creates appropriate formatter instances

3. **DatasetPipeline** (base.py)
   - Orchestrates loading, validation, formatting, and train/test splitting
   - Provides `preview()` for raw data and `preview_formatted()` for formatted samples
   - Handles column mapping to standardize dataset schemas
   - Main interface: `prepare()` returns (train_dataset, eval_dataset)

**Frontend (frontend/)**
- Pure HTML/CSS/JS interface (no build step)
- Wizard theme with animations
- Dataset configuration UI with live preview capabilities
- Real-time job status polling

### Key Design Patterns

- **Strategy Pattern**: Interchangeable loaders and formatters
- **Factory Pattern**: `get_formatter()` creates formatter instances
- **Facade Pattern**: `DatasetPipeline` simplifies complex dataset operations
- **Dependency Injection**: Pipeline receives loader and formatter instances

### Dataset Schema

All datasets must have these columns after loading and column mapping:
- `prompt` (required) - User input/question
- `chosen` (required) - Preferred response
- `rejected` (required for ORPO, not used in SFT) - Non-preferred response
- `system` (optional) - System message
- `reasoning` (optional) - Reasoning/thinking process (used by Qwen3 format)

Formatters transform these into chat-formatted strings with proper special tokens.

**Training Mode Requirements**:
- **ORPO Mode**: Requires all three fields (`prompt`, `chosen`, `rejected`) for preference optimization
- **SFT Mode**: Only uses `prompt` and `chosen` fields. The `rejected` field is ignored if present.

**Reasoning Field**: When using the Qwen3 format, if a `reasoning` column is present, it will be wrapped in `<think>` tags and prepended to both the chosen and rejected responses. This allows training models with explicit reasoning steps.

### Model Loading

Merlina supports loading models from two sources:

**1. HuggingFace Hub** (default)
- Use standard HuggingFace model IDs: `"meta-llama/Meta-Llama-3-8B-Instruct"`
- Format: `organization/model-name`
- Automatically downloads models from huggingface.co
- Gated models (like Llama) require `hf_token` in configuration

**2. Local Directory Paths**
- Use absolute paths: `"/home/user/models/my-llama-model"`
- Use relative paths: `"./models/my-model"` or `"../models/my-model"`
- Model directory must contain required files (config.json, model weights)
- Useful for pre-downloaded models or custom fine-tuned models

**Path Detection Logic** (see `src/preflight_checks.py:is_local_model_path()`)
- Paths starting with `/`, `./`, or `../` are treated as local
- Paths with multiple slashes (e.g., `models/sub/folder`) are treated as local
- Paths with backslashes (Windows) are treated as local
- Existing directories with model files (config.json, *.safetensors) are treated as local
- Format `org/model` with single slash is treated as HuggingFace ID

**Pre-flight Validation**:
- Local paths: Validates directory exists and contains config.json
- HuggingFace models: Checks for gated models and required tokens
- Both: Automatically detected via `is_local_model_path()` function

### Job Queue System (v1.1)

Jobs are managed through a priority queue system with configurable concurrency:

**Job States:**
- `queued` - Job waiting in queue for execution
- `running`/`initializing`/`loading_model`/`loading_dataset`/`training` - Job executing
- `completed` - Job finished successfully
- `failed` - Job failed with error
- `stopped` - Job stopped by user request
- `cancelled` - Job cancelled while in queue

**Queue Features:**
- Priority levels: LOW, NORMAL, HIGH (higher priority jobs run first)
- Configurable max concurrent jobs (default: 1)
- Thread-safe job management with worker threads
- Automatic queue processing
- Job cancellation support (queued or running)
- Queue position tracking

**Configuration:**
- `MAX_CONCURRENT_JOBS` in `.env` (default: 1)
- Set to 1 for single GPU to prevent OOM
- Increase for multi-GPU systems with sufficient VRAM

**Key Classes:**
- `JobQueue` (src/job_queue.py) - Main queue manager
- `QueuedJob` - Represents a job in the queue
- `JobPriority` - Enum for priority levels

### Training Modes

Merlina supports two training modes, selectable via the `training_mode` configuration parameter:

**1. ORPO (Odds Ratio Preference Optimization)** - Default
- Requires: `prompt`, `chosen`, and `rejected` fields
- Trains the model to prefer "chosen" responses over "rejected" responses
- Uses preference optimization loss to push chosen responses up and rejected responses down
- Best for: Alignment, RLHF-style training, improving response quality
- Trainer: `ORPOTrainer` from TRL library
- Parameters: Includes `beta` parameter to control preference strength

**2. SFT (Supervised Fine-Tuning)**
- Requires: `prompt` and `chosen` fields only (rejected field ignored)
- Traditional supervised learning on chosen responses
- Trains the model to predict the chosen response given the prompt
- Best for: General instruction following, adapting to new tasks, style transfer
- Trainer: `SFTTrainer` from TRL library
- Parameters: Does not use `beta` parameter

**Choosing a Mode**:
- Use **ORPO** when you have paired chosen/rejected responses and want to optimize for preference
- Use **SFT** when you only have good examples or want traditional fine-tuning

### Training Flow

1. User submits training config via POST /train with optional priority
2. Job created with unique job_id and added to queue
3. Worker thread picks up job when slot available
4. `run_training()` executes:
   - Load model and tokenizer with optional 4-bit quantization
   - Create DatasetPipeline with appropriate loader and formatter
   - Call `pipeline.prepare()` to get formatted train/eval datasets
   - Setup LoRA config (if enabled)
   - Choose trainer based on `training_mode`:
     - **ORPO Mode**: Setup ORPOTrainer and train with preference optimization (uses chosen vs rejected)
     - **SFT Mode**: Setup SFTTrainer and train with supervised fine-tuning (uses only chosen)
   - Save to ./models/{output_name}
   - Optionally merge and push to HuggingFace Hub
4. Frontend polls GET /status/{job_id} for progress updates

### API Endpoints

**Training**
- POST /train?priority=normal - Submit training job to queue (priority: low/normal/high)
- GET /status/{job_id} - Get job progress and queue status
- GET /jobs - List all jobs
- POST /jobs/{job_id}/stop - Cancel queued job or stop running job

**Queue Management**
- GET /queue/status - Get overall queue statistics and job lists
- GET /queue/jobs - List queued and running jobs with positions

**Dataset Management**
- POST /dataset/preview - Preview raw dataset (10 samples)
- POST /dataset/preview-formatted - Preview with formatting applied (5 samples)
- POST /dataset/upload-file - Upload file, returns dataset_id
- GET /dataset/uploads - List uploaded datasets

## Important Implementation Notes

### Tokenizer Format

When using `format_type: "tokenizer"`, the system automatically uses the model's native chat template from `tokenizer_config.json`. This is the recommended approach as it ensures format compatibility with the base model.

The TokenizerFormatter:
- Uses `tokenizer.apply_chat_template()` with message dictionaries
- Adds generation prompt for the prompt field
- Extracts proper suffix formatting for chosen/rejected responses
- Falls back to simple concatenation if chat_template is missing

### Memory Management

The training function includes explicit cleanup:
```python
del trainer, model
gc.collect()
torch.cuda.empty_cache()
```

This is critical for running multiple sequential training jobs.

### 4-bit Quantization

When `use_4bit=True`:
- Uses BitsAndBytesConfig with NF4 quantization
- Model prepared with `prepare_model_for_kbit_training()`
- Significantly reduces VRAM requirements (7B model: ~10GB instead of ~28GB)

### Flash Attention

Automatically enabled for GPUs with compute capability >= 8 (Ampere and newer). Falls back to "eager" attention for older GPUs.

### HuggingFace Hub Privacy Control

When pushing models to HuggingFace Hub (`push_to_hub=True`):
- `hf_hub_private=True` (default): Creates **private repository** (only you can access)
- `hf_hub_private=False`: Creates **public repository** (anyone can access)
- Both model and tokenizer are pushed with the same privacy setting
- Requires valid `hf_token` for authentication
- Repository visibility can be changed later on HuggingFace.co

**Best Practices:**
- Use private repositories for proprietary or experimental models
- Use public repositories for open-source contributions
- Never commit HF tokens to version control
- Use tokens with appropriate write permissions

## Testing Strategy

Tests are structured for independent execution (not using pytest fixtures):
- Each test file can run standalone: `python tests/test_*.py`
- Tests use `sys.path.insert(0, '/path/to/merlina')` for imports
- Fixtures stored in `tests/fixtures/` directory
- Test data: `tests/fixtures/test_dataset.json`

## Common Workflows

### Adding a New Dataset Formatter

1. Create formatter class in `dataset_handlers/formatters.py`:
```python
class NewFormatter(DatasetFormatter):
    def format(self, row: dict) -> dict:
        # Transform row to chat format
        return {"prompt": ..., "chosen": ..., "rejected": ...}

    def get_format_info(self) -> dict:
        return {"format_type": "new_format", ...}
```

2. Add case to `get_formatter()` factory function
3. Update frontend dropdown in `frontend/index.html`
4. Create test in `tests/test_*.py`

### Adding a New Dataset Loader

1. Create loader class in `dataset_handlers/loaders.py`:
```python
class NewLoader(DatasetLoader):
    def load(self) -> Dataset:
        # Load from your source
        return dataset

    def get_source_info(self) -> dict:
        return {"source_type": "new_source", ...}
```

2. Add handling in `run_training()` and preview endpoints in `merlina.py`
3. Update Pydantic models if needed
4. Add UI controls in frontend

### Modifying Training Parameters

ORPO-specific parameters are in the `ORPOConfig` class:
- `beta` - Controls preference optimization strength (higher = stronger preference)
- `max_length` - Total sequence length (prompt + completion)
- `max_prompt_length` - Maximum prompt length
- `max_completion_length` - Calculated automatically

LoRA parameters are in `LoraConfig`:
- `r` - Rank (controls adapter size)
- `lora_alpha` - Scaling factor (typically 2x rank)
- `target_modules` - Which layers to apply LoRA (q_proj, v_proj, etc.)

## New Features (v1.1)

### Persistent Job Storage (job_manager.py)

Jobs are now stored in SQLite database (`./data/jobs.db`) and persist across server restarts.

**Key classes:**
- `JobManager` - Main interface for job CRUD operations
- `JobRecord` - Dataclass representing a job with all fields

**Important methods:**
- `create_job(job_id, config)` - Create new job
- `update_job(job_id, **kwargs)` - Update job fields
- `get_job(job_id)` - Retrieve job
- `list_jobs(status, limit, offset)` - Query jobs with filtering
- `add_metric(job_id, step, loss, ...)` - Record training metrics
- `get_metrics(job_id)` - Retrieve time-series metrics

**Database schema:**
- `jobs` table - Main job records
- `training_metrics` table - Time-series training data

### WebSocket Real-time Updates (websocket_manager.py)

Live training updates via WebSocket connections.

**Key class:**
- `WebSocketManager` - Manages connections and broadcasts

**Message types sent to clients:**
- `status_update` - Training status, progress, loss, GPU memory
- `metrics` - Detailed metrics at specific steps
- `completed` - Training finished successfully
- `error` - Training failed with error message

**WebSocket endpoint:**
- `GET /ws/{job_id}` - Connect to receive real-time updates

### Pre-flight Validation (preflight_checks.py)

Validates configuration before training to prevent wasted GPU time.

**Key class:**
- `PreflightValidator` - Runs all validation checks

**Checks performed:**
1. GPU availability and compute capability
2. VRAM estimation vs available memory
3. Disk space for checkpoints
4. Model access and gating (HF token check)
5. Dataset configuration validity
6. Training hyperparameter sanity checks
7. API token validation (W&B, HF)
8. Optional dependency checks (Flash Attention, xformers)

**Returns:**
- List of errors (blocking issues)
- List of warnings (recommendations)
- Detailed check results per category

### Enhanced Training Runner (training_runner.py)

New modular training function with WebSocket integration.

**Key features:**
- `WebSocketCallback` - Custom TRL callback that broadcasts metrics
- Async WebSocket updates during training
- Better error handling and logging
- GPU memory tracking
- Metric persistence to database

**Functions:**
- `run_training_sync(job_id, config, job_manager, uploaded_datasets)` - Main training function
- Called by FastAPI background tasks

## Project Structure

```
merlina/
├── merlina.py                    # Main FastAPI application
├── config.py                     # Configuration management
├── .env                          # User configuration (gitignored)
├── .env.example                  # Configuration template
├── src/                          # Source modules (v1.1)
│   ├── job_manager.py           # Job persistence
│   ├── websocket_manager.py     # WebSocket manager
│   ├── preflight_checks.py      # Validation
│   └── training_runner.py       # Training logic
├── dataset_handlers/             # Dataset pipeline
│   ├── base.py
│   ├── loaders.py
│   ├── formatters.py
│   └── validators.py
├── frontend/                     # Web UI
├── examples/                     # Example scripts
├── tests/                        # Test suite
└── data/                         # Runtime data
    └── jobs.db                   # SQLite database
```

## File Locations

- **Source code:** `src/` directory
- **Models:** `./models/{output_name}/`
- **Training artifacts:** `./results/{job_id}/`
- **Job database:** `./data/jobs.db` (SQLite)
- **Frontend:** `./frontend/`
- **Test fixtures:** `./tests/fixtures/`

## Configuration System

Merlina uses a centralized configuration system (`config.py`) with support for environment variables and `.env` files.

### Configuration Files

**`.env` file** (create from `.env.example`):
```bash
# Copy the example file to get started
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Key Settings:**
- **Server:** `HOST`, `PORT`, `DOMAIN` - Server binding and public URL
- **Paths:** `DATA_DIR`, `MODELS_DIR`, `DATABASE_PATH` - Storage locations
- **External Services:** `WANDB_API_KEY`, `HF_TOKEN` - API keys for W&B and HuggingFace
- **System:** `CUDA_VISIBLE_DEVICES`, `LOG_LEVEL` - GPU selection and logging
- **Security:** `CORS_ORIGINS`, `MAX_UPLOAD_SIZE_MB` - Access control and limits

### Frontend URL Detection

The frontend automatically detects the API URL from the browser's current location using `window.location`. This means:
- Works on `localhost` during development
- Works on any production domain (e.g., `https://merlina.example.com`)
- Works behind reverse proxies
- Automatically handles HTTP/HTTPS and WebSocket protocols

No frontend configuration needed!

## New API Endpoints

**Validation:**
- `POST /validate` - Validate configuration before training

**Job Management:**
- `GET /jobs/history?status=&limit=&offset=` - Get paginated job history
- `GET /jobs/{job_id}/metrics` - Get detailed training metrics
- `GET /stats` - Database and system statistics

**WebSocket:**
- `WebSocket /ws/{job_id}` - Real-time training updates

## Testing New Features

```bash
# Test validation
python examples/validate_and_train.py

# Monitor training via WebSocket
python examples/websocket_monitor.py <job_id>

# View job history and metrics
python examples/job_history.py
python examples/job_history.py <job_id>
```
