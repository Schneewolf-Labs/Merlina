# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Merlina is a magical LLM training system with ORPO (Odds Ratio Preference Optimization) support. It provides a delightful wizard-themed web interface for fine-tuning language models with LoRA adapters. The system supports flexible dataset loading from multiple sources and automatic chat template formatting.

**New in v1.1:**
- Persistent job storage with SQLite database
- Real-time WebSocket updates during training
- Pre-flight validation to catch errors before training
- Detailed metrics tracking and history

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
- `rejected` (required) - Non-preferred response
- `system` (optional) - System message

Formatters transform these into chat-formatted strings with proper special tokens.

### Training Flow

1. User submits training config via POST /train
2. Job created with unique job_id
3. `run_training()` executes in background:
   - Load model and tokenizer with optional 4-bit quantization
   - Create DatasetPipeline with appropriate loader and formatter
   - Call `pipeline.prepare()` to get formatted train/eval datasets
   - Setup LoRA config and ORPOTrainer
   - Train with ORPO preference optimization
   - Save to ./models/{output_name}
   - Optionally merge and push to HuggingFace Hub
4. Frontend polls GET /status/{job_id} for progress updates

### API Endpoints

**Training**
- POST /train - Start training job
- GET /status/{job_id} - Get job progress
- GET /jobs - List all jobs

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

## Environment Variables

Optional configuration:
- `WANDB_API_KEY` - Weights & Biases logging
- `HF_TOKEN` - HuggingFace Hub authentication
- `CUDA_VISIBLE_DEVICES` - GPU selection

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
