# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Merlina is a magical LLM training system with ORPO (Odds Ratio Preference Optimization) support. It provides a delightful wizard-themed web interface for fine-tuning language models with LoRA adapters. The system supports flexible dataset loading from multiple sources and automatic chat template formatting.

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
- `run_training()` - Main training orchestrator that handles ORPO fine-tuning
- Job management with in-memory storage (jobs dict)
- Uploaded dataset storage (uploaded_datasets dict)
- Background task execution using FastAPI BackgroundTasks

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

## File Locations

- Models saved to: `./models/{output_name}/`
- Training artifacts: `./results/{job_id}/`
- Frontend files: `./frontend/` (or current directory as fallback)
- Test fixtures: `./tests/fixtures/`

## Environment Variables

Optional configuration:
- `WANDB_API_KEY` - Weights & Biases logging
- `HF_TOKEN` - HuggingFace Hub authentication
- `CUDA_VISIBLE_DEVICES` - GPU selection
