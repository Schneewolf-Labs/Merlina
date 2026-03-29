# Changelog

All notable changes to Merlina will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.1] - 2026-03-28

### Fixed
- **VLM loading on transformers v5+**: `AutoModelForVision2Seq` was removed in transformers v5. Now imports `AutoModelForImageTextToText` (v5+) with fallback to `AutoModelForVision2Seq` (v4).
- Bumped minimum `transformers` requirement to `>=5.0.0`.

## [1.4.0] - 2026-03-24 "Crystal Ball"

### Added
- **VLM Support**: Auto-detects vision-language models (Qwen-VL, LLaVA, etc.) and loads with correct `AutoModelForVision2Seq` class, preserving vision capabilities. Manual override via `model_type` dropdown (Auto/CausalLM/VLM).
- **Suggested Settings Presets**: Paper-backed recommended hyperparameters for all 7 training methods. "Apply Suggested Settings" button fills in optimal learning rate, beta, epochs, etc. per method. Available via `GET /presets/{mode}` API.
- **Multi-Dataset Concatenation**: Add multiple HuggingFace datasets to combine for training. Each additional source supports its own column mapping. All sources concatenated before formatting and splitting.
- **Dataset Deduplication**: Remove duplicate samples before training with configurable strategies (prompt, chosen, prompt+chosen, exact match). Toggle in Advanced Options.
- **Dataset Previewer Navigation**: Browse dataset samples one at a time with Prev/Next buttons, jump-to-index, and position indicator. Preview endpoints support `offset`/`limit` pagination.
- **Section Navigation UI**: Sticky tabbed banner (Model, Dataset, Training, Jobs) replaces scrolling through one long page. Auto-switches to Jobs tab after submitting training.
- **Output Name Generator**: "Generate" button creates model names from base model + dataset + training method (e.g. `Qwen2.5-7B-Instruct-MyDataset-ORPO`).
- Per-source `column_mapping` field on `DatasetSource` for multi-dataset workflows.
- `GET /presets` endpoint listing all available presets.

### Changed
- SimPO suggested beta is 2.0 (was defaulting to 0.1, matching DPO — 20x too low per the SimPO paper).
- IPO suggested beta is 0.01 (was 0.1 — IPO's beta has inverted meaning, smaller = stronger margin).
- SFT suggested learning rate is 2e-4 for LoRA (was 5e-6 — LoRA rates should be ~10x full fine-tuning).
- KTO suggested batch_size is 4 (per-step batch must be >= 4 for stable KL estimation per the KTO paper).
- All preference methods suggest 1 epoch and lora_dropout=0 (paper consensus: preference methods overfit fast, dropout interferes with preference signal).
- Jobs section is always accessible via nav tab (no longer auto-hidden when empty).

### Fixed
- VLM models (e.g. `Qwen3_5ForConditionalGeneration`) were silently loaded as text-only `Qwen3_5ForCausalLM`, stripping vision components. Fixed in both training and LoRA merge paths.



## [1.3.0] - 2026-03-14 "Seven Spells"

### Added
- **Messages Format Support**: Automatic detection and conversion of the common "messages" chat dataset format
  - Multi-turn conversation support (user/assistant turns combined with double newlines)
  - System message extraction into dedicated `system` field
  - Toggleable via UI checkbox or `convert_messages_format` API parameter (enabled by default)
  - New module: `dataset_handlers/messages_converter.py`
- **DPO Mode** (Direct Preference Optimization): Log-ratio preference learning with `beta` and `label_smoothing` parameters
- **SimPO Mode** (Simple Preference Optimization): Reference-free DPO variant with length-normalized rewards and configurable `gamma` margin
- **CPO Mode** (Contrastive Preference Optimization): Reference-free contrastive learning with `label_smoothing` support
- **IPO Mode** (Identity Preference Optimization): Squared-loss DPO variant, more robust to noisy preferences
- **KTO Mode** (Kahneman-Tversky Optimization): Binary feedback optimization using prospect theory; works with unpaired data and optional rejected responses
- Dynamic hyperparameter UI: `gamma` field appears for SimPO, `label_smoothing` for DPO/CPO
- Messages format detection banner in dataset preview UI
- Example scripts for messages format usage
- Test coverage for messages format conversion and toggle behavior

### Changed
- Training mode selector expanded from 2 modes to 7 (ORPO, SFT, DPO, SimPO, CPO, IPO, KTO)
- `TrainingConfig` now accepts `beta`, `label_smoothing`, and `gamma` hyperparameters
- Dataset preview endpoints enhanced with messages format detection and conversion
- Frontend dynamically adapts parameter fields based on selected training mode



### Planned
- Semantic versioning system
- Version tracking and changelog management
- Automated version bumping tools

## [1.2.0] - 2024-12-15 "Magical Memories"

### Added
- **SFT Mode**: New Supervised Fine-Tuning mode alongside ORPO
  - Train with only chosen responses (rejected field not required)
  - Traditional supervised learning for instruction following
  - Configurable via `training_mode` parameter
- Dynamic UI that adapts based on selected training mode
- `SFTTrainer` integration from TRL library

### Changed
- Dataset requirements now flexible based on training mode
- UI displays relevant fields based on ORPO vs SFT mode selection
- Training pipeline automatically selects appropriate trainer

### Documentation
- Updated CLAUDE.md with training mode documentation
- Added SFT vs ORPO usage guidelines

## [1.1.0] - 2024-11-30 "Persistent Power"

### Added
- **Persistent Job Storage**: SQLite database for job history
  - `JobManager` class for CRUD operations
  - Job history survives server restarts
  - Training metrics time-series storage
- **Real-time WebSocket Updates**: Live training progress
  - WebSocket connections per job
  - Real-time status, metrics, and GPU memory updates
  - `WebSocketManager` for connection handling
- **Pre-flight Validation**: Configuration validation before training
  - GPU availability and VRAM checks
  - Disk space validation
  - Model access and gating checks
  - Dataset configuration validation
  - Hyperparameter sanity checks
- **Job Queue System**: Priority-based job queue
  - Configurable concurrent job limit
  - Priority levels: LOW, NORMAL, HIGH
  - Queue position tracking
  - Job cancellation support
- **Local Model Support**: Load models from local directories
  - Absolute and relative path support
  - Automatic path detection
  - Pre-flight validation for local paths
- **HuggingFace Hub Privacy Control**: Public/private repository selection
  - `hf_hub_private` parameter
  - Configurable during push_to_hub

### Changed
- Training logic moved to modular `training_runner.py`
- Enhanced error handling and logging
- GPU memory tracking during training
- Configuration system centralized in `config.py`

### Added - API Endpoints
- `POST /validate` - Validate configuration
- `GET /jobs/history` - Paginated job history
- `GET /jobs/{job_id}/metrics` - Detailed training metrics
- `GET /stats` - Database and system statistics
- `WebSocket /ws/{job_id}` - Real-time training updates
- `GET /queue/status` - Queue statistics
- `POST /jobs/{job_id}/stop` - Cancel/stop jobs

### Added - Files
- `src/job_manager.py` - Job persistence layer
- `src/websocket_manager.py` - WebSocket management
- `src/preflight_checks.py` - Configuration validation
- `src/training_runner.py` - Enhanced training logic
- `src/job_queue.py` - Job queue management
- `data/jobs.db` - SQLite database (runtime)

### Documentation
- Comprehensive v1.1 feature documentation in CLAUDE.md
- Example scripts for new features
- WebSocket integration guide

## [1.0.0] - 2024-11-01 "Magical Beginnings"

### Added
- **Initial Release**: ORPO training system
- **Dataset Pipeline**: Modular loader and formatter system
  - `DatasetLoader` abstraction (HuggingFace, Local, Upload)
  - `DatasetFormatter` abstraction (ChatML, Llama3, Mistral, Tokenizer)
  - `DatasetPipeline` orchestration
- **Web UI**: Wizard-themed interface
  - Dataset configuration and preview
  - Training job submission
  - Status polling and progress tracking
- **Training Features**:
  - ORPO (Odds Ratio Preference Optimization)
  - LoRA adapter support
  - 4-bit quantization for memory efficiency
  - Flash Attention support (Ampere GPUs and newer)
- **Model Support**:
  - HuggingFace Hub model loading
  - Automatic tokenizer chat template detection
  - Multi-format support (ChatML, Llama3, Mistral, etc.)
- **Dataset Sources**:
  - HuggingFace Hub datasets
  - Local JSON/JSONL files
  - Direct file upload
- **API Endpoints**:
  - `POST /train` - Submit training job
  - `GET /status/{job_id}` - Job status
  - `POST /dataset/preview` - Raw dataset preview
  - `POST /dataset/preview-formatted` - Formatted preview
  - `POST /dataset/upload-file` - File upload
- **Configuration**:
  - Environment variable support (.env)
  - Flexible training parameters
  - GPU selection and optimization settings

### Core Components
- `merlina.py` - FastAPI application
- `dataset_handlers/` - Dataset pipeline
  - `base.py` - Abstract interfaces
  - `loaders.py` - Dataset loaders
  - `formatters.py` - Format strategies
  - `validators.py` - Validation logic
- `frontend/` - Web interface (HTML/CSS/JS)
- `config.py` - Configuration management

### Documentation
- `CLAUDE.md` - Comprehensive development guide
- `README.md` - User documentation
- `.env.example` - Configuration template
- Example training scripts

---

## Version Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

## Semantic Versioning Guide

- **MAJOR (X.0.0)**: Breaking changes, incompatible API changes
- **MINOR (1.X.0)**: New features, backwards-compatible additions
- **PATCH (1.0.X)**: Bug fixes, backwards-compatible fixes

## Links

- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [Merlina Repository](https://github.com/Schneewolf-Labs/Merlina)
