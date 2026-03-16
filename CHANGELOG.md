# Changelog

All notable changes to Merlina will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [1.5.0] - 2026-03-15 "Grimoire Rewards"

### Added
- **GRPO Answer Match Reward (R1-style)**: Compare model-extracted answers against ground-truth values from dataset
  - Configurable answer extraction regex (supports `\boxed{}`, `#### ...`, `<answer>...</answer>`, etc.)
  - Case-sensitive/insensitive comparison
  - Requires mapping a ground-truth column to "answer" in dataset column mapping
- **GRPO Regex Reward**: Score completions based on matching a custom regex pattern
  - Great for enforcing structured output (e.g. `<think>...</think>` tags, JSON format)
- **GRPO Answer + Format Composite Reward**: Weighted combination of answer accuracy and format matching
  - Configurable accuracy/format weights (default: 0.8/0.2)
  - Mirrors DeepSeek-R1's dual reward signal approach
- New `src/reward_functions.py` module with factory pattern for building reward functions
- "Answer" column support in dataset column mapping for GRPO ground-truth data
- Frontend UI for configuring new reward types with contextual field visibility
- Frontend validation for answer column mapping when answer-based rewards are selected
- 31 new unit tests for all reward function types

### Changed
- GRPO reward function dropdown expanded from 3 to 6 options
- Reward function construction moved from inline closures to dedicated module
- Dataset column mapping now supports optional "answer" target column

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
