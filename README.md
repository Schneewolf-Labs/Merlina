# 🧙‍♀️ Merlina - Magical Model Training

Train LLMs with ORPO, DPO, SimPO, CPO, IPO, KTO, and SFT using a delightful web interface powered by magic ✨

![Merlina Banner](https://img.shields.io/badge/Merlina-Magical%20Training-c042ff?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==)

[![Tests](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/tests.yml/badge.svg)](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/tests.yml)
[![Quick Tests](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/quick-test.yml/badge.svg)](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/quick-test.yml)
[![codecov](https://codecov.io/gh/Schneewolf-Labs/Merlina/branch/main/graph/badge.svg)](https://codecov.io/gh/Schneewolf-Labs/Merlina)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![API Coverage](https://img.shields.io/badge/API%20Coverage-100%25-brightgreen.svg)](tests/TEST_SUMMARY.md)

## Features

- 🎨 **Beautiful Web Interface** - Cute wizard-themed UI with animations
- 🧪 **7 Training Modes** - ORPO, DPO, SimPO, CPO, IPO, KTO, and SFT
- 💬 **Messages Format Support** - Automatic detection and conversion of common chat dataset formats
- 📚 **Flexible Datasets** - HuggingFace, upload files (JSON/CSV/Parquet), or local paths
- 📝 **Multiple Formats** - ChatML, Llama 3, Mistral, custom templates, or **automatic tokenizer-based formatting**
- 🤖 **Tokenizer Format** - Automatically uses the model's native chat template
- 🗜️ **4-bit Quantization** - Train large models on consumer GPUs
- 📊 **Real-time Monitoring** - WebSocket updates with live metrics and GPU stats
- 💾 **Persistent Job Storage** - SQLite database preserves jobs across restarts
- 📋 **Job Queue** - Priority-based queue with configurable concurrency
- ✅ **Pre-flight Validation** - Catch configuration errors before training starts
- 🤗 **HuggingFace Integration** - Push models directly to the Hub (public or private)
- 📈 **W&B Logging** - Detailed experiment tracking

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Schneewolf-Labs/Merlina.git
cd Merlina

pip install -r requirements.txt
```

> **GPU environments (RunPod, Colab, Lambda, etc.):** These come with torch
> pre-installed and matched to CUDA. The `requirements.txt` intentionally
> excludes torch/torchvision/torchaudio so it won't break your existing setup.
> Just run `pip install -r requirements.txt` and you're good to go.
>
> **No torch pre-installed?** Install it first with the correct CUDA version:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt
> ```

### 2. Configure (Optional)

```bash
cp .env.example .env
# Edit .env with your settings (HF token, W&B key, etc.)
```

### 3. Run Merlina

```bash
python merlina.py
```

Visit http://localhost:8000 and start training! 🎉

API docs are available at http://localhost:8000/api/docs.

## Training Modes

Merlina supports 7 training modes — pick the one that fits your data and goals:

| Mode | Description | Requires Rejected? |
|------|-------------|-------------------|
| **ORPO** | Odds Ratio Preference Optimization — single-pass preference + SFT | Yes |
| **DPO** | Direct Preference Optimization — log-ratio preference learning | Yes |
| **SimPO** | Simple Preference Optimization — reference-free with length normalization | Yes |
| **CPO** | Contrastive Preference Optimization — reference-free contrastive learning | Yes |
| **IPO** | Identity Preference Optimization — squared-loss DPO, robust to noise | Yes |
| **KTO** | Kahneman-Tversky Optimization — binary feedback, works with unpaired data | Optional |
| **SFT** | Supervised Fine-Tuning — train on good examples only | No |

**How to choose:**
- Have paired chosen/rejected responses? → **ORPO, DPO, SimPO, CPO, or IPO**
- Have binary feedback (thumbs up/down)? → **KTO**
- Only have good examples? → **SFT**

## Dataset Configuration

Merlina supports flexible dataset sources and formats.

**Sources:**
- **HuggingFace Hub** — Load any dataset from the Hub
- **Upload Files** — JSON, JSONL, CSV, or Parquet files
- **Local Path** — Use datasets from your filesystem

**Formats:**
- **Tokenizer** (Recommended) — Automatically uses the model's chat template
- ChatML, Llama 3, Mistral — Manual format selection
- Custom templates — Define your own format

**Messages Format** (New in v1.3):
Merlina automatically detects and converts datasets in the common "messages" format used by many chat datasets. Multi-turn conversations are supported. Toggle this on/off via the UI or the `convert_messages_format` API parameter.

See the **[Dataset Guide](docs/user/dataset-guide.md)** for detailed instructions.

## Training Configuration

The interface lets you configure:

- **Base Model** — HuggingFace model IDs or local directory paths
- **Training Mode** — ORPO, DPO, SimPO, CPO, IPO, KTO, or SFT
- **LoRA Settings** — Rank, alpha, dropout, target modules
- **Training Parameters** — Learning rate, epochs, batch size
- **Mode-specific Hyperparameters** — Beta, gamma (SimPO), label smoothing (DPO/CPO)
- **Options** — 4-bit quantization, W&B logging, HF Hub upload (public/private)

The UI dynamically adapts based on the selected training mode, showing only the relevant parameters.

## Directory Structure

```
merlina/
├── merlina.py                    # Backend server (main entry point)
├── config.py                     # Configuration management
├── version.py                    # Version information
├── bump_version.py               # Version bumping tool
├── requirements.txt              # Python dependencies
├── Makefile                      # Build/run shortcuts
├── .env.example                  # Configuration template
│
├── src/                          # Source modules
│   ├── job_manager.py            # SQLite job persistence
│   ├── job_queue.py              # Priority-based job queue
│   ├── websocket_manager.py      # Real-time updates
│   ├── preflight_checks.py       # Configuration validation
│   ├── training_runner.py        # Training logic
│   ├── config_manager.py         # Configuration management
│   ├── gpu_utils.py              # GPU detection and monitoring
│   ├── model_card.py             # Model card generation
│   ├── constants.py              # Shared constants
│   ├── exceptions.py             # Custom exceptions
│   └── utils.py                  # Utilities
│
├── dataset_handlers/             # Dataset pipeline
│   ├── base.py                   # Abstract interfaces
│   ├── loaders.py                # Dataset loaders (HF, local, upload)
│   ├── formatters.py             # Format strategies (ChatML, Llama3, etc.)
│   ├── messages_converter.py     # Messages format auto-conversion
│   ├── factory.py                # Formatter factory
│   └── validators.py             # Validation logic
│
├── frontend/                     # Web interface (no build step)
│   ├── index.html
│   ├── css/
│   └── js/
│
├── docs/                         # Documentation
│   ├── user/                     # User guides
│   └── dev/                      # Developer docs
│
├── examples/                     # Example scripts and configs
├── tests/                        # Test suite
└── data/                         # Runtime data (created automatically)
    └── jobs.db                   # SQLite database
```

## API Endpoints

**Training & Jobs:**
- `POST /train?priority=normal` — Submit training job (priority: low/normal/high)
- `GET /status/{job_id}` — Get job progress and queue status
- `GET /jobs` — List all jobs
- `GET /jobs/history` — Paginated job history
- `GET /jobs/{job_id}/metrics` — Detailed training metrics
- `POST /jobs/{job_id}/stop` — Cancel queued or stop running job

**Queue Management:**
- `GET /queue/status` — Queue statistics and job lists
- `GET /queue/jobs` — List queued and running jobs

**Dataset Management:**
- `POST /dataset/preview` — Preview raw dataset (10 samples)
- `POST /dataset/preview-formatted` — Preview with formatting applied
- `POST /dataset/upload-file` — Upload dataset file
- `GET /dataset/uploads` — List uploaded datasets

**Validation & Info:**
- `POST /validate` — Validate configuration before training
- `GET /version` — Current version info
- `GET /stats` — Database and system statistics
- `GET /api/docs` — Interactive API documentation

**WebSocket:**
- `WS /ws/{job_id}` — Real-time training updates

Full API documentation: **[API.md](API.md)**

## GPU Memory Requirements

With 4-bit quantization enabled (default):

| Model Size | VRAM Required |
|------------|---------------|
| 3B params  | ~6 GB         |
| 7B params  | ~10 GB        |
| 13B params | ~16 GB        |

## Environment Variables

Configure via `.env` file or environment variables:

```bash
# External services
WANDB_API_KEY=your_wandb_key       # Weights & Biases logging
HF_TOKEN=your_huggingface_token    # HuggingFace Hub access

# Server
HOST=0.0.0.0                       # Server bind address
PORT=8000                           # Server port

# System
CUDA_VISIBLE_DEVICES=0              # GPU selection
MAX_CONCURRENT_JOBS=1               # Queue concurrency (increase for multi-GPU)
LOG_LEVEL=INFO                      # Logging verbosity
```

See `.env.example` for all available options.

## Docker Support

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.10 python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python3", "merlina.py"]
```

Build and run:
```bash
docker build -t merlina .
docker run --gpus all -p 8000:8000 merlina
```

## Documentation

- 📖 **[Quick Start Guide](docs/user/quick-start.md)** — Get started in minutes
- 📚 **[Dataset Guide](docs/user/dataset-guide.md)** — Configure datasets (HuggingFace, local, uploads)
- 🤖 **[Tokenizer Format Guide](docs/user/tokenizer-format.md)** — Automatic chat formatting (recommended!)
- ⚙️ **[Configuration Management](docs/user/config_management.md)** — Save and reuse training configs
- 🆕 **[New Features Guide](docs/user/new-features.md)** — What's new in v1.1+
- 📋 **[Examples](examples/)** — Ready-to-use training configurations
- 📡 **[API Reference](API.md)** — Full REST and WebSocket API docs

**For Developers:**
- 🏗️ **[Developer Docs](docs/dev/)** — Implementation details and architecture
- 🧪 **[Tests](tests/)** — Test suite and fixtures

## Tips

- 🎯 Start with 1-2 epochs for testing
- 💾 Models are saved to `./models/{output_name}`
- 📊 Enable W&B for detailed metrics
- 🔧 Adjust LoRA rank based on your GPU memory
- ✨ Try the Konami code in the UI for a surprise!

## Credits

Created with 💜 by Schneewolf Labs

## License

MIT License — feel free to use for your magical model training!
