# 🧙‍♀️ Merlina - Magical Model Training

Train LLMs with ORPO, DPO, SimPO, CPO, IPO, KTO, and SFT using a delightful web interface powered by magic ✨

![Merlina Banner](https://img.shields.io/badge/Merlina-Magical%20Training-c042ff?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==)

[![Tests](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/tests.yml/badge.svg)](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/tests.yml)
[![Quick Tests](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/quick-test.yml/badge.svg)](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/quick-test.yml)
[![codecov](https://codecov.io/gh/Schneewolf-Labs/Merlina/branch/main/graph/badge.svg)](https://codecov.io/gh/Schneewolf-Labs/Merlina)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![API Coverage](https://img.shields.io/badge/API%20Coverage-100%25-brightgreen.svg)](tests/TEST_SUMMARY.md)

## Merlina 3 — Wishcraft ✨

Merlina 3 makes the Hugging Face Hub optional. Datasets, base models,
checkpoints and finished models can all live in your own S3-compatible
storage: MinIO on your LAN for private data, and Cloudflare R2 for anything a
rented GPU needs to reach. The same models train here, train or serve on
rented RunPod GPUs, and publish back to storage, with the Hub as one more
destination.

- **Your own storage.** Push or mirror a model once, then use it anywhere
  as `store:<name>`. Datasets live in the store as `swl://<name>@<rev>`.
  With `MODEL_MIRROR` / `DATASET_MIRROR`, Hub models and datasets are copied
  on first use and keep working through Hub outages and deletions
  ([models](docs/user/store-models.md), [storage tiers](docs/storage-tiers.md)).
- **Rented GPUs, with caps.** Remote training and vLLM serving on your own
  RunPod account. Every run gets a free dry-run plan first, then a hard
  runtime cap and spend cap, and the instance is always torn down. The
  **Remote GPUs** page plans, starts, watches and stops serve sessions, and
  can chat with them ([training](docs/user/remote-training.md),
  [inference](docs/user/remote-inference.md)).
- **Train → serve → eval.** Serve a stored model, a Hub repo, or a remote
  job's adapter behind an OpenAI-compatible endpoint, point your eval
  harness at it, and stop it.

Upgrading from 2.x? The storage settings were renamed (`merlina config
migrate-env` rewrites your `.env`), and `ARTIFACT_STORE` now defaults to R2
when R2 is configured. See [Upgrading from 2.x](CHANGELOG.md).

The repo also contains a separate hosted pilot app (accounts, prepaid
execution, a managed studio and an optional LLM advisor). Paid execution and
advisor calls are disabled by default, and the public service at
`merlina.swl.ai` is not launched by this release. See the
[implementation status](docs/dev/merlina-3.md) and the
[hosted pilot runbook](docs/operations/hosted-pilot.md).

## Features

- 🎨 **Beautiful Web Interface** - Cute wizard-themed UI with animations
- 🧪 **7 Training Modes** - ORPO, DPO, SimPO, CPO, IPO, KTO, and SFT
- 💬 **Messages Format Support** - Automatic detection and conversion of common chat dataset formats
- 📚 **Flexible Datasets** - HuggingFace, your own object store (`swl://`), upload files (JSON/CSV/Parquet), or local paths
- 🗄️ **Self-hosted Storage** - MinIO/R2 tiers for datasets, base models (`store:<name>`), run artifacts and published models, with Hub mirroring on first use
- 📝 **Multiple Formats** - ChatML, Llama 3, Mistral, custom templates, or **automatic tokenizer-based formatting**
- 🤖 **Tokenizer Format** - Automatically uses the model's native chat template
- 🗜️ **4-bit Quantization** - Train large models on consumer GPUs
- 🪄 **GGUF Export** - Quantize any trained model to GGUF via llama.cpp (Q4_K_M, Q5_K_M, Q8_0, F16, and more)
- 🔮 **llama-server Inference** - Swap between transformers (base + LoRA) and a llama-server GGUF backend from the UI
- ☁️ **Remote Training & Inference** - Train on rented RunPod GPUs, then serve the result with vLLM behind an OpenAI-compatible endpoint for evals. Runs are spend-capped and always torn down, and the Remote GPUs page manages serve sessions ([training](docs/user/remote-training.md), [inference](docs/user/remote-inference.md))
- 📦 **Export & Artifacts** - Dedicated Export section for post-hoc GGUF, HuggingFace uploads, and per-model artifact browsing / cleanup
- 📊 **Real-time Monitoring** - WebSocket updates with live metrics and GPU stats
- 💾 **Persistent Job Storage** - SQLite database preserves jobs across restarts
- 📋 **Job Queue** - Priority-based queue with configurable concurrency
- ✅ **Pre-flight Validation** - Catch configuration errors before training starts
- 🤗 **Publish anywhere** - Push to the HuggingFace Hub (public or private) and/or publish to your model store, with upload history for both
- 📈 **W&B Logging** - Detailed experiment tracking

## Quick Start

Pick your path — all of them end at the same web UI:

| | |
|---|---|
| 🐍 **pip** | `pip install merlina` then `merlina serve` |
| 🐳 **Docker** | `docker compose up` with GPU passthrough |
| ☁️ **RunPod** | One-line setup script or GHCR image — [guide](docs/user/runpod.md) |
| 📓 **Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Schneewolf-Labs/Merlina/blob/main/notebooks/Merlina_Colab.ipynb) free GPU, three cells |
| 🛠️ **From source** | `git clone` + `pip install -r requirements.txt` (below) |

> **About torch:** Merlina deliberately does **not** install
> torch/torchvision/torchaudio — GPU environments (RunPod, Colab, Lambda,
> etc.) ship CUDA-matched builds that pip would silently replace with broken
> generic ones. If you don't have torch yet, install it first:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
> ```
> torchvision is required for VLM training (Qwen-VL, LLaVA, etc.) — without
> it the model still trains, but the image-processor files are silently
> skipped and the uploaded checkpoint misses the vision side of the processor.

### Option A: pip install

```bash
pip install merlina          # add [vlm] / [diffusion] / [all] extras as needed
merlina serve                # then open http://localhost:8000
```

### Option B: Docker

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU access.

```bash
docker run --gpus all -p 8000:8000 \
  -v ./data:/app/data -v ./models:/app/models \
  ghcr.io/schneewolf-labs/merlina:latest
```

Or clone the repo and `docker compose up -d` (see `docker-compose.yml` for
volumes and `.env` wiring).

### Option C: From source

```bash
git clone https://github.com/Schneewolf-Labs/Merlina.git
cd Merlina

pip install -r requirements.txt

cp .env.example .env   # optional: HF token, W&B key, etc.

python merlina.py
```

Visit http://localhost:8000 and start training! 🎉

Interactive API docs (Swagger UI) are available at http://localhost:8000/docs, with ReDoc at http://localhost:8000/redoc and the raw OpenAPI spec at http://localhost:8000/openapi.json.

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
- **HuggingFace Hub** — Load any dataset from the Hub. With `DATASET_MIRROR=true`, each one is copied into your store on first use and served from there afterwards (remote runs too).
- **Internal store** — `swl://name` or `swl://name@rev` from MinIO/R2. Put your own files there with `merlina datasets put <file> <name>`, or save an upload with `POST /dataset/uploads/{id}/store`.
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

- **Base Model** — HuggingFace model IDs, `store:<name>` models from your object store, or local directory paths
- **Training Mode** — ORPO, DPO, SimPO, CPO, IPO, KTO, or SFT
- **LoRA Settings** — Rank, alpha, dropout, target modules
- **Training Parameters** — Learning rate, epochs, batch size
- **Mode-specific Hyperparameters** — Beta, gamma (SimPO), label smoothing (DPO/CPO)
- **Options** — 4-bit quantization, W&B logging, HF Hub upload (public/private), publish to the model store

The UI dynamically adapts based on the selected training mode, showing only the relevant parameters.

## Export & Quantization

After training, the **Export** section (step 6 in the nav) lets you work on any model under `./models/` without re-training. Three sub-panels:

- **🪄 GGUF Export** — Merge LoRA (if any) and produce one GGUF per selected quant (F16, Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q3_K_M, Q2_K). Progress streams live over WebSocket. Artifacts land in `./models/{name}/gguf/` with a `manifest.json` sidecar that the Inference picker reads.
- **🤗 HuggingFace Upload** — Repo ID override, visibility, commit message, license/tags/description, and include-what checkboxes (LoRA adapter, merged full model, GGUF files, README). A "last uploaded" banner shows whether local files have been modified since your last push, plus full history of every upload attempt (success + failure).
- **🗂️ Artifacts** — Categorized file browser (adapter, merged, GGUF, tokenizer, processor, config, readme) with sizes and per-file delete. Core files (`config.json`, `adapter_config.json`, tokenizer) are 🔒-protected from the API.

### llama.cpp integration

GGUF export and the llama-server inference backend are both powered by a local [llama.cpp](https://github.com/ggerganov/llama.cpp) install. Merlina finds the binaries in this order:

1. `LLAMA_CPP_DIR` — path to a llama.cpp checkout (must contain `convert_hf_to_gguf.py` and `build/bin/`)
2. `LLAMA_CPP_BIN_DIR` — path to just the built binaries
3. System `$PATH` — if `llama-quantize`, `llama-server`, etc. are installed globally
4. `./vendor/llama.cpp` — conventional local clone

If none are found, the GGUF and llama-server features stay disabled in the UI and pre-flight emits an advisory warning (never a blocking error). Check `GET /llama-cpp/status` to see what the resolver found.

### Inference backends

The Inference section exposes two backends:

- **transformers** — base model + LoRA adapter, runs on GPU with optional 4-bit quantization. The default.
- **llama_cpp** — GGUF served by `llama-server` as a subprocess. Picked automatically when you select a GGUF entry from the model dropdown (they appear as indented sub-options per model).

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
│   ├── training_runner.py        # Training logic + sync merge orchestration
│   ├── merge_artifact.py         # Ref-counted shared merged-model dir
│   ├── gguf_exporter.py          # LoRA merge + GGUF convert/quantize pipeline
│   ├── llama_cpp_resolver.py     # llama.cpp binary discovery
│   ├── llama_server.py           # llama-server subprocess inference backend
│   ├── upload_state.py           # Per-model HF upload history sidecar
│   ├── model_artifacts.py        # Artifact inventory + safe deletion
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
- `POST /dataset/uploads/{id}/store` — Save an upload into the internal store as `swl://name@rev`

**Export & Artifacts:**
- `GET /models/{name}/artifacts` — Categorized file inventory (adapter, merged, GGUF, tokenizer, etc.)
- `DELETE /models/{name}/artifacts?path=…` — Delete a file (protected core files refused)
- `GET /models/{name}/upload-state` — Upload history + "local is newer" freshness hint
- `POST /models/{name}/upload` — Post-hoc HuggingFace upload with advanced options
- `POST /models/{name}/export-gguf` — Post-hoc GGUF export (merges LoRA + runs llama-quantize)
- `POST /models/{name}/publish` — Publish a model (optionally merged) to the model store as `store:<name>`
- `GET /models/local` — Base models available without the Hub (HF cache, `./models`, model store)
- `GET /llama-cpp/status` — Resolver status (available binaries, supported quant types)

**Inference:**
- `GET /inference/models` — List local models (plus discovered GGUF artifacts per model)
- `POST /inference/load` — Load a model (`backend: "transformers" | "llama_cpp"`)
- `POST /inference/chat` — Send a chat completion to the loaded backend
- `POST /inference/unload` — Free VRAM / stop llama-server
- `GET /inference/status` — Current backend + model

**Remote Inference (rented GPUs, vLLM):**
- `POST /remote/inference/plan` — Dry run: GPU, $/hr, VRAM, deadline, max total cost
- `POST /remote/inference` — Start a session (HF repo, `store_model`, or a remote training job's adapter/merged artifact)
- `GET /remote/inference/sources` — Stored models and remote jobs that can be served
- `GET /remote/inference` — List sessions
- `GET /remote/inference/{id}` — Status, `base_url`, `served_model_name`, `api_key`
- `POST /remote/inference/{id}/chat` — One chat completion through the control plane
- `POST /remote/inference/{id}/stop` — Stop and terminate the instance

**Object Storage:**
- `GET /storage/tiers`, `/storage/buckets`, `/storage/ls`, `/storage/object`, `/storage/preview` — Browse tiers and buckets
- `POST /storage/sync` — Preview, then copy or move a prefix between tiers (private data never goes to a shared tier)

**Validation & Info:**
- `POST /validate` — Validate configuration before training
- `GET /version` — Current version info
- `GET /stats` — Database and system statistics
- `GET /docs` — Interactive API documentation (Swagger UI)
- `GET /redoc` — API documentation (ReDoc)
- `GET /openapi.json` — OpenAPI specification

**WebSocket:**
- `WS /ws/{job_id}` — Real-time training and export updates (status, metrics, `gguf_progress`)

Full API documentation: **[API.md](API.md)**

## Command Line

`merlina serve` runs the server. The other commands manage storage and work on any machine with the server's `.env`:

```bash
merlina models push ./models/my-model my-model     # upload a model dir as store:my-model
merlina models mirror Qwen/Qwen3-8B                 # copy a Hub model in, as store:hf/Qwen/Qwen3-8B
merlina models info my-model
merlina datasets put pairs.jsonl b3/pr-review       # -> swl://b3/pr-review@<content hash>
merlina datasets promote swl://name --to external   # internal (MinIO) -> external (R2); private data refused
merlina storage tiers | ls | du | sync | rm          # browse and move object storage
merlina config migrate-env [--write]                # rename 2.x storage settings in .env
```

## MCP Server

Merlina ships an **[MCP](https://modelcontextprotocol.io) server** so an LLM
agent (Claude Desktop, Claude Code, Cursor, …) can drive the workshop
conversationally — queue training jobs, watch progress, preview datasets, and
manage the GPU queue. It's a thin HTTP client to a running Merlina server, so
it pulls in no torch.

```bash
# 1. Start Merlina as usual
merlina serve            # or: python merlina.py

# 2. In another shell (or wired into your MCP client), run the MCP server
pip install 'merlina[mcp]'
MERLINA_API_URL=http://localhost:8000 merlina-mcp
```

Wire it into an MCP client (e.g. Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "merlina": {
      "command": "merlina-mcp",
      "env": { "MERLINA_API_URL": "http://localhost:8000" }
    }
  }
}
```

Exposed tools include `start_training`, `get_job_status`, `get_job_history`,
`get_job_metrics`, `stop_job`, `queue_status`, `validate_training_config`,
`preview_dataset`, `list_local_models`, `list_gpus`, and `get_stats`. See
**[docs/user/mcp.md](docs/user/mcp.md)** for the full reference.

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
# External services — setting these here means clients don't need to send
# tokens in HTTP request bodies. Tokens in requests still override env.
WANDB_API_KEY=your_wandb_key       # Weights & Biases logging
HF_TOKEN=your_huggingface_token    # HuggingFace Hub access (gated models, uploads)

# Server
HOST=0.0.0.0                       # Server bind address
PORT=8000                           # Server port

# System
CUDA_VISIBLE_DEVICES=0              # GPU selection
MAX_CONCURRENT_JOBS=1               # Queue concurrency (increase for multi-GPU)
LOG_LEVEL=INFO                      # Logging verbosity

# Object storage (optional; see docs/storage-tiers.md)
STORAGE_INTERNAL_ENDPOINT=http://127.0.0.1:9000  # MinIO on the LAN: private datasets
STORAGE_INTERNAL_BUCKET=datasets                 # + _ACCESS_KEY / _SECRET_KEY
STORAGE_EXTERNAL_ENDPOINT=https://<account>.r2.cloudflarestorage.com
STORAGE_EXTERNAL_BUCKET=merlina                  # + _ACCESS_KEY / _SECRET_KEY
ARTIFACT_STORE=external             # run artifacts + store: models (default: external if configured, else hf_hub)
DATASET_MIRROR=true                 # mirror Hub datasets into the internal store on first use
MODEL_MIRROR=true                   # mirror Hub base models into the model store on first use

# Rented GPUs (remote training / inference)
RUNPOD_API_KEY=your_runpod_key

# llama.cpp (optional — enables GGUF export and llama-server inference)
LLAMA_CPP_DIR=/opt/llama.cpp        # Path to a llama.cpp checkout (preferred)
LLAMA_CPP_BIN_DIR=                  # Or just the binary directory
```

See `.env.example` for all available options.

## Docker Support

Two images are published to GHCR from `main` and release tags:

- `ghcr.io/schneewolf-labs/merlina` — the full app (`Dockerfile`), used for local runs and remote training workers.
- `ghcr.io/schneewolf-labs/merlina-serve` — vLLM plus the serve worker (`Dockerfile.serve`), used by remote inference sessions.

`docker compose up -d` runs the app with `./data` (jobs DB, uploads, HF cache, `store:` model cache), `./models` and `./results` mounted, and reads `.env` (see [Quick Start](#option-b-docker)).

## Documentation

- 📖 **[Quick Start Guide](docs/user/quick-start.md)** — Get started in minutes
- 📚 **[Dataset Guide](docs/user/dataset-guide.md)** — Configure datasets (HuggingFace, local, uploads)
- 🤖 **[Tokenizer Format Guide](docs/user/tokenizer-format.md)** — Automatic chat formatting (recommended!)
- ⚙️ **[Configuration Management](docs/user/CONFIG_MANAGEMENT.md)** — Save and reuse training configs
- 🆕 **[New Features Guide](docs/user/new-features.md)** — What's new in v1.1+
- 🔌 **[MCP Server Guide](docs/user/mcp.md)** — Drive Merlina from an LLM agent via Model Context Protocol
- ☁️ **[Remote Training](docs/user/remote-training.md)** — Train on rented GPUs with spend caps
- ☁️ **[Remote Inference](docs/user/remote-inference.md)** — Serve models on rented GPUs for evals
- 🗄️ **[Models in Object Storage](docs/user/store-models.md)** — `store:` models, Hub mirroring, publishing to the store
- 🪣 **[Storage Tiers](docs/storage-tiers.md)** — MinIO/R2 setup, settings, sync and promotion
- 📋 **[Examples](examples/)** — Ready-to-use training configurations
- 📡 **[API Reference](API.md)** — Full REST and WebSocket API docs

**For Developers:**
- 🏗️ **[Developer Docs](docs/dev/)** — Implementation details and architecture
- 🧪 **[Tests](tests/)** — Test suite and fixtures

## Tips

- 🎯 Start with 1-2 epochs for testing
- 💾 Models are saved to `./models/{output_name}`; tick **Publish to the model store** to also get `store:{output_name}`
- 📊 Enable W&B for detailed metrics
- 🔧 Adjust LoRA rank based on your GPU memory
- ✨ Try the Konami code in the UI for a surprise!

## Credits

Created with 💜 by Schneewolf Labs

## License

MIT License — feel free to use for your magical model training!
