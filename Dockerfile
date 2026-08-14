# Merlina - Magical Model Training
#
# GPU image based on the official PyTorch CUDA runtime (torch, torchvision,
# torchaudio pre-installed and CUDA-matched, per the project's torch policy —
# see requirements.txt).
#
# Build:  docker build -t merlina .
# Run:    docker run --gpus all -p 8000:8000 -v ./data:/app/data -v ./models:/app/models merlina
# Or use docker-compose.yml (recommended).

# ---------------------------------------------------------------------------
# Stage 1: build the causal_conv1d wheel.
#
# Hybrid linear-attention models (Qwen3.5/3.6/3.8) train ~7x faster per step
# when `flash-linear-attention` and `causal_conv1d` are both present —
# transformers gates the fused path on all four kernel symbols, so one without
# the other is the same as neither. Measured on ReAligned-Qwen3.5-4B at 4096
# tokens: 13.67 s/step on the reference path, 1.84 s/step fused, cosine 0.99996.
#
# `fla` is pure Python plus Triton and installs from requirements.txt. But
# causal_conv1d compiles CUDA from source and needs nvcc, which the -runtime
# image does not ship. Building it here and copying only the wheel keeps the
# final image on -runtime instead of dragging in the whole toolkit.
#
# Best-effort on purpose: this is an optional accelerator, and a toolchain that
# cannot build it should still yield a working image. Preflight warns at job
# submission when the kernels are absent, so a slow image is loud rather than
# silent.
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS kernels

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /wheels \
    && (CAUSAL_CONV1D_FORCE_BUILD=TRUE pip wheel --no-build-isolation \
          --no-deps -w /wheels causal-conv1d \
        || echo "causal_conv1d wheel build failed; image will use the reference path")

# ---------------------------------------------------------------------------
# Stage 2: the runtime image
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# git: HuggingFace Hub downloads; curl: healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so code changes don't bust the dependency layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the prebuilt kernel wheel if stage 1 produced one. Guarded so a failed
# kernel build degrades speed, never the image.
COPY --from=kernels /wheels /tmp/wheels
RUN if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir /tmp/wheels/*.whl; \
    else \
        echo "no causal_conv1d wheel — hybrid models will train on the reference path"; \
    fi \
    && rm -rf /tmp/wheels

COPY . .

# Keep all mutable state (jobs DB, uploads, HF cache) under /app/data and
# trained models under /app/models so two bind mounts persist everything.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    RESULTS_DIR=/app/results \
    UPLOADS_DIR=/app/data/uploads \
    DATABASE_PATH=/app/data/jobs.db \
    HF_HOME=/app/data/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "merlina.py"]
