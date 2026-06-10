# Merlina - Magical Model Training
#
# GPU image based on the official PyTorch CUDA runtime (torch, torchvision,
# torchaudio pre-installed and CUDA-matched, per the project's torch policy —
# see requirements.txt).
#
# Build:  docker build -t merlina .
# Run:    docker run --gpus all -p 8000:8000 -v ./data:/app/data -v ./models:/app/models merlina
# Or use docker-compose.yml (recommended).

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# git: HuggingFace Hub downloads; curl: healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so code changes don't bust the dependency layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
