#!/usr/bin/env bash
# Merlina one-shot setup for RunPod (and similar GPU pods: Vast.ai, Lambda...)
#
# Designed for the standard RunPod PyTorch templates, which ship a
# CUDA-matched torch. Installs Merlina into the persistent /workspace volume
# so models, jobs, and the HF cache survive pod restarts.
#
# Usage (from a pod terminal or as a startup command):
#   curl -fsSL https://raw.githubusercontent.com/Schneewolf-Labs/Merlina/main/scripts/runpod_setup.sh | bash
#
# Then expose HTTP port 8000 on the pod and open it via RunPod's Connect menu.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MERLINA_DIR="${MERLINA_DIR:-$WORKSPACE/Merlina}"
MERLINA_BRANCH="${MERLINA_BRANCH:-main}"

echo "🧙‍♀️ Setting up Merlina in $MERLINA_DIR ..."

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: torch is not installed. Use a PyTorch pod template, or install it first:" >&2
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" >&2
    exit 1
fi

if [ -d "$MERLINA_DIR/.git" ]; then
    echo "Existing install found — updating..."
    git -C "$MERLINA_DIR" pull --ff-only origin "$MERLINA_BRANCH"
else
    git clone --branch "$MERLINA_BRANCH" https://github.com/Schneewolf-Labs/Merlina.git "$MERLINA_DIR"
fi

cd "$MERLINA_DIR"
pip install --no-cache-dir -r requirements.txt

# Persist all state on the /workspace volume
export DATA_DIR="$MERLINA_DIR/data"
export MODELS_DIR="$MERLINA_DIR/models"
export RESULTS_DIR="$MERLINA_DIR/results"
export UPLOADS_DIR="$MERLINA_DIR/data/uploads"
export DATABASE_PATH="$MERLINA_DIR/data/jobs.db"
export HF_HOME="${HF_HOME:-$WORKSPACE/huggingface}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo ""
echo "✨ Starting Merlina on port $PORT — expose this port on your pod and"
echo "   open it from RunPod's Connect menu."
exec python merlina.py
