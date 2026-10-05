#!/usr/bin/env bash
# Bootstrap a Merlina REMOTE WORKER on a bare PyTorch pod.
#
# This is the alternative to the prebuilt Docker image for instances that
# start from a provider's standard PyTorch template: it clones Merlina,
# installs dependencies, and execs the worker entry point. The Merlina
# control plane passes the job via MERLINA_REMOTE_JOB_B64 /
# MERLINA_WORKER_TOKEN environment variables on the instance.
#
# Usage (as the instance's start command):
#   bash -c "curl -fsSL https://raw.githubusercontent.com/Schneewolf-Labs/Merlina/main/scripts/remote_worker_bootstrap.sh | bash"

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MERLINA_DIR="${MERLINA_DIR:-$WORKSPACE/Merlina}"
MERLINA_BRANCH="${MERLINA_BRANCH:-main}"

echo "🧙‍♀️ Bootstrapping Merlina remote worker in $MERLINA_DIR ..."

if [ -z "${MERLINA_REMOTE_JOB_B64:-}" ]; then
    echo "ERROR: MERLINA_REMOTE_JOB_B64 is not set — this script is meant to be" >&2
    echo "launched by the Merlina remote orchestrator, not by hand." >&2
    exit 1
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: torch is not installed. Use a PyTorch pod template." >&2
    exit 1
fi

if [ -d "$MERLINA_DIR/.git" ]; then
    git -C "$MERLINA_DIR" pull --ff-only origin "$MERLINA_BRANCH"
else
    git clone --branch "$MERLINA_BRANCH" https://github.com/Schneewolf-Labs/Merlina.git "$MERLINA_DIR"
fi

cd "$MERLINA_DIR"
pip install --no-cache-dir -r requirements.txt

# Keep all state (HF cache, models, worker job DB) on the volume when present
export HF_HOME="${HF_HOME:-$WORKSPACE/huggingface}"
export MERLINA_WORKER_DB="${MERLINA_WORKER_DB:-$MERLINA_DIR/data/worker_jobs.db}"

exec python -m src.remote.worker_entry
