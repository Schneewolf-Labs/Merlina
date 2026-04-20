#!/usr/bin/env bash
#
# Entrypoint for Merlina training running inside a HuggingFace Job.
#
# Expected environment (provided via run_job secrets/env):
#   MERLINA_CONFIG_B64   base64-encoded TrainingConfig JSON (required)
#   MERLINA_JOB_ID       job identifier used by the parent server (required)
#   HF_TOKEN             HuggingFace token with write access (required for push_to_hub)
#   WANDB_API_KEY        W&B key (optional, only if use_wandb=true)
#
# Progress is tailed from the worker's JSONL file to stdout, each line
# prefixed with "MERLINA_PROGRESS:" so the parent server can discriminate
# progress events from other log noise when streaming logs via
# huggingface_hub.fetch_job_logs().

set -euo pipefail

: "${MERLINA_CONFIG_B64:?MERLINA_CONFIG_B64 env var is required}"
: "${MERLINA_JOB_ID:?MERLINA_JOB_ID env var is required}"

CONFIG_PATH="/tmp/merlina_config.json"
PROGRESS_FILE="/tmp/merlina_progress.jsonl"

echo "MERLINA_PROGRESS: {\"type\":\"status\",\"status\":\"initializing\",\"message\":\"HF Job starting\"}"

# Decode the config into place.
echo "$MERLINA_CONFIG_B64" | base64 -d > "$CONFIG_PATH"

# Create an empty progress file so tail -F has something to follow.
: > "$PROGRESS_FILE"

# Tail the progress file in the background, emitting each line as a
# sentinel-prefixed log entry so the parent can parse it out of the
# streamed job logs.
( tail -n +1 -F "$PROGRESS_FILE" 2>/dev/null \
    | awk '{ printf "MERLINA_PROGRESS: %s\n", $0; fflush() }' ) &
TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT

# The worker will skip JobManager/DB writes when --db-path is empty.
# Progress is written only to the JSONL file, which we stream above.
cd /opt/merlina

python -u -m src.train_worker \
    --config-path "$CONFIG_PATH" \
    --job-id "$MERLINA_JOB_ID" \
    --progress-file "$PROGRESS_FILE" \
    --db-path ""

# Give the background tail a moment to flush the final lines.
sleep 2
