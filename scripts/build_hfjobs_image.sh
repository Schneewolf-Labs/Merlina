#!/usr/bin/env bash
#
# Build and (optionally) push the Merlina HuggingFace Jobs image.
#
# Usage:
#   scripts/build_hfjobs_image.sh                 # build + push default tag
#   scripts/build_hfjobs_image.sh --no-push       # build only (local smoke test)
#   IMAGE_TAG=ghcr.io/me/merlina-hfjobs:dev \
#     scripts/build_hfjobs_image.sh               # custom registry/tag
#
# Requires: docker, a logged-in container registry
#   echo "$GITHUB_TOKEN" | docker login ghcr.io -u <user> --password-stdin

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-ghcr.io/schneewolf-labs/merlina-hfjobs:latest}"
PUSH=1

for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=0 ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo ">> Building $IMAGE_TAG (this is slow on first build — CUDA+PyTorch base is large)"
docker build -f Dockerfile.hfjobs -t "$IMAGE_TAG" .

if [[ "$PUSH" -eq 1 ]]; then
    echo ">> Pushing $IMAGE_TAG"
    docker push "$IMAGE_TAG"
    echo ">> Done. Remember: HF Jobs can only pull this image if it's public"
    echo "   (or you wire registry auth into run_job())."
else
    echo ">> Skipped push. Smoke test with:"
    echo "   docker run --rm -it --entrypoint bash $IMAGE_TAG \\"
    echo "     -c 'python -c \"import torch, transformers, peft, grimoire\"'"
fi
