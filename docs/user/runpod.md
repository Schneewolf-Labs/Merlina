# Running Merlina on RunPod ☁️

Two ways to get Merlina running on a rented GPU: the Docker image (cleanest)
or a setup script on a standard PyTorch pod (most flexible).

Both expose the web UI on **HTTP port 8000** — add it to your pod's exposed
HTTP ports and open it from RunPod's **Connect** menu.

> ⚠️ A pod's HTTP proxy URL is reachable by anyone who has it. Merlina has no
> built-in authentication yet, so don't share the URL, and stop the pod when
> you're done.

## Option A — Docker image (custom template)

Create a RunPod template with:

| Setting | Value |
|---|---|
| Container image | `ghcr.io/schneewolf-labs/merlina:latest` |
| Expose HTTP ports | `8000` |
| Volume mount path | `/app/data` (jobs DB + uploads + HF cache) |
| Container disk | 30 GB+ (more for large base models) |

Optional environment variables (all of `.env.example` works):

- `HF_TOKEN` — for gated models and Hub uploads
- `WANDB_API_KEY` — for experiment tracking
- `MAX_CONCURRENT_JOBS` — keep at `1` per GPU

Trained models land in `/app/models` inside the container; mount a second
volume there if you want them to survive pod termination, or push to the
HuggingFace Hub from the Export tab.

## Option B — Setup script on a PyTorch pod

Start any official RunPod **PyTorch** pod (torch comes pre-installed and
CUDA-matched), expose HTTP port 8000, then run:

```bash
curl -fsSL https://raw.githubusercontent.com/Schneewolf-Labs/Merlina/main/scripts/runpod_setup.sh | bash
```

The script:

1. Clones (or updates) Merlina into `/workspace/Merlina` — the persistent volume
2. Installs Python dependencies (torch is left untouched)
3. Points all state (jobs DB, models, results, HF cache) at `/workspace`
4. Starts the server on port 8000

To use it as a pod **startup command** instead of an interactive step:

```bash
bash -c "curl -fsSL https://raw.githubusercontent.com/Schneewolf-Labs/Merlina/main/scripts/runpod_setup.sh | bash"
```

Restarting later (already installed):

```bash
cd /workspace/Merlina && bash scripts/runpod_setup.sh
```

## Sizing cheatsheet

| Model size | 4-bit QLoRA | bf16 LoRA |
|---|---|---|
| 1–3 B | 8–12 GB (RTX 3080/4000-class) | 16–24 GB |
| 7–8 B | ~16 GB (RTX 4080/A4500) | 48 GB (A6000/L40S) |
| 12–14 B | ~24 GB (RTX 4090/A5000) | 80 GB (A100/H100) |

Pre-flight validation in the UI will warn you before a job that won't fit.
