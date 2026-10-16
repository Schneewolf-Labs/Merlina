# Remote Inference on Rented GPUs 🔮☁️

Train remotely, then serve what you trained on a rented GPU behind an
OpenAI-compatible API, point an eval harness at it, and stop it. That is
the whole loop: **train → serve → eval**, without the model ever touching
your machine.

A remote inference session rents one instance on **your** RunPod account
(the same `RUNPOD_API_KEY` remote training uses), runs
[vLLM](https://docs.vllm.ai) on it, and hands you a `base_url` and an
`api_key`. Merlina always terminates the instance: when you stop the
session, when it goes idle, when its runtime cap passes, or when its spend
cap is reached.

> Local inference (`/inference/load` + the chat tab) is unchanged. Remote
> sessions are API-only for now.

## Setup

1. Configure remote mode as in [Remote Training](remote-training.md):
   `RUNPOD_API_KEY` in `.env`, plus `HF_TOKEN` for gated/private repos.
2. The instance runs the serve image (`REMOTE_SERVE_IMAGE`, default
   `ghcr.io/schneewolf-labs/merlina-serve:latest`, built from
   `Dockerfile.serve`). To use your own build:
   ```bash
   docker build -f Dockerfile.serve -t ghcr.io/you/merlina-serve:dev .
   docker push ghcr.io/you/merlina-serve:dev
   # then REMOTE_SERVE_IMAGE=ghcr.io/you/merlina-serve:dev, or "worker_image" per request
   ```

## What to serve

**A HuggingFace repo** (optionally pinned to a revision):

```json
{"model": {"hf_repo": "Qwen/Qwen3-8B", "revision": "main"}}
```

**A finished remote training job.** The instance pulls the job's artifact
straight from that job's artifact store (S3/R2 or the private HF run repo):

```json
{"model": {"job_id": "job_20261016_101500_ab12cd", "artifact": "adapter"}}
```

- `"adapter"` (default) serves the job's **base model + LoRA adapter** with
  vLLM's LoRA support. One endpoint answers two model names: the adapter
  under the job's output name (or `served_model_name`) and the bare base
  model under its HF id, which is handy for base-vs-tuned evals.
- `"merged"` serves merged weights. These are only in the store when the
  job used `remote.merge_strategy: "remote"` without `push_to_hub`. If the
  merge was published to the Hub, serve that repo with `hf_repo` instead.
  The planner tells you which case applies.

Only **remote** jobs work as a source: a locally trained model is on your
disk, where a rented instance cannot reach it. Push it to the Hub and serve
it by `hf_repo`.

> Adapters that fully retrain modules (`modules_to_save`, e.g.
> `embed_tokens`/`lm_head`) can't be applied by vLLM's LoRA path. The plan
> warns you; serve the merged model for those.

## Plan first, then approve the spend

`POST /remote/inference/plan` is a dry run. It rents nothing. It shows
exactly what `POST /remote/inference` would do with the same body:

```bash
curl -s localhost:8000/remote/inference/plan -H 'Content-Type: application/json' -d '{
  "model": {"job_id": "job_20261016_101500_ab12cd", "artifact": "adapter"},
  "max_runtime_hours": 2,
  "max_total_cost_usd": 5,
  "idle_timeout_minutes": 20
}' | jq '{gpu: .sizing.gpu_type_id, count: .sizing.gpu_count,
          per_hr: .sizing.est_cost_per_hr, vram: .vram_breakdown,
          hours: .effective_max_hours, ends_by: .deadline_reason,
          max_cost: .est_max_total_cost_usd, cmd: (.vllm_command | join(" "))}'
```

How the plan is sized:

- **Weights at their real size.** The model's `model.safetensors.index.json`
  `total_size` is what vLLM loads, so FP8/FP4 checkpoints are sized as they
  really are. A ~167 GB FP8 MoE is sized at ~156 GiB, not at bf16 × the
  parameter count. Without an index, weights fall back to a bf16 estimate,
  and the plan warns you.
- **KV cache for your concurrency.** `max_num_seqs` sequences at full
  `max_model_len`. Per-token KV comes from `config.json`: layers × KV heads
  × head dim for standard/GQA attention, or the compressed latent
  (`kv_lora_rank + qk_rope_head_dim`) for MLA models. When those fields are
  missing it assumes 1 MiB/token, which is more than common decoders need.
  `max_model_len` defaults to `min(model max, 8192)`; `max_num_seqs`
  defaults to 8. Lower either to shrink the instance.
- **Overhead.** Activations and CUDA graphs (max(2 GiB, 5% of weights)),
  plus 2 GiB for an adapter. Per-GPU CUDA context is budgeted on top.
- **GPU pick.** This is the cheapest feasible type/count from live RunPod
  offers, using the same picker as remote training. Counts are limited to
  ones vLLM tensor parallelism can use (1/2/4/8, dividing the attention
  heads). `tensor_parallel_size` defaults to the GPU count. Pin a GPU with
  `gpu_type_id`/`gpu_count`, or cap the price with `max_cost_per_hr`.

## Start, use, stop

```bash
SID=$(curl -s localhost:8000/remote/inference -H 'Content-Type: application/json' \
  -d @session.json | jq -r .session_id)

# Poll until ready: provisioning -> starting (download + load) -> ready
curl -s localhost:8000/remote/inference/$SID | jq '{status, base_url, served_model_name, api_key}'
```

Once `status` is `ready`, the response has `base_url` (ends in `/v1`),
`served_model_name` and `api_key`. Any OpenAI client works:

```python
from openai import OpenAI
client = OpenAI(base_url=session["base_url"], api_key=session["api_key"])
reply = client.chat.completions.create(
    model=session["served_model_name"],
    messages=[{"role": "user", "content": "Cast a small spell."}],
)
```

Point an eval harness at it the same way. For example, with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
OPENAI_API_KEY=$API_KEY lm_eval --model local-chat-completions \
  --model_args model=$SERVED_NAME,base_url=$BASE_URL/chat/completions,num_concurrent=8 \
  --apply_chat_template --tasks gsm8k
```

Keep `num_concurrent` at or below the session's `max_num_seqs`. Requests
beyond that queue on the server.

When you're done:

```bash
curl -s -X POST localhost:8000/remote/inference/$SID/stop
```

The instance is terminated within one poll interval
(`REMOTE_POLL_INTERVAL_SECONDS`, default 10 s). `GET /remote/inference`
lists every session, newest first. Ended sessions keep their `stop_reason`
(`user`, `idle_timeout`, `max_runtime_hours`, `max_total_cost_usd`,
`worker_stopped`, `error`, `orphaned`). They also keep `metrics.est_cost_usd`,
an estimate from the observed rate, not your invoice. When vLLM fails to
start, the last lines of its log are in `metrics.log_tail`.

## Costs and caps

Every session carries these rails:

| Field | Default | Effect |
|---|---|---|
| `max_runtime_hours` | **2** (max 24) | Hard deadline from session start, boot included |
| `max_total_cost_usd` | off | Deadline = start + cap ÷ $/hr when that is earlier. Re-derived from the instance's **actual** rate once it is known (it can only move earlier) |
| `idle_timeout_minutes` | off | Stop after this long with no requests. In-flight streams count as activity |
| `max_cost_per_hr` | off | Refuse instance picks above this rate |
| `REMOTE_MAX_INFERENCE_SESSIONS` | 2 | Concurrent sessions per server (409 beyond) |

The deadline is durable. It lives in the same lifecycle tables as remote
training, so the remote reconciler enforces it even if the session's
controller dies. If Merlina restarts mid-session, the session is marked
`failed` (`orphaned`), and its instance is terminated once the dead
controller's lease lapses (about 2 minutes). Sessions don't resume.

`keep_instance_on_failure: true` keeps the instance after a **failure** so
you can debug it. It still bills, up to the session deadline. Stops, idle
timeouts and caps always terminate.

Provider outages can delay a shutdown past a cap. Treat the caps as a
strong guard, not an invoice guarantee.

## Security

- **Provider proxy URLs are guessable** (`https://<pod-id>-8001.proxy.runpod.net`).
  The endpoint therefore never runs without a key. Each session gets a
  fresh random API key (separate from the worker's control token), and an
  authenticating proxy on the instance requires
  `Authorization: Bearer <api_key>` on **every** path.
- vLLM itself listens only on the instance's loopback, because its own
  `--api-key` guards just some path prefixes (`/v1`, ...). Routes such as
  `/invocations`, `/tokenize` or `/metrics` would otherwise be open.
- The key reaches vLLM through its environment, never its command line, so
  it isn't in vLLM's startup log or `ps`. It is also scrubbed from any log
  tail Merlina reports.
- Merlina keeps the key **in memory only** and returns it only from
  `GET /remote/inference/{id}`. List responses never include it. It is
  never written to the database or logs. After a restart the key is gone,
  and so is the session.
- Like the worker token, the key is in the pod's environment, so anyone
  with access to your RunPod account can read it. Anyone who can reach
  Merlina's API can read it too, and can also rent GPUs on your account.
  Don't expose Merlina's API publicly without authentication in front of it.

## API summary

| Endpoint | Purpose |
|---|---|
| `POST /remote/inference/plan` | Dry run: GPU, count, TP, $/hr, VRAM breakdown, deadline, max total cost |
| `POST /remote/inference` | Start a session (same body). Returns `session_id` and the plan |
| `GET /remote/inference` | List sessions (no keys) |
| `GET /remote/inference/{id}` | Status, `base_url`, `served_model_name`, `api_key` |
| `POST /remote/inference/{id}/stop` | Stop the session and terminate the instance |

Request fields beyond the ones above: `served_model_name`, `hf_token`
(used, never stored; defaults to the training job's token, then the server's
`HF_TOKEN`), `gpu_type_id`, `gpu_count`, `cloud_type`, `container_disk_gb`,
`max_model_len`, `max_num_seqs`, `tensor_parallel_size`,
`gpu_memory_utilization` (default 0.92), `vllm_args` (extra `vllm serve`
flags such as `["--enable-prefix-caching"]`; host, port, API key, served
name and LoRA modules are managed by Merlina and rejected here), and
`worker_image`.
