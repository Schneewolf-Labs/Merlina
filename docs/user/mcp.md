# Merlina MCP Server

Merlina ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server that exposes the training API as tools, so an LLM agent can run your
fine-tuning workshop conversationally:

> "Queue an SFT run of Llama-3-8B on `org/my-dataset`, then tell me when the
> loss drops below 0.5."

The MCP server (`mcp_server.py`, installed as the `merlina-mcp` command) is a
**thin HTTP client** to a running Merlina server. It never imports torch — it
just talks to the FastAPI app over HTTP. That means you can run it anywhere
that can reach the server (locally, or pointed at a remote training box).

## Install

The MCP dependencies are an optional extra:

```bash
pip install 'merlina[mcp]'
# or, standalone:
pip install 'mcp[cli]' httpx
```

## Run

Start Merlina first, then start the MCP server pointed at it:

```bash
# Terminal 1 — the training server
merlina serve                       # or: python merlina.py

# Terminal 2 — the MCP server (stdio transport by default)
MERLINA_API_URL=http://localhost:8000 merlina-mcp
```

### Configuration

| Variable                 | Default                 | Purpose                                      |
|--------------------------|-------------------------|----------------------------------------------|
| `MERLINA_API_URL`        | `http://localhost:8000` | Base URL of the running Merlina server       |
| `MERLINA_MCP_TIMEOUT`    | `60`                    | Per-request timeout in seconds               |
| `MERLINA_MCP_TRANSPORT`  | `stdio`                 | Transport: `stdio`, `sse`, or `streamable-http` |

The transport can also be set with `merlina-mcp --transport streamable-http`.

## Wire it into an MCP client

### Claude Desktop

Add to `claude_desktop_config.json`:

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

### Claude Code

```bash
claude mcp add merlina --env MERLINA_API_URL=http://localhost:8000 -- merlina-mcp
```

## Tools

| Tool                       | What it does                                                        |
|----------------------------|--------------------------------------------------------------------|
| `health_check`             | Confirm the Merlina server is reachable                            |
| `get_version`              | Server version                                                     |
| `start_training`           | Queue a training job (common knobs + an `overrides` dict for the rest) |
| `validate_training_config` | Pre-flight a full config without queueing                          |
| `list_jobs`                | All jobs with status/progress                                      |
| `get_job_status`           | Progress, steps, loss, queue position, errors for one job          |
| `get_job_history`          | Paginated history, filterable by status                            |
| `get_job_metrics`          | Time-series loss/metrics for a job                                 |
| `stop_job`                 | Stop a running job or cancel a queued one                          |
| `queue_status`             | Queue statistics and queued/running jobs                           |
| `get_stats`                | Database and system statistics                                     |
| `list_gpus`                | Available GPUs with memory/utilization                             |
| `list_local_models`        | Base models already on disk (for offline use)                      |
| `list_uploaded_datasets`   | Datasets uploaded to the server                                    |
| `preview_dataset`          | Preview raw samples from a HuggingFace or uploaded dataset         |

### `start_training`

The most-used knobs are direct arguments (`output_name`, `base_model`,
`training_mode`, `dataset_repo_id`, `learning_rate`, `num_epochs`, `use_lora`,
`use_4bit`, `push_to_hub`, `priority`). Anything else in `TrainingConfig` goes
in `overrides`, which is merged at the top level and wins over the named args:

```jsonc
// overrides example
{
  "lora_r": 32,
  "lora_alpha": 64,
  "beta": 0.2,
  "gradient_checkpointing": true,
  "export_gguf": true,
  "gguf_quant_types": ["Q4_K_M"]
}
```

The server runs pre-flight validation before queueing; if it fails, the tool
result contains the validation errors. Use `validate_training_config` to check
a config without committing GPU time.

## How it fits together

```
LLM client ──stdio──▶ merlina-mcp ──HTTP──▶ Merlina server ──▶ GPU / queue
 (Claude)             (this server)          (merlina serve)
```

Because it's a plain REST client, errors come back as readable JSON (a
connection refusal becomes a hint to start the server, an HTTP 404 keeps the
server's `detail` message) rather than opaque tool failures.

## Troubleshooting

- **"Could not reach Merlina …"** — the server isn't running or
  `MERLINA_API_URL` is wrong. Start it with `merlina serve` and confirm the
  URL/port.
- **`http_400` on `start_training`** — pre-flight validation failed; the
  `detail` lists the specific errors (missing GPU, bad hyperparameters, etc.).
- **Tools don't appear in the client** — check the client picked up the config
  and that `merlina-mcp` is on `PATH` (it's installed by `pip install
  'merlina[mcp]'`).
