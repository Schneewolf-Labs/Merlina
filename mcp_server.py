"""
Merlina MCP server — drive the training workshop from any MCP client.

Exposes Merlina's REST API as Model Context Protocol tools so an LLM agent
(Claude Desktop, Claude Code, Cursor, ...) can queue training jobs, watch
their progress, inspect datasets, and manage the GPU queue conversationally.

Design: this is a thin HTTP client to a *running* Merlina server, mirroring
``merlina_cli.py``'s import-light philosophy. The MCP process never imports
torch/transformers — it just talks to the FastAPI app over HTTP. Start the
server (``merlina serve``) first, then point this at it:

    MERLINA_API_URL=http://localhost:8000 merlina-mcp

Or wire it into an MCP client config, e.g. Claude Desktop:

    {
      "mcpServers": {
        "merlina": {
          "command": "merlina-mcp",
          "env": { "MERLINA_API_URL": "http://localhost:8000" }
        }
      }
    }

The heavy `mcp` SDK is imported lazily so ``--help`` and import-time errors
stay friendly when it isn't installed (``pip install 'merlina[mcp]'``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

# httpx is the only hard runtime dependency of the MCP layer beyond the SDK.
import httpx

# Base URL of the running Merlina server. Override with MERLINA_API_URL.
BASE_URL = os.environ.get("MERLINA_API_URL", "http://localhost:8000").rstrip("/")

# How long to wait on a single API call. Training submits run a synchronous
# pre-flight validation server-side (up to ~30s), so keep this comfortably above.
TIMEOUT = float(os.environ.get("MERLINA_MCP_TIMEOUT", "60"))

MCP_INSTALL_HINT = """\
The Merlina MCP server needs the Model Context Protocol SDK, which is an
optional dependency. Install it with:

    pip install 'merlina[mcp]'

or directly:

    pip install 'mcp[cli]'
"""

# Module-level client cache. Tests may inject a stub via set_client().
_client_instance: Optional[httpx.AsyncClient] = None


def set_client(client: Optional[httpx.AsyncClient]) -> None:
    """Inject (or reset) the HTTP client. Used by tests to mock the API."""
    global _client_instance
    _client_instance = client


def _client() -> httpx.AsyncClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT)
    return _client_instance


async def _request(
    method: str,
    path: str,
    *,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
) -> Any:
    """
    Make one API call and return parsed JSON, or a structured error dict.

    Errors are returned (not raised) so the LLM sees a readable explanation
    instead of an opaque tool failure — e.g. a connection refused becomes a
    hint to start the Merlina server.
    """
    client = _client()
    try:
        resp = await client.request(method, path, params=params, json=json_body)
    except httpx.HTTPError as exc:
        return {
            "error": "connection_failed",
            "detail": f"Could not reach Merlina at {BASE_URL}: {exc}",
            "hint": "Is the Merlina server running? Start it with `merlina serve`.",
        }

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return {"error": f"http_{resp.status_code}", "detail": detail}

    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def _dump(value: Any) -> str:
    """Serialize a tool result to a readable JSON string."""
    return json.dumps(value, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool implementations
#
# These are plain async functions so they can be unit-tested directly against
# a mocked client. build_server() registers each one as an MCP tool.
# ---------------------------------------------------------------------------


async def health_check() -> str:
    """Check that the Merlina server is up and reachable."""
    return _dump(await _request("GET", "/health"))


async def get_version() -> str:
    """Get the running Merlina server version."""
    return _dump(await _request("GET", "/version"))


async def list_jobs() -> str:
    """List all training jobs with their status and progress."""
    return _dump(await _request("GET", "/jobs"))


async def get_job_status(job_id: str) -> str:
    """
    Get detailed status for a single training job: progress, current/total
    steps, latest loss, queue position, and any error.

    Args:
        job_id: The job identifier returned by start_training.
    """
    return _dump(await _request("GET", f"/status/{job_id}"))


async def get_job_history(limit: int = 20, offset: int = 0, status: Optional[str] = None) -> str:
    """
    Get paginated job history, optionally filtered by status.

    Args:
        limit: Maximum number of jobs to return (default 20).
        offset: Number of jobs to skip for pagination (default 0).
        status: Filter by status, e.g. 'completed', 'failed', 'running',
            'queued', 'stopped'. Omit for all statuses.
    """
    params: dict = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    return _dump(await _request("GET", "/jobs/history", params=params))


async def get_job_metrics(job_id: str) -> str:
    """
    Get the time-series training metrics (loss per step, etc.) for a job.

    Args:
        job_id: The job identifier.
    """
    return _dump(await _request("GET", f"/jobs/{job_id}/metrics"))


async def stop_job(job_id: str) -> str:
    """
    Stop a running job or cancel a queued one.

    Args:
        job_id: The job identifier.
    """
    return _dump(await _request("POST", f"/jobs/{job_id}/stop"))


async def queue_status() -> str:
    """Get overall queue statistics and the list of queued/running jobs."""
    return _dump(await _request("GET", "/queue/status"))


async def get_stats() -> str:
    """Get database and system statistics (job counts, disk usage, etc.)."""
    return _dump(await _request("GET", "/stats"))


async def list_gpus() -> str:
    """List available GPUs with memory and utilization details."""
    return _dump(await _request("GET", "/gpu/list"))


async def list_local_models() -> str:
    """
    List base models already available on disk (HuggingFace cache plus full
    models in ./models). Useful for choosing a base_model offline.
    """
    return _dump(await _request("GET", "/models/local"))


async def list_uploaded_datasets() -> str:
    """List datasets previously uploaded to the Merlina server."""
    return _dump(await _request("GET", "/dataset/uploads"))


async def preview_dataset(
    repo_id: str,
    source_type: str = "huggingface",
    split: str = "train",
    format_type: str = "chatml",
    training_mode: str = "orpo",
    limit: int = 5,
    offset: int = 0,
) -> str:
    """
    Preview raw samples from a dataset before training.

    Args:
        repo_id: HuggingFace dataset id (for source_type='huggingface') or the
            uploaded dataset id (for source_type='upload').
        source_type: 'huggingface' (default) or 'upload'.
        split: Dataset split (default 'train').
        format_type: Format to validate against: chatml, llama3, mistral,
            qwen3, custom (default 'chatml').
        training_mode: Affects which columns are required (default 'orpo').
        limit: Number of samples to return (default 5).
        offset: Number of samples to skip (default 0).
    """
    source: dict = {"source_type": source_type, "split": split}
    if source_type == "upload":
        source["dataset_id"] = repo_id
    else:
        source["repo_id"] = repo_id

    body = {
        "source": source,
        "format": {"format_type": format_type},
        "training_mode": training_mode,
    }
    result = await _request(
        "POST", "/dataset/preview", params={"limit": limit, "offset": offset}, json_body=body
    )
    return _dump(result)


async def validate_training_config(config: dict) -> str:
    """
    Run pre-flight validation on a full training config without queueing it.
    Returns errors and warnings (GPU/VRAM/disk/model-access/hyperparameter
    checks). Use this to sanity-check a config before start_training.

    Args:
        config: A TrainingConfig object as a dict (same shape as the /train
            request body). At minimum requires 'output_name'.
    """
    return _dump(await _request("POST", "/validate", json_body=config))


async def start_training(
    output_name: str,
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    training_mode: str = "orpo",
    dataset_repo_id: str = "schneewolflabs/Athanorlite-DPO",
    dataset_source_type: str = "huggingface",
    dataset_split: str = "train",
    dataset_format: str = "chatml",
    learning_rate: float = 5e-6,
    num_epochs: int = 2,
    use_lora: bool = True,
    use_4bit: bool = True,
    push_to_hub: bool = False,
    priority: str = "normal",
    overrides: Optional[dict] = None,
) -> str:
    """
    Queue a new training job. Covers the common knobs directly; anything else
    in TrainingConfig can be set via `overrides` (merged at the top level).

    The server runs pre-flight validation before queueing — if it fails, the
    returned object includes the validation errors. Call validate_training_config
    first if you want to check without queueing.

    Args:
        output_name: Name for the output model (required).
        base_model: HuggingFace model id or local path to fine-tune.
        training_mode: 'sft', 'orpo', 'dpo', 'simpo', 'cpo', 'ipo', 'kto', or
            a 'vlm_*'/'diffusion_*' mode.
        dataset_repo_id: HuggingFace dataset id, or uploaded dataset id when
            dataset_source_type='upload'.
        dataset_source_type: 'huggingface' (default) or 'upload'.
        dataset_split: Dataset split (default 'train').
        dataset_format: chatml, llama3, mistral, qwen3, tokenizer, or custom.
        learning_rate: Learning rate (default 5e-6).
        num_epochs: Number of epochs (default 2).
        use_lora: Train with LoRA adapters (default True).
        use_4bit: Use 4-bit quantization to save VRAM (default True).
        push_to_hub: Push the result to HuggingFace Hub when done (default False).
        priority: Queue priority: 'low', 'normal', or 'high'.
        overrides: Optional dict merged into the config for any advanced field
            (e.g. {"lora_r": 32, "beta": 0.2, "gradient_checkpointing": true}).
            Keys here win over the named arguments.
    """
    source: dict = {"source_type": dataset_source_type, "split": dataset_split}
    if dataset_source_type == "upload":
        source["dataset_id"] = dataset_repo_id
    else:
        source["repo_id"] = dataset_repo_id

    config: dict = {
        "output_name": output_name,
        "base_model": base_model,
        "training_mode": training_mode,
        "use_lora": use_lora,
        "use_4bit": use_4bit,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "push_to_hub": push_to_hub,
        "dataset": {
            "source": source,
            "format": {"format_type": dataset_format},
            "training_mode": training_mode,
        },
    }
    if overrides:
        config.update(overrides)

    return _dump(
        await _request("POST", "/train", params={"priority": priority}, json_body=config)
    )


# All tools exposed by the server, in a single registry so build_server() and
# the tests share one source of truth.
TOOLS = [
    health_check,
    get_version,
    list_jobs,
    get_job_status,
    get_job_history,
    get_job_metrics,
    stop_job,
    queue_status,
    get_stats,
    list_gpus,
    list_local_models,
    list_uploaded_datasets,
    preview_dataset,
    validate_training_config,
    start_training,
]


def build_server():
    """
    Construct the FastMCP server and register every tool. Imports the MCP SDK
    lazily so a missing dependency yields a friendly hint instead of a
    traceback at module import.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised via main()
        raise SystemExit(MCP_INSTALL_HINT) from exc

    server = FastMCP(
        "merlina",
        instructions=(
            "Tools for Merlina, a magical LLM/VLM/diffusion fine-tuning server. "
            "Queue training jobs, monitor progress, preview datasets, and manage "
            "the GPU queue. The server must be running (`merlina serve`); the "
            f"API base URL is {BASE_URL} (set MERLINA_API_URL to change it)."
        ),
    )
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main(argv=None) -> int:
    """Console entry point (installed as `merlina-mcp`)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="merlina-mcp",
        description=(
            "Merlina MCP server — expose the Merlina training API as Model "
            "Context Protocol tools. Talks to a running Merlina server (set "
            "MERLINA_API_URL, default http://localhost:8000)."
        ),
    )
    parser.add_argument(
        "--transport",
        default=os.environ.get("MERLINA_MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport (default: stdio).",
    )
    args = parser.parse_args(argv)

    server = build_server()
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
