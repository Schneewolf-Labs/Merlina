"""
llama-server Process Manager

Thin wrapper around ``llama-server`` from llama.cpp, used as the inference
backend when the user picks a GGUF file instead of a HuggingFace / LoRA
checkpoint. Handles spawn, readiness polling, chat proxying, and shutdown.

Kept standalone (no FastAPI / Merlina singletons) so it's testable in
isolation. The ``/inference/*`` endpoints instantiate one instance and
store it in the global inference state.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .llama_cpp_resolver import LlamaCppResolution, resolve_llama_cpp

logger = logging.getLogger(__name__)


READY_TIMEOUT_SECONDS = 120   # time to wait for llama-server to report healthy
READY_POLL_INTERVAL = 0.5
CHAT_REQUEST_TIMEOUT = 600


class LlamaServerError(RuntimeError):
    """Raised when the llama-server backend cannot fulfil a request."""


class LlamaServerProcess:
    """
    Manages a single ``llama-server`` subprocess bound to a local port.

    The server speaks an OpenAI-compatible API; we forward chat requests
    to ``POST /v1/chat/completions`` and shape the response to match
    Merlina's existing ``/inference/chat`` output.
    """

    def __init__(
        self,
        gguf_path: Path,
        *,
        port: int = 8080,
        host: str = "127.0.0.1",
        n_gpu_layers: int = -1,
        context_size: int = 4096,
        resolution: Optional[LlamaCppResolution] = None,
        extra_args: Optional[List[str]] = None,
    ):
        self.gguf_path = Path(gguf_path)
        if not self.gguf_path.is_file():
            raise LlamaServerError(f"GGUF file not found: {self.gguf_path}")

        self.port = port
        self.host = host
        self.n_gpu_layers = n_gpu_layers
        self.context_size = context_size
        self.extra_args = list(extra_args or [])

        self.resolution = resolution or resolve_llama_cpp()
        server_bin = self.resolution.binary("llama-server")
        if server_bin is None:
            raise LlamaServerError(
                "llama-server binary not available. Install llama.cpp or set "
                "LLAMA_CPP_DIR / LLAMA_CPP_BIN_DIR."
            )
        self.server_bin = server_bin

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._last_log_lines: List[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Spawn llama-server and wait until it reports healthy."""
        if self.is_running():
            raise LlamaServerError("llama-server is already running")

        if _port_in_use(self.host, self.port):
            raise LlamaServerError(
                f"Port {self.port} is already in use — pick a different "
                "llama_server_port or stop the other process."
            )

        args = [
            str(self.server_bin),
            "-m", str(self.gguf_path),
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.context_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            *self.extra_args,
        ]
        logger.info("Spawning llama-server: %s", " ".join(args))

        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(
            target=self._consume_stdout,
            name=f"LlamaServer-{self.port}",
            daemon=True,
        )
        self._reader_thread.start()

        self._await_ready()
        logger.info("llama-server ready at %s (model=%s)", self.base_url, self.gguf_path.name)

    def stop(self, timeout: float = 10.0) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            self._proc = None

        if proc.poll() is not None:
            return

        logger.info("Stopping llama-server (pid=%s)", proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit in %ss, killing", timeout)
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("llama-server failed to die after kill")

    def _consume_stdout(self) -> None:
        """Drain stdout so the subprocess doesn't block on a full pipe."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # Keep a short tail for diagnostic reporting on failure.
            self._last_log_lines.append(line)
            if len(self._last_log_lines) > 20:
                self._last_log_lines.pop(0)
            logger.debug("llama-server[%d]: %s", self.port, line)

    def _await_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                break
            try:
                with urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
                    # llama-server reports 503 while the model is still
                    # loading; treat that as "keep waiting".
            except URLError:
                pass
            except Exception as exc:  # pragma: no cover — best-effort polling
                logger.debug("Health probe failed: %s", exc)
            time.sleep(READY_POLL_INTERVAL)

        # Failed to become healthy.
        tail = "\n".join(self._last_log_lines[-10:]) or "(no output)"
        self.stop(timeout=2.0)
        raise LlamaServerError(
            f"llama-server at {self.base_url} did not become ready within "
            f"{READY_TIMEOUT_SECONDS}s. Last output:\n{tail}"
        )

    # ------------------------------------------------------------------
    # Chat proxy
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """Forward a chat completion request to the running llama-server."""
        if not self.is_running():
            raise LlamaServerError("llama-server is not running")

        payload: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0 if not do_sample else temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "stream": False,
        }

        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=CHAT_REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
        except URLError as exc:
            raise LlamaServerError(f"llama-server request failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LlamaServerError(
                f"llama-server returned non-JSON response: {body[:200]!r}"
            ) from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
        except (KeyError, IndexError) as exc:
            raise LlamaServerError(
                f"Unexpected llama-server response shape: {data!r}"
            ) from exc

        usage = data.get("usage", {})
        return {
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }

    def info(self) -> Dict[str, Any]:
        return {
            "gguf_path": str(self.gguf_path),
            "base_url": self.base_url,
            "port": self.port,
            "running": self.is_running(),
            "n_gpu_layers": self.n_gpu_layers,
            "context_size": self.context_size,
        }


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.bind((host, port))
        except OSError:
            return True
        return False


def resolve_gguf_for_model(
    models_dir: Path,
    model_name: str,
    *,
    preferred_quant: Optional[str] = None,
) -> Optional[Path]:
    """
    Find a GGUF file for ``model_name`` inside ``models_dir``.

    Strategy: read ``models_dir/{model_name}/gguf/manifest.json`` and pick
    an entry matching ``preferred_quant`` (case-insensitive). If preferred
    quant is not specified or not present, fall back to the first
    artifact in the manifest. Returns ``None`` if no manifest exists.
    """
    model_dir = models_dir / model_name
    gguf_dir = model_dir / "gguf"
    manifest_path = gguf_dir / "manifest.json"
    if not manifest_path.is_file():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        return None

    if preferred_quant:
        want = preferred_quant.strip().upper()
        for entry in artifacts:
            if entry.get("quant_type", "").upper() == want:
                candidate = Path(entry.get("path", ""))
                if candidate.is_file():
                    return candidate
                fallback = gguf_dir / entry.get("filename", "")
                if fallback.is_file():
                    return fallback

    # First viable entry.
    for entry in artifacts:
        candidate = Path(entry.get("path", ""))
        if candidate.is_file():
            return candidate
        fallback = gguf_dir / entry.get("filename", "")
        if fallback.is_file():
            return fallback

    return None
