"""
Serve stage — runs ON the rented instance, inside the vLLM-based serve image.

``run_serve_stage`` turns one worker into an OpenAI-compatible endpoint:

1. Pull the model from the artifact store when the session serves a
   training job's output ("merged" weights, or an "adapter" on its base).
2. Launch ``vllm serve`` as a subprocess bound to the instance's loopback
   (``127.0.0.1:8002``). Its stdout/stderr go to a log file whose tail is
   reported in ``/status`` when anything goes wrong.
3. Once vLLM's ``/health`` answers, start the API-key proxy on the public
   serve port (8001) and report the ``serving`` state.
4. Hold until the control plane posts ``/stop`` (or vLLM dies), then shut
   vLLM down.

Why a proxy instead of exposing vLLM directly: provider proxy URLs are
guessable, and vLLM's own ``--api-key`` only guards some path prefixes
(``/v1``, ``/v2``, ...) — routes such as ``/invocations``, ``/tokenize`` or
``/metrics`` stay open. The proxy requires the key on *every* path, and
also counts requests so the control plane can stop idle sessions. vLLM
still gets the key too (``VLLM_API_KEY``), as defense in depth.

The key travels in the environment (never argv, which vLLM logs and
``ps`` shows) and is scrubbed from any log tail this module reports.

Import surface: stdlib only (+ the artifact store for pulls), because the
serve image is vLLM's image plus a handful of Merlina modules — not the
training stack. ``tests/test_remote_serve_image.py`` enforces that the
serve Dockerfile ships every module this imports.
"""

from __future__ import annotations

import hmac
import http.client
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.remote.worker_client import SERVE_PORT, STATE_RUNNING, STATE_SERVING

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "MERLINA_SERVE_API_KEY"
VLLM_INTERNAL_PORT = 8002
OPENAI_BASE_PATH = "/v1"
DEFAULT_VLLM_EXECUTABLE = ("vllm",)
DEFAULT_STARTUP_TIMEOUT_MIN = 60.0
LOG_TAIL_LINES = 60
ROOT_CAUSE_CONTEXT_LINES = 30
ERROR_TAIL_LINES = 15
STOP_GRACE_SECONDS = 30.0
UPSTREAM_TIMEOUT_S = 600.0
# vLLM's accepted --max-lora-rank values (vllm/config/lora.py MaxLoRARanks).
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
# Flags the worker owns: overriding them would bypass the auth proxy or
# break the advertised endpoint, so pass-through vllm_args may not set them.
RESERVED_VLLM_FLAGS = ("--api-key", "--host", "--port", "--served-model-name",
                       "--lora-modules", "--uds")
# Process environment vLLM must not inherit (least privilege).
_WORKER_ONLY_ENV = ("MERLINA_WORKER_TOKEN", "MERLINA_REMOTE_JOB_B64", API_KEY_ENV_VAR)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
})


class ServeError(RuntimeError):
    """vLLM could not be started or died; carries the log tail."""

    def __init__(self, message: str, log_tail: Optional[List[str]] = None):
        tail = log_tail or []
        detail = "\n".join(tail[-ERROR_TAIL_LINES:])
        super().__init__(f"{message}\n--- vLLM log tail ---\n{detail}" if detail else message)
        self.log_tail = tail


# ---------------------------------------------------------------------------
# vLLM command
# ---------------------------------------------------------------------------

def lora_rank_for(r: Optional[int]) -> int:
    """Smallest vLLM-accepted ``--max-lora-rank`` that holds rank ``r``."""
    r = int(r or 16)
    for rank in VLLM_LORA_RANKS:
        if rank >= r:
            return rank
    raise ValueError(f"LoRA rank {r} exceeds vLLM's maximum ({VLLM_LORA_RANKS[-1]})")


def check_vllm_args(args: Sequence[str]) -> List[str]:
    """Reject pass-through flags that would undercut the worker's setup."""
    problems = []
    for arg in args:
        flag = str(arg).split("=", 1)[0]
        if flag in RESERVED_VLLM_FLAGS:
            problems.append(f"vllm_args may not set {flag} (Merlina manages it)")
    return problems


def build_vllm_command(serve: Dict[str, Any], *, model_ref: str,
                       lora_path: Optional[str] = None,
                       port: int = VLLM_INTERNAL_PORT,
                       executable: Sequence[str] = DEFAULT_VLLM_EXECUTABLE) -> List[str]:
    """
    The ``vllm serve`` argv for a serve spec. Never contains the API key —
    it travels as ``VLLM_API_KEY`` in the environment.

    For a LoRA session the base model is served under
    ``base_served_model_name`` and the adapter as a LoRA module named
    ``served_model_name``, so one endpoint answers both (base-vs-tuned evals).
    """
    lora = bool(serve.get("lora"))
    cmd = [*executable, "serve", model_ref,
           "--host", "127.0.0.1", "--port", str(port),
           "--served-model-name",
           str(serve.get("base_served_model_name") if lora else serve["served_model_name"]),
           "--tensor-parallel-size", str(int(serve.get("tensor_parallel_size") or 1)),
           "--max-model-len", str(int(serve["max_model_len"])),
           "--max-num-seqs", str(int(serve["max_num_seqs"])),
           "--gpu-memory-utilization", str(serve.get("gpu_memory_utilization") or 0.92)]
    if serve.get("revision") and serve.get("source_kind") == "hf":
        cmd += ["--revision", str(serve["revision"])]
    if lora:
        cmd += ["--enable-lora",
                "--lora-modules", f"{serve['served_model_name']}={lora_path or '<adapter>'}",
                "--max-lora-rank", str(int(serve.get("max_lora_rank") or 16))]
    cmd += [str(a) for a in serve.get("vllm_args") or []]
    return cmd


# ---------------------------------------------------------------------------
# API-key proxy
# ---------------------------------------------------------------------------

class ProxyStats:
    """Request accounting for idle detection (thread-safe)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self.requests = 0
        self.rejected = 0
        self.inflight = 0
        self._last_activity = clock()

    def begin(self) -> None:
        with self._lock:
            self.requests += 1
            self.inflight += 1
            self._last_activity = self._clock()

    def end(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            self._last_activity = self._clock()

    def reject(self) -> None:
        with self._lock:
            self.rejected += 1

    def reset_idle(self) -> None:
        with self._lock:
            self._last_activity = self._clock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            idle = 0.0 if self.inflight else max(0.0, self._clock() - self._last_activity)
            return {"requests": self.requests, "rejected_requests": self.rejected,
                    "inflight": self.inflight, "idle_seconds": round(idle, 1)}


def _bearer_ok(header: Optional[str], api_key: str) -> bool:
    scheme, _, token = (header or "").partition(" ")
    return (bool(api_key) and scheme.lower() == "bearer"
            and hmac.compare_digest(token.strip().encode(), api_key.encode()))


def make_proxy_handler(api_key: str, upstream_port: int, stats: ProxyStats,
                       upstream_host: str = "127.0.0.1"):
    """Handler that requires ``Authorization: Bearer <key>`` on every path and
    streams the request/response to vLLM (SSE included)."""

    class ServeProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, code: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> Optional[bytes]:
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                chunks = []
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        self.rfile.readline()
                        break
                    chunks.append(self.rfile.read(size))
                    self.rfile.readline()
                return b"".join(chunks)
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else None

        def _proxy(self) -> None:
            if not _bearer_ok(self.headers.get("Authorization"), api_key):
                stats.reject()
                self.close_connection = True        # the request body was never read
                self._reply(401, {"error": "Unauthorized"})
                return
            stats.begin()
            conn = None
            headers_sent = False
            try:
                body = self._read_body()
                conn = http.client.HTTPConnection(upstream_host, upstream_port,
                                                  timeout=UPSTREAM_TIMEOUT_S)
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                self.send_response(resp.status, resp.reason)
                for key, value in resp.getheaders():
                    if key.lower() not in _HOP_BY_HOP:
                        self.send_header(key, value)
                length = resp.getheader("Content-Length")
                if length is not None:
                    self.send_header("Content-Length", length)
                    self.end_headers()
                    headers_sent = True
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                else:
                    # Streamed (SSE) responses: relay each piece as it arrives.
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    headers_sent = True
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True       # client went away mid-stream
            except Exception as e:
                logger.warning("serve proxy upstream error: %s", type(e).__name__)
                if not headers_sent:
                    self._reply(502, {"error": "upstream unavailable"})
                self.close_connection = True
            finally:
                if conn is not None:
                    conn.close()
                stats.end()

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _proxy

        def log_message(self, fmt, *args):  # never log request lines/headers
            pass

    return ServeProxyHandler


def start_auth_proxy(api_key: str, port: int, upstream_port: int,
                     stats: ProxyStats) -> ThreadingHTTPServer:
    if not api_key:
        raise ServeError("Refusing to expose an inference endpoint without an API key")
    server = ThreadingHTTPServer(("0.0.0.0", port), make_proxy_handler(api_key, upstream_port, stats))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="merlina-serve-proxy").start()
    logger.info("Serve proxy listening on :%d (upstream vLLM on 127.0.0.1:%d)",
                server.server_address[1], upstream_port)
    return server


# ---------------------------------------------------------------------------
# vLLM process management
# ---------------------------------------------------------------------------

def read_log_tail(log_path: Path, lines: int = LOG_TAIL_LINES,
                  secrets: Sequence[str] = ()) -> List[str]:
    """Last ``lines`` lines of the vLLM log, with any secret redacted."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    all_lines = text.splitlines()
    tail = list(deque(all_lines, maxlen=lines))
    # vLLM's engine processes log the real error first; the API server then prints a long generic
    # traceback ("Engine core initialization failed. See root cause above") that fills the tail.
    cause = _root_cause(all_lines[:len(all_lines) - len(tail)])
    if cause:
        tail = cause + ["..."] + tail
    for secret in secrets:
        if secret:
            tail = [line.replace(secret, "***") for line in tail]
    return tail


_EXCEPTION_LINE = re.compile(r"\b\w*(?:Error|Exception)\b(?::|$)")


def _root_cause(lines: List[str], context: int = ROOT_CAUSE_CONTEXT_LINES) -> List[str]:
    """The first exception raised outside the API server process, with the lines leading up to it."""
    for i, line in enumerate(lines):
        if "(APIServer" not in line and _EXCEPTION_LINE.search(line) and "Traceback" not in line:
            return lines[max(0, i - context):i + 1]
    return []


def vllm_healthy(port: int, host: str = "127.0.0.1", timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop_process(proc: subprocess.Popen, grace_s: float = STOP_GRACE_SECONDS) -> None:
    """SIGTERM the process group, SIGKILL after ``grace_s``."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=10)


PULL_WORKERS = 8


def _resolve_model(payload: Dict[str, Any], serve: Dict[str, Any], work_dir: Path):
    """Return ``(model_ref, lora_path)``, pulling artifacts when needed."""
    if serve.get("source_kind") != "artifact":
        return serve["model_id"], None
    base = serve["model_id"]
    if serve.get("lora") and str(base).startswith("store:"):
        # The adapter's base is in object storage (store registered from the payload).
        from src.model_refs import resolve_model_ref
        base = resolve_model_ref(base)
    from src.remote.worker_entry import _resolve_store
    store = _resolve_store(payload)
    artifact = serve["artifact"]
    # Direct S3/R2 pulls run in parallel: a 150 GB model is dozens of shards,
    # and the GPUs bill while they download. Brokered pulls stay sequential.
    parallel = (store.kind == "s3"
                and (payload.get("artifact_store") or {}).get("credential_transport") != "broker")
    if parallel:
        local = store.pull_dir(artifact, work_dir / artifact, workers=PULL_WORKERS)
    else:
        local = store.pull_dir(artifact, work_dir / artifact)
    if serve.get("lora"):
        return base, str(local)
    return str(local), None


def _stop_requested(job_manager, job_id: str) -> bool:
    job = job_manager.get_job(job_id) if job_manager is not None else None
    return bool(job and job.stop_requested)


def run_serve_stage(payload: Dict[str, Any], job_manager, state, *,
                    executable: Optional[Sequence[str]] = None,
                    proxy_port: int = SERVE_PORT,
                    upstream_port: int = VLLM_INTERNAL_PORT,
                    work_dir: Optional[Path] = None,
                    poll_interval_s: float = 2.0,
                    env: Optional[Dict[str, str]] = None) -> None:
    """
    Serve until stopped. Returns normally after a requested stop; raises
    :class:`ServeError` when vLLM fails to start or dies (the caller marks
    the worker failed, and the log tail is already in ``state``).
    """
    env = dict(os.environ if env is None else env)
    api_key = env.get(API_KEY_ENV_VAR, "")
    if not api_key:
        raise ServeError(f"{API_KEY_ENV_VAR} is not set — refusing to serve an "
                         "unauthenticated inference endpoint")
    job_id = payload["job_id"]
    serve = payload["serve"]
    work_dir = Path(work_dir or "./serve_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    if job_manager is not None:
        job_manager.create_job(job_id, {"stage": "serve"})

    state.set_state(STATE_RUNNING)
    state.set_serve(phase="preparing_model", served_model_name=serve["served_model_name"],
                    base_served_model_name=serve.get("base_served_model_name"),
                    openai_base_path=OPENAI_BASE_PATH, port=proxy_port)
    model_ref, lora_path = _resolve_model(payload, serve, work_dir)

    cmd = build_vllm_command(serve, model_ref=model_ref, lora_path=lora_path,
                             port=upstream_port,
                             executable=executable or DEFAULT_VLLM_EXECUTABLE)
    proc_env = {k: v for k, v in env.items() if k not in _WORKER_ONLY_ENV}
    proc_env["VLLM_API_KEY"] = api_key
    log_path = work_dir / "vllm.log"
    logger.info("Launching vLLM: %s", shlex.join(cmd))
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=proc_env,
                                start_new_session=True)

    def tail(lines: int = LOG_TAIL_LINES) -> List[str]:
        return read_log_tail(log_path, lines, secrets=(api_key,))

    def fail(message: str) -> ServeError:
        lines = tail()
        state.set_serve(phase="failed", log_tail=lines)
        return ServeError(message, lines)

    proxy = None
    try:
        state.set_serve(phase="starting_vllm")
        startup_timeout_s = float(serve.get("startup_timeout_minutes")
                                  or DEFAULT_STARTUP_TIMEOUT_MIN) * 60
        started = time.monotonic()
        while not vllm_healthy(upstream_port):
            if proc.poll() is not None:
                raise fail(f"vLLM exited with code {proc.returncode} during startup")
            if _stop_requested(job_manager, job_id):
                logger.info("Stop requested before vLLM became ready")
                return
            if time.monotonic() - started > startup_timeout_s:
                raise fail(f"vLLM did not become healthy within {startup_timeout_s / 60:.0f} minutes")
            time.sleep(poll_interval_s)

        stats = ProxyStats()
        proxy = start_auth_proxy(api_key, proxy_port, upstream_port, stats)
        state.set_serve(phase="serving", port=proxy.server_address[1])
        state.set_serve_probe(stats.snapshot)
        state.set_state(STATE_SERVING)
        logger.info("Serving %r via the OpenAI-compatible API on :%d%s",
                    serve["served_model_name"], proxy.server_address[1], OPENAI_BASE_PATH)

        while True:
            if proc.poll() is not None:
                raise fail(f"vLLM exited unexpectedly with code {proc.returncode}")
            if _stop_requested(job_manager, job_id):
                logger.info("Stop requested; shutting vLLM down")
                break
            time.sleep(poll_interval_s)
        state.set_serve(phase="stopped")
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        stop_process(proc)
