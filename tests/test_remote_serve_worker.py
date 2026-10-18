"""
Tests for the worker-side serve stage (src/remote/serve_worker.py).

A tiny Python HTTP server (tests/fixtures/fake_vllm.py) stands in for
``vllm serve``; the worker status API, the API-key proxy and the fake vLLM
are exercised together over real localhost HTTP, the same way the control
plane and an eval harness would reach them through the provider proxy.

Run standalone: python -m pytest tests/test_remote_serve_worker.py
"""

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_manager import JobManager
from src.remote.serve_worker import (
    API_KEY_ENV_VAR,
    ServeError,
    build_vllm_command,
    check_vllm_args,
    lora_rank_for,
    read_log_tail,
    run_serve_stage,
)
from src.remote.worker_client import (
    STATE_FAILED,
    STATE_RUNNING,
    STATE_SERVING,
    HttpWorkerClient,
)
from src.remote.worker_entry import STAGE_RUNNERS, WorkerState, start_status_server

FAKE_VLLM = Path(__file__).parent / "fixtures" / "fake_vllm.py"
WORKER_TOKEN = "worker-token-abc"
API_KEY = "serve-key-0123456789abcdef0123456789abcdef"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_payload(**serve):
    spec = {"source_kind": "hf", "model_id": "org/model", "revision": None, "lora": False,
            "served_model_name": "model", "base_served_model_name": None,
            "tensor_parallel_size": 1, "max_model_len": 2048, "max_num_seqs": 4,
            "gpu_memory_utilization": 0.9, "vllm_args": [], "startup_timeout_minutes": 1}
    spec.update(serve)
    return {"job_id": "inf_test", "stage": "serve", "serve": spec,
            "config": {}, "artifact_store": {}}


class Harness:
    """Runs run_serve_stage on a thread with the worker status API up."""

    def __init__(self, tmp_path, *, mode="ok", payload=None, env_extra=None):
        import os
        self.tmp = tmp_path
        self.job_manager = JobManager(db_path=str(tmp_path / "worker.db"))
        self.state = WorkerState("inf_test", "serve")
        self.status_server = start_status_server(self.state, self.job_manager, WORKER_TOKEN, port=0)
        self.client = HttpWorkerClient(
            f"http://127.0.0.1:{self.status_server.server_address[1]}", WORKER_TOKEN)
        self.proxy_port, self.upstream_port = free_port(), free_port()
        self.base = f"http://127.0.0.1:{self.proxy_port}"
        self.record = tmp_path / "fake_vllm.json"
        self.env = {"PATH": os.environ.get("PATH", ""), API_KEY_ENV_VAR: API_KEY,
                    "MERLINA_WORKER_TOKEN": WORKER_TOKEN, "MERLINA_REMOTE_JOB_B64": "e30=",
                    "FAKE_VLLM_MODE": mode, "FAKE_VLLM_RECORD": str(self.record),
                    **(env_extra or {})}
        self.payload = payload or serve_payload()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            run_serve_stage(self.payload, self.job_manager, self.state,
                            executable=[sys.executable, str(FAKE_VLLM)],
                            proxy_port=self.proxy_port, upstream_port=self.upstream_port,
                            work_dir=self.tmp / "serve_work", poll_interval_s=0.05, env=self.env)
        except Exception as e:      # what worker_entry.main does
            self.error = e
            self.state.fail(str(e))

    def start(self):
        self.thread.start()
        return self

    def wait_for(self, *states, timeout=20):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            status = self.client.status()
            if status.state in states:
                return status
            time.sleep(0.05)
        raise AssertionError(f"worker never reached {states}; last {status}")

    def close(self):
        self.job_manager.request_stop("inf_test")
        self.thread.join(timeout=60)
        self.status_server.shutdown()


@pytest.fixture
def serving(tmp_path):
    harness = Harness(tmp_path).start()
    harness.wait_for(STATE_SERVING)
    yield harness
    harness.close()


def auth(key=API_KEY):
    return {"Authorization": f"Bearer {key}"}


class TestServingRoundTrip:
    def test_status_reports_serving_endpoint(self, serving):
        status = serving.client.status()
        assert status.state == STATE_SERVING
        assert status.serve["openai_base_path"] == "/v1"
        assert status.serve["served_model_name"] == "model"
        assert status.serve["port"] == serving.proxy_port
        assert status.serve["phase"] == "serving"

    def test_every_path_requires_the_key(self, serving):
        for method, path in (("GET", "/v1/models"), ("GET", "/health"), ("GET", "/metrics"),
                             ("POST", "/invocations"), ("POST", "/v1/chat/completions")):
            assert requests.request(method, serving.base + path, json={}).status_code == 401, path
        assert requests.get(serving.base + "/v1/models", headers=auth("wrong")).status_code == 401
        assert requests.get(serving.base + "/v1/models",
                            headers={"Authorization": API_KEY}).status_code == 401
        assert serving.client.status().serve["rejected_requests"] >= 7

    def test_openai_requests_pass_through(self, serving):
        models = requests.get(serving.base + "/v1/models", headers=auth())
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "model"
        chat = requests.post(serving.base + "/v1/chat/completions", headers=auth(), json={
            "model": "model", "messages": [{"role": "user", "content": "hello wizard"}]})
        assert chat.status_code == 200
        assert chat.json()["choices"][0]["message"]["content"] == "echo: hello wizard"

    def test_streaming_responses_relay(self, serving):
        resp = requests.post(serving.base + "/v1/chat/completions", headers=auth(), stream=True,
                             json={"model": "model", "stream": True,
                                   "messages": [{"role": "user", "content": "one two three"}]})
        assert resp.status_code == 200
        events = [line for line in resp.iter_lines() if line]
        assert events[-1] == b"data: [DONE]"
        words = [json.loads(e[6:])["choices"][0]["delta"]["content"] for e in events[:-1]]
        assert words == ["echo:", "one", "two", "three"]

    def test_request_accounting_for_idle_detection(self, serving):
        before = serving.client.status().serve
        time.sleep(0.3)
        assert serving.client.status().serve["idle_seconds"] >= 0.2
        requests.get(serving.base + "/v1/models", headers=auth())
        after = serving.client.status().serve
        assert after["requests"] == before["requests"] + 1
        assert after["idle_seconds"] < 0.2
        assert after["inflight"] == 0

    def test_vllm_gets_key_via_env_only(self, serving):
        record = json.loads(serving.record.read_text())
        argv = record["argv"]
        assert argv[argv.index("--host") + 1] == "127.0.0.1"     # never public
        assert argv[argv.index("--port") + 1] == str(serving.upstream_port)
        assert "--api-key" not in argv and API_KEY not in argv
        assert record["env"]["VLLM_API_KEY"] == API_KEY
        # Worker credentials are not handed to vLLM.
        assert record["env"]["MERLINA_WORKER_TOKEN"] is None
        assert record["env"]["MERLINA_SERVE_API_KEY"] is None
        assert record["env"]["MERLINA_REMOTE_JOB_B64"] is None

    def test_stop_shuts_vllm_down(self, tmp_path):
        harness = Harness(tmp_path).start()
        harness.wait_for(STATE_SERVING)
        assert harness.client.request_stop()
        harness.thread.join(timeout=60)
        assert not harness.thread.is_alive()
        assert harness.error is None
        assert harness.state.snapshot()["serve"]["phase"] == "stopped"
        with pytest.raises(requests.ConnectionError):
            requests.get(f"http://127.0.0.1:{harness.upstream_port}/health", timeout=2)
        with pytest.raises(requests.ConnectionError):
            requests.get(harness.base + "/v1/models", headers=auth(), timeout=2)
        harness.status_server.shutdown()


class TestServeFailures:
    def test_startup_crash_reports_scrubbed_log_tail(self, tmp_path):
        harness = Harness(tmp_path, mode="crash").start()
        status = harness.wait_for(STATE_FAILED)
        harness.thread.join(timeout=10)
        assert isinstance(harness.error, ServeError)
        assert "exited with code 3 during startup" in status.error
        assert "CUDA out of memory" in status.error
        tail = status.serve["log_tail"]
        assert any("CUDA out of memory" in line for line in tail)
        assert API_KEY not in json.dumps(status.serve) and API_KEY not in status.error
        assert any("api_key=***" in line for line in tail)
        harness.status_server.shutdown()

    def test_vllm_dying_while_serving_fails_the_worker(self, tmp_path):
        harness = Harness(tmp_path, mode="die").start()
        status = harness.wait_for(STATE_FAILED)
        assert "exited unexpectedly with code 4" in status.error
        harness.thread.join(timeout=10)
        harness.status_server.shutdown()

    def test_refuses_to_serve_without_a_key(self, tmp_path):
        harness = Harness(tmp_path, env_extra={API_KEY_ENV_VAR: ""}).start()
        status = harness.wait_for(STATE_FAILED)
        assert "refusing" in status.error
        assert not harness.record.exists()          # vLLM never launched
        harness.thread.join(timeout=10)
        harness.status_server.shutdown()

    def test_stop_before_ready_returns_cleanly(self, tmp_path):
        harness = Harness(tmp_path, mode="slow").start()     # never becomes healthy
        harness.wait_for(STATE_RUNNING)
        assert harness.client.request_stop()
        harness.thread.join(timeout=60)
        assert harness.error is None
        assert harness.state.snapshot()["state"] == STATE_RUNNING   # main() then marks done
        harness.status_server.shutdown()


class TestArtifactSource:
    def test_adapter_is_pulled_and_served_as_lora(self, tmp_path):
        from src.remote.artifacts import LocalArtifactStore
        store = LocalArtifactStore(tmp_path / "store")
        adapter = tmp_path / "adapter_src"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        store.push_dir(adapter, "adapter")
        payload = serve_payload(source_kind="artifact", artifact="adapter", lora=True,
                                model_id="org/base", base_served_model_name="org/base",
                                served_model_name="my-tune", max_lora_rank=64)
        payload["artifact_store"] = store.describe()
        harness = Harness(tmp_path, payload=payload).start()
        try:
            harness.wait_for(STATE_SERVING)
            argv = json.loads(harness.record.read_text())["argv"]
            assert argv[1] == "org/base"
            assert argv[argv.index("--served-model-name") + 1] == "org/base"
            name, path = argv[argv.index("--lora-modules") + 1].split("=", 1)
            assert name == "my-tune"
            assert (Path(path) / "adapter_config.json").exists()
            assert argv[argv.index("--max-lora-rank") + 1] == "64"
        finally:
            harness.close()


class TestHelpers:
    def test_serve_stage_registered(self):
        assert "serve" in STAGE_RUNNERS

    def test_lora_rank_rounding(self):
        assert lora_rank_for(8) == 8
        assert lora_rank_for(48) == 64
        assert lora_rank_for(None) == 16
        with pytest.raises(ValueError):
            lora_rank_for(1024)

    def test_reserved_flags(self):
        assert check_vllm_args(["--enable-prefix-caching", "--dtype", "bfloat16"]) == []
        assert check_vllm_args(["--api-key=x"]) and check_vllm_args(["--host", "0.0.0.0"])

    def test_command_merged_artifact_uses_local_path(self):
        cmd = build_vllm_command(serve_payload(source_kind="artifact")["serve"],
                                 model_ref="/w/merged")
        assert cmd[:3] == ["vllm", "serve", "/w/merged"]
        assert "--enable-lora" not in cmd and "--revision" not in cmd

    def test_log_tail_redacts(self, tmp_path):
        log = tmp_path / "log"
        log.write_text("".join(f"line {i} key={API_KEY}\n" for i in range(100)))
        tail = read_log_tail(log, 10, secrets=(API_KEY,))
        assert len(tail) == 10 and tail[-1] == "line 99 key=***"
        assert read_log_tail(tmp_path / "missing") == []

    def test_log_tail_keeps_engine_root_cause(self, tmp_path):
        log = tmp_path / "log"
        lines = [f"(EngineCore_DP0 pid=7) INFO line {i}" for i in range(50)]
        lines += ["(EngineCore_DP0 pid=7) Traceback (most recent call last):",
                  "(EngineCore_DP0 pid=7)   File \"x.py\", line 1, in f",
                  f"(EngineCore_DP0 pid=7) AttributeError: 'Linear' has no attribute 'weight_scale_inv' {API_KEY}"]
        lines += ["(APIServer pid=1) Traceback (most recent call last):"]
        lines += [f"(APIServer pid=1)   frame {i}" for i in range(80)]
        lines += ["(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above."]
        log.write_text("\n".join(lines) + "\n")
        tail = read_log_tail(log, 10, secrets=(API_KEY,))
        cut = tail.index("...")
        assert tail[cut - 1].endswith("AttributeError: 'Linear' has no attribute 'weight_scale_inv' ***")
        assert tail[cut - 3].endswith("Traceback (most recent call last):")
        assert tail[cut + 1:] == lines[-10:]
        # no error before the tail: nothing extra
        log.write_text("".join(f"line {i}\n" for i in range(100)))
        assert "..." not in read_log_tail(log, 10)
