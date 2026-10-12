"""
Tests for the worker status server (src/remote/worker_entry.py) and the
HTTP client the orchestrator uses to poll it — exercised together over a
real localhost HTTP roundtrip.

Run standalone: python -m pytest tests/test_remote_worker_protocol.py
"""

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_manager import JobManager
from src.remote.worker_client import (
    STATE_DONE,
    STATE_FAILED,
    STATE_RUNNING,
    HttpWorkerClient,
)
from src.remote.worker_entry import (
    JOB_ENV_VAR,
    WorkerState,
    build_status_payload,
    decode_job_payload,
    start_status_server,
)

TOKEN = "test-token-123"


@pytest.fixture
def job_manager(tmp_path):
    return JobManager(db_path=str(tmp_path / "jobs.db"))


@pytest.fixture
def worker(job_manager):
    state = WorkerState("job_1", "train")
    server = start_status_server(state, job_manager, TOKEN, port=0)
    port = server.server_address[1]
    yield state, job_manager, f"http://127.0.0.1:{port}"
    server.shutdown()


class TestWorkerState:
    def test_lifecycle(self):
        state = WorkerState("j", "train")
        assert state.snapshot()["state"] == "starting"
        state.set_state(STATE_RUNNING)
        state.add_artifact("adapter")
        snap = state.snapshot()
        assert snap["state"] == STATE_RUNNING
        assert snap["artifacts"] == ["adapter"]

    def test_fail_records_error(self):
        state = WorkerState("j", "train")
        state.fail("CUDA OOM")
        snap = state.snapshot()
        assert snap["state"] == STATE_FAILED
        assert snap["error"] == "CUDA OOM"


class TestStatusPayload:
    def test_mirrors_job_and_filters_metrics(self, job_manager):
        state = WorkerState("job_1", "train")
        job_manager.create_job("job_1", {"output_name": "m"})
        job_manager.update_job("job_1", status="training", progress=0.5,
                               current_step=30, total_steps=100, loss=1.23)
        for step in (10, 20, 30):
            job_manager.add_metric("job_1", step=step, loss=2.0 - step / 100)

        payload = build_status_payload(state, job_manager, since_step=10)
        assert payload["job"]["status"] == "training"
        assert payload["job"]["loss"] == 1.23
        assert [m["step"] for m in payload["new_metrics"]] == [20, 30]

    def test_no_job_yet(self, job_manager):
        state = WorkerState("job_unknown", "train")
        payload = build_status_payload(state, job_manager)
        assert payload["job"] == {}
        assert payload["new_metrics"] == []


class TestHttpRoundtrip:
    def test_health_and_status(self, worker):
        state, jm, url = worker
        client = HttpWorkerClient(url, TOKEN)
        assert client.health()

        jm.create_job("job_1", {})
        jm.update_job("job_1", status="training", progress=0.4)
        state.set_state(STATE_RUNNING)

        status = client.status()
        assert status.state == STATE_RUNNING
        assert status.job["status"] == "training"
        assert not status.is_terminal

    def test_terminal_state_with_artifacts(self, worker):
        state, jm, url = worker
        state.set_state(STATE_DONE)
        state.add_artifact("adapter")
        status = HttpWorkerClient(url, TOKEN).status()
        assert status.is_terminal
        assert status.artifacts == ["adapter"]

    def test_wrong_token_rejected(self, worker):
        _, _, url = worker
        client = HttpWorkerClient(url, "wrong-token")
        assert not client.health()
        with pytest.raises(Exception):
            client.status()

    def test_stop_sets_flag_in_db(self, worker):
        state, jm, url = worker
        jm.create_job("job_1", {})
        assert HttpWorkerClient(url, TOKEN).request_stop()
        assert jm.get_job("job_1").stop_requested

    def test_since_step_filters(self, worker):
        state, jm, url = worker
        jm.create_job("job_1", {})
        for step in (5, 15):
            jm.add_metric("job_1", step=step, loss=1.0)
        status = HttpWorkerClient(url, TOKEN).status(since_step=5)
        assert [m["step"] for m in status.new_metrics] == [15]


class TestPayloadDecoding:
    def test_roundtrip(self):
        payload = {"job_id": "j", "stage": "train", "config": {"output_name": "m"}}
        env = {JOB_ENV_VAR: base64.b64encode(json.dumps(payload).encode()).decode()}
        assert decode_job_payload(env) == payload

    def test_missing_env_raises(self):
        with pytest.raises(RuntimeError, match=JOB_ENV_VAR):
            decode_job_payload({})


class TestMergeStagePublishing:
    """push_to_hub publishes only to the Hub; otherwise the store gets it."""

    def _run(self, tmp_path, monkeypatch, job_manager, push_to_hub):
        import types
        import huggingface_hub
        from src.remote import worker_entry
        from src.remote.artifacts import LocalArtifactStore

        monkeypatch.chdir(tmp_path)
        store = LocalArtifactStore(tmp_path / "store")
        adapter = tmp_path / "adapter-src"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        store.push_dir(adapter, "adapter")
        monkeypatch.setattr(worker_entry, "_resolve_store", lambda payload: store)

        def fake_merge(base, adapter_dir, merged_dir, model_type="auto"):
            Path(merged_dir).mkdir(parents=True)
            (Path(merged_dir) / "model.safetensors").write_text("merged")
        monkeypatch.setitem(sys.modules, "src.gguf_exporter",
                            types.SimpleNamespace(merge_lora_to_directory=fake_merge))
        uploads = []

        class FakeApi:
            def __init__(self, token=None):
                pass

            def create_repo(self, *a, **kw):
                pass

            def upload_folder(self, *, folder_path, repo_id, **kw):
                uploads.append(repo_id)
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

        state = WorkerState("job_m", "merge")
        worker_entry.run_merge_stage(
            {"job_id": "job_m", "stage": "merge", "artifacts_out": ["merged"],
             "config": {"base_model": "base", "output_name": "me/model",
                        "push_to_hub": push_to_hub}},
            job_manager, state)
        return store, uploads, state.snapshot()["artifacts"]

    def test_push_to_hub_skips_store_upload(self, tmp_path, monkeypatch, job_manager):
        store, uploads, artifacts = self._run(tmp_path, monkeypatch, job_manager, True)
        assert uploads == ["me/model"]
        assert not store.exists("merged")
        assert artifacts == ["hub:me/model"]

    def test_without_hub_pushes_to_store(self, tmp_path, monkeypatch, job_manager):
        store, uploads, artifacts = self._run(tmp_path, monkeypatch, job_manager, False)
        assert uploads == []
        assert store.exists("merged")
        assert artifacts == ["merged"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
