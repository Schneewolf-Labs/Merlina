"""
API tests for remote inference sessions (/remote/inference*) in merlina.py.
Provider calls, Hub lookups and the session thread are all stubbed.

Run standalone: python -m pytest tests/test_remote_inference_api.py
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import merlina
from src.job_manager import JobManager
from src.remote.sessions import InferenceSessionStore, forget_api_key, remember_api_key
from src.remote.sizing import parse_model_specs
from tests.test_remote_sizing import LLAMA_7B_CONFIG

BODY = {"model": {"hf_repo": "org/model"}, "max_runtime_hours": 1, "hf_token": "hf_secret"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    jobs = JobManager(db_path=str(tmp_path / "jobs.db"))
    sessions = InferenceSessionStore(jobs.db_path)
    launched = []

    async def no_offers(provider_name):      # never contact a real provider
        return None

    from src.remote import inference
    monkeypatch.setattr(inference, "fetch_model_specs",
                        lambda model_id, hf_token=None, revision=None:
                        parse_model_specs(model_id, LLAMA_7B_CONFIG))
    monkeypatch.setattr(merlina, "_live_gpu_offers", no_offers)
    monkeypatch.setattr(merlina, "job_manager", jobs)
    monkeypatch.setattr(merlina, "_inference_sessions", sessions)
    monkeypatch.setattr(merlina, "_launch_inference_session",
                        lambda sid, request, plan: launched.append((sid, request, plan)))
    monkeypatch.setattr(merlina.settings, "runpod_api_key", "test-only-key")
    monkeypatch.setattr(merlina.settings, "remote_max_inference_sessions", 2)
    client = TestClient(merlina.app)
    client.sessions, client.launched, client.jobs = sessions, launched, jobs
    return client


class TestPlanEndpoint:
    def test_plan_is_a_dry_run(self, api):
        resp = api.post("/remote/inference/plan", json=BODY)
        assert resp.status_code == 200, resp.text
        plan = resp.json()
        assert plan["sizing"]["gpu_type_id"]
        assert plan["sizing"]["est_cost_per_hr"] > 0
        assert plan["vram_breakdown"]["total_gb"] > 0
        assert plan["tensor_parallel_size"] == plan["sizing"]["gpu_count"]
        assert plan["effective_max_hours"] == 1
        assert plan["est_max_total_cost_usd"] > 0
        assert api.launched == [] and api.sessions.list() == []
        assert "hf_secret" not in resp.text

    def test_plan_errors_are_400(self, api):
        resp = api.post("/remote/inference/plan",
                        json={"model": {"job_id": "missing"}})
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]

    def test_validation_errors_are_422(self, api):
        assert api.post("/remote/inference/plan", json={"model": {}}).status_code == 422
        assert api.post("/remote/inference/plan",
                        json={**BODY, "max_runtime_hours": 48}).status_code == 422
        assert api.post("/remote/inference/plan",
                        json={**BODY, "vllm_args": ["--api-key", "x"]}).status_code == 422


class TestSessionLifecycleEndpoints:
    def test_start_creates_session_and_launches(self, api):
        resp = api.post("/remote/inference", json=BODY)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        sid = data["session_id"]
        assert sid.startswith("inf_")
        assert data["status"] == "queued"
        assert data["plan"]["served_model_name"] == "model"
        assert len(api.launched) == 1 and api.launched[0][0] == sid
        assert api.launched[0][1].hf_token == "hf_secret"      # handed to the thread only
        stored = api.sessions.get(sid)
        assert "hf_token" not in stored.request
        assert stored.plan["sizing"] == data["plan"]["sizing"]

    def test_start_requires_provider_key(self, api, monkeypatch):
        monkeypatch.setattr(merlina.settings, "runpod_api_key", None)
        resp = api.post("/remote/inference", json=BODY)
        assert resp.status_code == 400
        assert api.launched == []

    def test_concurrency_cap(self, api):
        for _ in range(2):
            assert api.post("/remote/inference", json=BODY).status_code == 200
        resp = api.post("/remote/inference", json=BODY)
        assert resp.status_code == 409
        assert len(api.launched) == 2

    def test_get_returns_key_only_while_active_and_list_never(self, api):
        sid = api.post("/remote/inference", json=BODY).json()["session_id"]
        api.sessions.update(sid, status="ready", base_url="https://pod-8001.example/v1",
                            served_model_name="model")
        remember_api_key(sid, "sk-live-key")
        try:
            one = api.get(f"/remote/inference/{sid}").json()
            assert one["api_key"] == "sk-live-key"
            assert one["base_url"] == "https://pod-8001.example/v1"
            assert one["openai"]["model"] == "model"
            assert "sk-live-key" not in one["openai"]["example"]
            listing = api.get("/remote/inference")
            assert listing.status_code == 200
            assert [s["session_id"] for s in listing.json()["sessions"]] == [sid]
            assert "sk-live-key" not in listing.text and "api_key" not in listing.text
            api.sessions.update(sid, status="stopped")
            assert api.get(f"/remote/inference/{sid}").json()["api_key"] is None
        finally:
            forget_api_key(sid)

    def test_stop(self, api):
        sid = api.post("/remote/inference", json=BODY).json()["session_id"]
        api.sessions.update(sid, status="ready")
        resp = api.post(f"/remote/inference/{sid}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopping"
        assert "api_key" not in resp.json()
        assert api.sessions.get(sid).stop_requested is True
        # Stopping a finished session is a no-op
        api.sessions.update(sid, status="stopped")
        assert api.post(f"/remote/inference/{sid}/stop").json()["status"] == "stopped"

    def test_unknown_session_404(self, api):
        assert api.get("/remote/inference/inf_nope").status_code == 404
        assert api.post("/remote/inference/inf_nope/stop").status_code == 404

    def test_sessions_do_not_appear_as_jobs(self, api):
        api.post("/remote/inference", json=BODY)
        assert api.jobs.list_jobs() == []


class TestSourcesEndpoint:
    def test_lists_store_models_and_remote_jobs(self, api, monkeypatch):
        from src.remote import artifacts
        from src.remote.lifecycle import RemoteLifecycle
        from tests.test_remote_inference_plan import finished_remote_job

        class Store:
            tier = "external"

            def list_artifacts(self):
                return [{"name": "C0", "files": 53, "size": 166_000_000_000, "created_at": "x"}]

        monkeypatch.setattr(artifacts, "model_store", lambda settings=None: Store())
        db = (api.jobs, RemoteLifecycle(api.jobs.db_path))
        finished_remote_job(db, "job_r", artifacts=("adapter", "merged"))
        finished_remote_job(db, "job_local", remote=False)
        resp = api.get("/remote/inference/sources")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["store_models"][0]["name"] == "C0"
        assert data["store_tier"] == "external" and data["store_error"] is None
        assert [j["job_id"] for j in data["jobs"]] == ["job_r"]
        assert data["jobs"][0]["artifacts"] == ["adapter", "merged"]
        assert "hf_job" not in resp.text

    def test_store_error_is_reported_not_raised(self, api, monkeypatch):
        from src.remote import artifacts
        from src.remote.artifacts import ArtifactStoreError

        def no_store(settings=None):
            raise ArtifactStoreError("needs ARTIFACT_STORE=external")

        monkeypatch.setattr(artifacts, "model_store", no_store)
        data = api.get("/remote/inference/sources").json()
        assert data["store_models"] == [] and "ARTIFACT_STORE" in data["store_error"]


class TestChatEndpoint:
    def test_relays_to_the_session_endpoint_with_its_key(self, api, monkeypatch):
        import io
        import json as _json
        import urllib.request

        sid = api.post("/remote/inference", json=BODY).json()["session_id"]
        api.sessions.update(sid, status="ready", base_url="https://pod-8001.example/v1",
                            served_model_name="model")
        remember_api_key(sid, "sk-live-key")
        seen = {}

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            seen["url"], seen["auth"] = req.full_url, req.headers.get("Authorization")
            seen["ua"] = req.get_header("User-agent")
            seen["body"] = _json.loads(req.data)
            return Resp(_json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]}).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        try:
            resp = api.post(f"/remote/inference/{sid}/chat",
                            json={"messages": [{"role": "user", "content": "hello"}]})
            assert resp.status_code == 200, resp.text
            assert resp.json()["choices"][0]["message"]["content"] == "hi"
            assert seen["url"] == "https://pod-8001.example/v1/chat/completions"
            assert seen["auth"] == "Bearer sk-live-key"
            assert seen["ua"].startswith("merlina/"), "Cloudflare blocks the urllib default"
            assert seen["body"]["model"] == "model"
        finally:
            forget_api_key(sid)

    def test_refuses_sessions_that_are_not_ready(self, api):
        sid = api.post("/remote/inference", json=BODY).json()["session_id"]
        resp = api.post(f"/remote/inference/{sid}/chat",
                        json={"messages": [{"role": "user", "content": "x"}]})
        assert resp.status_code == 409
        assert api.post("/remote/inference/inf_nope/chat",
                        json={"messages": [{"role": "user", "content": "x"}]}).status_code == 404
