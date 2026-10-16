"""
Tests for remote inference sessions in the orchestrator
(run_inference_remote) and their persistence/reconciliation
(src/remote/sessions.py) — fake provider and scripted worker, no network.

Run standalone: python -m pytest tests/test_remote_inference_orchestrator.py
"""

import base64
import json
import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_manager import JobManager
from src.remote.cleanup import reconcile_once
from src.remote.inference import RemoteInferenceRequest
from src.remote.lifecycle import RemoteLifecycle
from src.remote.orchestrator import run_inference_remote, run_training_remote
from src.remote.sessions import (
    InferenceSessionStore,
    RemoteRecords,
    api_key_for,
    new_session_id,
    with_inference_sessions,
)
from src.remote.sizing import parse_model_specs
from src.remote.spec import RemoteInstance
from src.remote.worker_client import (
    STATE_DONE,
    STATE_FAILED,
    STATE_RUNNING,
    STATE_SERVING,
    WorkerClient,
    WorkerStatus,
)
from tests.test_remote_orchestrator import FakeProvider
from tests.test_remote_sizing import LLAMA_7B_CONFIG

API_KEY_VAR = "MERLINA_SERVE_API_KEY"


@pytest.fixture(autouse=True)
def fake_model_specs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.remote import inference
    monkeypatch.setattr(inference, "fetch_model_specs",
                        lambda model_id, hf_token=None, revision=None:
                        parse_model_specs(model_id, LLAMA_7B_CONFIG))


@pytest.fixture
def sessions(tmp_path):
    return InferenceSessionStore(tmp_path / "jobs.db")


class ConfirmingProvider(FakeProvider):
    """FakeProvider whose inventory confirms terminations (like RunPod's 404)."""

    def get_instance(self, instance_id):
        status = "terminated" if instance_id in self.terminated else "running"
        return RemoteInstance(instance_id=instance_id, provider=self.name, status=status)


class ServeWorker(WorkerClient):
    """Scripted worker; ``hook(poll_number, worker)`` runs before each status."""

    def __init__(self, script, hook=None):
        self.script = list(script)
        self.hook = hook
        self.polls = 0
        self.stop_requests = 0

    def health(self):
        return True

    def status(self, since_step=0):
        self.polls += 1
        if self.hook:
            self.hook(self.polls, self)
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]

    def request_stop(self):
        self.stop_requests += 1
        return True


SERVING = WorkerStatus(state=STATE_SERVING, serve={
    "openai_base_path": "/v1", "served_model_name": "model", "requests": 3,
    "inflight": 0, "idle_seconds": 1.0})


def start(sessions, worker, *, provider=None, guard=None, job_manager=None, **request):
    provider = provider or ConfirmingProvider()
    body = {"model": {"hf_repo": "org/model"}}
    body.update(request)
    req = RemoteInferenceRequest(**body)
    session_id = new_session_id()
    sessions.create(session_id, req.model_dump())
    run_inference_remote(
        session_id, req, sessions, provider=provider, job_manager=job_manager,
        worker_client_factory=lambda url, token: worker,
        poll_interval_s=0.01, boot_timeout_min=0.01, execution_guard=guard,
    )
    return session_id, provider


def stop_after(polls, sessions):
    def hook(n, worker):
        if n == polls:
            sid = sessions.list()[0].session_id
            sessions.request_stop(sid)
    return hook


class TestServeHappyPath:
    def test_ready_then_stop_then_terminated(self, sessions):
        seen = {}

        def hook(n, worker):
            sid = sessions.list()[0].session_id
            current = sessions.get(sid)
            if current.status == "ready" and "ready" not in seen:
                seen["ready"] = current
                seen["key"] = api_key_for(sid)
                sessions.request_stop(sid)

        worker = ServeWorker([WorkerStatus(state=STATE_RUNNING), SERVING], hook=hook)
        sid, provider = start(sessions, worker)

        ready = seen["ready"]
        assert ready.base_url == "http://fake/inst_1:8001/v1"
        assert ready.served_model_name == "model"
        assert ready.ready_at is not None
        final = sessions.get(sid)
        assert final.status == "stopped"
        assert final.stop_reason == "user"
        assert provider.terminated == ["inst_1"]
        assert final.metrics["remote_resources"][0]["cleanup_status"] == "terminated"
        # The key was available while ready and is forgotten afterwards.
        assert seen["key"] == provider.provisioned[0].env[API_KEY_VAR]
        assert api_key_for(sid) is None

    def test_spec_exposes_worker_and_serve_ports(self, sessions):
        worker = ServeWorker([SERVING], hook=stop_after(2, sessions))
        _, provider = start(sessions, worker)
        spec = provider.provisioned[0]
        assert spec.http_ports == [8000, 8001]
        assert spec.docker_start_cmd == ["python3", "-m", "src.remote.worker_entry"]
        assert spec.image == "ghcr.io/schneewolf-labs/merlina-serve:latest"

    def test_api_key_in_env_separate_from_worker_token_and_never_logged(self, sessions, caplog):
        caplog.set_level(logging.DEBUG)
        worker = ServeWorker([SERVING], hook=stop_after(3, sessions))
        sid, provider = start(sessions, worker, hf_token="hf_req")
        env = provider.provisioned[0].env
        key = env[API_KEY_VAR]
        assert len(key) >= 40
        assert key != env["MERLINA_WORKER_TOKEN"]
        assert env["HF_TOKEN"] == "hf_req"
        payload = json.loads(base64.b64decode(env["MERLINA_REMOTE_JOB_B64"]))
        assert payload["stage"] == "serve"
        assert payload["serve"]["served_model_name"] == "model"
        assert key not in json.dumps(payload) and "hf_req" not in json.dumps(payload)
        assert key not in caplog.text
        # Nothing secret is persisted with the session either.
        row = json.dumps(sessions.get(sid).to_public_dict())
        assert key not in row and "hf_req" not in row

    def test_serve_stats_mirrored(self, sessions):
        worker = ServeWorker([SERVING], hook=stop_after(3, sessions))
        sid, _ = start(sessions, worker)
        assert sessions.get(sid).metrics["serve"]["requests"] == 3

    def test_lifecycle_labels_the_serve_stage(self, sessions):
        worker = ServeWorker([SERVING], hook=stop_after(2, sessions))
        sid, provider = start(sessions, worker)
        attempts = RemoteLifecycle(sessions.db_path).attempts()
        assert [(a["job_id"], a["stage_name"], a["state"]) for a in attempts] == [
            (sid, "serve", "completed")]
        assert provider.provisioned[0].name == attempts[0]["label"]

    def test_plan_built_from_provider_offers(self, sessions):
        worker = ServeWorker([SERVING], hook=stop_after(2, sessions))
        sid, provider = start(sessions, worker)
        plan = sessions.get(sid).plan
        assert plan["sizing"]["gpu_type_id"] in ("FAKE-24G", "FAKE-141G")
        assert provider.provisioned[0].gpu_type_id == plan["sizing"]["gpu_type_id"]


class TestServeFailures:
    def test_worker_failure_terminates_and_records_log_tail(self, sessions):
        failed = WorkerStatus(state=STATE_FAILED, error="vLLM exited with code 1 during startup",
                              serve={"log_tail": ["torch.OutOfMemoryError: CUDA out of memory"]})
        sid, provider = start(sessions, ServeWorker([WorkerStatus(state=STATE_RUNNING), failed]))
        final = sessions.get(sid)
        assert final.status == "failed"
        assert "exited with code 1" in final.error
        assert final.metrics["log_tail"] == ["torch.OutOfMemoryError: CUDA out of memory"]
        assert provider.terminated == ["inst_1"]
        assert api_key_for(sid) is None

    def test_vllm_dies_while_serving(self, sessions):
        worker = ServeWorker([SERVING, SERVING, WorkerStatus(state=STATE_FAILED, error="vLLM died")])
        sid, provider = start(sessions, worker)
        assert sessions.get(sid).status == "failed"
        assert provider.terminated == ["inst_1"]

    def test_keep_instance_on_failure_only_for_failures(self, sessions):
        failed = WorkerStatus(state=STATE_FAILED, error="boom")
        _, provider = start(sessions, ServeWorker([failed]), keep_instance_on_failure=True)
        assert provider.terminated == []
        _, provider = start(sessions, ServeWorker([SERVING], hook=stop_after(2, sessions)),
                            keep_instance_on_failure=True)
        assert provider.terminated == ["inst_1"]

    def test_worker_exit_before_serving_fails(self, sessions):
        sid, provider = start(sessions, ServeWorker([WorkerStatus(state=STATE_DONE)]))
        assert sessions.get(sid).status == "failed"
        assert provider.terminated == ["inst_1"]

    def test_provision_failure_before_instance(self, sessions):
        class Broken(ConfirmingProvider):
            def provision(self, spec):
                from src.remote.providers.base import ProvisionRejected
                raise ProvisionRejected("no capacity")
        sid, provider = start(sessions, ServeWorker([SERVING]), provider=Broken())
        assert sessions.get(sid).status == "failed"
        assert "no capacity" in sessions.get(sid).error

    def test_bad_plan_fails_without_provisioning(self, sessions):
        sid, provider = start(sessions, ServeWorker([SERVING]), gpu_type_id="FAKE-24G",
                              gpu_count=1, max_num_seqs=1024, max_model_len=131072)
        assert sessions.get(sid).status == "failed"
        assert provider.provisioned == []


class TestSpendRails:
    def test_runtime_deadline_terminates(self, sessions):
        start_t = time.monotonic()
        sid, provider = start(sessions, ServeWorker([SERVING]), max_runtime_hours=0.0003)
        final = sessions.get(sid)
        assert final.status == "expired"
        assert final.stop_reason == "max_runtime_hours"
        assert provider.terminated == ["inst_1"]
        assert time.monotonic() - start_t < 10

    def test_total_cost_cap_uses_actual_instance_price(self, sessions):
        class Pricey(ConfirmingProvider):
            def provision(self, spec):
                instance = super().provision(spec)
                instance.cost_per_hr = 3600.0       # $1/second
                return instance

        sid, provider = start(sessions, ServeWorker([SERVING]), provider=Pricey(),
                              max_total_cost_usd=1.0)
        final = sessions.get(sid)
        assert final.status == "expired"
        assert final.stop_reason == "max_total_cost_usd"
        assert provider.terminated == ["inst_1"]
        run = RemoteLifecycle(sessions.db_path).run(sid)
        assert run["deadline"] - run["started_at"] <= 1.5
        assert final.deadline_at == pytest.approx(run["deadline"])

    def test_plan_deadline_from_total_cost(self, sessions):
        worker = ServeWorker([SERVING], hook=stop_after(2, sessions))
        sid, _ = start(sessions, worker, gpu_type_id="FAKE-141G", max_total_cost_usd=2.0)
        run = RemoteLifecycle(sessions.db_path).run(sid)
        # FAKE-141G costs $4/hr -> $2 buys 30 minutes, well under the 2 h runtime cap
        assert run["deadline"] - run["started_at"] == pytest.approx(1800, abs=5)

    def test_idle_timeout_stops(self, sessions):
        idle = WorkerStatus(state=STATE_SERVING, serve={"openai_base_path": "/v1",
                                                        "idle_seconds": 61.0, "requests": 0})
        sid, provider = start(sessions, ServeWorker([idle]), idle_timeout_minutes=1)
        final = sessions.get(sid)
        assert final.status == "stopped"
        assert final.stop_reason == "idle_timeout"
        assert provider.terminated == ["inst_1"]

    def test_stop_during_boot(self, sessions):
        worker = ServeWorker([WorkerStatus(state=STATE_RUNNING)], hook=stop_after(2, sessions))
        sid, provider = start(sessions, worker)
        assert sessions.get(sid).status == "stopped"
        assert provider.terminated == ["inst_1"]

    def test_estimated_spend_recorded(self, sessions):
        class Priced(ConfirmingProvider):
            def provision(self, spec):
                instance = super().provision(spec)
                instance.cost_per_hr = 2.0
                return instance
        worker = ServeWorker([SERVING], hook=stop_after(2, sessions))
        sid, _ = start(sessions, worker, provider=Priced())
        assert sessions.get(sid).metrics["est_cost_usd"] >= 0


class TestExecutionGuardHooks:
    def test_guard_sees_serve_stage_and_settles(self, sessions):
        calls = []

        class Guard:
            def before_provision(self, job_id, stage, spec):
                calls.append(("before", stage, spec.http_ports))
            def usage(self, job_id, resources):
                calls.append(("usage",))
            def settle(self, job_id, resources):
                calls.append(("settle", len(resources)))

        worker = ServeWorker([SERVING], hook=stop_after(2, sessions))
        start(sessions, worker, guard=Guard())
        assert calls[0] == ("before", "serve", [8000, 8001])
        assert ("usage",) in calls
        assert calls[-1] == ("settle", 1)

    def test_guard_budget_exhaustion_terminates(self, sessions):
        class Guard:
            def before_provision(self, *a): pass
            def usage(self, *a): raise RuntimeError("approved budget exhausted")
            def settle(self, *a): pass

        sid, provider = start(sessions, ServeWorker([SERVING]), guard=Guard(),
                              keep_instance_on_failure=True)
        assert provider.terminated == ["inst_1"]
        assert "budget exhausted" in sessions.get(sid).error


class TestTrainServeLoop:
    def test_serves_adapter_from_a_finished_remote_job(self, sessions, tmp_path, monkeypatch):
        from src.remote.artifacts import LocalArtifactStore
        from tests.test_remote_orchestrator import ScriptedWorker, make_config, make_remote_cfg

        job_manager = JobManager(db_path=str(tmp_path / "jobs.db"))
        store = LocalArtifactStore(Path("store"))

        def push_adapter():
            Path("trained").mkdir(exist_ok=True)
            (Path("trained") / "adapter_config.json").write_text("{}")
            store.push_dir(Path("trained"), "adapter")

        job_manager.create_job("job_1", {})
        train_worker = ScriptedWorker([WorkerStatus(state=STATE_DONE, artifacts=["adapter"])],
                                      on_done=push_adapter)
        config = make_config(remote=make_remote_cfg(merge_strategy="skip"))
        from src.remote import plan as plan_module
        monkeypatch.setattr(plan_module, "fetch_model_specs", lambda model_id, hf_token=None:
                            parse_model_specs(model_id, LLAMA_7B_CONFIG))
        run_training_remote("job_1", config, job_manager, {}, provider=FakeProvider(), store=store,
                            worker_client_factory=lambda url, token: train_worker,
                            poll_interval_s=0.01, boot_timeout_min=0.01)
        assert job_manager.get_job("job_1").status == "completed"
        # The real API persists TrainingConfig.model_dump(); FakeConfig's is partial.
        import sqlite3
        with sqlite3.connect(job_manager.db_path) as db:
            db.execute("UPDATE jobs SET config=? WHERE job_id='job_1'", (json.dumps({
                "base_model": "fake/llama-7b", "output_name": "my-model", "use_lora": True,
                "lora_r": 64, "hf_token": "hf_x", "remote": {"enabled": True}}),))

        worker = ServeWorker([SERVING], hook=stop_after(2, sessions))
        sid, provider = start(sessions, worker, job_manager=job_manager,
                              model={"job_id": "job_1", "artifact": "adapter"})
        final = sessions.get(sid)
        assert final.status == "stopped", final.error
        payload = json.loads(base64.b64decode(provider.provisioned[0].env["MERLINA_REMOTE_JOB_B64"]))
        assert payload["serve"]["lora"] is True
        assert payload["serve"]["model_id"] == "fake/llama-7b"
        assert payload["artifact_store"]["kind"] == "local"
        assert payload["artifacts_in"] == ["adapter"]
        assert provider.provisioned[0].env["HF_TOKEN"] == "hf_x"


# ---------------------------------------------------------------------------
# Persistence + reconciliation
# ---------------------------------------------------------------------------

class TestSessionStore:
    def test_request_is_scrubbed(self, sessions):
        session = sessions.create("inf_x", {"hf_token": "secret", "model": {"hf_repo": "a/b"}})
        assert "hf_token" not in session.request
        assert session.status == "queued"

    def test_request_stop_and_active_count(self, sessions):
        sessions.create("inf_a", {})
        sessions.create("inf_b", {})
        sessions.update("inf_b", status="stopped")
        assert sessions.count_active() == 1
        assert sessions.request_stop("inf_a").status == "stopping"

    def test_fail_orphans(self, sessions):
        sessions.create("inf_a", {})
        sessions.update("inf_a", status="ready")
        sessions.create("inf_b", {})
        sessions.update("inf_b", status="stopped")
        cutoff = time.time() + 1
        sessions.create("inf_new", {})
        assert sessions.fail_orphans("restart", created_before=sessions.get("inf_new").created_at) \
            == ["inf_a"]
        assert sessions.get("inf_a").status == "failed"
        assert sessions.get("inf_a").stop_reason == "orphaned"
        assert sessions.get("inf_b").status == "stopped"
        assert sessions.get("inf_new").status == "queued"     # started after the scan began
        assert sessions.fail_orphans("restart", created_before=cutoff) == ["inf_new"]

    def test_records_facade(self, sessions, tmp_path):
        jobs = JobManager(db_path=str(tmp_path / "jobs.db"))
        jobs.create_job("job_1", {})
        sessions.create("inf_a", {})
        records = with_inference_sessions(jobs)
        assert isinstance(records, RemoteRecords)
        assert with_inference_sessions(records) is records
        assert records.get_job("job_1").job_id == "job_1"
        assert records.get_job("inf_a").session_id == "inf_a"
        records.update_job("inf_a", status="failed", error="x")
        assert sessions.get("inf_a").error == "x"
        assert records.db_path == jobs.db_path


class InventoryProvider(ConfirmingProvider):
    def provision(self, spec):
        instance = super().provision(spec)
        instance.raw = {"name": spec.name}
        return instance

    def list_instances(self):
        return [RemoteInstance(f"inst_{i + 1}", self.name, "running", raw={"name": s.name})
                for i, s in enumerate(self.provisioned) if f"inst_{i + 1}" not in self.terminated]


class TestReconciler:
    def _orphan(self, sessions, clock):
        """A serve pod whose controller is gone (e.g. the API restarted)."""
        lifecycle = RemoteLifecycle(sessions.db_path, clock=clock)
        provider = InventoryProvider()
        sessions.create("inf_live", {})
        owner = lifecycle.claim("inf_live", 2)
        label = lifecycle.intent("inf_live", owner, "serve", provider.name)
        from src.remote.spec import InstanceSpec
        instance = provider.provision(InstanceSpec(name=label, image="img"))
        lifecycle.record_instance("inf_live", "serve", label, instance)
        lifecycle.stage_state("inf_live", "serve", "running")
        sessions.update("inf_live", status="ready")
        return lifecycle, provider

    def test_live_session_pod_is_not_mistaken_for_a_deleted_job(self, sessions, tmp_path):
        now = [1000.0]
        lifecycle, provider = self._orphan(sessions, lambda: now[0])
        jobs = JobManager(db_path=str(sessions.db_path))
        reconcile_once(lifecycle, {provider.name: provider}, job_manager=jobs, stop_grace_seconds=0)
        assert provider.terminated == []

    def test_orphaned_session_pod_terminated_after_lease_expires(self, sessions):
        now = [1000.0]
        lifecycle, provider = self._orphan(sessions, lambda: now[0])
        jobs = JobManager(db_path=str(sessions.db_path))
        now[0] += lifecycle.lease_seconds + 1
        reconcile_once(lifecycle, {provider.name: provider}, job_manager=jobs, stop_grace_seconds=0)
        assert provider.terminated == ["inst_1"]
        session = sessions.get("inf_live")
        assert session.status == "failed"
        assert session.metrics["remote_resources"][0]["cleanup_status"] == "terminated"

    def test_session_deadline_enforced_by_reconciler(self, sessions):
        now = [1000.0]
        lifecycle, provider = self._orphan(sessions, lambda: now[0])
        now[0] += 2 * 3600 + 1
        reconcile_once(lifecycle, {provider.name: provider},
                       job_manager=JobManager(db_path=str(sessions.db_path)), stop_grace_seconds=0)
        assert provider.terminated == ["inst_1"]
