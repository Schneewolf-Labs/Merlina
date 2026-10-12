"""
Tests for the remote orchestrator (src/remote/orchestrator.py) with fake
provider / store / worker client — full lifecycle without any network.

Run standalone: python -m pytest tests/test_remote_orchestrator.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_manager import JobManager
from src.remote.artifacts import LocalArtifactStore
from src.remote.orchestrator import run_training_remote
from src.remote.providers.base import ComputeProvider, ProviderError
from src.remote.spec import GpuOffer, RemoteInstance
from src.remote.worker_client import (
    STATE_DONE,
    STATE_FAILED,
    STATE_PUSHING,
    STATE_RUNNING,
    WorkerClient,
    WorkerStatus,
)
from tests.test_remote_sizing import KIMI_K2_CONFIG, LLAMA_7B_CONFIG


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeProvider(ComputeProvider):
    name = "fake"

    def __init__(self):
        self.provisioned = []          # InstanceSpec list
        self.terminated = []           # instance ids
        self._counter = 0

    def provision(self, spec):
        self._counter += 1
        self.provisioned.append(spec)
        return RemoteInstance(instance_id=f"inst_{self._counter}",
                              provider=self.name, status="running")

    def get_instance(self, instance_id):
        return RemoteInstance(instance_id=instance_id, provider=self.name,
                              status="running")

    def terminate(self, instance_id):
        self.terminated.append(instance_id)

    def list_gpu_offers(self):
        return [
            GpuOffer("FAKE-24G", "Fake 24G", 24, price_per_hr_secure=0.5),
            GpuOffer("FAKE-141G", "Fake 141G", 141, price_per_hr_secure=4.0),
        ]

    def proxy_url(self, instance, port):
        return f"http://fake/{instance.instance_id}:{port}"


class ScriptedWorker(WorkerClient):
    """Plays back a scripted sequence of WorkerStatus responses."""

    def __init__(self, script: List[WorkerStatus], on_done=None,
                 unhealthy_polls: int = 0):
        self.script = list(script)
        self.stop_requests = 0
        self.on_done = on_done
        self._unhealthy_polls = unhealthy_polls

    def health(self):
        if self._unhealthy_polls > 0:
            self._unhealthy_polls -= 1
            return False
        return True

    def status(self, since_step=0):
        status = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if status.is_terminal and self.on_done:
            self.on_done()
        return status

    def request_stop(self):
        self.stop_requests += 1
        return True


def make_remote_cfg(**overrides):
    base = dict(
        enabled=True, provider="fake", gpu_type_id=None, gpu_count=None,
        cloud_type="secure", container_disk_gb=None, volume_gb=None,
        max_cost_per_hr=None, max_runtime_hours=48.0, merge_strategy="skip",
        artifact_repo=None, worker_image=None, keep_instance_on_failure=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeConfig(SimpleNamespace):
    def model_dump(self):
        return {
            "base_model": self.base_model,
            "output_name": self.output_name,
            "hf_token": self.hf_token,
            "push_to_hub": self.push_to_hub,
            "training_mode": self.training_mode,
            "dataset": {"source": vars(self.dataset.source),
                        "additional_sources": [vars(src) for src in self.dataset.additional_sources]},
        }


def make_config(**overrides):
    dataset = SimpleNamespace(
        source=SimpleNamespace(source_type="huggingface"),
        additional_sources=[],
    )
    cfg = FakeConfig(
        base_model="fake/llama-7b",
        output_name="my-model",
        hf_token="hf_x",
        hf_hub_private=True,
        push_to_hub=False,
        use_4bit=True,
        lora_r=64,
        batch_size=1,
        max_length=2048,
        training_mode="sft",
        model_type="auto",
        dataset=dataset,
        wandb_key=None,
        remote=make_remote_cfg(),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def job_manager(tmp_path):
    return JobManager(db_path=str(tmp_path / "jobs.db"))


@pytest.fixture(autouse=True)
def patched_environment(tmp_path, monkeypatch):
    """Run in a temp cwd and stub out the Hub model-spec fetch."""
    monkeypatch.chdir(tmp_path)
    from src.remote import plan as plan_module
    from src.remote.sizing import parse_model_specs

    def fake_fetch(model_id, hf_token=None):
        cfg = KIMI_K2_CONFIG if "kimi" in model_id.lower() else LLAMA_7B_CONFIG
        return parse_model_specs(model_id, cfg)

    monkeypatch.setattr(plan_module, "fetch_model_specs", fake_fetch)


def run(config, job_manager, *, provider=None, store=None, worker=None,
        tmp_path=Path(".")):
    """Drive run_training_remote with fast polling and a pre-loaded store."""
    provider = provider or FakeProvider()
    if store is None:
        store = LocalArtifactStore(Path("store"))
    if worker is None:
        # Worker "trains", then pushes the adapter into the store before
        # reporting done — mirroring what the real worker_entry does.
        def push_adapter():
            adapter_src = Path("trained_adapter")
            adapter_src.mkdir(exist_ok=True)
            (adapter_src / "adapter_config.json").write_text("{}")
            store.push_dir(adapter_src, "adapter")
        worker = ScriptedWorker([
            WorkerStatus(state=STATE_RUNNING,
                         job={"status": "training", "progress": 0.5,
                              "current_step": 50, "total_steps": 100, "loss": 1.5},
                         new_metrics=[{"step": 50, "loss": 1.5}]),
            WorkerStatus(state=STATE_PUSHING),
            WorkerStatus(state=STATE_DONE, artifacts=["adapter"]),
        ], on_done=push_adapter)

    job_manager.create_job("job_1", {})
    run_training_remote(
        "job_1", config, job_manager, {},
        provider=provider,
        store=store,
        worker_client_factory=lambda url, token: worker,
        poll_interval_s=0.01,
        boot_timeout_min=0.01,
    )
    return provider, store, worker


class TestHappyPath:
    def test_completes_and_pulls_adapter(self, job_manager):
        provider, store, worker = run(make_config(), job_manager)

        job = job_manager.get_job("job_1")
        assert job.status == "completed"
        assert job.progress == 1.0
        assert Path("models/my-model/adapter_config.json").exists()
        assert job.output_dir == str(Path("models/my-model"))

    def test_instance_always_terminated(self, job_manager):
        provider, _, _ = run(make_config(), job_manager)
        assert provider.terminated == ["inst_1"]

    def test_progress_and_metrics_mirrored(self, job_manager):
        run(make_config(), job_manager)
        job = job_manager.get_job("job_1")
        assert job.loss == 1.5
        metrics = job_manager.get_metrics("job_1")
        assert any(m["step"] == 50 for m in metrics)

    def test_plan_recorded_in_job_metadata(self, job_manager):
        run(make_config(), job_manager)
        job = job_manager.get_job("job_1")
        assert "remote_plan" in job.metrics
        assert job.metrics["artifact_store"]["kind"] == "local"

    def test_worker_env_carries_job_payload_and_token(self, job_manager):
        provider, _, _ = run(make_config(), job_manager)
        env = provider.provisioned[0].env
        assert "MERLINA_REMOTE_JOB_B64" in env
        assert env["MERLINA_WORKER_TOKEN"]
        assert env["HF_TOKEN"] == "hf_x"

    def test_train_payload_disables_hub_push_and_remote(self, job_manager):
        import base64, json
        provider, _, _ = run(make_config(push_to_hub=True,
                                         remote=make_remote_cfg(merge_strategy="skip")),
                             job_manager)
        payload = json.loads(base64.b64decode(
            provider.provisioned[0].env["MERLINA_REMOTE_JOB_B64"]))
        assert payload["config"]["push_to_hub"] is False
        assert payload["config"]["remote"] is None
        assert payload["stage"] == "train"
        assert payload["artifacts_out"] == ["adapter"]


class TestKimiK2Sizing:
    def test_big_moe_provisions_multi_gpu(self, job_manager):
        provider, _, _ = run(make_config(base_model="fake/kimi-k2"), job_manager)
        spec = provider.provisioned[0]
        assert spec.gpu_type_id == "FAKE-141G"
        assert spec.gpu_count > 1
        # disk must cover the ~2 TB bf16 weights download
        assert spec.container_disk_gb > 1000

    def test_model_parallel_strategy_forwarded(self, job_manager):
        import base64, json
        provider, _, _ = run(make_config(base_model="fake/kimi-k2"), job_manager)
        payload = json.loads(base64.b64decode(
            provider.provisioned[0].env["MERLINA_REMOTE_JOB_B64"]))
        assert payload["config"]["multi_gpu_strategy"] == "single"


class TestFailure:
    def test_worker_failure_fails_job_and_terminates(self, job_manager):
        worker = ScriptedWorker([WorkerStatus(state=STATE_FAILED, error="CUDA OOM")])
        provider = FakeProvider()
        run(make_config(), job_manager, provider=provider, worker=worker)

        job = job_manager.get_job("job_1")
        assert job.status == "failed"
        assert "CUDA OOM" in job.error
        assert provider.terminated == ["inst_1"]

    def test_keep_instance_on_failure(self, job_manager):
        worker = ScriptedWorker([WorkerStatus(state=STATE_FAILED, error="boom")])
        provider = FakeProvider()
        cfg = make_config(remote=make_remote_cfg(keep_instance_on_failure=True))
        run(cfg, job_manager, provider=provider, worker=worker)
        assert provider.terminated == []

    def test_boot_timeout_fails_job(self, job_manager):
        worker = ScriptedWorker([WorkerStatus(state=STATE_RUNNING)],
                                unhealthy_polls=10_000)
        provider = FakeProvider()
        run(make_config(), job_manager, provider=provider, worker=worker)
        job = job_manager.get_job("job_1")
        assert job.status == "failed"
        assert "healthy" in job.error
        assert provider.terminated == ["inst_1"]

    def test_expired_runtime_prevents_provisioning(self, job_manager):
        worker = ScriptedWorker([WorkerStatus(state=STATE_RUNNING)] * 2)
        provider = FakeProvider()
        cfg = make_config(remote=make_remote_cfg(max_runtime_hours=1e-7))
        run(cfg, job_manager, provider=provider, worker=worker)
        job = job_manager.get_job("job_1")
        assert job.status == "failed"
        assert "max_runtime_hours" in job.error
        assert provider.provisioned == []

    def test_runtime_deadline_mid_stage_terminates(self, job_manager):
        from src.remote.lifecycle import RemoteLifecycle

        class ExpiresWhileRunning(ScriptedWorker):
            def status(self, since_step=0):
                # The shared total deadline lapses while the worker is training.
                with RemoteLifecycle(job_manager.db_path).connect() as db:
                    db.execute("UPDATE remote_runs SET deadline=0")
                return super().status(since_step)

        worker = ExpiresWhileRunning([WorkerStatus(state=STATE_RUNNING)] * 2)
        provider = FakeProvider()
        run(make_config(), job_manager, provider=provider, worker=worker)
        job = job_manager.get_job("job_1")
        assert job.status == "failed"
        assert "max_runtime_hours" in job.error
        assert len(provider.provisioned) == 1
        assert provider.terminated == ["inst_1"]


class TestStop:
    @staticmethod
    def _stop_after_first_status(job_manager, script, on_stop=None):
        class StopMidRun(ScriptedWorker):
            def status(self, since_step=0):
                job_manager.request_stop("job_1")
                return super().status(since_step)

            def request_stop(self):
                super().request_stop()
                if on_stop:
                    on_stop(self)
                return True
        return StopMidRun(script)

    def test_stop_while_running_forwarded_then_terminated(self, job_manager, monkeypatch):
        """No cooperation from the worker: the grace window ends and we terminate."""
        from config import settings
        monkeypatch.setattr(settings, "remote_stop_grace_minutes", 0.002, raising=False)
        worker = self._stop_after_first_status(job_manager, [WorkerStatus(state=STATE_RUNNING)] * 2)
        provider = FakeProvider()
        run(make_config(), job_manager, provider=provider, worker=worker)
        assert worker.stop_requests == 1
        assert job_manager.get_job("job_1").status == "stopped"
        assert provider.terminated == ["inst_1"]

    def test_stop_while_running_honoured_by_worker(self, job_manager, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "remote_stop_grace_minutes", 5.0, raising=False)

        def worker_stops(worker):
            worker.script = [WorkerStatus(state=STATE_DONE, job={"status": "stopped"})]
        worker = self._stop_after_first_status(
            job_manager, [WorkerStatus(state=STATE_RUNNING)] * 2, on_stop=worker_stops)
        provider = FakeProvider()
        run(make_config(), job_manager, provider=provider, worker=worker)
        assert worker.stop_requests == 1
        assert job_manager.get_job("job_1").status == "stopped"
        assert provider.terminated == ["inst_1"]

    def test_stop_before_provisioning_rents_nothing(self, job_manager, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "remote_stop_grace_minutes", 0.0, raising=False)

        running_forever = ScriptedWorker([WorkerStatus(state=STATE_RUNNING)] * 2)
        provider = FakeProvider()

        job_manager.create_job("job_1", {})
        job_manager.request_stop("job_1")
        run_training_remote(
            "job_1", make_config(), job_manager, {},
            provider=provider,
            store=LocalArtifactStore(Path("store")),
            worker_client_factory=lambda url, token: running_forever,
            poll_interval_s=0.01,
            boot_timeout_min=0.01,
        )

        assert running_forever.stop_requests == 0
        assert job_manager.get_job("job_1").status == "stopped"
        assert provider.provisioned == []


class TestProvisionOutcome:
    def _run_failing_provision(self, job_manager, error):
        from src.remote.lifecycle import RemoteLifecycle

        class Failing(FakeProvider):
            def provision(self, spec):
                self.provisioned.append(spec)
                raise error
        provider = Failing()
        run(make_config(), job_manager, provider=provider)
        return provider, RemoteLifecycle(job_manager.db_path).attempts()[0]

    def test_definite_rejection_is_failed_not_ambiguous(self, job_manager):
        from src.remote.providers.base import ProvisionRejected
        provider, attempt = self._run_failing_provision(
            job_manager, ProvisionRejected("no instances currently available"))
        assert attempt["state"] == "failed"
        assert attempt["error"] == "ProvisionRejected"
        assert provider.terminated == []
        assert job_manager.get_job("job_1").status == "failed"

    @pytest.mark.parametrize("error", [ProviderError("RunPod API request failed: read timeout"),
                                       TimeoutError("read timeout")])
    def test_ambiguous_create_stays_unknown(self, job_manager, error):
        _, attempt = self._run_failing_provision(job_manager, error)
        assert attempt["state"] == "provisioning_unknown"


class TestHeartbeat:
    def test_transient_errors_retried_until_lease_lost(self, caplog):
        import logging
        import sqlite3
        import threading
        from src.remote.lifecycle import RemoteLeaseLost
        from src.remote.orchestrator import _heartbeat_loop

        class Lifecycle:
            lease_seconds = 120.0
            def __init__(self):
                self.calls = 0
            def heartbeat(self, job_id, owner):
                self.calls += 1
                if self.calls <= 3:
                    raise sqlite3.OperationalError("database is locked")
                if self.calls == 5:
                    raise RemoteLeaseLost("expired")

        lifecycle = Lifecycle()
        stop = threading.Event()
        with caplog.at_level(logging.WARNING, logger="src.remote.orchestrator"):
            thread = threading.Thread(target=_heartbeat_loop, args=(lifecycle, "job", "owner", stop),
                                      kwargs={"interval": 0.01, "max_backoff": 0.01})
            thread.start()
            thread.join(timeout=5)
        assert not thread.is_alive()           # exited on RemoteLeaseLost
        assert lifecycle.calls == 5            # kept going through transient errors
        assert sum("database is locked" in r.getMessage() for r in caplog.records) == 3

    def test_stop_event_ends_loop(self):
        import threading
        from src.remote.orchestrator import _heartbeat_loop

        class Lifecycle:
            lease_seconds = 120.0
            calls = 0
            def heartbeat(self, job_id, owner):
                self.calls += 1
        stop = threading.Event()
        thread = threading.Thread(target=_heartbeat_loop, args=(Lifecycle(), "job", "owner", stop),
                                  kwargs={"interval": 0.01})
        thread.start()
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


class TestMergeStage:
    def test_local_merge_invoked(self, job_manager, monkeypatch):
        merged = []
        from src.remote import orchestrator
        def fake_merge(config, adapter_dir):
            merged.append(adapter_dir)
            result = Path("merged-result")
            result.mkdir()
            (result / "weights").write_text("merged")
            return result
        monkeypatch.setattr(orchestrator, "_run_local_merge", fake_merge)
        cfg = make_config(remote=make_remote_cfg(merge_strategy="local"))
        run(cfg, job_manager)
        assert merged == [Path("models/my-model")]
        # Local merges stay on local disk; they are not re-uploaded.
        assert not Path("store/merged").exists()
        assert job_manager.get_job("job_1").status == "completed"

    def test_remote_merge_provisions_second_instance(self, job_manager):
        store = LocalArtifactStore(Path("store"))

        def push_artifact():
            src = Path("stage_out")
            src.mkdir(exist_ok=True)
            (src / "f").write_text("x")
            store.push_dir(src, "adapter")

        # One scripted worker serves both stages: train (pushes adapter),
        # then merge.
        worker = ScriptedWorker([
            WorkerStatus(state=STATE_RUNNING, job={"status": "training", "progress": 0.5}),
            WorkerStatus(state=STATE_DONE, artifacts=["adapter"]),
            WorkerStatus(state=STATE_DONE, artifacts=["merged"]),
        ], on_done=push_artifact)
        provider = FakeProvider()
        cfg = make_config(remote=make_remote_cfg(merge_strategy="remote"))
        run(cfg, job_manager, provider=provider, store=store, worker=worker)

        assert len(provider.provisioned) == 2
        assert provider.provisioned[1].name.endswith("-merge")
        assert provider.terminated == ["inst_1", "inst_2"]
        assert job_manager.get_job("job_1").status == "completed"


class TestUploadedDatasetStaging:
    def test_uploaded_dataset_fails_fast(self, job_manager):
        cfg = make_config()
        cfg.dataset.source.source_type = "upload"
        provider = FakeProvider()
        job_manager.create_job("job_1", {})
        run_training_remote(
            "job_1", cfg, job_manager, {"ds_1": {"content": b"x"}},
            provider=provider,
            store=LocalArtifactStore(Path("store")),
            worker_client_factory=lambda url, token: ScriptedWorker(
                [WorkerStatus(state=STATE_DONE)]),
            poll_interval_s=0.01,
        )
        job = job_manager.get_job("job_1")
        assert job.status == "failed"
        assert provider.provisioned == []

    def test_missing_uploaded_content_fails_before_provisioning(self, job_manager):
        cfg = make_config()
        cfg.dataset.source = SimpleNamespace(source_type="upload", dataset_id="expired")
        provider = FakeProvider()
        run(cfg, job_manager, provider=provider)
        assert job_manager.get_job("job_1").status == "failed"
        assert provider.provisioned == []

    def test_upload_uses_portable_payload_before_provisioning(self, job_manager, monkeypatch):
        import base64
        import json
        from config import settings
        from src.remote.s3_artifacts import S3ArtifactStore
        from tests.test_s3_artifacts import FakeS3
        # Operator data is only staged into an operator-owned S3/R2 store.
        monkeypatch.setattr(settings, "merlina_artifact_store", "s3")
        cfg = make_config()
        cfg.dataset.source = SimpleNamespace(source_type="upload", dataset_id="uploaded")
        store = S3ArtifactStore("bucket", "job_1", fs=FakeS3())
        provider = FakeProvider()
        data = b'{"prompt":"hi","chosen":"hello"}'
        def finished():
            adapter = Path("adapter-source")
            adapter.mkdir(exist_ok=True)
            (adapter / "adapter_config.json").write_text("{}")
            store.push_dir(adapter, "adapter")
        worker = ScriptedWorker([WorkerStatus(state=STATE_DONE)], on_done=finished)
        job_manager.create_job("job_1", cfg.model_dump())
        run_training_remote("job_1", cfg, job_manager,
                            {"uploaded": {"filename": "/home/me/customer.json", "content": data}},
                            provider=provider, store=store,
                            worker_client_factory=lambda *_: worker, poll_interval_s=0.01)
        assert job_manager.get_job("job_1").status == "completed"
        payload = json.loads(base64.b64decode(provider.provisioned[0].env["MERLINA_REMOTE_JOB_B64"]))
        source = payload["config"]["dataset"]["source"]
        assert source["file_path"].startswith("artifact://input-0-")
        assert "dataset_id" not in source
        assert "/home/me/customer" not in json.dumps(payload)
        assert store.exists(source["file_path"].removeprefix("artifact://"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
