import base64
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.job_manager import JobManager
from src.remote.checkpoints import DurableCheckpointCallback, publish_checkpoint, remote_checkpoint_callback
from src.remote.s3_artifacts import S3ArtifactStore, artifact_metadata, safe_path
from src.remote.artifacts import ArtifactStoreError
from tests.test_s3_artifacts import FakeS3


@pytest.fixture
def fixture(tmp_path):
    manager = JobManager(str(tmp_path / "jobs.db"))
    manager.create_job("job", {})
    fs = FakeS3()
    store = S3ArtifactStore("bucket", "job", fs=fs)
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "optimizer.bin").write_bytes(b"optimizer")
    (checkpoint / "random_states_0.pkl").write_bytes(b"rng")
    barriers = []
    trainer = SimpleNamespace(global_step=5, accelerator=SimpleNamespace(
        is_main_process=True, wait_for_everyone=lambda: barriers.append("barrier")))
    return manager, fs, store, checkpoint, trainer, barriers


def test_completed_save_publishes_and_restores_verified_checkpoint(fixture, tmp_path):
    manager, fs, store, checkpoint, trainer, barriers = fixture

    def runner(command, **kwargs):
        assert kwargs["timeout"] == 120
        request = json.loads(Path(command[-1]).read_text())
        assert manager.get_job("job").metrics["checkpoint_publication"]["state"] == "publishing"
        assert barriers == ["barrier"]
        publish_checkpoint(request, store=store)

    callback = DurableCheckpointCallback("job", manager, store.describe(), runner=runner)
    callback.on_save(trainer, str(checkpoint))
    assert barriers == ["barrier", "barrier"]
    assert manager.get_job("job").metrics["checkpoint_publication"]["last_committed_step"] == 5
    restored = store.pull_dir("checkpoint", tmp_path / "restored")
    assert (restored / "optimizer.bin").read_bytes() == b"optimizer"
    assert store._manifest("checkpoint")["metadata"]["checkpoint"]["step"] == 5


def test_failed_checkpoint_visible_and_final_adapter_still_succeeds(fixture):
    manager, fs, store, checkpoint, trainer, barriers = fixture
    publish_checkpoint({"step": 4, "path": str(checkpoint)}, store=store)
    callback = DurableCheckpointCallback("job", manager, store.describe())
    callback._record("committed", 4)
    fs.fail_upload = True

    def runner(command, **kwargs):
        publish_checkpoint(json.loads(Path(command[-1]).read_text()), store=store)

    callback.runner = runner
    callback.on_save(trainer, checkpoint)
    status = manager.get_job("job").metrics["checkpoint_publication"]
    assert status["state"] == "failed"
    assert status["last_committed_step"] == 4
    assert "secret" not in status["error"]
    assert store._manifest("checkpoint")["metadata"]["checkpoint"]["step"] == 4
    fs.fail_upload = False
    store.push_dir(checkpoint, "adapter")
    assert store.exists("adapter")


def test_timeout_does_not_stop_training_or_claim_commit(fixture):
    manager, fs, store, checkpoint, trainer, barriers = fixture

    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    callback = DurableCheckpointCallback("job", manager, store.describe(), timeout=1, runner=runner)
    callback.on_save(trainer, checkpoint)
    status = manager.get_job("job").metrics["checkpoint_publication"]
    assert status["state"] == "failed" and "timed out" in status["error"]
    assert "last_committed_step" not in status
    assert barriers == ["barrier", "barrier"]


def test_non_main_rank_joins_both_barriers_without_upload(fixture):
    manager, fs, store, checkpoint, trainer, barriers = fixture
    trainer.accelerator.is_main_process = False
    callback = DurableCheckpointCallback("job", manager, store.describe(), runner=lambda *a, **kw: pytest.fail("rank uploaded"))
    callback.on_save(trainer, checkpoint)
    assert barriers == ["barrier", "barrier"]
    assert not store.exists("checkpoint")


def test_only_matching_remote_s3_train_gets_callback(monkeypatch, fixture):
    manager, fs, store, checkpoint, trainer, barriers = fixture
    monkeypatch.delenv("MERLINA_REMOTE_JOB_B64", raising=False)
    assert remote_checkpoint_callback("job", manager) is None
    for stage, kind, matches in [("train", "s3", True), ("merge", "s3", False), ("train", "hf_hub", False)]:
        payload = {"job_id": "job", "stage": stage, "artifact_store": {"kind": kind}, "config": {}}
        monkeypatch.setenv("MERLINA_REMOTE_JOB_B64", base64.b64encode(json.dumps(payload).encode()).decode())
        assert (remote_checkpoint_callback("job", manager) is not None) == matches
        assert remote_checkpoint_callback("another-job", manager) is None


def test_metadata_omits_original_filesystem_paths():
    metadata = artifact_metadata({"dataset": {"source": {"file_path": "/private/source.csv"}},
                                  "base_model": "/private/model", "output_dir": "relative/private",
                                  "nested": {"cache_path": "C:\\private", "hf_token": "secret"}})
    assert "private" not in json.dumps(metadata)
    assert "secret" not in json.dumps(metadata)


@pytest.mark.parametrize("path", ["%2e%2e/escape", "safe%2funsafe", "percent%25", "a%00b"])
def test_percent_encoded_paths_are_rejected(path):
    with pytest.raises(ArtifactStoreError):
        safe_path(path)


def test_checkpoint_status_reaches_control_plane_without_overwriting_plan(fixture, tmp_path):
    from src.remote.worker_entry import WorkerState, build_status_payload
    from src.remote.worker_client import WorkerStatus
    from src.remote.orchestrator import _mirror_status
    manager, fs, store, checkpoint, trainer, barriers = fixture
    callback = DurableCheckpointCallback("job", manager, store.describe())
    callback._record("committed", 5)
    control = JobManager(str(tmp_path / "control.db"))
    control.create_job("job", {})
    control.update_job("job", metrics={"remote_plan": {"stages": ["train"]}})
    payload = build_status_payload(WorkerState("job", "train"), manager)
    status = WorkerStatus.from_dict(payload)
    _mirror_status(status, "job", control, None, 0)
    metrics = control.get_job("job").metrics
    assert metrics["checkpoint_publication"]["last_committed_step"] == 5
    assert metrics["remote_plan"] == {"stages": ["train"]}
