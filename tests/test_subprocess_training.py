#!/usr/bin/env python3
"""
Tests for isolated single-process training (src/train_single.py +
run_training_subprocess / _monitor_subprocess_job in src/training_runner.py).

The two stuck-worker paths these pin down:

1. Model loading used to run on a queue worker thread inside the API
   process, starving the event loop — now it runs in a subprocess.
2. A stop request could leave the worker hung forever (stop was only
   checked at step boundaries, and a thread wedged in a C call can't be
   killed) — now stops are enforced with SIGTERM → SIGKILL escalation
   against the subprocess's process group.

Runnable standalone (`python tests/test_subprocess_training.py`) or under
pytest. ML dependencies are mocked — no torch/GPU required; real child
processes are plain `python -c` one-liners.
"""

import os
import sys
import json
import time
import signal
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep any merlina import in this process on the legacy path (mirrors conftest).
os.environ.setdefault("TRAINING_ISOLATION", "thread")

# ── Mock ML dependencies (mirrors tests/conftest.py for standalone runs) ─────

if 'torch' not in sys.modules:
    class FakeTensor:
        pass

    class FakeModule:
        pass

    mock_torch = MagicMock()
    mock_torch.__spec__ = MagicMock()
    mock_torch.Tensor = FakeTensor
    mock_torch.nn = MagicMock()
    mock_torch.nn.Module = FakeModule
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0
    mock_torch.cuda.empty_cache = Mock()
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.float16 = "float16"
    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = mock_torch.nn

if 'transformers' not in sys.modules:
    class FakeTokenizerBase:
        pass

    mock_transformers = MagicMock()
    mock_transformers.PreTrainedTokenizerBase = FakeTokenizerBase
    sys.modules['transformers'] = mock_transformers

for _module in [
    'trl', 'trl.experimental', 'trl.experimental.orpo',
    'peft', 'accelerate', 'bitsandbytes', 'wandb', 'psutil', 'pynvml',
    'grimoire', 'grimoire.losses', 'grimoire.data',
    'artemis_vlm',
    'atelier', 'atelier.adapters', 'atelier.losses', 'atelier.data',
    'diffusers',
]:
    if _module not in sys.modules:
        sys.modules[_module] = MagicMock()

for _module in ['datasets', 'datasets.arrow_dataset', 'datasets.load',
                'pyarrow', 'huggingface_hub']:
    try:
        __import__(_module)
    except ImportError:
        sys.modules[_module] = MagicMock()

from src.job_manager import JobManager  # noqa: E402
import src.training_runner as tr  # noqa: E402
import config as config_module  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_job_manager():
    tmp = tempfile.mkdtemp(prefix="merlina_test_")
    jm = JobManager(db_path=os.path.join(tmp, "jobs.db"))
    return jm, tmp


def _spawn(code: str) -> subprocess.Popen:
    """Spawn a python one-liner in its OWN process group (required: the
    monitor signals the whole group, and without start_new_session the
    test process itself would be signalled)."""
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _cleanup_proc(proc):
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    proc.wait()


# ── _monitor_subprocess_job ──────────────────────────────────────────────────

def test_monitor_marks_crashed_job_failed():
    """A subprocess that dies without writing a terminal status (segfault,
    OOM kill, import error) must leave the job 'failed', not stuck."""
    jm, tmp = _make_job_manager()
    try:
        jm.create_job("job_crash", {})
        jm.update_job("job_crash", status="training", progress=0.5)

        proc = _spawn("import sys; sys.exit(3)")
        try:
            detached = tr._monitor_subprocess_job(proc, "job_crash", jm, event_loop=None)
        finally:
            _cleanup_proc(proc)

        assert detached is False
        job = jm.get_job("job_crash")
        assert job.status == "failed"
        assert "exit code 3" in (job.error or "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_monitor_stops_job_with_sigterm():
    """stop_requested → SIGTERM to the group; when the process exits without
    a terminal status the job is marked 'stopped' (not failed)."""
    jm, tmp = _make_job_manager()
    try:
        jm.create_job("job_stop", {})
        jm.update_job("job_stop", status="training", progress=0.4)
        jm.request_stop("job_stop")

        # Child exits on SIGTERM (default handler).
        proc = _spawn("import time; time.sleep(60)")
        try:
            t0 = time.monotonic()
            detached = tr._monitor_subprocess_job(proc, "job_stop", jm, event_loop=None)
            elapsed = time.monotonic() - t0
        finally:
            _cleanup_proc(proc)

        assert detached is False
        assert elapsed < 30, f"stop took {elapsed:.0f}s — SIGTERM was not sent promptly"
        job = jm.get_job("job_stop")
        assert job.status == "stopped"
        assert not job.error
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_monitor_force_kills_hung_job():
    """A job that ignores SIGTERM (hung load / wedged step) must be
    SIGKILLed after the grace period — the queue worker can never hang."""
    jm, tmp = _make_job_manager()
    try:
        jm.create_job("job_hung", {})
        jm.update_job("job_hung", status="loading_model", progress=0.1)
        jm.request_stop("job_hung")

        proc = _spawn(
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(120)"
        )
        try:
            with patch.object(config_module.settings, "stop_grace_loading_seconds", 2.0):
                t0 = time.monotonic()
                detached = tr._monitor_subprocess_job(proc, "job_hung", jm, event_loop=None)
                elapsed = time.monotonic() - t0
        finally:
            _cleanup_proc(proc)

        assert detached is False
        assert elapsed < 30, f"kill escalation took {elapsed:.0f}s"
        job = jm.get_job("job_hung")
        assert job.status == "stopped"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_monitor_detaches_when_upload_runs_in_background():
    """Once the job reaches 'uploading', the monitor must release the queue
    slot (return True) even though the subprocess is still alive running
    its background upload/GGUF threads."""
    jm, tmp = _make_job_manager()
    try:
        jm.create_job("job_bg", {})
        jm.update_job("job_bg", status="uploading", progress=0.95)

        proc = _spawn("import time; time.sleep(60)")
        try:
            t0 = time.monotonic()
            detached = tr._monitor_subprocess_job(proc, "job_bg", jm, event_loop=None)
            elapsed = time.monotonic() - t0

            assert detached is True
            assert elapsed < 10, f"detach took {elapsed:.0f}s"
            assert proc.poll() is None, "monitor must not kill the background-work process"
        finally:
            _cleanup_proc(proc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── _StopEscalation unit behavior ────────────────────────────────────────────

def test_escalation_sends_sigterm_once_then_kills_only_killable_statuses():
    jm, tmp = _make_job_manager()
    try:
        jm.create_job("job_esc", {})
        jm.request_stop("job_esc")

        sent = []
        with patch.object(tr, "_signal_process_group", lambda proc, sig: sent.append(sig)):
            esc = tr._StopEscalation(proc=MagicMock(), job_id="job_esc")

            with patch.object(config_module.settings, "stop_grace_seconds", 0.0), \
                 patch.object(config_module.settings, "stop_grace_loading_seconds", 0.0):
                # Post-training statuses are never force-killed: the worker is
                # already winding down (killing would corrupt the checkpoint).
                jm.update_job("job_esc", status="saving")
                job = jm.get_job("job_esc")
                esc.check(job)   # first check → SIGTERM
                esc.check(job)   # zero grace, but status not killable → no kill
                assert sent == [signal.SIGTERM]
                assert not esc.killed

                # A wedged killable status does get the SIGKILL.
                jm.update_job("job_esc", status="training")
                job = jm.get_job("job_esc")
                esc.check(job)
                assert sent == [signal.SIGTERM, signal.SIGKILL]
                assert esc.killed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── run_training_subprocess launcher wiring ──────────────────────────────────

class _FakeConfig:
    """Minimal stand-in for the Pydantic TrainingConfig."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_launcher_spawns_worker_with_config_and_datasets():
    """The launcher must serialize the config + uploaded datasets, spawn the
    entry script with the right arguments, and respect the terminal status
    the worker writes to the DB."""
    jm, tmp = _make_job_manager()
    stub_out = os.path.join(tmp, "stub_args.json")
    project_root = str(Path(__file__).parent.parent)

    stub_script = os.path.join(tmp, "stub_worker.py")
    with open(stub_script, "w") as f:
        f.write(f"""
import argparse, json, os, sys
sys.path.insert(0, {project_root!r})
parser = argparse.ArgumentParser()
parser.add_argument("--config-path", required=True)
parser.add_argument("--job-id", required=True)
parser.add_argument("--db-path", required=True)
parser.add_argument("--uploaded-datasets-dir", default=None)
args = parser.parse_args()

with open(args.config_path) as f:
    config = json.load(f)

record = {{
    "job_id": args.job_id,
    "config": config,
    "has_uploads_meta": bool(
        args.uploaded_datasets_dir
        and os.path.exists(os.path.join(args.uploaded_datasets_dir, "meta.json"))
    ),
}}
with open(os.environ["MERLINA_STUB_OUT"], "w") as f:
    json.dump(record, f)

# Write the terminal status directly (src.job_manager pulls in torch via
# src/__init__.py, which this mocked test environment doesn't have).
import sqlite3
conn = sqlite3.connect(args.db_path)
conn.execute(
    "UPDATE jobs SET status = ?, progress = ?, output_dir = ? WHERE job_id = ?",
    ("completed", 1.0, "./models/stub", args.job_id),
)
conn.commit()
conn.close()
""")

    try:
        jm.create_job("job_launch", {})
        config = _FakeConfig({"base_model": "gpt2", "output_name": "stub"})
        uploads = {"ds1": {"content": b"{}", "filename": "ds1.json", "size": 2}}

        os.environ["MERLINA_STUB_OUT"] = stub_out
        try:
            with patch.object(tr, "_TRAIN_SINGLE_SCRIPT", stub_script):
                tr.run_training_subprocess("job_launch", config, jm, uploads, event_loop=None)
        finally:
            os.environ.pop("MERLINA_STUB_OUT", None)

        job = jm.get_job("job_launch")
        assert job.status == "completed", f"unexpected status {job.status!r} ({job.error!r})"
        assert job.output_dir == "./models/stub"

        with open(stub_out) as f:
            record = json.load(f)
        assert record["job_id"] == "job_launch"
        assert record["config"] == {"base_model": "gpt2", "output_name": "stub"}
        assert record["has_uploads_meta"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── uploaded-dataset round trip with the entry script ────────────────────────

def test_uploaded_datasets_round_trip():
    """_serialize_uploaded_datasets (launcher) → _load_uploaded_datasets
    (train_single entry) must reproduce the in-memory dict shape."""
    from src.train_single import _load_uploaded_datasets

    tmp = tempfile.mkdtemp(prefix="merlina_test_")
    try:
        uploads = {
            "abc": {"content": b'{"prompt": "hi"}', "filename": "abc.json", "size": 16},
            "xyz": {"content": b"p,c\n1,2\n", "filename": "xyz.csv", "size": 8},
        }
        ds_dir = tr._serialize_uploaded_datasets(uploads, tmp)
        assert ds_dir is not None

        loaded = _load_uploaded_datasets(ds_dir)
        assert set(loaded.keys()) == {"abc", "xyz"}
        for key in uploads:
            assert loaded[key]["content"] == uploads[key]["content"]
            assert loaded[key]["filename"] == uploads[key]["filename"]

        assert tr._serialize_uploaded_datasets({}, tmp) is None
        assert _load_uploaded_datasets(None) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── dispatch: TRAINING_ISOLATION setting ─────────────────────────────────────

def test_callback_dispatch_honors_isolation_setting():
    """_make_training_callback routes non-DDP jobs to the subprocess runner
    by default, and to the legacy in-process runner in 'thread' mode."""
    import merlina

    config_dict = {"base_model": "gpt2", "output_name": "dispatch_test"}

    with patch.object(tr, "_get_distributed_gpu_count", return_value=0), \
         patch.object(tr, "run_training_subprocess") as sub_mock, \
         patch.object(tr, "run_training_sync") as sync_mock, \
         patch.object(tr, "run_training_distributed") as ddp_mock:

        with patch.object(config_module.settings, "training_isolation", "subprocess"):
            merlina._make_training_callback(None)("job_d1", config_dict)
        assert sub_mock.call_count == 1
        assert sync_mock.call_count == 0

        with patch.object(config_module.settings, "training_isolation", "thread"):
            merlina._make_training_callback(None)("job_d2", config_dict)
        assert sync_mock.call_count == 1
        assert sub_mock.call_count == 1

        assert ddp_mock.call_count == 0


if __name__ == "__main__":
    print("=" * 60)
    print("Isolated subprocess training tests")
    print("=" * 60)
    failures = 0
    for test in [
        test_monitor_marks_crashed_job_failed,
        test_monitor_stops_job_with_sigterm,
        test_monitor_force_kills_hung_job,
        test_monitor_detaches_when_upload_runs_in_background,
        test_escalation_sends_sigterm_once_then_kills_only_killable_statuses,
        test_launcher_spawns_worker_with_config_and_datasets,
        test_uploaded_datasets_round_trip,
        test_callback_dispatch_honors_isolation_setting,
    ]:
        try:
            test()
            print(f"✅ {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"❌ {test.__name__}: {exc}")
    print("=" * 60)
    if failures:
        print(f"❌ {failures} test(s) failed")
        sys.exit(1)
    print("✅ All tests passed!")
