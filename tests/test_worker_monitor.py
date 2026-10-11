"""
Tests for src/worker_monitor.py - the single loop that watches a detached
training worker: relays progress, forwards stop requests, reconciles the
job on exit. Runs without GPUs or the ML stack.
"""

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_manager import JobManager  # noqa: E402
from src import worker_monitor  # noqa: E402
from src.worker_monitor import monitor_worker, forward_progress_record  # noqa: E402


@pytest.fixture
def jm(tmp_path):
    return JobManager(db_path=str(tmp_path / "jobs.db"))


@pytest.fixture
def published(monkeypatch):
    """Capture everything the monitor would push to WebSocket clients."""
    captured = []
    monkeypatch.setattr(
        worker_monitor.websocket_manager, "publish",
        lambda job_id, message, loop=None: captured.append((job_id, message)),
    )
    return captured


def _spawn_sleeper(job_id, seconds=60):
    # Own session, like the real launcher, so killpg() has a group to hit.
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})", "--job-id", job_id],
        start_new_session=True,
    )


def _wait_until(pred, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_forward_progress_record_shapes(published):
    forward_progress_record({"type": "metrics", "loss": 1.5, "current_step": 3, "total_steps": 10, "progress": 0.3}, "j")
    forward_progress_record({"type": "eval", "eval_loss": 0.9}, "j")
    forward_progress_record({"type": "status", "status": "saving", "message": "writing"}, "j")
    forward_progress_record({"type": "completed", "output_dir": "/m"}, "j")
    forward_progress_record({"type": "error", "error": "boom"}, "j")
    forward_progress_record({"type": "eval"}, "j")        # no eval_loss -> dropped
    forward_progress_record({"type": "mystery"}, "j")     # unknown -> dropped

    types = [m["type"] for _, m in published]
    assert types == ["status_update", "status_update", "status_update", "completed", "error"]
    metrics = published[0][1]
    assert metrics["loss"] == 1.5 and metrics["current_step"] == 3 and metrics["progress"] == 0.3
    assert published[1][1]["eval_loss"] == 0.9
    assert published[2][1]["status"] == "saving" and published[2][1]["message"] == "writing"
    assert published[3][1]["output_dir"] == "/m"
    assert published[4][1]["error"] == "boom"
    assert all(jid == "j" for jid, _ in published)


def test_worker_dying_without_terminal_status_marks_job_failed(jm, tmp_path, published):
    job_id = "job_dies"
    run_dir = tmp_path / "worker_runs" / job_id
    run_dir.mkdir(parents=True)
    progress = run_dir / "progress.jsonl"
    progress.write_text("")

    proc = _spawn_sleeper(job_id)
    jm.create_job(job_id, {})
    jm.update_job(job_id, status="training", worker_pid=proc.pid, progress_file=str(progress))

    t = threading.Thread(
        target=monitor_worker,
        args=(job_id, proc.pid, str(progress), jm),
        kwargs={"is_alive": lambda: proc.poll() is None, "returncode": lambda: proc.returncode,
                "poll_interval": 0.05},
        daemon=True,
    )
    t.start()
    time.sleep(0.2)
    assert jm.get_job(job_id).status == "training"

    proc.kill()
    t.join(timeout=10)
    assert not t.is_alive(), "monitor did not return after the worker died"

    job = jm.get_job(job_id)
    assert job.status == "failed"
    assert "exited without reporting a final status" in job.error
    assert "exit code" in job.error
    assert job.worker_pid is None
    assert not run_dir.exists(), "run dir should be removed once the worker is gone"
    assert any(m["type"] == "error" for _, m in published)


def test_worker_that_reported_completion_is_left_alone(jm, tmp_path, published):
    job_id = "job_ok"
    run_dir = tmp_path / "worker_runs" / job_id
    run_dir.mkdir(parents=True)
    progress = run_dir / "progress.jsonl"
    progress.write_text("")

    proc = _spawn_sleeper(job_id, seconds=1)
    jm.create_job(job_id, {})
    jm.update_job(job_id, status="training", worker_pid=proc.pid, progress_file=str(progress))

    # The worker writes its own terminal status and a final record, then exits 0.
    with open(progress, "a") as f:
        f.write(json.dumps({"type": "metrics", "loss": 0.5, "progress": 0.9}) + "\n")
        f.write(json.dumps({"type": "completed", "output_dir": "/models/x"}) + "\n")
    jm.update_job(job_id, status="completed", progress=1.0)

    monitor_worker(job_id, proc.pid, str(progress), jm,
                   is_alive=lambda: proc.poll() is None, returncode=lambda: proc.returncode,
                   poll_interval=0.05)

    job = jm.get_job(job_id)
    assert job.status == "completed"
    assert job.error is None
    assert job.worker_pid is None
    assert [m["type"] for _, m in published] == ["status_update", "completed"]


def test_stop_request_is_forwarded_as_sigterm(jm, tmp_path, published):
    job_id = "job_stop"
    proc = _spawn_sleeper(job_id)
    jm.create_job(job_id, {})
    jm.update_job(job_id, status="training", worker_pid=proc.pid)

    t = threading.Thread(
        target=monitor_worker, args=(job_id, proc.pid, None, jm),
        kwargs={"is_alive": lambda: proc.poll() is None, "returncode": lambda: proc.returncode,
                "poll_interval": 0.05},
        daemon=True,
    )
    t.start()
    time.sleep(0.2)
    assert proc.poll() is None

    jm.request_stop(job_id)
    t.join(timeout=10)
    assert not t.is_alive()
    assert proc.returncode == -signal.SIGTERM, f"expected SIGTERM, got {proc.returncode}"
    # The worker was killed before it could report, so the monitor fails it.
    assert jm.get_job(job_id).status == "failed"


def test_sigterm_escalates_to_sigkill(jm, monkeypatch):
    job_id = "job_stubborn"
    # A worker that ignores SIGTERM.
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
         "--job-id", job_id],
        start_new_session=True,
    )
    jm.create_job(job_id, {})
    jm.update_job(job_id, status="training", worker_pid=proc.pid)
    monkeypatch.setattr(worker_monitor, "STOP_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(worker_monitor.websocket_manager, "publish", lambda *a, **k: None)

    # Give the child time to install its handler before we ask it to stop.
    time.sleep(0.5)
    jm.request_stop(job_id)
    started = time.monotonic()
    monitor_worker(job_id, proc.pid, None, jm,
                   is_alive=lambda: proc.poll() is None, returncode=lambda: proc.returncode,
                   poll_interval=0.05)
    assert proc.returncode == -signal.SIGKILL
    assert time.monotonic() - started < 10


def test_monitor_survives_transient_db_errors(jm, published):
    """A DB hiccup while polling must not kill the monitor and orphan the job."""
    job_id = "job_flaky_db"
    proc = _spawn_sleeper(job_id)
    jm.create_job(job_id, {})
    jm.update_job(job_id, status="training", worker_pid=proc.pid)

    calls = {"n": 0}
    real_get_job = jm.get_job

    def flaky_get_job(jid):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("database is locked")
        return real_get_job(jid)

    jm.get_job = flaky_get_job

    t = threading.Thread(
        target=monitor_worker, args=(job_id, proc.pid, None, jm),
        kwargs={"is_alive": lambda: proc.poll() is None, "returncode": lambda: proc.returncode,
                "poll_interval": 0.05},
        daemon=True,
    )
    t.start()
    assert _wait_until(lambda: calls["n"] > 3), "monitor stopped polling after a DB error"
    assert t.is_alive()

    proc.kill()
    t.join(timeout=10)
    assert not t.is_alive()
    assert real_get_job(job_id).status == "failed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
