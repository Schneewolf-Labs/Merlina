"""Per-component trainer metrics (ORPO nll/or loss, rewards, margins, ...)
are persisted with each metrics row, not just the total loss."""

import json
import math
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.job_manager import JobManager
from src.trainer_metrics import extract_extra_metrics, finite_or_none

# What grimoire's trainer hands on_log for an ORPO step.
ORPO_LOG = {
    "train/loss": 1.25,
    "train/learning_rate": 5e-6,
    "train/epoch": 0.5,
    "train/global_step": 4,
    "train/progress": 0.5,
    "train/nll_loss": 1.1,
    "train/or_loss": 0.15,
    "train/chosen_rewards": -0.2,
    "train/rejected_rewards": -0.3,
    "train/log_odds_ratio": 0.4,
    "train/reward_margin": 0.1,
    "train/reward_accuracy": 0.75,
}
ORPO_EVAL = {"eval/loss": 1.3, "eval/reward_margin": 0.08, "eval/reward_accuracy": 0.5}


@pytest.fixture
def jm(tmp_path):
    jm = JobManager(str(tmp_path / "jobs.db"))
    jm.create_job("job_1", {})
    return jm


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

class TestExtractExtraMetrics:
    def test_keeps_components_drops_columns_and_bookkeeping(self):
        extra = extract_extra_metrics(ORPO_LOG)
        assert extra == {
            "train/epoch": 0.5,
            "train/nll_loss": 1.1,
            "train/or_loss": 0.15,
            "train/chosen_rewards": -0.2,
            "train/rejected_rewards": -0.3,
            "train/log_odds_ratio": 0.4,
            "train/reward_margin": 0.1,
            "train/reward_accuracy": 0.75,
        }

    def test_eval_loss_is_a_column(self):
        assert extract_extra_metrics(ORPO_EVAL) == {
            "eval/reward_margin": 0.08, "eval/reward_accuracy": 0.5,
        }

    def test_non_finite_becomes_none_and_non_numeric_dropped(self):
        extra = extract_extra_metrics({
            "train/a": float("nan"), "train/b": float("inf"),
            "train/c": "text", "train/d": [1, 2], "train/e": None, "train/f": 3,
        })
        assert extra == {"train/a": None, "train/b": None, "train/f": 3.0}
        json.dumps(extra, allow_nan=False)  # API-safe

    def test_tensor_like_scalars_and_bools(self):
        class Scalar:
            def item(self):
                return 0.5
        assert extract_extra_metrics({"train/x": Scalar(), "train/flag": True}) == {
            "train/x": 0.5, "train/flag": 1.0,
        }

    def test_empty(self):
        assert extract_extra_metrics(None) == {}
        assert extract_extra_metrics({"train/loss": 1.0}) == {}

    def test_finite_or_none(self):
        assert finite_or_none(None) is None
        assert finite_or_none(float("nan")) is None
        assert finite_or_none("x") is None
        assert finite_or_none(2) == 2.0


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

class TestJobManagerExtras:
    def test_round_trip(self, jm):
        jm.add_metric("job_1", step=1, loss=1.0, learning_rate=1e-5,
                      extra={"train/nll_loss": 0.9, "train/reward_margin": 0.1})
        (row,) = jm.get_metrics("job_1")
        assert row["loss"] == 1.0 and row["learning_rate"] == 1e-5
        assert row["extra"] == {"train/nll_loss": 0.9, "train/reward_margin": 0.1}

    def test_rows_without_extras_report_empty_dict(self, jm):
        jm.add_metric("job_1", step=1, loss=1.0)
        assert jm.get_metrics("job_1")[0]["extra"] == {}

    def test_migrates_old_schema(self, tmp_path):
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                timestamp TEXT NOT NULL, step INTEGER NOT NULL, loss REAL,
                eval_loss REAL, learning_rate REAL, gpu_memory_used REAL
            )
        """)
        conn.execute("INSERT INTO training_metrics (job_id, timestamp, step, loss)"
                     " VALUES ('old', '2025-01-01T00:00:00', 5, 0.7)")
        conn.commit()
        conn.close()

        jm = JobManager(str(db))
        old = jm.get_metrics("old")
        assert old[0]["loss"] == 0.7 and old[0]["extra"] == {}
        jm.add_metric("old", step=6, loss=0.6, extra={"train/or_loss": 0.2})
        assert jm.get_metrics("old")[1]["extra"] == {"train/or_loss": 0.2}

    def test_eval_merges_into_same_step_row(self, jm):
        jm.add_metric("job_1", step=4, loss=1.0, extra={"train/nll_loss": 0.9})
        jm.record_eval_metric("job_1", step=4, eval_loss=1.2,
                              extra={"eval/reward_margin": 0.05})
        (row,) = jm.get_metrics("job_1")
        assert row["loss"] == 1.0 and row["eval_loss"] == 1.2
        assert row["extra"] == {"train/nll_loss": 0.9, "eval/reward_margin": 0.05}

    def test_eval_without_train_row_inserts(self, jm):
        jm.add_metric("job_1", step=4, loss=1.0)
        jm.record_eval_metric("job_1", step=5, eval_loss=1.2, extra=None)
        rows = jm.get_metrics("job_1")
        assert [r["step"] for r in rows] == [4, 5]
        assert rows[1]["loss"] is None and rows[1]["eval_loss"] == 1.2

    def test_eval_extras_only_keeps_existing_eval_loss(self, jm):
        jm.record_eval_metric("job_1", step=3, eval_loss=1.5)
        jm.record_eval_metric("job_1", step=3, eval_loss=None, extra={"eval/x": 1.0})
        (row,) = jm.get_metrics("job_1")
        assert row["eval_loss"] == 1.5 and row["extra"] == {"eval/x": 1.0}


# --------------------------------------------------------------------------- #
# Callbacks (the subprocess worker and the in-process runner)
# --------------------------------------------------------------------------- #

def _trainer(step=4, max_steps=8):
    return SimpleNamespace(global_step=step, max_steps=max_steps)


class _CallbackBase:
    """Stand-in for grimoire.TrainerCallback (conftest mocks grimoire, and a
    class deriving from a MagicMock is itself a MagicMock)."""


def _load_with_real_callback_base(module: str):
    """Import a private copy of ``src.<module>`` whose callbacks are real classes.

    The shared ``src.<module>`` in sys.modules is left untouched.
    """
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "src" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"src._metrics_test_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    with patch.object(sys.modules["grimoire"], "TrainerCallback", _CallbackBase, create=True):
        spec.loader.exec_module(mod)
    # Other tests reconfigure the session-wide torch mock; pin "no CUDA" here.
    mod.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    return mod


@pytest.fixture(scope="module")
def train_worker_mod():
    return _load_with_real_callback_base("train_worker")


@pytest.fixture(scope="module")
def training_runner_mod():
    return _load_with_real_callback_base("training_runner")


class TestFileProgressCallback:
    def test_on_log_and_on_evaluate_persist_components(self, jm, tmp_path, train_worker_mod):
        FileProgressCallback = train_worker_mod.FileProgressCallback

        progress = tmp_path / "progress.jsonl"
        cb = FileProgressCallback("job_1", str(progress), jm)
        cb.on_log(_trainer(), ORPO_LOG)
        cb.on_evaluate(_trainer(), ORPO_EVAL)

        (row,) = jm.get_metrics("job_1")
        assert row["step"] == 4 and row["loss"] == 1.25 and row["eval_loss"] == 1.3
        for key in ("train/nll_loss", "train/or_loss", "train/chosen_rewards",
                    "train/rejected_rewards", "train/reward_margin",
                    "train/reward_accuracy", "eval/reward_margin"):
            assert key in row["extra"], key
        assert "train/loss" not in row["extra"]

        records = [json.loads(line) for line in progress.read_text().splitlines()]
        assert records[0]["type"] == "metrics"
        assert records[0]["extra"]["train/or_loss"] == 0.15
        assert records[1]["type"] == "eval" and records[1]["eval_loss"] == 1.3
        assert records[1]["extra"] == {"eval/reward_margin": 0.08, "eval/reward_accuracy": 0.5}
        assert jm.get_job("job_1").eval_loss == 1.3

    def test_sft_log_has_no_extras_beyond_epoch(self, jm, tmp_path, train_worker_mod):
        FileProgressCallback = train_worker_mod.FileProgressCallback

        cb = FileProgressCallback("job_1", str(tmp_path / "p.jsonl"), jm)
        cb.on_log(_trainer(), {"train/loss": 2.0, "train/learning_rate": 1e-5,
                               "train/global_step": 4, "train/progress": 0.5})
        (row,) = jm.get_metrics("job_1")
        assert row["loss"] == 2.0 and row["extra"] == {}

    def test_nan_eval_loss_is_not_written(self, jm, tmp_path, train_worker_mod):
        FileProgressCallback = train_worker_mod.FileProgressCallback

        cb = FileProgressCallback("job_1", str(tmp_path / "p.jsonl"), jm)
        cb.on_evaluate(_trainer(), {"eval/loss": float("nan")})
        assert jm.get_metrics("job_1") == []


class TestWebSocketCallback:
    def test_on_log_and_on_evaluate_persist_components(self, jm, training_runner_mod):
        training_runner = training_runner_mod

        sent = []
        with patch.object(training_runner, "send_websocket_update",
                          lambda coro, loop=None: (sent.append(coro), coro.close())):
            cb = training_runner.WebSocketCallback("job_1", jm)
            cb.on_log(_trainer(), ORPO_LOG)
            cb.on_evaluate(_trainer(), ORPO_EVAL)

        (row,) = jm.get_metrics("job_1")
        assert row["loss"] == 1.25 and row["eval_loss"] == 1.3
        assert row["extra"]["train/nll_loss"] == 1.1
        assert row["extra"]["train/or_loss"] == 0.15
        assert row["extra"]["eval/reward_accuracy"] == 0.5
        assert sent, "live status update still sent"


# --------------------------------------------------------------------------- #
# Live relay, remote ingestion and API
# --------------------------------------------------------------------------- #

def test_worker_monitor_forwards_extras():
    from src import worker_monitor

    published = []
    with patch.object(worker_monitor.websocket_manager, "publish",
                      lambda job_id, message, loop=None: published.append(message)):
        worker_monitor.forward_progress_record(
            {"type": "metrics", "loss": 1.0, "current_step": 2,
             "extra": {"train/or_loss": 0.1}}, "j")
        worker_monitor.forward_progress_record(
            {"type": "eval", "current_step": 2, "extra": {"eval/reward_margin": 0.3}}, "j")
    assert published[0]["extra_metrics"] == {"train/or_loss": 0.1}
    assert published[1]["extra_metrics"] == {"eval/reward_margin": 0.3}
    assert published[1]["current_step"] == 2


def test_remote_status_mirror_keeps_extras(jm):
    from src.remote.orchestrator import _mirror_status

    status = SimpleNamespace(job={}, new_metrics=[
        {"step": 3, "loss": 1.0, "extra": {"train/nll_loss": 0.8}},
        {"step": 4, "loss": 0.9, "extra": {}},
    ])
    with patch("src.remote.orchestrator._send_ws", lambda *a, **k: None):
        since = _mirror_status(status, "job_1", jm, None, 0)
    assert since == 4
    rows = jm.get_metrics("job_1")
    assert rows[0]["extra"] == {"train/nll_loss": 0.8}
    assert rows[1]["extra"] == {}


def test_metrics_endpoint_returns_extras(jm):
    from fastapi.testclient import TestClient
    import merlina

    jm.add_metric("job_1", step=1, loss=1.0, extra={"train/reward_margin": 0.1, "train/x": None})
    jm.add_metric("job_1", step=2, loss=0.9)
    with patch.object(merlina, "job_manager", jm):
        response = TestClient(merlina.app).get("/jobs/job_1/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    first, second = body["metrics"]
    # Existing fields are unchanged; extras ride alongside.
    for field in ("step", "loss", "eval_loss", "learning_rate", "gpu_memory_used", "timestamp"):
        assert field in first
    assert first["extra"] == {"train/reward_margin": 0.1, "train/x": None}
    assert second["extra"] == {}
    assert not math.isnan(first["loss"])
