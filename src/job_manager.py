"""
Job persistence (SQLite).

The DB is the single source of truth for job status: the API process, the
queue workers and the detached training subprocess (``src/train_worker.py``)
all write to it. SQLite in WAL mode with a busy timeout is what makes that
safe across processes - every call opens its own short-lived connection and
commits on exit, so no connection is ever shared between threads.

Status vocabulary (``ACTIVE_STATUSES`` / ``TERMINAL_STATUSES``) is defined
here so the queue, the worker monitor and the restart reconciler agree on
what "still running" means.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Statuses that mean "this job should have a live worker or queue entry".
ACTIVE_STATUSES = frozenset({
    "started", "queued", "initializing", "loading_model", "loading_dataset",
    "training", "saving", "stopping", "running",
})

# Statuses a job sets on its own way out; never overwrite these.
TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped", "cancelled", "uploading"})


@dataclass
class JobRecord:
    """One row of the ``jobs`` table."""
    job_id: str
    status: str
    progress: float
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    wandb_url: Optional[str] = None
    stop_requested: bool = False
    upload_error: Optional[str] = None
    gguf_error: Optional[str] = None
    # Subprocess bookkeeping so an API restart can find (and re-attach to)
    # a still-running training worker instead of orphaning it.
    worker_pid: Optional[int] = None
    progress_file: Optional[str] = None


# Columns that update_job() may touch, with their SQL type (used both for
# the CREATE TABLE and for adding columns to databases from older versions).
_UPDATABLE_COLUMNS: Dict[str, str] = {
    "status": "TEXT",
    "progress": "REAL",
    "current_step": "INTEGER",
    "total_steps": "INTEGER",
    "loss": "REAL",
    "eval_loss": "REAL",
    "learning_rate": "REAL",
    "error": "TEXT",
    "output_dir": "TEXT",
    "metrics": "TEXT",
    "wandb_url": "TEXT",
    "stop_requested": "INTEGER",
    "upload_error": "TEXT",
    "gguf_error": "TEXT",
    "worker_pid": "INTEGER",
    "progress_file": "TEXT",
}

# For most columns ``None`` means "leave unchanged" (callers pass through
# whatever they have). These two must be clearable: a job with no live worker
# must not keep advertising a stale pid.
_NULLABLE_COLUMNS = frozenset({"worker_pid", "progress_file"})


def _dump_extra(extra: Optional[Dict[str, Any]]) -> Optional[str]:
    return json.dumps(extra) if extra else None


def _load_extra(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class JobManager:
    """CRUD for jobs and their time-series metrics."""

    def __init__(self, db_path: str = "./data/jobs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        # timeout= installs SQLite's busy handler: a writer blocked by another
        # process/thread waits up to 30 s instead of failing immediately.
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            # WAL is persistent in the database file; setting it once here
            # (rather than on every connection) avoids taking a write lock on
            # each read.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config TEXT NOT NULL
                )
            """)
            existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            for column, sql_type in _UPDATABLE_COLUMNS.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {sql_type}")
                    logger.info(f"Added {column} column to jobs table")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL,
                    eval_loss REAL,
                    learning_rate REAL,
                    gpu_memory_used REAL,
                    extra TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
            """)
            # Databases from before per-component metrics lack ``extra``.
            metric_columns = {row[1] for row in conn.execute("PRAGMA table_info(training_metrics)")}
            if "extra" not in metric_columns:
                conn.execute("ALTER TABLE training_metrics ADD COLUMN extra TEXT")
                logger.info("Added extra column to training_metrics table")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_created ON jobs(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_job ON training_metrics(job_id, step)")
        logger.info(f"Database initialized at {self.db_path}")

    # ------------------------------------------------------------------ #
    # Jobs
    # ------------------------------------------------------------------ #

    def create_job(self, job_id: str, config: Dict[str, Any]) -> JobRecord:
        now = datetime.now().isoformat()
        job = JobRecord(
            job_id=job_id, status="started", progress=0.0,
            created_at=now, updated_at=now, config=config,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, progress, created_at, updated_at, config, stop_requested)"
                " VALUES (?, ?, ?, ?, ?, ?, 0)",
                (job_id, job.status, job.progress, now, now, json.dumps(config)),
            )
        logger.info(f"Created job {job_id}")
        return job

    def update_job(self, job_id: str, **fields: Any) -> bool:
        """
        Update columns of a job. Returns True if the row exists.

        ``None`` leaves a column unchanged, except for ``worker_pid`` and
        ``progress_file`` where an explicit ``None`` clears it. Unknown
        column names are a programming error and raise ``TypeError``.
        """
        unknown = set(fields) - set(_UPDATABLE_COLUMNS)
        if unknown:
            raise TypeError(f"update_job() got unknown fields: {sorted(unknown)}")

        assignments: Dict[str, Any] = {}
        for column, value in fields.items():
            if value is None and column not in _NULLABLE_COLUMNS:
                continue
            if column == "metrics":
                value = json.dumps(value)
            elif column == "stop_requested":
                value = int(bool(value))
            assignments[column] = value

        if not assignments:
            return False

        assignments["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{column} = ?" for column in assignments)
        params = [*assignments.values(), job_id]

        with self._connect() as conn:
            cursor = conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", params)
            updated = cursor.rowcount > 0

        if updated:
            logger.debug(f"Updated job {job_id}: {assignments}")
        return updated

    def request_stop(self, job_id: str) -> bool:
        """Ask a running job to stop gracefully (the job polls this flag)."""
        return self.update_job(job_id, stop_requested=True)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[JobRecord]:
        where, params = "", []
        if status:
            where, params = "WHERE status = ?", [status]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM training_metrics WHERE job_id = ?", (job_id,))
            deleted = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,)).rowcount > 0
        if deleted:
            logger.info(f"Deleted job {job_id}")
        return deleted

    def clear_all_jobs(self) -> int:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            conn.execute("DELETE FROM training_metrics")
            conn.execute("DELETE FROM jobs")
        logger.info(f"Cleared all jobs ({count} jobs deleted)")
        return count

    # ------------------------------------------------------------------ #
    # Metrics / stats
    # ------------------------------------------------------------------ #

    def add_metric(
        self,
        job_id: str,
        step: int,
        loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gpu_memory_used: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one metrics row.

        ``extra`` holds every other metric the trainer logged for this step
        (e.g. ORPO's ``train/nll_loss`` / ``train/or_loss``, DPO's reward
        margin), keyed as the trainer logged them. Stored as JSON.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO training_metrics"
                " (job_id, timestamp, step, loss, eval_loss, learning_rate, gpu_memory_used, extra)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, datetime.now().isoformat(), step, loss, eval_loss, learning_rate,
                 gpu_memory_used, _dump_extra(extra)),
            )

    def record_eval_metric(
        self,
        job_id: str,
        step: int,
        eval_loss: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach evaluation results to the metrics row for ``step``.

        The trainer logs the training row for a step before it evaluates, so
        eval results are merged into that row (one row per step keeps the
        chart aligned). If no row exists for the step - eval ran on a step
        that wasn't a logging step - a new row is inserted.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, eval_loss, extra FROM training_metrics"
                " WHERE job_id = ? AND step = ? ORDER BY id DESC LIMIT 1",
                (job_id, step),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO training_metrics (job_id, timestamp, step, eval_loss, extra)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (job_id, datetime.now().isoformat(), step, eval_loss, _dump_extra(extra)),
                )
                return
            merged = {**_load_extra(row["extra"]), **(extra or {})}
            conn.execute(
                "UPDATE training_metrics SET eval_loss = ?, extra = ? WHERE id = ?",
                (eval_loss if eval_loss is not None else row["eval_loss"],
                 _dump_extra(merged), row["id"]),
            )

    def get_metrics(self, job_id: str) -> List[Dict[str, Any]]:
        """Metrics rows, oldest step first.

        Each row has the fixed columns (``step``, ``loss``, ``eval_loss``,
        ``learning_rate``, ``gpu_memory_used``, ``timestamp``) plus ``extra``:
        a dict of every other logged metric (``{}`` for rows recorded before
        extras were persisted).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM training_metrics WHERE job_id = ? ORDER BY step ASC, id ASC", (job_id,)
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["extra"] = _load_extra(data.get("extra"))
            result.append(data)
        return result

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            by_status = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
            total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            total_metrics = conn.execute("SELECT COUNT(*) FROM training_metrics").fetchone()[0]
        return {"total_jobs": total_jobs, "total_metrics": total_metrics, "by_status": by_status}

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        data = dict(row)
        data["config"] = json.loads(data["config"])
        data["metrics"] = json.loads(data["metrics"]) if data.get("metrics") else None
        data["stop_requested"] = bool(data.get("stop_requested"))
        return JobRecord(**{k: v for k, v in data.items() if k in JobRecord.__dataclass_fields__})
