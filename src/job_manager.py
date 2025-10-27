"""
Job Manager with SQLite Persistence
Handles job storage, tracking, and history
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """Represents a training job"""
    job_id: str
    status: str  # started, initializing, loading_model, loading_dataset, training, saving, uploading, completed, failed, stopped
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


class JobManager:
    """
    Manages training jobs with SQLite persistence.
    Provides CRUD operations and job tracking.
    """

    def __init__(self, db_path: str = "./data/jobs.db"):
        """
        Initialize job manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config TEXT NOT NULL,
                    current_step INTEGER,
                    total_steps INTEGER,
                    loss REAL,
                    eval_loss REAL,
                    learning_rate REAL,
                    error TEXT,
                    output_dir TEXT,
                    metrics TEXT,
                    wandb_url TEXT,
                    stop_requested INTEGER DEFAULT 0
                )
            """)

            # Migration: Add wandb_url column if it doesn't exist
            cursor.execute("PRAGMA table_info(jobs)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'wandb_url' not in columns:
                cursor.execute("ALTER TABLE jobs ADD COLUMN wandb_url TEXT")
                logger.info("Added wandb_url column to jobs table")

            # Migration: Add stop_requested column if it doesn't exist
            cursor.execute("PRAGMA table_info(jobs)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'stop_requested' not in columns:
                cursor.execute("ALTER TABLE jobs ADD COLUMN stop_requested INTEGER DEFAULT 0")
                logger.info("Added stop_requested column to jobs table")

            # Training metrics table (for time-series data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL,
                    eval_loss REAL,
                    learning_rate REAL,
                    gpu_memory_used REAL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_status
                ON jobs(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_created
                ON jobs(created_at DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_job
                ON training_metrics(job_id, step)
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def create_job(self, job_id: str, config: Dict[str, Any]) -> JobRecord:
        """
        Create a new job.

        Args:
            job_id: Unique job identifier
            config: Training configuration dictionary

        Returns:
            JobRecord instance
        """
        now = datetime.now().isoformat()

        job = JobRecord(
            job_id=job_id,
            status="started",
            progress=0.0,
            created_at=now,
            updated_at=now,
            config=config
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (
                    job_id, status, progress, created_at, updated_at, config,
                    current_step, total_steps, loss, eval_loss, learning_rate,
                    error, output_dir, metrics, wandb_url, stop_requested
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.status,
                job.progress,
                job.created_at,
                job.updated_at,
                json.dumps(job.config),
                job.current_step,
                job.total_steps,
                job.loss,
                job.eval_loss,
                job.learning_rate,
                job.error,
                job.output_dir,
                json.dumps(job.metrics) if job.metrics else None,
                job.wandb_url,
                int(job.stop_requested)
            ))

        logger.info(f"Created job {job_id}")
        return job

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        error: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        wandb_url: Optional[str] = None,
        stop_requested: Optional[bool] = None
    ) -> bool:
        """
        Update job fields.

        Args:
            job_id: Job identifier
            **kwargs: Fields to update

        Returns:
            True if job was updated
        """
        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)

        if current_step is not None:
            updates.append("current_step = ?")
            params.append(current_step)

        if total_steps is not None:
            updates.append("total_steps = ?")
            params.append(total_steps)

        if loss is not None:
            updates.append("loss = ?")
            params.append(loss)

        if eval_loss is not None:
            updates.append("eval_loss = ?")
            params.append(eval_loss)

        if learning_rate is not None:
            updates.append("learning_rate = ?")
            params.append(learning_rate)

        if error is not None:
            updates.append("error = ?")
            params.append(error)

        if output_dir is not None:
            updates.append("output_dir = ?")
            params.append(output_dir)

        if metrics is not None:
            updates.append("metrics = ?")
            params.append(json.dumps(metrics))

        if wandb_url is not None:
            updates.append("wandb_url = ?")
            params.append(wandb_url)

        if stop_requested is not None:
            updates.append("stop_requested = ?")
            params.append(int(stop_requested))

        if not updates:
            return False

        # Always update timestamp
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        # Add job_id for WHERE clause
        params.append(job_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
            cursor.execute(query, params)
            success = cursor.rowcount > 0

        if success:
            logger.debug(f"Updated job {job_id}: {dict(zip([u.split(' = ')[0] for u in updates], params[:-1]))}")

        return success

    def request_stop(self, job_id: str) -> bool:
        """
        Request a job to stop gracefully.
        Sets the stop_requested flag so training can detect and stop.

        Args:
            job_id: Job identifier

        Returns:
            True if flag was set successfully
        """
        return self.update_job(job_id, stop_requested=True)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """
        Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobRecord or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_job(row)

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[JobRecord]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of JobRecord instances
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute("""
                    SELECT * FROM jobs
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

            rows = cursor.fetchall()

        return [self._row_to_job(row) for row in rows]

    def add_metric(
        self,
        job_id: str,
        step: int,
        loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gpu_memory_used: Optional[float] = None
    ):
        """
        Add a training metric record.

        Args:
            job_id: Job identifier
            step: Training step number
            loss: Training loss
            eval_loss: Evaluation loss
            learning_rate: Current learning rate
            gpu_memory_used: GPU memory in GB
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_metrics (
                    job_id, timestamp, step, loss, eval_loss,
                    learning_rate, gpu_memory_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                datetime.now().isoformat(),
                step,
                loss,
                eval_loss,
                learning_rate,
                gpu_memory_used
            ))

    def get_metrics(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get all metrics for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of metric dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_metrics
                WHERE job_id = ?
                ORDER BY step ASC
            """, (job_id,))
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job and its metrics.

        Args:
            job_id: Job identifier

        Returns:
            True if job was deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Delete metrics first
            cursor.execute("DELETE FROM training_metrics WHERE job_id = ?", (job_id,))

            # Delete job
            cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            success = cursor.rowcount > 0

        if success:
            logger.info(f"Deleted job {job_id}")

        return success

    def clear_all_jobs(self) -> int:
        """
        Delete all jobs and metrics from the database.

        Returns:
            Number of jobs deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Count jobs before deletion
            cursor.execute("SELECT COUNT(*) FROM jobs")
            count = cursor.fetchone()[0]

            # Delete all metrics
            cursor.execute("DELETE FROM training_metrics")

            # Delete all jobs
            cursor.execute("DELETE FROM jobs")

        logger.info(f"Cleared all jobs ({count} jobs deleted)")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with stats
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Count by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM jobs
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            # Total jobs
            cursor.execute("SELECT COUNT(*) FROM jobs")
            total_jobs = cursor.fetchone()[0]

            # Total metrics
            cursor.execute("SELECT COUNT(*) FROM training_metrics")
            total_metrics = cursor.fetchone()[0]

        return {
            "total_jobs": total_jobs,
            "total_metrics": total_metrics,
            "by_status": status_counts
        }

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        """Convert database row to JobRecord"""
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            progress=row["progress"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            config=json.loads(row["config"]),
            current_step=row["current_step"],
            total_steps=row["total_steps"],
            loss=row["loss"],
            eval_loss=row["eval_loss"],
            learning_rate=row["learning_rate"],
            error=row["error"],
            output_dir=row["output_dir"],
            metrics=json.loads(row["metrics"]) if row["metrics"] else None,
            wandb_url=row["wandb_url"] if "wandb_url" in row.keys() else None,
            stop_requested=bool(row["stop_requested"]) if "stop_requested" in row.keys() else False
        )
