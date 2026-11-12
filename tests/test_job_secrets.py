#!/usr/bin/env python3
"""Job configs must never store credentials (hf_token, wandb_key) at rest."""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_manager import JobManager


def _stored(db_path, job_id):
    with sqlite3.connect(db_path) as conn:
        return json.loads(conn.execute("SELECT config FROM jobs WHERE job_id = ?", (job_id,)).fetchone()[0])


def test_create_job_strips_secrets(tmp_path):
    db = tmp_path / "jobs.db"
    jm = JobManager(db_path=str(db))
    jm.create_job("job_a", {"base_model": "m", "hf_token": "hf_secret", "wandb_key": "wb_secret"})
    stored = _stored(db, "job_a")
    assert stored == {"base_model": "m"}
    assert jm.get_job("job_a").config == {"base_model": "m"}


def test_existing_secrets_scrubbed_on_open(tmp_path):
    db = tmp_path / "jobs.db"
    JobManager(db_path=str(db)).create_job("job_b", {"base_model": "m"})
    # Simulate a row written by an older version.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE jobs SET config = ? WHERE job_id = 'job_b'",
                     (json.dumps({"base_model": "m", "hf_token": "hf_old"}),))
    JobManager(db_path=str(db))
    assert _stored(db, "job_b") == {"base_model": "m"}
