"""Durable remote ownership. Never retry an ambiguous paid provisioning call.

Only a new job ID authorizes a new execution. A lost controller is failed after
its lease expires; checkpoint/worker resume is deliberately not implied.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class RemoteAlreadyClaimed(RuntimeError):
    pass


class RemoteLeaseLost(RuntimeError):
    pass


class RemoteLifecycle:
    def __init__(self, db_path, *, lease_seconds=120.0, clock=time.time):
        self.db_path = str(db_path)
        self.lease_seconds = lease_seconds
        self.clock = clock
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS remote_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS remote_cleanup_alerts (
                    alert_key TEXT PRIMARY KEY, message TEXT NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS remote_cancellations (
                    job_id TEXT PRIMARY KEY, requested_at REAL NOT NULL,
                    cleanup_after REAL NOT NULL, emergency INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS remote_runs (
                    job_id TEXT PRIMARY KEY, owner TEXT NOT NULL, state TEXT NOT NULL,
                    started_at REAL NOT NULL, deadline REAL NOT NULL, lease_until REAL NOT NULL,
                    error TEXT);
                CREATE TABLE IF NOT EXISTS remote_attempts (
                    job_id TEXT NOT NULL, stage_name TEXT NOT NULL, attempt INTEGER NOT NULL,
                    label TEXT UNIQUE NOT NULL, provider TEXT NOT NULL, state TEXT NOT NULL,
                    created_at REAL NOT NULL, artifacts TEXT NOT NULL DEFAULT '[]', error TEXT,
                    PRIMARY KEY(job_id, stage_name, attempt));
                CREATE TABLE IF NOT EXISTS remote_resources (
                    provider TEXT NOT NULL, instance_id TEXT NOT NULL, job_id TEXT,
                    stage_name TEXT, label TEXT NOT NULL, started_at REAL NOT NULL,
                    terminated_at REAL, cost_per_hr REAL,
                    cleanup_status TEXT NOT NULL DEFAULT 'active', cleanup_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT, PRIMARY KEY(provider, instance_id));
            """)
            db.execute("INSERT OR IGNORE INTO remote_metadata VALUES ('namespace', ?)", (uuid.uuid4().hex[:16],))
            self.namespace = db.execute("SELECT value FROM remote_metadata WHERE key='namespace'").fetchone()[0]

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def claim(self, job_id, max_runtime_hours):
        now = self.clock()
        owner = uuid.uuid4().hex
        with self.connect() as db:
            result = db.execute(
                "INSERT OR IGNORE INTO remote_runs VALUES (?, ?, 'running', ?, ?, ?, NULL)",
                (job_id, owner, now, now + max_runtime_hours * 3600, now + self.lease_seconds))
            if not result.rowcount:
                raise RemoteAlreadyClaimed(f"Remote job {job_id} already has a durable execution record")
        return owner

    def run(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM remote_runs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def heartbeat(self, job_id, owner):
        now = self.clock()
        with self.connect() as db:
            result = db.execute("""UPDATE remote_runs SET lease_until=?
                WHERE job_id=? AND owner=? AND state='running' AND lease_until>? AND deadline>?""",
                (now + self.lease_seconds, job_id, owner, now, now))
            if not result.rowcount:
                raise RemoteLeaseLost("Remote execution lease or total job deadline expired")

    def intent(self, job_id, owner, stage_name, provider):
        """Commit ownership and stable discovery label BEFORE calling the provider."""
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT * FROM remote_runs WHERE job_id=?", (job_id,)).fetchone()
            if not run or run['owner'] != owner or run['state'] != 'running' or run['lease_until'] <= now or run['deadline'] <= now:
                raise RemoteLeaseLost("Cannot provision after lease or total job deadline expired")
            label = f"merlina-{self.namespace}-{hashlib.sha256(job_id.encode()).hexdigest()[:16]}-{stage_name}"
            try:
                db.execute("""INSERT INTO remote_attempts
                    (job_id, stage_name, attempt, label, provider, state, created_at)
                    VALUES (?, ?, 1, ?, ?, 'provisioning', ?)""", (job_id, stage_name, label, provider, now))
            except sqlite3.IntegrityError as exc:
                raise RemoteAlreadyClaimed("Stage already attempted; automatic provisioning retry is unsafe") from exc
        return label

    def record_instance(self, job_id, stage_name, label, instance):
        with self.connect() as db:
            # Ambiguous creates are charged conservatively from the persisted intent.
            attempt = db.execute("SELECT created_at FROM remote_attempts WHERE label=?", (label,)).fetchone()
            started_at = attempt[0] if attempt else self.clock()
            db.execute("""INSERT OR IGNORE INTO remote_resources
                (provider, instance_id, job_id, stage_name, label, started_at, cost_per_hr)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (instance.provider, instance.instance_id, job_id, stage_name, label, started_at, instance.cost_per_hr))

    def stage_state(self, job_id, stage_name, state, *, artifacts=None, error=None):
        with self.connect() as db:
            db.execute("""UPDATE remote_attempts SET state=?, artifacts=?, error=?
                WHERE job_id=? AND stage_name=?""", (state, json.dumps(artifacts or []), error, job_id, stage_name))

    def finish(self, job_id, owner, state, error=None):
        with self.connect() as db:
            db.execute("UPDATE remote_runs SET state=?, error=? WHERE job_id=? AND owner=? AND state='running'",
                       (state, error, job_id, owner))

    def attempts(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM remote_attempts")]

    def resources(self, job_id=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM remote_resources" + (" WHERE job_id=?" if job_id else ""),
                              (job_id,) if job_id else ())
            return [dict(row) for row in rows]

    def cleanup_result(self, provider, instance_id, *, confirmed=False, error=None, retained=False):
        with self.connect() as db:
            db.execute("""UPDATE remote_resources SET cleanup_status=?, cleanup_attempts=cleanup_attempts+1,
                last_error=?, terminated_at=CASE WHEN ? THEN COALESCE(terminated_at, ?) ELSE terminated_at END
                WHERE provider=? AND instance_id=? AND terminated_at IS NULL""",
                ('terminated' if confirmed else 'retained' if retained else 'pending', error, confirmed,
                 self.clock(), provider, instance_id))

    def request_cancel(self, job_id, grace_seconds, *, emergency=False):
        """Persist a grace deadline once; later observers cannot extend it."""
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT deadline FROM remote_runs WHERE job_id=?", (job_id,)).fetchone()
            cleanup_after = min(run[0], now + max(0, grace_seconds)) if run else now
            if emergency:
                cleanup_after = now
            db.execute("""INSERT INTO remote_cancellations VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET cleanup_after=MIN(cleanup_after, excluded.cleanup_after),
                emergency=MAX(emergency, excluded.emergency)""", (job_id, now, cleanup_after, emergency))
            return db.execute("SELECT cleanup_after FROM remote_cancellations WHERE job_id=?", (job_id,)).fetchone()[0]

    def alert(self, key, message):
        with self.connect() as db:
            if message is None:
                db.execute("DELETE FROM remote_cleanup_alerts WHERE alert_key=?", (key,))
            else:
                db.execute("INSERT OR REPLACE INTO remote_cleanup_alerts VALUES (?, ?, ?)",
                           (key, message, self.clock()))

    def alerts(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM remote_cleanup_alerts")]

    def fail_run(self, job_id, reason):
        with self.connect() as db:
            db.execute("UPDATE remote_runs SET state='failed', error=? WHERE job_id=? AND state='running'",
                       (reason, job_id))

    def expire_runs(self):
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT job_id FROM remote_runs WHERE state='running' AND (lease_until<=? OR deadline<=?)",
                              (now, now)).fetchall()
            ids = [r[0] for r in rows]
            for job_id in ids:
                db.execute("UPDATE remote_runs SET state='failed', error=? WHERE job_id=?",
                           ('Remote controller lease or total job deadline expired; resume unsupported', job_id))
            return ids


def terminate_confirmed(lifecycle, provider, instance_id):
    """An accepted DELETE is not proof that billing stopped."""
    try:
        provider.terminate(instance_id)
        confirmed = provider.get_instance(instance_id).status == 'terminated'
        error = None if confirmed else 'Provider has not confirmed termination; resource may still be billing'
    except Exception as exc:
        confirmed, error = False, f'{type(exc).__name__}: termination confirmation failed'
    if not confirmed:
        # Billing-critical: keep this loud even though the record retries it.
        logger.error("FAILED TO TERMINATE %s instance %s (%s) — it may still be billing; "
                     "terminate it manually in the provider console if cleanup does not confirm it",
                     provider.name, instance_id, error)
    lifecycle.cleanup_result(provider.name, instance_id, confirmed=confirmed, error=error)
    return any(r["instance_id"] == instance_id and r["provider"] == provider.name
               and r["cleanup_status"] == "terminated" for r in lifecycle.resources())
