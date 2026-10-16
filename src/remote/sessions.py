"""
Remote inference sessions — persistence and the reconciler bridge.

A session is a rented instance serving one model over an OpenAI-compatible
API until it is stopped, hits a deadline/budget, or goes idle. Sessions live
in their own table (``remote_inference_sessions``) in the same SQLite DB as
jobs, so they never show up as training jobs, while the durable ownership
records (claims, attempts, resources) are the shared ``remote_*`` tables in
:mod:`src.remote.lifecycle` — which is what lets the existing reconciler
terminate an orphaned serve pod exactly like an orphaned training pod.

The reconciler looks records up through a JobManager-shaped interface
(``get_job`` / ``update_job``); :func:`with_inference_sessions` wraps a
JobManager so session ids resolve to session rows.

The endpoint's API key is deliberately NOT persisted: like the worker
token, it lives in this process's memory (and the pod's environment) only.
A session never outlives the process that drives it (after a restart the
reconciler terminates its pod), so a key on disk would only be a leak.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SESSION_ID_PREFIX = "inf_"

# Lifecycle of a session row.
ACTIVE_SESSION_STATUSES = ("queued", "provisioning", "starting", "ready", "stopping")
TERMINAL_SESSION_STATUSES = ("stopped", "expired", "failed")

_COLUMNS = {
    "status": "TEXT", "created_at": "REAL", "updated_at": "REAL", "request": "TEXT",
    "plan": "TEXT", "base_url": "TEXT", "served_model_name": "TEXT", "error": "TEXT",
    "stop_requested": "INTEGER", "stop_reason": "TEXT", "metrics": "TEXT",
    "deadline_at": "REAL", "ready_at": "REAL", "ended_at": "REAL",
}
_JSON_COLUMNS = ("request", "plan", "metrics")
# The reconciler writes ``status``/``error``/``metrics``; the orchestrator
# everything else. Unknown names are a programming error.
_UPDATABLE = set(_COLUMNS) - {"created_at"}


def new_session_id() -> str:
    return f"{SESSION_ID_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def is_session_id(value: Optional[str]) -> bool:
    return bool(value) and str(value).startswith(SESSION_ID_PREFIX)


@dataclass
class InferenceSession:
    session_id: str
    status: str
    created_at: float
    updated_at: float
    request: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    base_url: Optional[str] = None
    served_model_name: Optional[str] = None
    error: Optional[str] = None
    stop_requested: bool = False
    stop_reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    deadline_at: Optional[float] = None
    ready_at: Optional[float] = None
    ended_at: Optional[float] = None

    @property
    def job_id(self) -> str:      # JobRecord-compatible for shared helpers
        return self.session_id

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_SESSION_STATUSES

    def to_public_dict(self) -> Dict[str, Any]:
        """API view. Never includes the API key or any token."""
        from dataclasses import asdict
        data = asdict(self)
        data["is_terminal"] = self.is_terminal
        return data


class InferenceSessionStore:
    """CRUD for session rows. Each call opens its own short connection."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS remote_inference_sessions (
                session_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL, request TEXT NOT NULL)""")
            existing = {row[1] for row in db.execute("PRAGMA table_info(remote_inference_sessions)")}
            for column, sql_type in _COLUMNS.items():
                if column not in existing:
                    db.execute(f"ALTER TABLE remote_inference_sessions ADD COLUMN {column} {sql_type}")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.db_path), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, session_id: str, request: Dict[str, Any]) -> InferenceSession:
        now = time.time()
        with self._connect() as db:
            db.execute("""INSERT INTO remote_inference_sessions
                (session_id, status, created_at, updated_at, request, stop_requested, metrics)
                VALUES (?, 'queued', ?, ?, ?, 0, '{}')""",
                       (session_id, now, now, json.dumps(scrub_request(request))))
        return self.get(session_id)

    def get(self, session_id: str) -> Optional[InferenceSession]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM remote_inference_sessions WHERE session_id=?",
                             (session_id,)).fetchone()
        return _row_to_session(row) if row else None

    def list(self, limit: int = 100) -> List[InferenceSession]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM remote_inference_sessions ORDER BY created_at DESC LIMIT ?",
                              (limit,)).fetchall()
        return [_row_to_session(r) for r in rows]

    def count_active(self) -> int:
        marks = ",".join("?" * len(ACTIVE_SESSION_STATUSES))
        with self._connect() as db:
            return db.execute(f"SELECT COUNT(*) FROM remote_inference_sessions WHERE status IN ({marks})",
                              ACTIVE_SESSION_STATUSES).fetchone()[0]

    def update(self, session_id: str, **fields: Any) -> bool:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise TypeError(f"Unknown session fields: {sorted(unknown)}")
        if not fields:
            return self.get(session_id) is not None
        values = []
        for key, value in fields.items():
            if key in _JSON_COLUMNS:
                value = json.dumps(value or {})
            elif key == "stop_requested":
                value = int(bool(value))
            values.append(value)
        sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
        with self._connect() as db:
            result = db.execute(f"UPDATE remote_inference_sessions SET {sets} WHERE session_id=?",
                                (*values, time.time(), session_id))
        return bool(result.rowcount)

    def request_stop(self, session_id: str) -> Optional[InferenceSession]:
        with self._connect() as db:
            db.execute("""UPDATE remote_inference_sessions SET stop_requested=1,
                status=CASE WHEN status IN ('queued','provisioning','starting','ready')
                       THEN 'stopping' ELSE status END, updated_at=?
                WHERE session_id=?""", (time.time(), session_id))
        return self.get(session_id)

    def fail_orphans(self, reason: str, *, created_before: Optional[float] = None) -> List[str]:
        """Mark non-terminal sessions failed (no process drives them any more).

        Called at API startup: session threads never survive a restart. The
        reconciler terminates their pods once the dead controller's lease lapses.
        ``created_before`` (this process's start time) keeps a session started
        while the startup scan runs from being swept up with the orphans.
        """
        marks = ",".join("?" * len(ACTIVE_SESSION_STATUSES))
        cutoff = time.time() if created_before is None else created_before
        with self._connect() as db:
            ids = [r[0] for r in db.execute(
                f"SELECT session_id FROM remote_inference_sessions WHERE status IN ({marks}) "
                "AND created_at < ?", (*ACTIVE_SESSION_STATUSES, cutoff))]
            now = time.time()
            for sid in ids:
                db.execute("""UPDATE remote_inference_sessions SET status='failed', error=?,
                    stop_reason='orphaned', ended_at=?, updated_at=? WHERE session_id=?""",
                           (reason, now, now, sid))
        for sid in ids:
            forget_api_key(sid)
        return ids

    # ---- JobManager-shaped surface (orchestrator + reconciler) -------------

    def get_job(self, session_id: str) -> Optional[InferenceSession]:
        return self.get(session_id)

    def update_job(self, session_id: str, **fields: Any) -> bool:
        return self.update(session_id, **{k: v for k, v in fields.items() if k in _UPDATABLE})


def _row_to_session(row: sqlite3.Row) -> InferenceSession:
    data = dict(row)
    for key in _JSON_COLUMNS:
        try:
            data[key] = json.loads(data.get(key) or "{}") or {}
        except (TypeError, ValueError):
            data[key] = {}
    data["stop_requested"] = bool(data.get("stop_requested"))
    return InferenceSession(**{k: data.get(k) for k in InferenceSession.__dataclass_fields__})


_SECRET_KEYS = ("token", "secret", "password", "api_key", "apikey")


def scrub_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Drop credential fields before a request is persisted or echoed."""
    if isinstance(request, dict):
        return {k: scrub_request(v) for k, v in request.items()
                if not any(s in str(k).lower() for s in _SECRET_KEYS)}
    if isinstance(request, list):
        return [scrub_request(v) for v in request]
    return request


# ---------------------------------------------------------------------------
# In-memory API keys (never persisted, never logged)
# ---------------------------------------------------------------------------

_api_keys: Dict[str, str] = {}
_api_keys_lock = threading.Lock()


def remember_api_key(session_id: str, api_key: str) -> None:
    with _api_keys_lock:
        _api_keys[session_id] = api_key


def api_key_for(session_id: str) -> Optional[str]:
    with _api_keys_lock:
        return _api_keys.get(session_id)


def forget_api_key(session_id: str) -> None:
    with _api_keys_lock:
        _api_keys.pop(session_id, None)


# ---------------------------------------------------------------------------
# Reconciler bridge
# ---------------------------------------------------------------------------

class RemoteRecords:
    """
    JobManager facade that also resolves inference session ids.

    The reconciler identifies the owner of every remote resource with
    ``get_job(owner_id)`` and treats a missing record as "deleted → terminate".
    Without this bridge a live serve pod would look deleted and be killed
    on the next pass.
    """

    def __init__(self, job_manager):
        self._jobs = job_manager
        self._sessions: Optional[InferenceSessionStore] = None

    @property
    def sessions(self) -> InferenceSessionStore:
        if self._sessions is None:
            self._sessions = InferenceSessionStore(self._jobs.db_path)
        return self._sessions

    def get_job(self, record_id):
        if is_session_id(record_id):
            return self.sessions.get(record_id)
        return self._jobs.get_job(record_id)

    def update_job(self, record_id, **fields):
        if is_session_id(record_id):
            return self.sessions.update_job(record_id, **fields)
        return self._jobs.update_job(record_id, **fields)

    def __getattr__(self, name):
        return getattr(self._jobs, name)


def with_inference_sessions(job_manager):
    """Wrap ``job_manager`` so session ids resolve; no-op when not applicable."""
    if job_manager is None or isinstance(job_manager, RemoteRecords):
        return job_manager
    if getattr(job_manager, "db_path", None) is None:
        return job_manager
    return RemoteRecords(job_manager)
