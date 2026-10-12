"""Job-scoped artifact capabilities for hosted workers; never shares AWS keys."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

router = APIRouter()


class ArtifactGrants:
    def __init__(self, db_path, *, clock=time.time):
        self.db_path, self.clock = str(db_path), clock
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS hosted_artifact_grants (
                    token_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL,
                    project_id TEXT NOT NULL, job_id TEXT NOT NULL, prefix TEXT NOT NULL,
                    expires_at REAL NOT NULL, max_bytes INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS hosted_artifact_allocations (
                    token_hash TEXT NOT NULL, key TEXT NOT NULL, size INTEGER NOT NULL,
                    PRIMARY KEY(token_hash, key)
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def issue(self, *, account_id, project_id, job_id, prefix, expires_at, max_bytes):
        if expires_at <= self.clock() or max_bytes <= 0:
            raise ValueError("Artifact access requires a future expiry and byte limit")
        if not _safe_key(prefix):
            raise ValueError("Invalid artifact prefix")
        token = secrets.token_urlsafe(48)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as db:
            db.execute("INSERT INTO hosted_artifact_grants VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                       (token_hash, account_id, project_id, job_id, prefix.rstrip('/') + '/', expires_at, max_bytes))
        return token

    def authorize(self, token, key, operation, size=None):
        if not _safe_key(key):
            raise HTTPException(403, "Artifact access denied")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            grant = db.execute("SELECT * FROM hosted_artifact_grants WHERE token_hash=?", (token_hash,)).fetchone()
            if not grant or grant['revoked'] or grant['expires_at'] <= self.clock() or not key.startswith(grant['prefix']):
                raise HTTPException(403, "Artifact access denied")
            if operation == 'put':
                if type(size) is not int or size < 0 or size > grant['max_bytes']:
                    raise HTTPException(413, "Artifact storage limit exceeded")
                previous = db.execute("SELECT size FROM hosted_artifact_allocations WHERE token_hash=? AND key=?", (token_hash, key)).fetchone()
                used = db.execute("SELECT COALESCE(SUM(size), 0) FROM hosted_artifact_allocations WHERE token_hash=?", (token_hash,)).fetchone()[0]
                allocation = max(size, previous[0] if previous else 0)
                if used - (previous[0] if previous else 0) + allocation > grant['max_bytes']:
                    raise HTTPException(413, "Artifact storage limit exceeded")
                db.execute("INSERT INTO hosted_artifact_allocations VALUES (?, ?, ?) ON CONFLICT(token_hash,key) DO UPDATE SET size=excluded.size", (token_hash, key, allocation))
            elif operation != 'get':
                raise HTTPException(403, "Artifact access denied")
            return min(300, max(1, int(grant['expires_at'] - self.clock())))

    def revoke(self, job_id):
        with self.connect() as db:
            db.execute("UPDATE hosted_artifact_grants SET revoked=1 WHERE job_id=?", (job_id,))


def _safe_key(key):
    return (isinstance(key, str) and bool(key) and len(key) <= 2048
            and not key.startswith('/') and not any(c in key for c in ('\\', '%', '\x00', '?', '#'))
            and all(p not in ('', '.', '..') for p in key.split('/'))
            and str(PurePosixPath(key)) == key)


class CapabilityRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: str = Field(min_length=1, max_length=2048)
    operation: Literal['get', 'put']
    size: int | None = Field(default=None, ge=0, strict=True)


@router.post('/api/hosted/worker-artifacts/capability')
def artifact_capability(body: CapabilityRequest, request: Request):
    header = request.headers.get('authorization', '')
    if not header.startswith('Bearer ') or len(header) > 512:
        raise HTTPException(403, 'Artifact access denied')
    grants = getattr(request.app.state, 'artifact_grants', None)
    signer = getattr(request.app.state, 'artifact_signer', None)
    if grants is None or signer is None:
        # Mounted always; configured only by the paid runtime.
        raise HTTPException(503, 'Artifact broker is not configured')
    expires = grants.authorize(header[7:], body.key, body.operation, body.size)
    try:
        kwargs = {'ContentLength': body.size} if body.operation == 'put' else {}
        url = signer.sign(
            body.key, expiration=expires,
            client_method='put_object' if body.operation == 'put' else 'get_object', **kwargs)
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Object store requires HTTPS')
    except Exception:
        raise HTTPException(503, 'Artifact storage is unavailable') from None
    return {'url': url, 'expires_in': expires}


class ScopedWorkerArtifactStore:
    """Control-plane operations delegate; workers reconstruct a scoped transport."""
    def __init__(self, store, broker_url, token):
        self.store, self.broker_url, self.token = store, broker_url, token

    def push_dir(self, local_dir, name):
        return self.store.push_dir(local_dir, name)

    def pull_dir(self, name, local_dir):
        return self.store.pull_dir(name, local_dir)

    def exists(self, name):
        return self.store.exists(name)

    def describe(self):
        return {**self.store.describe(), 'credential_transport': 'broker', 'broker_url': self.broker_url}

    def worker_environment(self):
        return {'MERLINA_ARTIFACT_BROKER_TOKEN': self.token}


def build_scoped_store(store, *, db_path, public_url, account_id, project_id, job_id, expires_at, max_bytes):
    parsed = urlsplit(public_url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Hosted artifact access requires an HTTPS public URL')
    description = store.describe()
    if description.get('kind') != 's3':
        raise ValueError('Hosted workers require S3 artifact storage')
    grants = ArtifactGrants(db_path)
    token = grants.issue(account_id=account_id, project_id=project_id, job_id=job_id,
                         prefix=f"{description['bucket']}/{description['prefix']}",
                         expires_at=expires_at, max_bytes=max_bytes)
    return ScopedWorkerArtifactStore(store, public_url.rstrip('/') + '/api/hosted/worker-artifacts/capability', token)
