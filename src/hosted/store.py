"""Single-control-plane SQLite store with explicit ownership at every public boundary."""
from contextlib import contextmanager
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid
from fastapi import HTTPException
from .auth import hash_password, verify_password, token_hash

SCHEMA_VERSION = 1
# Run an equivalent scrypt operation for missing accounts, avoiding cheap enumeration.
_DUMMY_HASH = hash_password('invalid-account-password')


def new_id() -> str:
    return uuid.uuid4().hex


def _decode(row):
    result = dict(row)
    for name in ('config', 'result', 'payload'):
        if name in result:
            result[name] = json.loads(result[name])
    return result


class HostedStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.transaction() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError('Hosted database is newer than this application; refuse downgrade')
            if version == 0:
                statements = [
                    'CREATE TABLE accounts (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at REAL NOT NULL)',
                    'CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id), expires_at REAL NOT NULL)',
                    'CREATE TABLE login_attempts (email TEXT PRIMARY KEY, failures INTEGER NOT NULL, window_start REAL NOT NULL)',
                    'CREATE TABLE projects (id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id), name TEXT NOT NULL, goal TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)',
                    "CREATE TABLE jobs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), account_id TEXT NOT NULL REFERENCES accounts(id), config TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft', result TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, updated_at REAL NOT NULL)",
                    'CREATE TABLE events (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), job_id TEXT REFERENCES jobs(id), kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL)',
                    'CREATE TABLE datasets (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), name TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at REAL NOT NULL)',
                    'CREATE TABLE artifacts (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), job_id TEXT NOT NULL REFERENCES jobs(id), name TEXT NOT NULL, object_key TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at REAL NOT NULL)',
                    'CREATE INDEX projects_owner ON projects(account_id)',
                    'CREATE INDEX jobs_project ON jobs(project_id)',
                    'CREATE INDEX events_project ON events(project_id,created_at)',
                ]
                for statement in statements:
                    db.execute(statement)
                db.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
        self.path.chmod(0o600)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=30000')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _project(db, account_id, project_id):
        row = db.execute('SELECT * FROM projects WHERE id=? AND account_id=?', (project_id, account_id)).fetchone()
        if not row:
            raise HTTPException(404, 'Project not found')
        return dict(row)

    @classmethod
    def _job(cls, db, account_id, project_id, job_id):
        cls._project(db, account_id, project_id)
        row = db.execute('SELECT * FROM jobs WHERE id=? AND project_id=? AND account_id=?', (job_id, project_id, account_id)).fetchone()
        if not row:
            raise HTTPException(404, 'Job not found')
        return _decode(row)

    def create_account(self, email, password):
        email = email.strip().lower()
        if len(email) > 254 or '@' not in email or any(c.isspace() for c in email):
            raise ValueError('Valid email required')
        encoded = hash_password(password)
        account = {'id': new_id(), 'email': email, 'created_at': time.time()}
        with self.transaction() as db:
            db.execute('INSERT INTO accounts VALUES (?,?,?,?)', (account['id'], email, encoded, account['created_at']))
        return account

    def login(self, email, password, ttl):
        email = email.strip().lower()
        now = time.time()
        # Reserve attempts atomically before the expensive KDF. Hashing the email
        # bounds key length and separates caller-controlled values from global keys.
        bucket = 'email:' + token_hash(email)
        with self.transaction() as db:
            db.execute('DELETE FROM login_attempts WHERE window_start<=?', (now - 900,))
            global_attempt = db.execute("SELECT * FROM login_attempts WHERE email='global'").fetchone()
            recent_global = global_attempt and now - global_attempt['window_start'] < 60
            if recent_global and global_attempt['failures'] >= 240:
                raise HTTPException(429, 'Login capacity reached; try again shortly')
            attempt = db.execute('SELECT * FROM login_attempts WHERE email=?', (bucket,)).fetchone()
            if attempt and attempt['failures'] >= 10:
                raise HTTPException(429, 'Too many login attempts; try again in 15 minutes')
            if not attempt and db.execute("SELECT COUNT(*) FROM login_attempts WHERE email != 'global'").fetchone()[0] >= 4096:
                raise HTTPException(429, 'Login capacity reached; try again shortly')
            db.execute('INSERT OR REPLACE INTO login_attempts VALUES (?,?,?)',
                       ('global', global_attempt['failures'] + 1 if recent_global else 1,
                        global_attempt['window_start'] if recent_global else now))
            db.execute('INSERT OR REPLACE INTO login_attempts VALUES (?,?,?)',
                       (bucket, attempt['failures'] + 1 if attempt else 1,
                        attempt['window_start'] if attempt else now))
            row = db.execute('SELECT * FROM accounts WHERE email=?', (email,)).fetchone()

        # No database transaction or writer lock remains held while scrypt runs.
        valid = verify_password(password, row['password_hash'] if row else _DUMMY_HASH)
        if not valid or row is None:
            raise HTTPException(401, 'Invalid email or password')
        with self.transaction() as db:
            # Recheck identity/password state after verification, so an operator
            # password change or account deletion cannot race session issuance.
            current = db.execute('SELECT id,password_hash FROM accounts WHERE id=?', (row['id'],)).fetchone()
            if current is None or current['password_hash'] != row['password_hash']:
                raise HTTPException(401, 'Invalid email or password')
            db.execute('DELETE FROM login_attempts WHERE email=?', (bucket,))
            db.execute('DELETE FROM sessions WHERE expires_at<=?', (time.time(),))
            token = secrets.token_urlsafe(32)
            expires = time.time() + ttl
            db.execute('INSERT INTO sessions VALUES (?,?,?)', (token_hash(token), row['id'], expires))
        return {'token': token, 'token_type': 'bearer', 'expires_at': expires,
                'account': {'id': row['id'], 'email': row['email']}}

    def authenticate(self, token):
        with self.transaction() as db:
            row = db.execute('SELECT a.id,a.email FROM sessions s JOIN accounts a ON a.id=s.account_id WHERE s.token_hash=? AND s.expires_at>?', (token_hash(token), time.time())).fetchone()
            return dict(row) if row else None

    def logout(self, token):
        with self.transaction() as db:
            db.execute('DELETE FROM sessions WHERE token_hash=?', (token_hash(token),))

    def require_project(self, account_id, project_id):
        with self.transaction() as db:
            return self._project(db, account_id, project_id)

    def require_job(self, account_id, project_id, job_id):
        with self.transaction() as db:
            return self._job(db, account_id, project_id, job_id)

    def create_project(self, account_id, name, goal=''):
        now = time.time()
        project_id = new_id()
        with self.transaction() as db:
            db.execute('INSERT INTO projects VALUES (?,?,?,?,?,?)', (project_id, account_id, name, goal, now, now))
            return self._project(db, account_id, project_id)

    def list_projects(self, account_id):
        with self.transaction() as db:
            return [dict(row) for row in db.execute('SELECT * FROM projects WHERE account_id=? ORDER BY created_at DESC', (account_id,))]

    def update_project(self, account_id, project_id, **fields):
        if set(fields) - {'name', 'goal'}:
            raise ValueError('Unsupported project field')
        with self.transaction() as db:
            current = self._project(db, account_id, project_id)
            db.execute('UPDATE projects SET name=?,goal=?,updated_at=? WHERE id=? AND account_id=?', (fields.get('name', current['name']), fields.get('goal', current['goal']), time.time(), project_id, account_id))
            return self._project(db, account_id, project_id)

    def create_job(self, account_id, project_id, config):
        job_id, now = new_id(), time.time()
        with self.transaction() as db:
            self._project(db, account_id, project_id)
            db.execute('INSERT INTO jobs(id,project_id,account_id,config,created_at,updated_at) VALUES (?,?,?,?,?,?)', (job_id, project_id, account_id, json.dumps(config, allow_nan=False), now, now))
            return self._job(db, account_id, project_id, job_id)

    def list_jobs(self, account_id, project_id):
        with self.transaction() as db:
            self._project(db, account_id, project_id)
            return [_decode(row) for row in db.execute('SELECT * FROM jobs WHERE project_id=? AND account_id=? ORDER BY created_at DESC', (project_id, account_id))]

    def update_job(self, account_id, project_id, job_id, *, status=None, result=None):
        with self.transaction() as db:
            current = self._job(db, account_id, project_id, job_id)
            db.execute('UPDATE jobs SET status=?,result=?,updated_at=? WHERE id=? AND account_id=? AND project_id=?', (status or current['status'], json.dumps(result if result is not None else current['result'], allow_nan=False), time.time(), job_id, account_id, project_id))
            return self._job(db, account_id, project_id, job_id)

    def add_event(self, account_id, project_id, kind, payload, job_id=None):
        event = {'id': new_id(), 'project_id': project_id, 'job_id': job_id, 'kind': kind, 'payload': payload, 'created_at': time.time()}
        with self.transaction() as db:
            self._project(db, account_id, project_id)
            if job_id:
                self._job(db, account_id, project_id, job_id)
            db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)', (event['id'], project_id, job_id, kind, json.dumps(payload, allow_nan=False), event['created_at']))
        return event

    def list_events(self, account_id, project_id, job_id=None):
        with self.transaction() as db:
            self._project(db, account_id, project_id)
            if job_id:
                self._job(db, account_id, project_id, job_id)
            return [_decode(row) for row in db.execute('SELECT * FROM events WHERE project_id=? AND (? IS NULL OR job_id=?) ORDER BY created_at,id', (project_id, job_id, job_id))]

    def register_dataset(self, account_id, project_id, dataset_id, name, sha256, size_bytes):
        with self.transaction() as db:
            self._project(db, account_id, project_id)
            db.execute('INSERT INTO datasets VALUES (?,?,?,?,?,?)', (dataset_id, project_id, name, sha256, size_bytes, time.time()))
        return self.require_dataset(account_id, project_id, dataset_id)

    def require_dataset(self, account_id, project_id, dataset_id):
        with self.transaction() as db:
            self._project(db, account_id, project_id)
            row = db.execute('SELECT * FROM datasets WHERE id=? AND project_id=?', (dataset_id, project_id)).fetchone()
            if not row:
                raise HTTPException(404, 'Dataset not found')
            return dict(row)

    def list_datasets(self, account_id, project_id):
        with self.transaction() as db:
            self._project(db, account_id, project_id)
            return [dict(row) for row in db.execute('SELECT * FROM datasets WHERE project_id=? ORDER BY created_at DESC', (project_id,))]

    def list_artifacts(self, account_id, project_id, job_id):
        with self.transaction() as db:
            self._job(db, account_id, project_id, job_id)
            return [dict(row) for row in db.execute('SELECT id,project_id,job_id,name,sha256,size_bytes,created_at FROM artifacts WHERE project_id=? AND job_id=?', (project_id, job_id))]

    def require_artifact(self, account_id, project_id, job_id, artifact_id):
        with self.transaction() as db:
            self._job(db, account_id, project_id, job_id)
            row = db.execute('SELECT * FROM artifacts WHERE id=? AND project_id=? AND job_id=?', (artifact_id, project_id, job_id)).fetchone()
            if not row:
                raise HTTPException(404, 'Artifact not found')
            return dict(row)

    def backup(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise ValueError('Backup destination already exists')
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        source = sqlite3.connect(str(self.path))
        target = sqlite3.connect(str(destination))
        try:
            source.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise RuntimeError('Backup integrity check failed')
        finally:
            source.close()
            target.close()
        destination.chmod(0o600)
