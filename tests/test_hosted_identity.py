"""Security boundaries exercised without importing local training services."""
import hashlib
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.hosted.app import create_hosted_app
from src.hosted.settings import HostedSettings
from src.hosted.store import HostedStore
from src.hosted.services import build_execution_config, resolve_uploaded_datasets, scoped_path


@pytest.fixture
def hosted(tmp_path):
    settings = HostedSettings(data_dir=tmp_path)
    app = create_hosted_app(settings)
    store = app.state.store
    accounts = [store.create_account(f'user{i}@example.com', 'secure password 123') for i in range(2)]
    clients = [TestClient(app) for _ in range(2)]
    for index, client in enumerate(clients):
        login = client.post('/api/hosted/session', json={'email': accounts[index]['email'], 'password': 'secure password 123'})
        assert login.status_code == 200
        client.headers['Authorization'] = 'Bearer ' + login.json()['token']
    return app, store, accounts, clients


def project(client):
    response = client.post('/api/hosted/projects', json={'name': 'Wishcraft', 'goal': 'Build a helper'})
    assert response.status_code == 201
    return response.json()['id']


def job(client, pid, **overrides):
    config = {'base_model': 'Qwen/Qwen3-0.6B', 'dataset': {'source': {'source_type': 'huggingface', 'repo_id': 'test/data'}}}
    config.update(overrides)
    return client.post(f'/api/hosted/projects/{pid}/jobs', json={'config': config})


def test_routes_require_identity_and_local_routes_absent(hosted):
    app, store, accounts, clients = hosted
    anonymous = TestClient(app)
    for endpoint in ['/projects', '/session', '/projects/x/jobs', '/projects/x/events']:
        assert anonymous.get('/api/hosted' + endpoint).status_code == 401
    for endpoint in ['/train', '/jobs/history', '/datasets', '/config', '/ws/x']:
        assert anonymous.get(endpoint).status_code == 404


def test_project_job_event_artifact_isolation(hosted):
    app, store, accounts, (alice, bob) = hosted
    pid = project(alice)
    bid = project(bob)
    response = job(alice, pid)
    assert response.status_code == 201
    jid = response.json()['id']
    store.add_event(accounts[0]['id'], pid, 'created', {'message': 'private'}, jid)
    with store.transaction() as db:
        db.execute('INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?)', ('artifact', pid, jid, 'adapter', 'private/key', 'abc', 1, 1))
    assert bob.get('/api/hosted/projects').json()[0]['id'] == bid
    for path in [f'/projects/{pid}', f'/projects/{pid}/jobs', f'/projects/{pid}/events',
                 f'/projects/{pid}/datasets', f'/projects/{pid}/jobs/{jid}',
                 f'/projects/{pid}/jobs/{jid}/events', f'/projects/{pid}/jobs/{jid}/artifacts',
                 f'/projects/{pid}/jobs/{jid}/artifacts/artifact/download', f'/projects/{bid}/jobs/{jid}']:
        assert bob.get('/api/hosted' + path).status_code == 404, path
    assert job(bob, pid).status_code == 404
    assert alice.get(f'/api/hosted/projects/{pid}/jobs/{jid}/artifacts').json()[0]['name'] == 'adapter'
    assert 'object_key' not in alice.get(f'/api/hosted/projects/{pid}/jobs/{jid}/artifacts').text


def test_sessions_are_hashed_revoked_expire_and_persist(hosted):
    app, store, accounts, (alice, _) = hosted
    token = alice.headers['Authorization'].split()[1]
    with store.transaction() as db:
        sessions = [dict(row) for row in db.execute('SELECT * FROM sessions')]
        assert all(row['token_hash'] != token for row in sessions)
    assert HostedStore(store.path).authenticate(token)['id'] == accounts[0]['id']
    assert alice.delete('/api/hosted/session').status_code == 204
    assert alice.get('/api/hosted/session').status_code == 401
    token = store.login(accounts[0]['email'], 'secure password 123', -1)['token']
    assert store.authenticate(token) is None


def test_login_failures_throttled_without_user_enumeration(hosted):
    app, store, accounts, _ = hosted
    client = TestClient(app)
    for _ in range(10):
        assert client.post('/api/hosted/session', json={'email': 'unknown@example.com', 'password': 'wrong'}).status_code == 401
    assert client.post('/api/hosted/session', json={'email': 'unknown@example.com', 'password': 'wrong'}).status_code == 429


@pytest.mark.parametrize('overrides', [
    {'base_model': '../secret'}, {'base_model': '/etc/passwd'}, {'base_model': 'https://internal/metadata'},
    {'output_name': '../../escape'}, {'trust_remote_code': True}, {'hf_token': 'private'},
    {'model_type': 'vlm'}, {'remote': {'worker_image': 'attacker/image'}},
    {'dataset': {'source': {'source_type': 'local_file', 'file_path': '/etc/passwd'}}},
    {'dataset': {'source': {'source_type': 'internal', 'uri': 'swl://other-tenant'}}},
    {'dataset': {'source': {'source_type': 'huggingface', 'repo_id': '../x'}}},
])
def test_untrusted_execution_config_rejected(hosted, overrides):
    *_, (alice, _) = hosted
    assert job(alice, project(alice), **overrides).status_code == 422


def test_upload_ownership_integrity_and_remote_defaults(hosted):
    app, store, accounts, (alice, bob) = hosted
    pid, bid = project(alice), project(bob)
    response = alice.post(f'/api/hosted/projects/{pid}/datasets', json={'name': '../../ignored.jsonl', 'content': '{"prompt":"Hi","chosen":"Hello"}\n'})
    assert response.status_code == 201
    dataset = response.json()
    config = {'dataset': {'source': {'source_type': 'upload', 'dataset_id': dataset['id']}}}
    assert job(bob, bid, **config).status_code == 404
    result = job(alice, pid, **config).json()
    clean = build_execution_config(store, app.state.settings, accounts[0]['id'], pid, result['id'])
    assert clean['remote']['merge_strategy'] == 'skip'
    assert clean['trust_remote_code'] is False
    assert clean['output_name'].endswith(result['id'])
    upload = resolve_uploaded_datasets(store, app.state.settings, accounts[0]['id'], pid, result['config'])
    assert upload[dataset['id']]['filename'] == 'dataset.jsonl'
    path = scoped_path(app.state.settings, accounts[0]['id'], pid, dataset['id'])
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='integrity'):
        resolve_uploaded_datasets(store, app.state.settings, accounts[0]['id'], pid, result['config'])
    with pytest.raises(ValueError):
        scoped_path(app.state.settings, '../escape', pid, dataset['id'])


def test_bad_upload_and_size_limit(hosted):
    app, store, accounts, (alice, _) = hosted
    pid = project(alice)
    for content in ['[]', '{bad', '', 'NaN', '{"x":NaN}']:
        assert alice.post(f'/api/hosted/projects/{pid}/datasets', json={'name': 'data', 'content': content}).status_code == 422
    app.state.settings.max_dataset_bytes = 4
    assert alice.post(f'/api/hosted/projects/{pid}/datasets', json={'name': 'data', 'content': '{"x":1}'}).status_code == 413


def test_cors_never_wildcard_and_cookie_not_identity(hosted):
    app, _, _, _ = hosted
    response = TestClient(app).get('/api/hosted/projects', headers={'Origin': 'https://evil.example'}, cookies={'session': 'anything'})
    assert response.status_code == 401
    assert 'access-control-allow-origin' not in response.headers
    with pytest.raises(ValueError):
        HostedSettings(cors_origins=('*',))


def test_concurrent_transactions_and_backup_restore(hosted, tmp_path):
    _, store, accounts, _ = hosted
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: store.create_project(accounts[0]['id'], f'project {i}'), range(20)))
    assert len(store.list_projects(accounts[0]['id'])) == 20
    backup = tmp_path / 'backup.db'
    store.backup(backup)
    restored = HostedStore(backup)
    assert restored.list_projects(accounts[0]['id']) == store.list_projects(accounts[0]['id'])
    with pytest.raises(ValueError):
        store.backup(backup)


def test_hosted_import_is_cpu_only(tmp_path):
    code = '''
import sys
from src.hosted.app import create_hosted_app
from src.hosted.settings import HostedSettings
app = create_hosted_app(HostedSettings(data_dir=sys.argv[1]))
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'merlina' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_same_account_cross_project_and_error_redaction(hosted):
    app, store, accounts, (alice, _) = hosted
    first, second = project(alice), project(alice)
    jid = job(alice, first).json()['id']
    assert alice.get(f'/api/hosted/projects/{second}/jobs/{jid}').status_code == 404
    response = job(alice, first, hf_token='sensitive-do-not-echo')
    assert response.status_code == 422
    assert 'sensitive-do-not-echo' not in response.text


def test_refuse_newer_schema_and_symlink_escape(tmp_path):
    path = tmp_path / 'newer.db'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=99')
    with pytest.raises(RuntimeError, match='newer'):
        HostedStore(path)
    settings = HostedSettings(data_dir=tmp_path / 'data')
    settings.data_dir.mkdir()
    (settings.data_dir / 'accounts').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='escapes'):
        scoped_path(settings, 'a' * 32, 'b' * 32, 'c' * 32)


def test_explicit_cors_only_allows_configured_origin(tmp_path):
    client = TestClient(create_hosted_app(HostedSettings(data_dir=tmp_path, cors_origins=('https://merlina.example',))))
    allowed = client.options('/api/hosted/projects', headers={'Origin': 'https://merlina.example', 'Access-Control-Request-Method': 'POST'})
    assert allowed.headers['access-control-allow-origin'] == 'https://merlina.example'
    assert 'access-control-allow-credentials' not in allowed.headers
    rejected = client.options('/api/hosted/projects', headers={'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'POST'})
    assert rejected.status_code == 400


def test_env_settings_are_explicit_and_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv('MERLINA_HOSTED_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('MERLINA_HOSTED_PAID_EXECUTION_ENABLED', 'true')
    monkeypatch.setenv('MERLINA_HOSTED_SESSION_TTL_SECONDS', '120')
    settings = HostedSettings.from_env()
    assert settings.paid_execution_enabled is True
    assert settings.session_ttl_seconds == 120
    assert settings.database_path == tmp_path / 'hosted.sqlite3'
    monkeypatch.setenv('MERLINA_HOSTED_PAID_EXECUTION_ENABLED', 'yes')
    with pytest.raises(ValueError, match='true or false'):
        HostedSettings.from_env()
    with pytest.raises(ValueError, match='positive'):
        HostedSettings(max_dataset_bytes=-1)


def test_unexpected_exception_response_does_not_leak(hosted):
    app, _, _, _ = hosted
    @app.get('/test-error')
    def error():
        raise RuntimeError('provider-secret-123')
    response = TestClient(app, raise_server_exceptions=False).get('/test-error')
    assert response.status_code == 500
    assert 'provider-secret' not in response.text


def test_password_kdf_does_not_hold_database_write_lock(hosted, monkeypatch):
    import threading
    import src.hosted.store as store_module
    _, store, accounts, _ = hosted
    entered, release = threading.Event(), threading.Event()
    def slow_verify(password, encoded):
        entered.set()
        assert release.wait(5)
        return True
    monkeypatch.setattr(store_module, 'verify_password', slow_verify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        login = pool.submit(store.login, accounts[0]['email'], 'secure password 123', 60)
        assert entered.wait(2)
        try:
            write = pool.submit(store.create_project, accounts[1]['id'], 'Concurrent billing/project write')
            assert write.result(timeout=2)['name'] == 'Concurrent billing/project write'
        finally:
            release.set()
        assert login.result()['account']['id'] == accounts[0]['id']


def test_login_throttle_bounded_pruned_and_reserved_before_kdf(hosted, monkeypatch):
    import time
    import src.hosted.store as store_module
    from fastapi import HTTPException
    _, store, accounts, _ = hosted
    calls = []
    monkeypatch.setattr(store_module, 'verify_password', lambda *args: calls.append(args) or False)
    with store.transaction() as db:
        db.execute('DELETE FROM login_attempts')
    for index in range(240):
        with pytest.raises(HTTPException) as error:
            store.login(f'random{index}@example.com', 'wrong', 60)
        assert error.value.status_code == 401
    with pytest.raises(HTTPException) as error:
        store.login('another@example.com', 'wrong', 60)
    assert error.value.status_code == 429
    assert len(calls) == 240
    with store.transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM login_attempts').fetchone()[0] == 241
        db.execute('UPDATE login_attempts SET window_start=?', (time.time() - 901,))
    with pytest.raises(HTTPException) as error:
        store.login('new@example.com', 'wrong', 60)
    assert error.value.status_code == 401
    with store.transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM login_attempts').fetchone()[0] == 2
        db.executemany('INSERT INTO login_attempts VALUES (?,?,?)',
                       [(f'old-{i}', 1, time.time()) for i in range(4095)])
    with pytest.raises(HTTPException) as error:
        store.login('over-cap@example.com', 'wrong', 60)
    assert error.value.status_code == 429
    assert len(calls) == 241
