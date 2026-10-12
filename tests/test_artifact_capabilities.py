from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.hosted.artifact_broker import ArtifactGrants, router, build_scoped_store
from src.remote.artifact_capabilities import CapabilityS3Client


@pytest.fixture
def grants(tmp_path):
    return ArtifactGrants(tmp_path / 'hosted.db', clock=lambda: 1000)


def issue(grants, **kw):
    return grants.issue(account_id='alice', project_id='p1', job_id='j1',
                        prefix='bucket/jobs/j1', expires_at=2000, max_bytes=100, **kw)


def test_scope_expiry_revocation_and_traversal(grants):
    token = issue(grants)
    assert grants.authorize(token, 'bucket/jobs/j1/adapter/manifest.json', 'get') == 300
    for key in ('bucket/jobs/j2/a', 'bucket/jobs/j1-other/a', 'bucket/jobs/j1/../j2/a',
                'bucket/jobs/j1/%2e%2e/a', 'bucket/jobs/j1//a', 'bucket/jobs/j1/a\\b'):
        with pytest.raises(HTTPException):
            grants.authorize(token, key, 'get')
    with pytest.raises(HTTPException):
        grants.authorize('wrong', 'bucket/jobs/j1/a', 'get')
    grants.clock = lambda: 2001
    with pytest.raises(HTTPException):
        grants.authorize(token, 'bucket/jobs/j1/a', 'get')
    grants.clock = lambda: 1000
    grants.revoke('j1')
    with pytest.raises(HTTPException):
        grants.authorize(token, 'bucket/jobs/j1/a', 'get')


def test_concurrent_capability_reservations_cannot_exceed_storage_budget(grants):
    token = issue(grants)
    def allocate(key):
        try:
            grants.authorize(token, f'bucket/jobs/j1/{key}', 'put', 60)
            return True
        except HTTPException:
            return False
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(allocate, ['a', 'b'])) == 1
    # A repeated upload cannot free a prior, still-valid larger signed PUT.
    with grants.connect() as db:
        key = db.execute('SELECT key FROM hosted_artifact_allocations').fetchone()[0]
        assert db.execute('SELECT token_hash FROM hosted_artifact_grants').fetchone()[0] != token
    grants.authorize(token, key, 'put', 1)
    with pytest.raises(HTTPException):
        grants.authorize(token, 'bucket/jobs/j1/c', 'put', 41)


def test_broker_signs_only_scoped_and_length_bound_requests(grants):
    app = FastAPI()
    app.include_router(router)
    app.state.artifact_grants = grants
    signer = app.state.artifact_signer = Mock()
    signer.sign.return_value = 'https://s3.example/bucket/obj?signed=private'
    token = issue(grants)
    client = TestClient(app)
    path = '/api/hosted/worker-artifacts/capability'
    body = {'key': 'bucket/jobs/j1/adapter/manifest.json', 'operation': 'put', 'size': 40}
    assert client.post(path, json=body).status_code == 403
    response = client.post(path, json=body, headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 200
    signer.sign.assert_called_once_with(body['key'], expiration=300, client_method='put_object', ContentLength=40)
    body['key'] = 'bucket/jobs/j2/a'
    assert client.post(path, json=body, headers={'Authorization': f'Bearer {token}'}).status_code == 403
    assert signer.sign.call_count == 1


def test_worker_never_sends_broker_token_to_object_store(tmp_path):
    broker, data = Mock(), Mock()
    broker.post.return_value = Mock(status_code=200, json=lambda: {'url': 'https://s3.example/key?signature=secret'})
    data.put.return_value = Mock(status_code=200)
    fs = CapabilityS3Client('https://merlina.example/api/hosted/worker-artifacts/capability', 'worker-secret', session=broker, data_session=data)
    fs.pipe_file('bucket/jobs/j1/a', b'abc')
    assert broker.post.call_args.kwargs['headers']['Authorization'] == 'Bearer worker-secret'
    assert data.put.call_args.kwargs['headers'] == {'Content-Length': '3'}
    assert not data.put.call_args.kwargs['allow_redirects']
    broker.post.return_value.status_code = 403
    with pytest.raises(OSError, match='Could not obtain artifact access') as error:
        fs.pipe_file('bucket/jobs/j2/a', b'abc')
    assert 'secret' not in str(error.value)


def test_signed_url_errors_are_redacted_and_missing_is_distinct():
    broker, data = Mock(), Mock()
    broker.post.return_value = Mock(status_code=200, json=lambda: {'url': 'https://s3.example/key?secret=abc'})
    fs = CapabilityS3Client('https://merlina.example/broker', 'token', session=broker, data_session=data)
    data.get.return_value = Mock(status_code=404)
    with pytest.raises(FileNotFoundError):
        fs.cat_file('bucket/jobs/j1/manifest.json')
    data.get.side_effect = RuntimeError('secret=abc')
    with pytest.raises(OSError) as exc:
        fs.cat_file('bucket/jobs/j1/manifest.json')
    assert 'abc' not in str(exc.value)


def test_scoped_store_description_excludes_token(tmp_path):
    underlying = Mock()
    underlying.describe.return_value = {'kind': 's3', 'bucket': 'bucket', 'prefix': 'jobs/j1'}
    store = build_scoped_store(underlying, db_path=tmp_path/'db', public_url='https://merlina.example',
                              account_id='a', project_id='p', job_id='j', expires_at=10**12, max_bytes=100)
    assert store.describe()['credential_transport'] == 'broker'
    assert store.worker_environment()['MERLINA_ARTIFACT_BROKER_TOKEN'] not in str(store.describe())
    store.push_dir(tmp_path, 'adapter')
    underlying.push_dir.assert_called_once_with(tmp_path, 'adapter')
    with pytest.raises(ValueError):
        build_scoped_store(underlying, db_path=tmp_path/'db', public_url='http://merlina.example',
                           account_id='a', project_id='p', job_id='j', expires_at=10**12, max_bytes=100)
