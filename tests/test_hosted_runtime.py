"""The hosted app installs execution only from explicit operator configuration."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.hosted.app import create_hosted_app
from src.hosted.settings import HostedSettings
from src.hosted.runtime import configure_runtime
from src.hosted.billing import BillingUnavailable


def test_default_hosted_startup_never_configures_paid_provider(tmp_path, monkeypatch):
    from src.hosted import remote_runtime
    def unexpected(*args, **kwargs):
        raise AssertionError('Disabled hosted startup must not initialize paid execution')
    monkeypatch.setattr(remote_runtime, 'install_runtime', unexpected)
    with TestClient(create_hosted_app(HostedSettings(data_dir=tmp_path))) as client:
        assert client.get('/health').json()['mode'] == 'hosted'
        assert client.post('/train', json={}).status_code == 404
        assert client.get('/model-files/private').status_code == 404


def test_paid_runtime_fails_closed_before_serving_without_https_origin(tmp_path):
    settings = HostedSettings(data_dir=tmp_path, paid_execution_enabled=True,
        billing_policy_version='test', billing_failure_policy='charge_metered',
        billing_overage_policy='operator_absorbs')
    with pytest.raises(BillingUnavailable, match='HTTPS origin'):
        with TestClient(create_hosted_app(settings)):
            pass


def test_runtime_builds_job_specific_broker_store(tmp_path, monkeypatch):
    from config import settings as server
    from src.hosted import remote_runtime
    monkeypatch.setattr(server, 'merlina_artifact_store', 's3')
    monkeypatch.setattr(server, 'merlina_artifact_s3_bucket', 'test-bucket')
    monkeypatch.setattr(server, 'merlina_artifact_s3_prefix', 'runs')
    settings = HostedSettings(data_dir=tmp_path, public_url='https://merlina.example',
        paid_execution_enabled=True, billing_policy_version='test',
        billing_failure_policy='charge_metered', billing_overage_policy='operator_absorbs')
    app = create_hosted_app(settings)
    captured = {}
    def install(app, factory):
        captured['factory'] = factory
        return SimpleNamespace(tick=lambda: None, billing=SimpleNamespace(claim=lambda job_id: True))
    monkeypatch.setattr(remote_runtime, 'install_runtime', install)
    from src.hosted.billing import BillingService
    monkeypatch.setattr(BillingService, 'run', lambda self, job_id: {'quote': {'stages': [
        {'boot_seconds': 60, 'runtime_seconds': 3600, 'shutdown_seconds': 60}]}})
    configure_runtime(app)
    job = {'account_id': 'a'*32, 'project_id': 'b'*32, 'id': 'c'*32, 'config': {}}
    store = captured['factory'](job)
    assert store.describe()['prefix'] == f"runs/{job['account_id']}/{job['project_id']}/{job['id']}"
    assert store.describe()['credential_transport'] == 'broker'
    env = store.worker_environment()
    assert list(env) == ['MERLINA_ARTIFACT_BROKER_TOKEN']
    grants = app.state.artifact_grants
    prefix = 'test-bucket/' + store.describe()['prefix']
    assert grants.authorize(env['MERLINA_ARTIFACT_BROKER_TOKEN'], prefix+'/adapter/manifest.json', 'get') > 0
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        grants.authorize(env['MERLINA_ARTIFACT_BROKER_TOKEN'], 'test-bucket/runs/other/adapter/manifest.json', 'get')
