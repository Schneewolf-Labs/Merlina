"""Restart and race tests for the durable paid-resource ownership boundary."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.remote.lifecycle import RemoteAlreadyClaimed, RemoteLeaseLost, RemoteLifecycle, terminate_confirmed
from src.remote.spec import RemoteInstance


def test_concurrent_dispatch_claims_exactly_once(tmp_path):
    path = tmp_path / 'jobs.db'
    RemoteLifecycle(path)
    def claim(_):
        try:
            return RemoteLifecycle(path).claim('job', 1)
        except RemoteAlreadyClaimed:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(bool(x) for x in pool.map(claim, range(16))) == 1


def test_restart_cannot_reprovision_ambiguous_intent(tmp_path):
    path = tmp_path / 'jobs.db'
    first = RemoteLifecycle(path)
    owner = first.claim('job', 1)
    label = first.intent('job', owner, 'train', 'runpod')
    restarted = RemoteLifecycle(path)
    assert restarted.namespace == first.namespace
    assert restarted.attempts()[0]['label'] == label
    with pytest.raises(RemoteAlreadyClaimed):
        restarted.claim('job', 1)
    with pytest.raises(RemoteAlreadyClaimed):
        restarted.intent('job', owner, 'train', 'runpod')


def test_expired_lease_cannot_be_resurrected(tmp_path):
    now = [1000.0]
    lifecycle = RemoteLifecycle(tmp_path / 'jobs.db', lease_seconds=10, clock=lambda: now[0])
    owner = lifecycle.claim('job', 1)
    now[0] += 11
    with pytest.raises(RemoteLeaseLost):
        lifecycle.heartbeat('job', owner)
    assert lifecycle.expire_runs() == ['job']
    with pytest.raises(RemoteLeaseLost):
        lifecycle.intent('job', owner, 'train', 'runpod')
    assert lifecycle.run('job')['state'] == 'failed'


def test_total_deadline_is_shared_between_stages(tmp_path):
    now = [1000.0]
    lifecycle = RemoteLifecycle(tmp_path / 'jobs.db', clock=lambda: now[0])
    owner = lifecycle.claim('job', 0.01)
    lifecycle.intent('job', owner, 'train', 'runpod')
    now[0] += 37
    with pytest.raises(RemoteLeaseLost):
        lifecycle.intent('job', owner, 'merge', 'runpod')


def test_cleanup_requires_positive_confirmation(tmp_path):
    lifecycle = RemoteLifecycle(tmp_path / 'jobs.db')
    owner = lifecycle.claim('job', 1)
    label = lifecycle.intent('job', owner, 'train', 'runpod')
    instance = RemoteInstance('pod', 'runpod', cost_per_hr=2)
    lifecycle.record_instance('job', 'train', label, instance)
    class Provider:
        name = 'runpod'
        state = 'unknown'
        def terminate(self, instance_id):
            pass
        def get_instance(self, instance_id):
            return RemoteInstance(instance_id, self.name, status=self.state)
    provider = Provider()
    assert not terminate_confirmed(lifecycle, provider, 'pod')
    record = lifecycle.resources('job')[0]
    assert record['terminated_at'] is None
    assert record['cleanup_status'] == 'pending'
    assert record['last_error']
    provider.state = 'terminated'
    assert terminate_confirmed(lifecycle, provider, 'pod')
    assert lifecycle.resources('job')[0]['terminated_at']
