"""Failure exercises: API death, ambiguous creation, retries, cancellation."""
import pytest

from src.job_manager import JobManager
from src.remote.cleanup import reconcile_once
from src.remote.lifecycle import RemoteLifecycle
from src.remote.spec import RemoteInstance


class InventoryProvider:
    name = 'runpod'
    def __init__(self):
        self.pods = {}
        self.deleted = []
        self.fail_delete = False
        self.fail_inventory = False
        self.confirm_delete = True
    def list_instances(self):
        if self.fail_inventory:
            raise TimeoutError('provider timeout')
        return list(self.pods.values())
    def terminate(self, instance_id):
        self.deleted.append(instance_id)
        if self.fail_delete:
            raise TimeoutError('provider timeout')
        if self.confirm_delete:
            self.pods.pop(instance_id, None)
    def get_instance(self, instance_id):
        return self.pods.get(instance_id, RemoteInstance(instance_id, self.name, 'terminated'))
    def create(self, name, instance_id='pod'):
        self.pods[instance_id] = RemoteInstance(instance_id, self.name, 'running', 2, raw={'name': name})
        return self.pods[instance_id]


@pytest.fixture
def state(tmp_path):
    now = [1000.0]
    manager = JobManager(str(tmp_path / 'jobs.db'))
    manager.create_job('job', {})
    lifecycle = RemoteLifecycle(manager.db_path, lease_seconds=10, clock=lambda: now[0])
    owner = lifecycle.claim('job', 1)
    label = lifecycle.intent('job', owner, 'train', 'runpod')
    provider = InventoryProvider()
    return now, manager, lifecycle, owner, label, provider


def test_crash_between_paid_create_and_id_persistence(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label)
    now[0] += 11
    restarted = RemoteLifecycle(manager.db_path, clock=lambda: now[0])
    result = reconcile_once(restarted, {'runpod': provider}, job_manager=manager)
    assert provider.deleted == ['pod']
    assert result['resources'][0]['cleanup_status'] == 'terminated'
    assert result['alerts'] == []
    assert manager.get_job('job').status == 'failed'
    assert restarted.attempts()[0]['state'] == 'failed'


def test_unconfirmed_delete_retries_after_restart(state):
    now, manager, lifecycle, owner, label, provider = state
    instance = provider.create(label)
    lifecycle.record_instance('job', 'train', label, instance)
    lifecycle.finish('job', owner, 'failed')
    provider.confirm_delete = False
    first = reconcile_once(lifecycle, {'runpod': provider})
    assert first['alerts']
    assert first['resources'][0]['terminated_at'] is None
    provider.confirm_delete = True
    second = reconcile_once(RemoteLifecycle(manager.db_path), {'runpod': provider})
    assert second['resources'][0]['cleanup_status'] == 'terminated'
    assert provider.deleted == ['pod', 'pod']
    assert second['alerts'] == []


def test_inventory_failure_still_cleans_known_ids(state):
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.record_instance('job', 'train', label, provider.create(label))
    lifecycle.finish('job', owner, 'failed')
    provider.fail_inventory = True
    result = reconcile_once(lifecycle, {'runpod': provider})
    assert provider.deleted == ['pod']
    assert result['alerts'][0]['alert_key'] == 'inventory:runpod'


def test_no_inventory_does_not_prove_ambiguous_create_failed(state):
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.finish('job', owner, 'failed')
    settlements = []
    class Guard:
        def settle(self, *args):
            settlements.append(args)
    result = reconcile_once(lifecycle, {'runpod': provider}, execution_guard_factory=lambda _: Guard())
    assert result['alerts'][0]['alert_key'].startswith('ambiguous:')
    assert settlements == []
    assert lifecycle.attempts()[0]['state'] == 'provisioning'


def test_scope_excludes_other_merlina_installations_and_unknown_names(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create('merlina-somebody-else-train', 'foreign')
    provider.create(f'merlina-{lifecycle.namespace}-unknown-train', 'unowned')
    provider.create(label, 'owned')
    lifecycle.finish('job', owner, 'failed')
    result = reconcile_once(lifecycle, {'runpod': provider})
    assert provider.deleted == ['owned']
    assert any(a['alert_key'] == 'unowned:runpod:unowned' for a in result['alerts'])


def test_active_lease_preserved_but_total_deadline_enforced(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label)
    reconcile_once(lifecycle, {'runpod': provider})
    assert provider.deleted == []
    now[0] += 3601
    reconcile_once(lifecycle, {'runpod': provider})
    assert provider.deleted == ['pod']


def test_cancel_grace_is_persisted_signalled_and_capped(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label)
    manager.request_stop('job')
    signals = []
    reconcile_once(lifecycle, {'runpod': provider}, job_manager=manager,
                   stop_grace_seconds=30, stop_worker=lambda resource: signals.append(resource['instance_id']))
    assert provider.deleted == []
    assert signals == ['pod']
    now[0] += 20
    restarted = RemoteLifecycle(manager.db_path, clock=lambda: now[0])
    reconcile_once(restarted, {'runpod': provider}, job_manager=manager, stop_grace_seconds=30)
    assert provider.deleted == []  # lost controller still has bounded checkpoint grace
    now[0] += 11
    reconcile_once(restarted, {'runpod': provider}, job_manager=manager, stop_grace_seconds=30)
    assert provider.deleted == ['pod']  # subsequent scans cannot extend deadline


def test_cancel_grace_cannot_extend_total_deadline(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label)
    manager.request_stop('job')
    now[0] += 3595
    reconcile_once(lifecycle, {'runpod': provider}, job_manager=manager, stop_grace_seconds=30)
    assert provider.deleted == []
    now[0] += 6
    reconcile_once(lifecycle, {'runpod': provider}, job_manager=manager, stop_grace_seconds=30)
    assert provider.deleted == ['pod']


def test_emergency_cancellation_bypasses_grace(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label)
    manager.request_stop('job')
    lifecycle.request_cancel('job', 30, emergency=True)
    reconcile_once(lifecycle, {'runpod': provider}, job_manager=manager, stop_grace_seconds=30)
    assert provider.deleted == ['pod']


def test_duplicate_provider_creations_fail_job_and_terminate_both(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label, 'one')
    provider.create(label, 'two')
    reconcile_once(lifecycle, {'runpod': provider}, job_manager=manager)
    assert set(provider.deleted) == {'one', 'two'}
    assert lifecycle.run('job')['state'] == 'failed'
    assert manager.get_job('job').status == 'failed'


def test_delete_timeout_stays_visible_without_exception_secrets(state):
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label)
    lifecycle.finish('job', owner, 'failed')
    provider.fail_delete = True
    result = reconcile_once(lifecycle, {'runpod': provider})
    assert result['alerts']
    assert result['resources'][0]['terminated_at'] is None
    assert result['resources'][0]['cleanup_attempts'] == 1


def test_retained_debug_instance_cannot_outlive_job_deadline(state):
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.record_instance('job', 'train', label, provider.create(label))
    lifecycle.finish('job', owner, 'failed')
    lifecycle.cleanup_result('runpod', 'pod', retained=True)
    reconcile_once(lifecycle, {'runpod': provider})
    assert provider.deleted == []
    now[0] += 3601
    reconcile_once(lifecycle, {'runpod': provider})
    assert provider.deleted == ['pod']


def test_startup_without_configured_provider_has_no_side_effects(tmp_path):
    from types import SimpleNamespace
    from src.remote.cleanup import reconcile_startup
    calls = []
    manager = JobManager(str(tmp_path / 'jobs.db'))
    result = reconcile_startup(manager, SimpleNamespace(runpod_api_key=None),
                               provider_factory=lambda *args, **kwargs: calls.append(args))
    assert result is None
    assert calls == []


def test_startup_reconciles_owned_remote_jobs(state):
    from types import SimpleNamespace
    from src.remote.cleanup import reconcile_startup
    now, manager, lifecycle, owner, label, provider = state
    provider.create(label)
    # Real-clock startup observes this injected historical lease as expired.
    reconcile_startup(manager, SimpleNamespace(runpod_api_key='test-only'),
                      provider_factory=lambda *args, **kwargs: provider)
    assert provider.deleted == ['pod']
    assert manager.get_job('job').status == 'failed'


def test_local_reattach_does_not_fail_live_remote_controller(tmp_path):
    from src.worker_reattach import reattach_orphaned_workers
    manager = JobManager(str(tmp_path / 'jobs.db'))
    manager.create_job('remote', {'remote': {'enabled': True}})
    manager.update_job('remote', status='training')
    assert reattach_orphaned_workers(manager)['failed'] == []
    assert manager.get_job('remote').status == 'training'


def test_cleanup_can_forward_authenticated_worker_stop_without_storing_token(state, monkeypatch):
    now, manager, lifecycle, owner, label, provider = state
    instance = provider.create(label)
    instance.raw['env'] = {'MERLINA_WORKER_TOKEN': 'scoped-test-token'}
    provider.proxy_url = lambda instance, port: 'https://worker.example'
    calls = []
    class Client:
        def __init__(self, url, token):
            calls.append((url, token))
        def request_stop(self):
            return True
    from src.remote import worker_client
    monkeypatch.setattr(worker_client, 'HttpWorkerClient', Client)
    manager.request_stop('job')
    reconcile_once(lifecycle, {'runpod': provider}, job_manager=manager, stop_grace_seconds=30)
    assert calls == [('https://worker.example', 'scoped-test-token')]
    assert provider.deleted == []
    assert 'scoped-test-token' not in repr(lifecycle.resources())


def _wait_for(predicate, timeout=5.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class CountingProvider(InventoryProvider):
    def __init__(self):
        super().__init__()
        self.scans = 0
    def list_instances(self):
        self.scans += 1
        return super().list_instances()


def test_api_restart_within_lease_is_cleaned_up_after_expiry(state):
    """One startup pass is not enough: the lease was still valid when it ran."""
    from types import SimpleNamespace
    from src.remote.cleanup import start_reconciler
    now, manager, lifecycle, owner, label, _ = state
    provider = CountingProvider()
    instance = provider.create(label)
    lifecycle.record_instance('job', 'train', label, instance)
    manager.update_job('job', status='training')
    # The controller died with the old process; its lease is still valid.
    restarted = RemoteLifecycle(manager.db_path, lease_seconds=10, clock=lambda: now[0])
    thread, stop = start_reconciler(manager, SimpleNamespace(runpod_api_key='test-only'),
                                    provider_factory=lambda *a, **k: provider,
                                    lifecycle=restarted, interval=0.01)
    try:
        assert _wait_for(lambda: provider.scans >= 2)
        assert provider.deleted == []                  # live lease respected
        assert manager.get_job('job').status == 'training'
        now[0] += 11                                   # lease lapses, nobody renews it
        assert _wait_for(lambda: provider.deleted == ['pod'])
        assert _wait_for(lambda: manager.get_job('job').status == 'failed')
        assert _wait_for(lambda: restarted.resources('job')[0]['cleanup_status'] == 'terminated')
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_reconciler_idle_and_local_installs_never_contact_provider(tmp_path):
    from types import SimpleNamespace
    from src.remote.cleanup import start_reconciler
    manager = JobManager(str(tmp_path / 'jobs.db'))
    calls = []
    for key in (None, 'test-only'):
        thread, stop = start_reconciler(manager, SimpleNamespace(runpod_api_key=key),
                                        provider_factory=lambda *a, **k: calls.append(a),
                                        interval=0.01)
        thread.join(timeout=0.1 if key else 5)
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert calls == []


def test_reconciler_survives_failed_pass(state):
    from types import SimpleNamespace
    from src.remote.cleanup import start_reconciler
    now, manager, lifecycle, owner, label, _ = state
    provider = CountingProvider()
    provider.fail_inventory = True
    lifecycle.finish('job', owner, 'failed')
    attempts = []
    def factory(*a, **k):
        attempts.append(a)
        if len(attempts) == 1:
            raise RuntimeError('transient')
        return provider
    thread, stop = start_reconciler(manager, SimpleNamespace(runpod_api_key='test-only'),
                                    provider_factory=factory, lifecycle=lifecycle, interval=0.01)
    try:
        assert _wait_for(lambda: provider.scans >= 1)
    finally:
        stop.set()
        thread.join(timeout=5)


def test_ambiguous_create_absent_across_grace_resolves_failed(state):
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.stage_state('job', 'train', 'provisioning_unknown', error='ProviderError')
    lifecycle.finish('job', owner, 'failed')
    settlements = []
    class Guard:
        def settle(self, *args):
            settlements.append(args)
    kwargs = dict(job_manager=manager, execution_guard_factory=lambda _: Guard(),
                  absent_scans=3, absent_grace_seconds=600)
    for _ in range(3):  # three scans, but within the grace window
        result = reconcile_once(lifecycle, {'runpod': provider}, **kwargs)
        now[0] += 100
    assert any(a['alert_key'] == f'ambiguous:{label}' for a in result['alerts'])
    assert lifecycle.attempts()[0]['state'] == 'provisioning_unknown'
    now[0] += 400                                       # 700s since first absence
    result = reconcile_once(lifecycle, {'runpod': provider}, **kwargs)
    assert lifecycle.attempts()[0]['state'] == 'failed'
    assert 'never appeared' in lifecycle.attempts()[0]['error']
    assert result['alerts'] == []
    assert len(settlements) == 1
    assert provider.deleted == []


def test_failed_inventory_scans_do_not_count_as_absence(state):
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.stage_state('job', 'train', 'provisioning_unknown')
    lifecycle.finish('job', owner, 'failed')
    provider.fail_inventory = True
    for _ in range(5):
        reconcile_once(lifecycle, {'runpod': provider}, absent_scans=3, absent_grace_seconds=600)
        now[0] += 600
    assert lifecycle.attempts()[0]['state'] == 'provisioning_unknown'


def test_late_appearance_resets_absence_and_is_terminated(state):
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.stage_state('job', 'train', 'provisioning_unknown')
    lifecycle.finish('job', owner, 'failed')
    kwargs = dict(absent_scans=3, absent_grace_seconds=600)
    for _ in range(2):
        reconcile_once(lifecycle, {'runpod': provider}, **kwargs)
        now[0] += 400
    provider.create(label)  # the listing lagged; the pod really exists
    reconcile_once(lifecycle, {'runpod': provider}, **kwargs)
    assert provider.deleted == ['pod']
    assert lifecycle.resources('job')[0]['cleanup_status'] == 'terminated'
    assert 'reconciled and terminated' in lifecycle.attempts()[0]['error']


def test_cli_resolve_records_operator_and_clears_alert(state, capsys):
    from src.remote.cleanup import main
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.stage_state('job', 'train', 'provisioning_unknown')
    lifecycle.finish('job', owner, 'failed')
    reconcile_once(lifecycle, {'runpod': provider})
    assert lifecycle.alerts()
    assert main(['--db', str(manager.db_path), '--resolve', 'job', '--as', 'failed',
                 '--by', 'nick', '--note', 'checked console']) == 0
    assert lifecycle.attempts()[0]['state'] == 'failed'
    assert 'by nick' in lifecycle.attempts()[0]['error']
    record = lifecycle.manual_resolutions()[0]
    assert (record['label'], record['resolved_as'], record['resolved_by'], record['note']) == (
        label, 'failed', 'nick', 'checked console')
    assert lifecycle.alerts() == []
    assert reconcile_once(lifecycle, {'runpod': provider})['alerts'] == []


def test_cli_resolve_terminated_confirms_unterminable_resource(state):
    from src.remote.cleanup import main
    now, manager, lifecycle, owner, label, provider = state
    lifecycle.record_instance('job', 'train', label, provider.create(label))
    lifecycle.finish('job', owner, 'failed')
    lifecycle.cleanup_result('runpod', 'pod', error='timeout')
    assert main(['--db', str(manager.db_path), '--resolve', label, '--as', 'terminated', '--by', 'nick']) == 0
    resource = lifecycle.resources('job')[0]
    assert resource['cleanup_status'] == 'terminated'
    assert 'nick' in resource['last_error']
    assert lifecycle.manual_resolutions()[0]['resolved_as'] == 'terminated'


def test_cli_resolve_refuses_live_run_and_unknown_target(state):
    from src.remote.cleanup import main
    now, manager, lifecycle, owner, label, provider = state
    with pytest.raises(SystemExit):
        main(['--db', str(manager.db_path), '--resolve', 'job', '--as', 'failed', '--by', 'nick'])
    with pytest.raises(SystemExit):
        main(['--db', str(manager.db_path), '--resolve', 'nope', '--as', 'failed', '--by', 'nick'])
    with pytest.raises(SystemExit):
        main(['--db', str(manager.db_path), '--resolve', 'job'])
    assert lifecycle.attempts()[0]['state'] == 'provisioning'
    assert lifecycle.manual_resolutions() == []
