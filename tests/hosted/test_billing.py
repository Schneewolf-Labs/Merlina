import hashlib
import hmac
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.hosted.billing import BillingError, BillingService, BillingUnavailable, BudgetExceeded, StageQuote, verify_stripe_event
from src.hosted.execution import ExecutionGuard, ExecutionService


class Store:
    def __init__(self, path):
        self.path = path
        self.jobs = {}
    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
    def require_job(self, account, project, job):
        value = self.jobs[job]
        if value['account_id'] != account or value['project_id'] != project:
            raise BillingError('Not owned')
        return value
    def update_job(self, account, project, job, **kwargs):
        self.require_job(account, project, job).update(kwargs)


@pytest.fixture
def billing(tmp_path):
    settings = SimpleNamespace(paid_execution_enabled=True, billing_currency='usd',
        billing_policy_version='pilot-v1', billing_failure_policy='charge_metered',
        billing_overage_policy='operator_absorbs')
    result = BillingService(Store(tmp_path/'ledger.db'), settings, clock=lambda: 1000)
    return result


def credit(billing, amount=1000, account='a', session='s', event='e', payment='p'):
    class Transport:
        def create_checkout(self, *args):
            return {'id': session, 'url': 'https://checkout.stripe.com/test'}
    billing.create_checkout(account, amount, Transport())
    payload = {'id': event, 'type':'checkout.session.completed', 'data':{'object':{
        'id':session, 'mode':'payment', 'payment_status':'paid', 'payment_intent':payment,
        'amount_total':amount, 'currency':'usd'}}}
    billing.confirmed_payment(payload)
    return payload


def stage(**kwargs):
    data = dict(name='train', provider='runpod', gpu_type_id='gpu', gpu_count=1,
        cloud_type='secure', rate_minor_per_hour=3600, rate_observed_at=1000,
        measurement_id='observed-run', measured_seconds=60, runtime_seconds=100,
        boot_seconds=20, shutdown_seconds=20, billing_granularity_seconds=1)
    return StageQuote(**(data | kwargs))


def quote(billing, job='j', account='a', config=None):
    return billing.quote(account,'proj',job,config or {},[stage()],fixed_minor=10)


def reserve(billing, job='j'):
    q=quote(billing,job)
    return billing.reserve('a','proj',job,q['quote_id'],{})


def test_confirmed_payment_idempotency_and_validation(billing):
    event = credit(billing)
    assert not billing.confirmed_payment(event)
    event['id']='other'
    assert not billing.confirmed_payment(event)
    assert billing.balance('a')['available_minor']==1000
    event['data']['object']['amount_total']=1
    with pytest.raises(BillingError,match='mismatch'):
        billing.confirmed_payment(event)


def test_atomic_concurrent_reservations(billing):
    credit(billing,amount=200)
    quotes=[quote(billing,'j'+str(i)) for i in range(12)]
    def attempt(q):
        try:
            billing.reserve('a','proj','j'+str(quotes.index(q)),q['quote_id'],{})
            return True
        except BillingError:
            return False
    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(attempt,quotes))==1
    assert billing.balance('a')['reserved_minor']==150


def test_raced_duplicate_payments(billing):
    event=credit(billing)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert not any(pool.map(billing.confirmed_payment,[event]*30))
    assert billing.balance('a')['balance_minor']==1000


def test_quote_ownership_change_staleness_and_policy(billing):
    credit(billing)
    q=quote(billing)
    for account, config in [('other',{}),('a',{'changed':True})]:
        with pytest.raises(BillingError):
            billing.reserve(account,'proj','j',q['quote_id'],config)
    billing.clock=lambda:1300
    with pytest.raises(BillingError,match='expired'):
        billing.reserve('a','proj','j',q['quote_id'],{})
    billing.clock=lambda:1000
    billing.settings.billing_policy_version='new'
    with pytest.raises(BillingError,match='policy'):
        billing.reserve('a','proj','j',q['quote_id'],{})


def test_funds_and_policy_block_before_provision(billing):
    with pytest.raises(BillingError,match='Insufficient'):
        reserve(billing)
    billing.settings.billing_failure_policy=None
    with pytest.raises(BillingUnavailable):
        quote(billing)


def test_settlement_restart_cleanup_and_overage(billing):
    credit(billing)
    reserve(billing)
    with pytest.raises(BillingError,match='unconfirmed'):
        billing.settle('j',40,cleanup_confirmed=False)
    assert billing.balance('a')['reserved_minor']==150
    restarted=BillingService(billing.store,billing.settings)
    assert restarted.settle('j',1000,cleanup_confirmed=True)==150
    assert restarted.settle('j',1,cleanup_confirmed=True)==150
    assert restarted.balance('a')['balance_minor']==850
    assert restarted.balance('a')['reserved_minor']==0
    assert restarted.run('j')['provider_cost']==1000


def test_signature_tampering_and_expiry():
    body=b'{"id":"evt"}'
    signature=hmac.new(b'secret',b'1000.'+body,hashlib.sha256).hexdigest()
    header='t=1000,v1='+signature
    assert verify_stripe_event(body,header,'secret',now=1000)['id']=='evt'
    for data, now in [(body+b' ',1000),(body,1301)]:
        with pytest.raises(BillingError):
            verify_stripe_event(data,header,'secret',now=now)


def test_guard_budget_cleanup_and_shape(billing):
    credit(billing)
    reserve(billing)
    assert billing.claim('j')
    guard=ExecutionGuard(billing,'j',rate_check=lambda *_:(3600,1000))
    spec=SimpleNamespace(gpu_type_id='gpu',gpu_count=1,cloud_type='secure',container_disk_gb=50,volume_gb=0)
    guard.before_provision('j','train',spec)
    spec.gpu_count=2
    with pytest.raises(BillingError,match='shape'):
        guard.before_provision('j','train',spec)
    resource={'stage_name':'train','started_at':880,'terminated_at':None,'cost_per_hr':36}
    with pytest.raises(BudgetExceeded):
        guard.usage('j',[resource])
    with pytest.raises(BillingError):
        guard.settle('j',[resource])
    resource['terminated_at']=1000
    assert guard.settle('j',[resource])==130


def test_stale_rates_and_multistage_quote(billing):
    with pytest.raises(BillingUnavailable,match='stale'):
        billing.quote('a','p','j',{},[stage(rate_observed_at=1)],fixed_minor=0)
    q=billing.quote('a','p','j',{},[stage(),stage(name='merge')],fixed_minor=10)
    assert q['amount_minor']==290


def test_dispatch_failure_retains_claim_and_funds(billing):
    credit(billing)
    reserve(billing)
    billing.store.jobs['j']={'id':'j','account_id':'a','project_id':'proj','config':{}}
    launched=[]
    reconciled=[]
    def launch(job,guard):
        launched.append(job['id'])
        raise TimeoutError('ambiguous provider create')
    service=ExecutionService(billing.store,billing,launch=launch,reconcile=lambda j,g:reconciled.append(j['id']))
    service.tick()
    restarted=ExecutionService(billing.store,billing,launch=launch,reconcile=lambda j,g:reconciled.append(j['id']))
    restarted.tick()
    assert launched==['j'] and reconciled==['j']
    assert billing.balance('a')['reserved_minor']==150
    assert billing.run('j')['last_error']=='TimeoutError'


def test_cancel_queued_does_not_provision(billing):
    credit(billing)
    reserve(billing)
    billing.store.jobs['j']={'id':'j','account_id':'a','project_id':'proj','config':{}}
    billing.cancel('a','proj','j')
    service=ExecutionService(billing.store,billing,launch=lambda *_:pytest.fail('provisioned'))
    service.tick()
    assert billing.run('j')['state']=='settled'
    assert billing.balance('a')['available_minor']==1000


def test_live_full_rates_reject_stale_missing_and_lower_gpu_price(billing):
    from src.hosted.remote_runtime import ManagedRemoteRuntime
    from src.remote.spec import GpuOffer
    runtime=ManagedRemoteRuntime.__new__(ManagedRemoteRuntime)
    runtime.billing=billing
    runtime.provider=SimpleNamespace(list_gpu_offers=lambda:[GpuOffer('gpu','GPU',80,price_per_hr_secure=1.20)])
    shape={'provider':'runpod','gpu_type_id':'gpu','gpu_count':1,'cloud_type':'secure','container_disk_gb':50,'volume_gb':0}
    feed={'currency':'usd','observed_at':1000,'rates':[shape | {'rate_minor_per_hour':130}]}
    runtime.fetch_prices=lambda:feed
    assert runtime.price(shape)==(130,1000)
    feed['rates'][0]['rate_minor_per_hour']=119
    with pytest.raises(BillingUnavailable,match='inconsistent'):
        runtime.price(shape)
    feed['observed_at']=0
    with pytest.raises(BillingUnavailable,match='stale'):
        runtime.price(shape)


def test_guard_revalidates_rate_increases_and_cleanup_unknown(billing):
    credit(billing)
    reserve(billing)
    billing.claim('j')
    guard=ExecutionGuard(billing,'j',rate_check=lambda *_:(9999,1000))
    spec=SimpleNamespace(gpu_type_id='gpu',gpu_count=1,cloud_type='secure',container_disk_gb=50,volume_gb=0)
    with pytest.raises(BillingError,match='changed'):
        guard.before_provision('j','train',spec)
    resource={'stage_name':'train','started_at':900,'terminated_at':1000,'cost_per_hr':None}
    with pytest.raises(BillingError,match='rate missing'):
        guard.settle('j',[resource])
    assert billing.balance('a')['reserved_minor']==150
