"""Hosted identity -> quote -> ledger -> real orchestrator -> verified S3 -> download.

Provider, object transport and worker are fake; lifecycle, storage manifests,
account isolation, job persistence and billing settlement are real code.
"""
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.hosted.app import create_hosted_app
from src.hosted.billing import BillingService, config_digest
from src.hosted.execution import ExecutionService
from src.hosted.remote_runtime import ManagedRemoteRuntime, RuntimeSettings
from src.hosted.settings import HostedSettings
from src.remote.s3_artifacts import S3ArtifactStore
from src.remote.spec import GpuOffer, ModelSpecs, RemoteInstance
from src.remote.worker_client import STATE_DONE, WorkerStatus
from tests.test_s3_artifacts import FakeS3
from tests.test_remote_orchestrator import ScriptedWorker


class Provider:
    name='runpod'
    def __init__(self):
        self.instances={}
        self.starts=0
    def list_gpu_offers(self):
        return [GpuOffer('gpu','GPU',80,price_per_hr_secure=1.2)]
    def provision(self,spec):
        self.starts+=1
        instance=RemoteInstance('instance','runpod','running',cost_per_hr=1.2,raw={'name':spec.name})
        self.instances[instance.instance_id]=instance
        return instance
    def proxy_url(self,instance,port):
        return 'https://worker.invalid'
    def get_instance(self,instance_id):
        return self.instances[instance_id]
    def terminate(self,instance_id):
        self.instances[instance_id].status='terminated'
    def list_instances(self):
        return list(self.instances.values())


class ScopedStore:
    def __init__(self,inner):
        self.inner=inner
    def __getattr__(self,name):
        return getattr(self.inner,name)
    def worker_environment(self):
        return {'MERLINA_ARTIFACT_BROKER_TOKEN':'scoped-test-token'}


@pytest.fixture
def execution(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings=HostedSettings(data_dir=tmp_path/'hosted',paid_execution_enabled=True,
        billing_policy_version='test',billing_failure_policy='charge_metered',billing_overage_policy='operator_absorbs')
    app=create_hosted_app(settings)
    store=app.state.store
    owner=store.create_account('a@example.com','long-password-here')
    project=store.create_project(owner['id'],'project','goal')
    from src.hosted.schemas import HostedJobConfig
    config=HostedJobConfig.model_validate({'base_model':'org/model','dataset':{'source':{'source_type':'huggingface','repo_id':'org/data'}}}).model_dump(exclude_none=True)
    job=store.create_job(owner['id'],project['id'],config)
    catalog=tmp_path/'measurements.json'
    entry={'measurement_id':'bench1','measured_seconds':10,'runtime_seconds':120,'boot_seconds':60,
        'shutdown_seconds':60,'billing_granularity_seconds':60,'fixed_minor':5,
        'training_config':{'remote':{'worker_image':'image@sha256:'+'a'*64,'gpu_type_id':'gpu','gpu_count':1,'container_disk_gb':100}}}
    catalog.write_text(json.dumps({config_digest(config):entry}))
    billing=BillingService(store,settings)
    provider=Provider()
    fs=FakeS3()
    artifact=ScopedStore(S3ArtifactStore('bucket','jobs/'+job['id'],fs=fs))
    rates=lambda:{'currency':'usd','observed_at':int(time.time()),'rates':[{
        'provider':'runpod','gpu_type_id':'gpu','gpu_count':1,'cloud_type':'secure',
        'container_disk_gb':100,'volume_gb':0,'rate_minor_per_hour':130}]}
    runtime=ManagedRemoteRuntime(store,settings,billing,lambda job:artifact,
        runtime_settings=RuntimeSettings(str(catalog),'https://price.invalid','test'),provider=provider,fetch_prices=rates)
    service=ExecutionService(store,billing,quote_builder=runtime.quote_builder,launch=runtime.launch,
        reconcile=runtime.reconcile,cancel=runtime.cancel,rate_check=runtime.rate_check,recover=runtime.recover_results)
    app.state.billing=billing
    app.state.execution=service
    app.state.artifact_download=runtime.download
    from src.remote import plan,orchestrator
    from src.remote.sizing import parse_model_specs
    from tests.test_remote_sizing import LLAMA_7B_CONFIG
    monkeypatch.setattr(plan,'fetch_model_specs',lambda *a,**kw:parse_model_specs('org/model',LLAMA_7B_CONFIG))
    # Broker transport itself has dedicated tests. Here no network is contacted.
    monkeypatch.setattr(orchestrator,'artifact_worker_env',lambda *a,**kw:{})
    source=tmp_path/'trained'
    source.mkdir()
    (source/'adapter_config.json').write_text('{}')
    (source/'adapter_model.safetensors').write_bytes(b'weights')
    worker=ScriptedWorker([WorkerStatus(state=STATE_DONE)],on_done=lambda:artifact.push_dir(source,'adapter'))
    actual=orchestrator.run_training_remote
    monkeypatch.setattr(orchestrator,'run_training_remote',lambda *args,**kw:actual(*args,**kw,
        worker_client_factory=lambda *_:worker,poll_interval_s=.001,boot_timeout_min=.01))
    class Checkout:
        def create_checkout(self,*args):
            return {'id':'cs1','url':'https://checkout.stripe.com/test'}
    billing.create_checkout(owner['id'],1000,Checkout())
    billing.confirmed_payment({'id':'evt1','type':'checkout.session.completed','data':{'object':{
        'id':'cs1','mode':'payment','payment_status':'paid','payment_intent':'pi1','amount_total':1000,'currency':'usd'}}})
    client=TestClient(app)
    client.headers['Authorization']='Bearer '+store.login(owner['email'],'long-password-here',3600)['token']
    return SimpleNamespace(app=app,store=store,job=job,client=client,billing=billing,provider=provider,runtime=runtime,service=service,catalog=catalog,artifact=artifact)


def endpoint(env,action):
    return f'/api/hosted/projects/{env.job["project_id"]}/jobs/{env.job["id"]}/{action}'


def approve(env):
    response=env.client.post(endpoint(env,'quote'))
    assert response.status_code==200,response.text
    response=env.client.post(endpoint(env,'approve'),json={'quote_id':response.json()['quote_id']})
    assert response.status_code==200,response.text


def test_paid_hosted_journey_and_download(execution):
    e=execution
    approve(e)
    e.service.tick()
    deadline=time.monotonic()+5
    while e.runtime.active and time.monotonic()<deadline:
        time.sleep(.01)
    e.runtime.sync(e.job)
    state=e.store.require_job(e.job['account_id'],e.job['project_id'],e.job['id'])
    assert state['status']=='completed',state
    assert e.provider.starts==1
    assert e.billing.run(e.job['id'])['state']=='settled'
    assert e.billing.balance(e.job['account_id'])['reserved_minor']==0
    artifacts=e.client.get(endpoint(e,'artifacts')).json()
    assert len(artifacts)==1
    download=e.client.get(endpoint(e,'artifacts')+'/'+artifacts[0]['id']+'/download')
    assert download.status_code==200
    assert download.content.startswith(b'PK')
    assert e.store.list_events(e.job['account_id'],e.job['project_id'],e.job['id'])
    e.service.tick()
    assert e.provider.starts==1


def test_profile_change_refunds_without_starting_provider(execution):
    e=execution
    approve(e)
    catalog=json.loads(e.catalog.read_text())
    next(iter(catalog.values()))['training_config']['remote']['container_disk_gb']=200
    e.catalog.write_text(json.dumps(catalog))
    e.service.tick()
    assert e.provider.starts==0
    assert e.billing.run(e.job['id'])['state']=='settled'
    assert e.billing.balance(e.job['account_id'])['available_minor']==1000
    state=e.store.require_job(e.job['account_id'],e.job['project_id'],e.job['id'])
    assert state['status']=='failed'
    assert 'no compute was charged' in state['result']['error']


def test_billing_routes_preserve_account_boundary(execution):
    e=execution
    other=e.store.create_account('b@example.com','long-password-here')
    client=TestClient(e.app)
    client.headers['Authorization']='Bearer '+e.store.login(other['email'],'long-password-here',3600)['token']
    assert client.post(endpoint(e,'quote')).status_code==404
    assert client.post(endpoint(e,'approve'),json={'quote_id':'anything'}).status_code==404
    assert client.get('/api/hosted/billing/balance').json()['balance_minor']==0
    assert client.post('/api/hosted/billing/credit',json={'amount_minor':1000}).status_code==404


def test_measurement_tool_uses_stored_config_and_explicit_evidence(execution,tmp_path):
    from src.hosted.measurement import record_measurement
    path=tmp_path/'catalog.json'
    key=record_measurement(path,execution.job['config'],measurement_id='observed-1',measured_seconds=10,
        runtime_seconds=20,boot_seconds=60,shutdown_seconds=60,billing_granularity_seconds=60,
        fixed_minor=5,gpu_type_id='gpu',gpu_count=1,container_disk_gb=100,worker_image='image@sha256:'+'a'*64)
    assert key==config_digest(execution.job['config'])
    assert json.loads(path.read_text())[key]['measurement_id']=='observed-1'


def test_ambiguous_provider_create_retains_then_reconciles_once(execution,monkeypatch):
    e=execution
    original=e.provider.provision
    def lost_response(spec):
        original(spec)
        raise TimeoutError('Provider accepted create but response was lost')
    monkeypatch.setattr(e.provider,'provision',lost_response)
    approve(e)
    e.service.tick()
    deadline=time.monotonic()+5
    while e.runtime.active and time.monotonic()<deadline:
        time.sleep(.01)
    assert e.provider.starts==1
    assert e.billing.run(e.job['id'])['state']=='dispatching'
    assert e.billing.balance(e.job['account_id'])['reserved_minor']>0
    e.service.tick()
    assert e.billing.run(e.job['id'])['state']=='settled'
    assert e.provider.instances['instance'].status=='terminated'
    charge=e.billing.run(e.job['id'])['charged']
    e.service.tick()
    assert e.provider.starts==1
    assert e.billing.run(e.job['id'])['charged']==charge


def test_sync_exports_finite_metrics_and_safe_failure_code(execution):
    e=execution
    approve(e)
    e.billing.claim(e.job['id'])
    e.runtime.jobs.create_job(e.job['id'],{})
    e.runtime.jobs.update_job(e.job['id'],status='failed',loss=.75,eval_loss=.8,current_step=12,
        total_steps=20,error='private provider https://user:token@internal/path')
    e.billing.record_failure(e.job['id'],'budget_exhausted')
    owner=e.runtime.lifecycle.claim(e.job['id'],1)
    label=e.runtime.lifecycle.intent(e.job['id'],owner,'train','runpod')
    instance=RemoteInstance('metrics-instance','runpod','running',cost_per_hr=1.2)
    e.runtime.lifecycle.record_instance(e.job['id'],'train',label,instance)
    e.runtime.lifecycle.cleanup_result('runpod','metrics-instance',confirmed=True)
    e.runtime.sync(e.job)
    result=e.store.require_job(e.job['account_id'],e.job['project_id'],e.job['id'])['result']
    assert result['loss']==.75 and result['eval_loss']==.8 and result['steps']==12
    assert result['runtime_seconds']>=0 and result['error_code']=='budget_exhausted'
    assert 'internal' not in json.dumps(result) and 'token' not in json.dumps(result)
    e.runtime.jobs.update_job(e.job['id'],loss=float('inf'))
    e.runtime.sync(e.job)
    assert 'loss' not in e.store.require_job(e.job['account_id'],e.job['project_id'],e.job['id'])['result']


def test_artifact_failure_after_settlement_recovers_after_restart(execution,monkeypatch):
    e=execution
    publish=e.runtime.publish_artifact
    monkeypatch.setattr(e.runtime,'publish_artifact',lambda *args: (_ for _ in ()).throw(OSError('private path')))
    approve(e)
    e.service.tick()
    deadline=time.monotonic()+5
    while e.runtime.active and time.monotonic()<deadline:
        time.sleep(.01)
    e.runtime.sync(e.job)
    assert e.billing.run(e.job['id'])['state']=='settled'
    state=e.store.require_job(e.job['account_id'],e.job['project_id'],e.job['id'])
    assert state['status']=='completed' and state['result']['artifact_status']=='pending'
    assert state['result']['artifacts_available'] is False
    assert 'private path' not in json.dumps(state['result'])
    # A new runtime loads the persisted recovery queue even though no billing run
    # is pending. It restores missing local output from the verified object store.
    import shutil
    shutil.rmtree(e.runtime.jobs.get_job(e.job['id']).output_dir)
    restarted=ManagedRemoteRuntime(e.store,e.runtime.settings,e.billing,lambda job:e.artifact,
        runtime_settings=e.runtime.config,provider=e.provider,fetch_prices=e.runtime.fetch_prices)
    restarted_service=ExecutionService(e.store,e.billing,recover=restarted.recover_results)
    restarted_service.tick()
    state=e.store.require_job(e.job['account_id'],e.job['project_id'],e.job['id'])
    assert state['result']['artifact_status']=='ready'
    e.app.state.artifact_download=restarted.download
    artifacts=e.client.get(endpoint(e,'artifacts')).json()
    assert len(artifacts)==1
    assert e.client.get(endpoint(e,'artifacts')+'/'+artifacts[0]['id']+'/download').status_code==200
    assert e.provider.starts==1


def backdate_claim(e,seconds):
    with e.store.transaction() as db:
        db.execute('UPDATE billing_runs SET claimed_at=claimed_at-? WHERE job_id=?',(seconds,e.job['id']))


def test_dispatch_crash_before_lifecycle_claim_releases_after_grace(execution):
    from src.hosted.remote_runtime import UNLAUNCHED_DISPATCH_GRACE_SECONDS
    from src.remote.lifecycle import RemoteAlreadyClaimed
    e=execution
    approve(e)
    # Crash between the billing claim and the worker's lifecycle claim.
    assert e.billing.claim(e.job['id'])
    e.service.tick()
    assert e.billing.run(e.job['id'])['state']=='dispatching'
    assert e.billing.balance(e.job['account_id'])['reserved_minor']>0
    backdate_claim(e,UNLAUNCHED_DISPATCH_GRACE_SECONDS+1)
    e.service.tick()
    run=e.billing.run(e.job['id'])
    assert run['state']=='settled' and run['charged']==0
    assert e.billing.balance(e.job['account_id'])=={'balance_minor':1000,'reserved_minor':0,'available_minor':1000,'currency':'usd'}
    state=e.store.require_job(e.job['account_id'],e.job['project_id'],e.job['id'])
    assert state['status']=='failed' and 'no compute was charged' in state['result']['error']
    # A late worker for the released claim can never reach provisioning.
    with pytest.raises(RemoteAlreadyClaimed):
        e.runtime.lifecycle.claim(e.job['id'],1)
    e.service.tick()
    assert e.provider.starts==0 and e.billing.run(e.job['id'])['charged']==0


def test_dispatch_with_live_worker_claim_is_not_released(execution):
    from src.hosted.remote_runtime import UNLAUNCHED_DISPATCH_GRACE_SECONDS
    e=execution
    approve(e)
    assert e.billing.claim(e.job['id'])
    e.runtime.lifecycle.claim(e.job['id'],1)
    backdate_claim(e,UNLAUNCHED_DISPATCH_GRACE_SECONDS+1)
    e.service.tick()
    assert e.billing.run(e.job['id'])['state']=='dispatching'
    assert e.billing.balance(e.job['account_id'])['reserved_minor']>0


def test_definitively_failed_provision_settles_without_charge(execution):
    e=execution
    approve(e)
    jid=e.job['id']
    assert e.billing.claim(jid)
    e.runtime.jobs.create_job(jid,{})
    # Lifecycle records a definite provider rejection: no instance was created.
    lifecycle=e.runtime.lifecycle
    owner=lifecycle.claim(jid,1)
    lifecycle.intent(jid,owner,'train','runpod')
    lifecycle.stage_state(jid,'train','failed',error='ProviderError')
    lifecycle.finish(jid,owner,'failed','Provider rejected create')
    e.runtime.jobs.update_job(jid,status='failed',error='Provider rejected create')
    e.service.tick()
    run=e.billing.run(jid)
    assert run['state']=='settled' and run['charged']==0
    assert e.billing.balance(e.job['account_id'])['available_minor']==1000
    assert e.store.require_job(e.job['account_id'],e.job['project_id'],jid)['status']=='failed'
    assert e.provider.starts==0
