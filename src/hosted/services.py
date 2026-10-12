"""Tenant-safe dataset paths and job configuration resolution (no provisioning)."""
import hashlib
import json
from pathlib import Path
import re
from fastapi import HTTPException
from .schemas import HostedJobConfig
from .store import new_id


def scoped_path(settings, account_id, project_id, resource_id, *, kind='datasets'):
    if kind not in ('datasets', 'jobs') or not all(re.fullmatch('[a-f0-9]{32}', value) for value in (account_id, project_id, resource_id)):
        raise ValueError('Invalid resource identifier')
    root = settings.data_dir.resolve()
    path = root / 'accounts' / account_id / 'projects' / project_id / kind / resource_id
    if not path.resolve().is_relative_to(root):
        raise ValueError('Resource path escapes hosted storage')
    return path


def save_dataset(store, settings, account_id, project_id, name, content):
    store.require_project(account_id, project_id)
    raw = content.encode('utf-8')
    if len(raw) > settings.max_dataset_bytes:
        raise HTTPException(413, 'Dataset exceeds upload limit')
    rows = 0
    try:
        for line in content.splitlines():
            if line.strip():
                value = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite number')))
                if not isinstance(value, dict):
                    raise ValueError('Every JSONL row must be an object')
                rows += 1
        if rows == 0:
            raise ValueError('Dataset is empty')
    except (ValueError, RecursionError):
        raise HTTPException(422, 'Dataset must contain JSON objects, one per line') from None
    dataset_id = new_id()
    path = scoped_path(settings, account_id, project_id, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with path.open('xb') as output:
            output.write(raw)
        path.chmod(0o600)
        return store.register_dataset(account_id, project_id, dataset_id, name, hashlib.sha256(raw).hexdigest(), len(raw))
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def validate_job_config(store, account_id, project_id, config):
    store.require_project(account_id, project_id)
    clean = HostedJobConfig.model_validate(config).model_dump(exclude_none=True)
    source = clean['dataset']['source']
    if source['source_type'] == 'upload':
        store.require_dataset(account_id, project_id, source['dataset_id'])
    return clean


def resolve_uploaded_datasets(store, settings, account_id, project_id, config):
    clean = validate_job_config(store, account_id, project_id, config)
    source = clean['dataset']['source']
    if source['source_type'] != 'upload':
        return {}
    dataset_id = source['dataset_id']
    record = store.require_dataset(account_id, project_id, dataset_id)
    raw = scoped_path(settings, account_id, project_id, dataset_id).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise RuntimeError('Stored dataset failed integrity verification')
    return {dataset_id: {'filename': 'dataset.jsonl', 'content': raw}}


def build_execution_config(store, settings, account_id, project_id, job_id):
    """Return safe worker fields; dispatcher supplies immutable server-owned remote settings."""
    job = store.require_job(account_id, project_id, job_id)
    clean = validate_job_config(store, account_id, project_id, job['config'])
    clean.update(output_name=f'hosted/{account_id}/{project_id}/{job_id}', use_lora=True,
                 push_to_hub=False, use_wandb=False, trust_remote_code=False)
    clean['remote'] = {'enabled': True, 'provider': 'runpod', 'merge_strategy': 'skip',
                       'keep_instance_on_failure': False, 'cloud_type': 'secure'}
    return clean
