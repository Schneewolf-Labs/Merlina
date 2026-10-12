"""Operator-only catalog writer for observed pilot workloads; never an HTTP tool.

Run with --help after measuring an equivalent workload and confirming the image,
shape, billing granularity, runtime allowance, and fixed storage/transfer charge.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

from .billing import config_digest


def record_measurement(catalog_path, config, *, measurement_id, measured_seconds,
                       runtime_seconds, boot_seconds, shutdown_seconds,
                       billing_granularity_seconds, fixed_minor, gpu_type_id,
                       gpu_count, container_disk_gb, worker_image):
    if not re.fullmatch(r'.+@sha256:[a-f0-9]{64}', worker_image):
        raise ValueError('Worker image must include its full SHA-256 digest')
    if not measurement_id or not gpu_type_id:
        raise ValueError('Measurement evidence and GPU shape are required')
    if any(type(n) is not int or n <= 0 for n in (measured_seconds, runtime_seconds, boot_seconds,
            shutdown_seconds, billing_granularity_seconds, gpu_count, container_disk_gb)):
        raise ValueError('Runtime, shape and billing bounds must be positive integers')
    if measured_seconds > runtime_seconds or runtime_seconds > 86400 or type(fixed_minor) is not int or fixed_minor < 0:
        raise ValueError('Invalid runtime or fixed charge')
    entry = dict(measurement_id=measurement_id, measured_seconds=measured_seconds,
                 runtime_seconds=runtime_seconds, boot_seconds=boot_seconds,
                 shutdown_seconds=shutdown_seconds, billing_granularity_seconds=billing_granularity_seconds,
                 fixed_minor=fixed_minor, training_config={'remote': dict(gpu_type_id=gpu_type_id,
                    gpu_count=gpu_count, container_disk_gb=container_disk_gb, worker_image=worker_image)})
    path = Path(catalog_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        digest = config_digest(config)
        data[digest] = entry
        descriptor, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(descriptor, 'w') as output:
                json.dump(data, output, indent=2, sort_keys=True)
                output.write('\n')
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return digest


def main():
    from .settings import HostedSettings
    from .store import HostedStore
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--job-id', required=True)
    for field in ('measurement-id', 'gpu-type-id', 'worker-image'):
        parser.add_argument('--'+field, required=True)
    for field in ('measured-seconds', 'runtime-seconds', 'boot-seconds', 'shutdown-seconds',
                  'billing-granularity-seconds', 'fixed-minor', 'gpu-count', 'container-disk-gb'):
        parser.add_argument('--'+field, type=int, required=True)
    args = vars(parser.parse_args())
    catalog = args.pop('catalog')
    job_id = args.pop('job_id')
    store = HostedStore(HostedSettings.from_env().database_path)
    with store.transaction() as db:
        row = db.execute('SELECT config FROM jobs WHERE id=?', (job_id,)).fetchone()
    if row is None:
        parser.error('Job not found')
    print(record_measurement(catalog, json.loads(row['config']), **args))


if __name__ == '__main__':
    main()
