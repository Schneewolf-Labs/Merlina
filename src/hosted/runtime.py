"""Hosted runtime composition and optional standalone supervised dispatcher."""
from __future__ import annotations

import argparse
import logging
import time
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


def configure_runtime(app):
    """Refuse incomplete paid configuration before accepting hosted traffic."""
    if getattr(app.state, 'execution', None) is not None:
        return app.state.execution
    settings = app.state.settings
    if not settings.paid_execution_enabled:
        return None
    from config import settings as server
    from src.remote.s3_artifacts import S3ArtifactStore, artifact_metadata
    from .artifact_broker import ArtifactGrants, build_scoped_store
    from .billing import BillingService, BillingUnavailable
    from .remote_runtime import install_runtime

    billing = BillingService(app.state.store, settings)
    billing.enabled()
    origin = urlsplit(settings.public_url or '')
    if (origin.scheme != 'https' or not origin.hostname or origin.username or origin.password
            or origin.path not in ('', '/') or origin.query or origin.fragment):
        raise BillingUnavailable('Configure MERLINA_HOSTED_PUBLIC_URL as an HTTPS origin')
    if server.merlina_artifact_store != 's3' or not server.merlina_artifact_s3_bucket:
        raise BillingUnavailable('Hosted execution requires configured S3 artifact storage')
    if server.merlina_artifact_s3_endpoint and not server.merlina_artifact_s3_endpoint.startswith('https://'):
        raise BillingUnavailable('Hosted S3 artifact endpoint requires HTTPS')

    def raw_store(job=None):
        prefix = server.merlina_artifact_s3_prefix
        if job is not None:
            prefix = '/'.join((prefix, job['account_id'], job['project_id'], job['id']))
        return S3ArtifactStore(server.merlina_artifact_s3_bucket, prefix,
            endpoint_url=server.merlina_artifact_s3_endpoint,
            region_name=server.merlina_artifact_s3_region,
            metadata=artifact_metadata(job['config']) if job else {})

    class LazySigner:
        def sign(self, *args, **kwargs):
            return raw_store().fs.sign(*args, **kwargs)

    app.state.artifact_grants = ArtifactGrants(settings.database_path)
    app.state.artifact_signer = LazySigner()

    def scoped_store(job):
        run = billing.run(job['id'])
        duration = sum(s['boot_seconds'] + s['runtime_seconds'] + s['shutdown_seconds']
                       for s in run['quote']['stages'])
        return build_scoped_store(raw_store(job), db_path=settings.database_path,
            public_url=settings.public_url, account_id=job['account_id'],
            project_id=job['project_id'], job_id=job['id'],
            expires_at=time.time() + duration + 300, max_bytes=settings.max_artifact_bytes)

    return install_runtime(app, scoped_store)


def tick(app):
    execution = getattr(app.state, 'execution', None)
    if execution:
        execution.tick()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    from .app import create_hosted_app
    app = create_hosted_app()
    if configure_runtime(app) is None:
        parser.error('Paid execution is disabled; nothing to dispatch')
    while True:
        tick(app)
        if args.once:
            return 0
        time.sleep(app.state.settings.dispatch_interval_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
