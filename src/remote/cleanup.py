"""Run separately from the API: python -m src.remote.cleanup --db data/jobs.db.

Uses the same durable database and server-side provider credentials. Never
provisions or reattaches a worker. Inventory matches exact persisted intent
labels, not a user-controllable generic name prefix.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from .lifecycle import RemoteLifecycle, terminate_confirmed

logger = logging.getLogger(__name__)


def reconcile_once(lifecycle, providers, *, job_manager=None, execution_guard_factory=None,
                   stop_grace_seconds=None, stop_worker=None):
    """Reconcile expired ownership, ambiguous creations, and termination.

    Idempotent across concurrent/restarted reconcilers. Failed API calls retain
    pending records and durable alerts; an empty inventory does not resolve an
    ambiguous creation (provider listings may lag).
    """
    if stop_grace_seconds is None:
        from config import settings
        stop_grace_seconds = settings.remote_stop_grace_minutes * 60
    expired = lifecycle.expire_runs()
    for job_id in expired:
        if job_manager:
            job_manager.update_job(job_id, status="failed", error=lifecycle.run(job_id)["error"])

    attempts = {a["label"]: a for a in lifecycle.attempts()}
    for name, provider in providers.items():
        try:
            inventory = provider.list_instances()
            lifecycle.alert(f"inventory:{name}", None)
        except Exception as exc:
            lifecycle.alert(f"inventory:{name}", f"{name} inventory unavailable ({type(exc).__name__}); scan incomplete")
            inventory = []
        matched = {}
        for instance in inventory:
            label = instance.raw.get("name", "")
            attempt = attempts.get(label)
            if not attempt or attempt["provider"] != name:
                # A database restore may lose ownership records. Alert for manual
                # review rather than delete resources without their intent.
                if label.startswith(f"merlina-{lifecycle.namespace}-"):
                    lifecycle.alert(f"unowned:{name}:{instance.instance_id}",
                                    f"Resource {name}/{instance.instance_id} has no provisioning intent; review manually")
                continue
            lifecycle.record_instance(attempt["job_id"], attempt["stage_name"], label, instance)
            matched.setdefault(label, []).append(instance)
        for label, instances in matched.items():
            attempt = attempts[label]
            if len(instances) > 1:
                lifecycle.fail_run(attempt["job_id"], "Multiple provider resources share a provisioning intent; cleanup required")

    for resource in lifecycle.resources():
        if resource["cleanup_status"] == "terminated":
            continue
        name, instance_id = resource["provider"], resource["instance_id"]
        run = lifecycle.run(resource["job_id"]) if resource["job_id"] else None
        attempt = attempts.get(resource["label"])
        job = job_manager.get_job(resource["job_id"]) if job_manager and resource["job_id"] else None
        cancelled = job is not None and job.stop_requested
        deleted = job_manager is not None and job is None
        deadline_expired = not run or run["deadline"] <= lifecycle.clock()
        retained = resource["cleanup_status"] == "retained" and not deadline_expired and not cancelled and not deleted
        due = (not run or run["state"] != "running" or deadline_expired or cancelled or deleted
               or resource["cleanup_status"] == "pending"
               or (attempt and attempt["state"] in ("completed", "failed")))
        if cancelled:
            stop_deadline = lifecycle.request_cancel(resource["job_id"], stop_grace_seconds)
            stage_finished = attempt and attempt["state"] in ("completed", "failed")
            if (lifecycle.clock() < stop_deadline and not stage_finished
                    and resource["cleanup_status"] != "pending"):
                # Give checkpoints the same persisted window as the controller.
                # Read a scoped worker token from provider metadata if available;
                # never persist it in lifecycle records or expose it in alerts.
                provider = providers.get(name)
                if provider:
                    try:
                        if stop_worker:
                            stop_worker(resource)
                        else:
                            _signal_stop(provider, instance_id)
                    except Exception:
                        logger.warning("Could not forward cancellation to %s/%s; cleanup remains scheduled", name, instance_id)
                continue
        if not due or retained:
            continue
        if name not in providers:
            lifecycle.alert(f"provider:{name}", f"No {name} credentials configured; cleanup pending")
            continue
        lifecycle.alert(f"provider:{name}", None)
        confirmed = terminate_confirmed(lifecycle, providers[name], instance_id)
        lifecycle.alert(f"resource:{name}:{instance_id}", None if confirmed else
                        f"Resource {name}/{instance_id} termination unconfirmed; may still be billing")

    for label, attempt in attempts.items():
        run = lifecycle.run(attempt["job_id"])
        resources = [r for r in lifecycle.resources(attempt["job_id"]) if r["label"] == label]
        if run and run["state"] != "running" and attempt["state"] in ("provisioning", "provisioning_unknown"):
            if resources and all(r["cleanup_status"] == "terminated" for r in resources):
                lifecycle.stage_state(attempt["job_id"], attempt["stage_name"], "failed",
                                      error="Interrupted provisioning reconciled and terminated")
                lifecycle.alert(f"ambiguous:{label}", None)
            else:
                lifecycle.alert(f"ambiguous:{label}", "Provisioning outcome unknown; retain reservation and review provider inventory")

    job_ids = {a["job_id"] for a in lifecycle.attempts()} | set(expired)
    for job_id in job_ids:
        run = lifecycle.run(job_id)
        resources = lifecycle.resources(job_id)
        if job_manager:
            job = job_manager.get_job(job_id)
            if job:
                changes = {"metrics": {**(job.metrics or {}), "remote_resources": resources}}
                if run and run["state"] == "failed" and job.status not in ("completed", "stopped", "cancelled"):
                    changes.update(status="failed", error=run["error"])
                job_manager.update_job(job_id, **changes)
        attempts_for_job = [a for a in lifecycle.attempts() if a["job_id"] == job_id]
        if (execution_guard_factory and run and run["state"] != "running"
                and all(a["state"] not in ("provisioning", "provisioning_unknown") for a in attempts_for_job)
                and all(r["cleanup_status"] == "terminated" for r in resources)):
            try:
                execution_guard_factory(job_id).settle(job_id, resources)
                lifecycle.alert(f"settlement:{job_id}", None)
            except Exception as exc:
                lifecycle.alert(f"settlement:{job_id}", f"Settlement pending ({type(exc).__name__})")
    return {"expired_jobs": expired, "alerts": lifecycle.alerts(), "resources": lifecycle.resources()}


def reconcile_startup(job_manager, settings, *, provider_factory=None):
    """Best-effort startup pass; local-only installs never contact a provider."""
    api_key = getattr(settings, "runpod_api_key", None)
    if not api_key:
        return None
    lifecycle = RemoteLifecycle(job_manager.db_path)
    if not lifecycle.attempts():
        return None
    if provider_factory is None:
        from .providers import get_provider
        provider_factory = get_provider
    provider = provider_factory("runpod", api_key=api_key)
    return reconcile_once(lifecycle, {"runpod": provider}, job_manager=job_manager,
                          stop_grace_seconds=getattr(settings, "remote_stop_grace_minutes", 15) * 60)


def _signal_stop(provider, instance_id):
    from .worker_client import HttpWorkerClient, WORKER_PORT
    instance = provider.get_instance(instance_id)
    env = instance.raw.get("env") or {}
    token = env.get("MERLINA_WORKER_TOKEN") if isinstance(env, dict) else None
    url = provider.proxy_url(instance, WORKER_PORT) if token else None
    if token and url:
        HttpWorkerClient(url, token).request_stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./data/jobs.db")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 1 or args.interval > 60:
        parser.error("--interval must be between 1 and 60 seconds")
    from config import settings
    from src.job_manager import JobManager
    from .providers import get_provider
    lifecycle = RemoteLifecycle(args.db)
    providers = {"runpod": get_provider("runpod", api_key=settings.runpod_api_key)}
    while True:
        try:
            result = reconcile_once(lifecycle, providers, job_manager=JobManager(args.db))
            print(json.dumps(result), flush=True)
            if args.once:
                return 1 if result["alerts"] else 0
        except Exception:
            logger.exception("Cleanup pass failed; retrying, resources may still be billing")
            if args.once:
                return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
