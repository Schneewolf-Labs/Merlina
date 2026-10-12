"""Durable paid-job dispatch and lifecycle guard, independent of the API process.

Run ``tick`` from one or more supervised dispatchers. A persisted dispatch claim
is never reset automatically: after a crash only lifecycle reconciliation may
continue or close it, so an uncertain provider response cannot start another pod.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING

from .billing import BillingError, BillingUnavailable, BudgetExceeded, StageQuote, config_digest


class ExecutionGuard:
    def __init__(self, billing, job_id, *, rate_check=None):
        self.billing, self.job_id, self.rate_check = billing, job_id, rate_check

    def before_provision(self, job_id, stage_name, spec):
        self.billing.enabled()
        run = self.billing.run(job_id)
        if job_id != self.job_id or run["state"] != "dispatching" or run["cancel_requested"]:
            raise BillingError("Execution is not authorized")
        stage = self._stage(run, stage_name)
        for field in ("gpu_type_id", "gpu_count", "cloud_type", "container_disk_gb", "volume_gb"):
            if getattr(spec, field) != getattr(stage, field):
                raise BillingError("Provider shape differs from approved quote")
        if stage.worker_image and spec.image != stage.worker_image:
            raise BillingError("Worker image differs from approved quote")
        if self.rate_check is None:
            raise BillingUnavailable("Live provider rate revalidation is required")
        rate, observed_at = self.rate_check(stage, spec)
        if (type(rate) is not int or rate <= 0 or rate > stage.rate_minor_per_hour
                or not 0 <= self.billing.clock()-observed_at <= 60):
            raise BillingError("Provider rate changed or is stale; do not provision")

    @staticmethod
    def _stage(run, name):
        for stage in run["quote"]["stages"]:
            if stage["name"] == name:
                return StageQuote(**stage)
        raise BillingError("Stage is not included in approved quote")

    def _cost(self, run, resources, *, final=False):
        total = run["quote"]["fixed_minor"] if resources else 0
        now = self.billing.clock()
        for resource in resources:
            stage = self._stage(run, resource["stage_name"])
            start = resource.get("started_at")
            end = resource.get("terminated_at")
            if start is None or (final and end is None):
                raise BillingError("Usage or termination evidence is missing")
            seconds = max(0, (end if end is not None else now)-start)
            # Never underbill a higher observed provider rate. Overage beyond the
            # approved reservation is separately recorded and absorbed by SWL.
            rate = resource.get("cost_per_hr")
            if rate is None:
                raise BillingError("Authoritative provider usage rate missing")
            actual_rate = int((Decimal(str(rate))*100).to_integral_value(rounding=ROUND_CEILING))
            if actual_rate <= 0:
                raise BillingError("Invalid provider usage rate")
            data = dict(stage.__dict__, rate_minor_per_hour=max(actual_rate, stage.rate_minor_per_hour))
            total += StageQuote(**data).charge(seconds)
            if not final and end is None and seconds >= stage.boot_seconds+stage.runtime_seconds:
                self.billing.record_failure(self.job_id, "budget_exhausted")
                raise BudgetExceeded("Approved stage runtime exhausted; cleanup required")
        return total

    def usage(self, job_id, resources):
        if job_id != self.job_id:
            raise BillingError("Guard job mismatch")
        run = self.billing.run(job_id)
        spent = self._cost(run, resources)
        # Project each live resource to the end of its shutdown allowance and
        # round elapsed+shutdown once, like the reservation. Rounding them
        # separately would forfeit up to a billing unit of approved runtime.
        now = self.billing.clock()
        projected = [r if r.get("terminated_at") is not None else
                     dict(r, terminated_at=max(now, r["started_at"])+self._stage(run, r["stage_name"]).shutdown_seconds)
                     for r in resources]
        if self._cost(run, projected) > run["amount"]:
            self.billing.record_failure(self.job_id, "budget_exhausted")
            raise BudgetExceeded("Approved budget exhausted; cleanup required")
        return spent

    def settle(self, job_id, resources, *, terminal=True):
        if job_id != self.job_id:
            raise BillingError("Guard job mismatch")
        if not terminal or any(r.get("terminated_at") is None for r in resources):
            raise BillingError("Cleanup is unconfirmed; reservation retained")
        run = self.billing.run(job_id)
        return self.billing.settle(job_id, self._cost(run, resources, final=True), cleanup_confirmed=True)


class ExecutionService:
    """Server-installed callbacks own credentials, worker capabilities and plans.

    ``quote_builder(job)`` returns (StageQuote list, fixed_minor), from live full
    provider rates and measured workload records. ``launch`` runs only after the
    durable claim; ``reconcile`` must NEVER provision a fresh uncertain attempt.
    Both receive (job, ExecutionGuard). The lifecycle service owns provider cleanup.
    """
    def __init__(self, store, billing, *, quote_builder=None, launch=None,
                 reconcile=None, cancel=None, rate_check=None, recover=None):
        self.store, self.billing = store, billing
        self.quote_builder, self.launch = quote_builder, launch
        self.reconcile, self.cancel_callback, self.rate_check = reconcile, cancel, rate_check
        self.recover = recover

    def quote(self, account_id, project_id, job_id):
        self.billing.enabled()
        job = self.store.require_job(account_id, project_id, job_id)
        if not all((self.quote_builder, self.launch, self.reconcile, self.cancel_callback, self.rate_check)):
            raise BillingUnavailable("Managed provider execution has not been configured")
        stages, fixed_minor = self.quote_builder(job)
        return self.billing.quote(account_id, project_id, job_id, job["config"], stages, fixed_minor=fixed_minor)

    def approve(self, account_id, project_id, job_id, quote_id):
        job = self.store.require_job(account_id, project_id, job_id)
        if not all((self.launch, self.reconcile, self.cancel_callback, self.rate_check)):
            raise BillingUnavailable("Managed provider execution has not been configured")
        result = self.billing.reserve(account_id, project_id, job_id, quote_id, job["config"])
        # A retry or concurrent dispatcher must not overwrite terminal/running
        # execution status with the reservation's earlier state.
        with self.store.transaction() as db:
            db.execute("UPDATE jobs SET status='approved',updated_at=? WHERE id=? AND account_id=? AND project_id=? AND status='draft' AND EXISTS (SELECT 1 FROM billing_runs WHERE job_id=? AND state='approved')",
                       (self.billing.clock(), job_id, account_id, project_id, job_id))
        self.billing.event(account_id, project_id, job_id, "execution.approved", {"quote_id": quote_id, "reserved_minor": result["reserved_minor"]})
        return result

    def cancel(self, account_id, project_id, job_id):
        job = self.store.require_job(account_id, project_id, job_id)
        result = self.billing.cancel(account_id, project_id, job_id)
        if self.cancel_callback:
            self.cancel_callback(job)
        self.billing.event(account_id, project_id, job_id, "execution.cancel_requested", {})
        return result

    def tick(self):
        """One scan, suitable for a supervisor. Persistent failures stay visible."""
        for run in self.billing.pending():
            jid = run["job_id"]
            try:
                job = self.store.require_job(run["account_id"], run["project_id"], jid)
                approved = self.billing.run(jid)
                if config_digest(job["config"]) != approved["quote"]["config_hash"]:
                    raise BillingError("Approved configuration changed")
                guard = ExecutionGuard(self.billing, jid, rate_check=self.rate_check)
                if self.billing.claim(jid):
                    # Cancel before provisioning is a confirmed zero-resource run.
                    if self.billing.run(jid)["cancel_requested"]:
                        guard.settle(jid, [], terminal=True)
                        self.store.update_job(run["account_id"], run["project_id"], jid, status="cancelled")
                    elif self.launch:
                        self.billing.event(run["account_id"], run["project_id"], jid, "execution.dispatching", {})
                        self.launch(job, guard)
                    else:
                        raise BillingUnavailable("Execution dispatcher not configured")
                else:
                    if run["cancel_requested"] and self.cancel_callback:
                        self.cancel_callback(job)
                    if self.reconcile:
                        self.reconcile(job, guard)
            except Exception as exc:
                # Do not release a reservation on a network failure: a provider
                # may have created a billable resource before the response failed.
                with self.store.transaction() as db:
                    db.execute("UPDATE billing_runs SET last_error=? WHERE job_id=?", (type(exc).__name__, jid))

        # Result publication has its own durable queue: billing may already be
        # settled when archiving fails or the controller restarts.
        if self.recover:
            self.recover()
