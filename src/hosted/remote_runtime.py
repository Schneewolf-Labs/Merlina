"""Operator-configured adapter-only RunPod pilot binding.

No public request can choose credentials, worker images, volumes or artifact
paths. A local measurement catalog pins full execution profiles to exact draft
configuration hashes. A configured HTTPS price service supplies current TOTAL
instance rates (GPU + disks); RunPod live GPU offers are cross-checked as well.
Without either evidence source paid launch fails closed.
"""
from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
import time
import zipfile
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from types import SimpleNamespace

from .billing import BillingError, BillingUnavailable, BillingService, StageQuote, config_digest
from .execution import ExecutionGuard, ExecutionService

# A dispatch claim whose worker never took lifecycle ownership is released after
# this long; a slower launch in another dispatcher then finds the claim taken.
UNLAUNCHED_DISPATCH_GRACE_SECONDS = 600


@dataclass(frozen=True)
class RuntimeSettings:
    measurement_catalog_path: str
    pricing_url: str
    runpod_api_key: str

    @classmethod
    def from_env(cls):
        return cls(os.getenv("MERLINA_MEASUREMENT_CATALOG", ""),
                   os.getenv("MERLINA_LIVE_PRICING_URL", ""),
                   os.getenv("RUNPOD_API_KEY", ""))


class ConfigNode(SimpleNamespace):
    """Only operator profiles and schema-validated hosted fields enter this tree."""
    def model_dump(self):
        def plain(value):
            if isinstance(value, ConfigNode):
                return {k: plain(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [plain(v) for v in value]
            return value
        return plain(self)


def config_node(value):
    if isinstance(value, dict):
        return ConfigNode(**{k: config_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [config_node(v) for v in value]
    return value


class ManagedRemoteRuntime:
    def __init__(self, store, settings, billing, artifact_store_factory, *, runtime_settings=None,
                 provider=None, fetch_prices=None):
        from src.job_manager import JobManager
        from src.remote.lifecycle import RemoteLifecycle
        from src.remote.providers import get_provider
        self.store, self.settings, self.billing = store, settings, billing
        self.config = runtime_settings or RuntimeSettings.from_env()
        if not self.config.measurement_catalog_path or not self.config.pricing_url.startswith("https://"):
            raise BillingUnavailable("Measured catalog and HTTPS live total-price source are required")
        self.provider = provider or get_provider("runpod", api_key=self.config.runpod_api_key)
        self.fetch_prices = fetch_prices or self._fetch_prices
        self.artifact_store_factory = artifact_store_factory
        self.jobs = JobManager(str(settings.data_dir / "remote-jobs.sqlite3"))
        self.lifecycle = RemoteLifecycle(self.jobs.db_path)
        self.active = set()
        self.lock = threading.Lock()
        with self.store.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS hosted_result_sync (job_id TEXT PRIMARY KEY, pending INTEGER NOT NULL DEFAULT 1)")
            db.execute("INSERT OR IGNORE INTO hosted_result_sync(job_id) SELECT job_id FROM billing_runs WHERE state!='approved'")

    def _fetch_prices(self):
        import requests
        response = requests.get(self.config.pricing_url, timeout=20, allow_redirects=False)
        response.raise_for_status()
        return response.json()

    def profile(self, job):
        from .services import build_execution_config
        catalog = json.loads(Path(self.config.measurement_catalog_path).read_text())
        entry = catalog.get(config_digest(job["config"]))
        if not entry or not entry.get("measurement_id"):
            raise BillingUnavailable("This configuration has no operator-validated workload measurement")
        execution = {"use_4bit": True, "gradient_checkpointing": True, **copy.deepcopy(entry["training_config"])}
        safe = build_execution_config(self.store, self.settings, job["account_id"], job["project_id"], job["id"])
        remote = {"max_cost_per_hr": None, "artifact_repo": None, **execution.get("remote", {})}
        execution.update(safe)
        remote.update(safe["remote"])
        remote["max_runtime_hours"] = (entry["boot_seconds"]+entry["runtime_seconds"]+entry["shutdown_seconds"])/3600
        execution["remote"] = remote
        execution["hf_token"] = None
        execution["wandb_key"] = None
        execution["export_gguf"] = False
        execution["trust_remote_code"] = False
        if not remote.get("worker_image") or "@sha256:" not in remote["worker_image"]:
            raise BillingUnavailable("Hosted worker image must be pinned by digest")
        if not remote.get("gpu_type_id") or not remote.get("gpu_count") or not remote.get("container_disk_gb"):
            raise BillingUnavailable("Measured worker shape and disk size must be pinned")
        remote["volume_gb"] = 0
        # Hosted first pilot is adapter-only. Pricing must cover all resource stages
        # before expanding the binding to a second merge instance.
        return entry, execution

    def price(self, shape):
        feed = self.fetch_prices()
        if feed.get("currency") != self.billing.currency:
            raise BillingUnavailable("Price-feed currency mismatch")
        observed = feed.get("observed_at")
        if type(observed) is not int or not 0 <= self.billing.clock()-observed <= 60:
            raise BillingUnavailable("Total provider prices are stale")
        matches = [r for r in feed.get("rates", []) if all(r.get(k) == shape[k] for k in
                   ("provider", "gpu_type_id", "gpu_count", "cloud_type", "container_disk_gb", "volume_gb"))]
        if len(matches) != 1 or type(matches[0].get("rate_minor_per_hour")) is not int:
            raise BillingUnavailable("No current total provider price for this exact instance")
        rate = matches[0]["rate_minor_per_hour"]
        offers = self.provider.list_gpu_offers()  # No static fallback on failure.
        offer = next((o for o in offers if o.gpu_type_id == shape["gpu_type_id"] and o.available), None)
        gpu_rate = offer.price_per_hr(shape["cloud_type"]) if offer else None
        if gpu_rate is None or rate < int((Decimal(str(gpu_rate))*100*shape["gpu_count"]).to_integral_value(rounding=ROUND_CEILING)):
            raise BillingUnavailable("Total rate is inconsistent with live RunPod GPU pricing")
        return rate, observed

    def quote_builder(self, job):
        entry, execution = self.profile(job)
        remote = execution["remote"]
        shape = {k: remote[k] for k in ("provider", "gpu_type_id", "gpu_count", "cloud_type", "container_disk_gb", "volume_gb")}
        rate, observed = self.price(shape)
        stage = StageQuote(name="train", **shape, rate_minor_per_hour=rate, rate_observed_at=observed,
            measurement_id=entry["measurement_id"], measured_seconds=entry["measured_seconds"],
            runtime_seconds=entry["runtime_seconds"], boot_seconds=entry["boot_seconds"],
            shutdown_seconds=entry["shutdown_seconds"], worker_image=remote["worker_image"],
            execution_config_hash=config_digest(execution),
            billing_granularity_seconds=entry["billing_granularity_seconds"])
        # fixed_minor is the disclosed prepaid charge for storage/transfer. Agent
        # usage is excluded: no agent calls happen in this execution adapter.
        return [stage], entry["fixed_minor"]

    def rate_check(self, stage, spec):
        return self.price({k: getattr(stage, k) for k in
            ("provider", "gpu_type_id", "gpu_count", "cloud_type", "container_disk_gb", "volume_gb")})

    def launch(self, job, guard):
        from .services import resolve_uploaded_datasets
        from src.remote.orchestrator import run_training_remote
        try:
            _, execution = self.profile(job)
            quote = self.billing.run(job["id"])["quote"]
            if config_digest(execution) != quote["stages"][0]["execution_config_hash"]:
                raise BillingError("Execution profile changed after approval")
            artifacts = self.artifact_store_factory(job)
            if not callable(getattr(artifacts, "worker_environment", None)):
                raise BillingUnavailable("Scoped per-job artifact capability required")
            uploads = resolve_uploaded_datasets(self.store, self.settings, job["account_id"], job["project_id"], job["config"])
            if self.jobs.get_job(job["id"]) is None:
                self.jobs.create_job(job["id"], execution)
            self.store.update_job(job["account_id"], job["project_id"], job["id"], status="running")
            with self.store.transaction() as db:
                db.execute("INSERT OR REPLACE INTO hosted_result_sync VALUES(?,1)", (job["id"],))
        except Exception as exc:
            # Provider runner has not been invoked. Persist that proof and release
            # the reservation atomically; this is not an ambiguous create refund.
            self.billing.abort_before_launch(job["id"], type(exc).__name__)
            self.store.update_job(job["account_id"], job["project_id"], job["id"], status="failed",
                                  result={"error": "Launch validation failed; no compute was charged."})
            raise
        with self.lock:
            self.active.add(job["id"])
        def work():
            try:
                run_training_remote(job["id"], config_node(execution), self.jobs, uploads,
                                    provider=self.provider, store=artifacts, execution_guard=guard,
                                    lifecycle=self.lifecycle)
            finally:
                with self.lock:
                    self.active.discard(job["id"])
                self.sync(job)
        threading.Thread(target=work, name="hosted-"+job["id"], daemon=True).start()

    def recover_results(self):
        with self.store.transaction() as db:
            rows = [dict(r) for r in db.execute("SELECT j.id,j.account_id,j.project_id FROM hosted_result_sync s JOIN jobs j ON j.id=s.job_id WHERE s.pending=1")]
        for row in rows:
            try:
                job = self.store.require_job(row["account_id"], row["project_id"], row["id"])
                self.sync(job)
            except Exception:
                # Leave the durable work item pending across restarts. Internal
                # details remain in operator logs, never customer-visible result.
                logging.getLogger(__name__).exception("Result synchronization pending for job %s", row["id"])

    def sync(self, job):
        record = self.jobs.get_job(job["id"])
        if record is None:
            if self.billing.run(job["id"])["state"] == "settled":
                with self.store.transaction() as db:
                    db.execute("UPDATE hosted_result_sync SET pending=0 WHERE job_id=?", (job["id"],))
            return
        current = self.store.require_job(job["account_id"], job["project_id"], job["id"])
        result = {"error": "The run failed. Contact support with the job ID." if record.error else None,
                  "artifacts_available": False, "artifact_status": "not_ready"}
        for key, value in (("progress", record.progress), ("loss", record.loss),
                           ("eval_loss", record.eval_loss), ("steps", record.current_step),
                           ("current_step", record.current_step), ("total_steps", record.total_steps)):
            if type(value) in (int, float) and math.isfinite(value):
                result[key] = value
        resources = self.lifecycle.resources(job["id"])
        if resources:
            starts = [r.get("started_at") for r in resources]
            ends = [r.get("terminated_at") if r.get("terminated_at") is not None else self.billing.clock() for r in resources]
            if all(type(v) in (int, float) and math.isfinite(v) for v in starts+ends):
                result["runtime_seconds"] = max(0, max(ends)-min(starts))
        if record.status == "failed":
            code = self.billing.run(job["id"]).get("failure_code")
            result["error_code"] = code if code == "budget_exhausted" else "training_failed"
        if record.status == "completed":
            try:
                source = Path(record.output_dir) if record.output_dir else None
                if source is not None and source.is_dir():
                    self.publish_artifact(job, source)
                else:
                    with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as temporary:
                        self.artifact_store_factory(job).pull_dir("adapter", Path(temporary))
                        self.publish_artifact(job, Path(temporary))
                result.update(artifacts_available=True, artifact_status="ready")
            except Exception:
                result.update(artifact_status="pending", artifact_error="Artifact preparation is pending; recovery will retry.")
                logging.getLogger(__name__).exception("Artifact preparation pending for job %s", job["id"])
        self.store.update_job(job["account_id"], job["project_id"], job["id"], status=record.status, result=result)
        if current["status"] != record.status or current.get("result", {}).get("artifact_status") != result["artifact_status"]:
            self.billing.event(job["account_id"], job["project_id"], job["id"], "execution.status",
                               {"status": record.status, "artifact_status": result["artifact_status"]})
        terminal = record.status in {"completed", "failed", "stopped", "cancelled"}
        if terminal and result["artifact_status"] != "pending":
            with self.store.transaction() as db:
                db.execute("UPDATE hosted_result_sync SET pending=0 WHERE job_id=?", (job["id"],))

    def publish_artifact(self, job, source):
        from .services import scoped_path
        target = scoped_path(self.settings, job["account_id"], job["project_id"], job["id"], kind="jobs") / "adapter.zip"
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not target.exists():
            if not source.is_dir() or not any(p.is_file() for p in source.rglob("*")):
                raise BillingError("Completed adapter has no verified files")
            descriptor, temp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
            os.close(descriptor)
            try:
                with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED) as archive:
                    for path in sorted(source.rglob("*")):
                        if path.is_symlink():
                            raise BillingError("Artifact archive cannot contain symlinks")
                        if path.is_file():
                            info = zipfile.ZipInfo(path.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                            info.external_attr = 0o100600 << 16
                            with path.open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                                shutil.copyfileobj(reader, writer)
                os.replace(temp, target)
            finally:
                Path(temp).unlink(missing_ok=True)
        digest = self.file_hash(target)
        artifact_id = hashlib.sha256((job["id"]+":adapter.zip").encode()).hexdigest()[:32]
        with self.store.transaction() as db:
            existing = db.execute("SELECT sha256 FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
            if existing and existing[0] != digest:
                raise BillingError("Artifact cache differs from registered result")
            db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?,?,?)",
                (artifact_id, job["project_id"], job["id"], "adapter.zip", "adapter", digest, target.stat().st_size, time.time()))
        return target

    @staticmethod
    def file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as reader:
            for block in iter(lambda: reader.read(1024*1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def download(self, artifact):
        from fastapi.responses import FileResponse
        from .services import scoped_path
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=? AND project_id=?", (artifact["job_id"], artifact["project_id"])).fetchone()
        if row is None:
            raise BillingError("Artifact job is missing")
        job = self.store.require_job(row["account_id"], row["project_id"], row["id"])
        path = scoped_path(self.settings, job["account_id"], job["project_id"], job["id"], kind="jobs") / "adapter.zip"
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as temporary:
                self.artifact_store_factory(job).pull_dir("adapter", Path(temporary))
                path = self.publish_artifact(job, Path(temporary))
        if self.file_hash(path) != artifact["sha256"] or path.stat().st_size != artifact["size_bytes"]:
            raise BillingError("Artifact cache failed integrity verification")
        return FileResponse(path, filename="adapter.zip", media_type="application/zip")

    def release_unlaunched(self, job):
        """Settle at zero a dispatch that crashed before any provider call.

        Provisioning requires ``lifecycle.intent``, which requires a running
        lifecycle run owned by the worker; ``lifecycle.claim`` is insert-only.
        Taking that claim ourselves therefore proves no worker can provision.
        """
        jid = job["id"]
        run = self.billing.run(jid)
        if run["state"] != "dispatching" or jid in self.active:
            return False
        from src.remote.lifecycle import RemoteAlreadyClaimed
        reason = "Dispatch interrupted before compute started"
        remote = self.lifecycle.run(jid)
        if remote is None:
            if self.billing.clock()-(run["claimed_at"] or 0) < UNLAUNCHED_DISPATCH_GRACE_SECONDS:
                return False
            try:
                owner = self.lifecycle.claim(jid, 0)
            except RemoteAlreadyClaimed:
                return False  # A worker claimed it meanwhile; normal reconciliation owns it.
            self.lifecycle.finish(jid, owner, "failed", reason)
        elif remote["state"] == "running":
            return False
        if any(a["job_id"] == jid for a in self.lifecycle.attempts()) or self.lifecycle.resources(jid):
            return False
        self.billing.abort_before_launch(jid, "dispatch_interrupted")
        if self.jobs.get_job(jid) is not None:
            self.jobs.update_job(jid, status="failed", error=reason)
        self.store.update_job(job["account_id"], job["project_id"], jid, status="failed",
                              result={"error": reason + "; no compute was charged."})
        with self.store.transaction() as db:
            db.execute("UPDATE hosted_result_sync SET pending=0 WHERE job_id=?", (jid,))
        self.billing.event(job["account_id"], job["project_id"], jid, "execution.status",
                           {"status": "failed", "artifact_status": "not_ready"})
        return True

    def reconcile(self, job, guard):
        from src.remote.cleanup import reconcile_once
        if self.release_unlaunched(job):
            return
        reconcile_once(self.lifecycle, {"runpod": self.provider}, job_manager=self.jobs,
                       execution_guard_factory=lambda jid: ExecutionGuard(self.billing, jid, rate_check=self.rate_check))
        self.sync(job)

    def cancel(self, job):
        self.jobs.update_job(job["id"], stop_requested=True)


def install_runtime(app, artifact_store_factory, *, runtime_settings=None):
    """Install only after operator setup; call from hosted app factory/lifespan."""
    billing = BillingService(app.state.store, app.state.settings)
    runtime = ManagedRemoteRuntime(app.state.store, app.state.settings, billing,
                                   artifact_store_factory, runtime_settings=runtime_settings)
    app.state.billing = billing
    app.state.remote_runtime = runtime
    app.state.artifact_download = runtime.download
    app.state.execution = ExecutionService(app.state.store, billing,
        quote_builder=runtime.quote_builder, launch=runtime.launch, reconcile=runtime.reconcile,
        cancel=runtime.cancel, rate_check=runtime.rate_check, recover=runtime.recover_results)
    return app.state.execution
