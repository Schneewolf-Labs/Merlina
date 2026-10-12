"""Bounded durable publication at Grimoire's completed checkpoint-save hook.

This module imports no training stack. Grimoire dispatches callbacks by method
name, so the callback need not inherit its heavyweight package's base class.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .artifacts import store_from_description
from .s3_artifacts import artifact_metadata

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 120


class DurableCheckpointCallback:
    def __init__(self, job_id, job_manager, description, metadata=None, *, timeout=DEFAULT_TIMEOUT_SECONDS, runner=None):
        self.job_id, self.job_manager = job_id, job_manager
        self.description, self.metadata = description, metadata or {}
        self.timeout = max(1, min(float(timeout), 900))
        self.runner = runner or subprocess.run

    def _record(self, state, step, error=None):
        if self.job_manager is None:
            return
        job = self.job_manager.get_job(self.job_id)
        metrics = dict(job.metrics or {}) if job else {}
        previous = metrics.get("checkpoint_publication", {})
        record = {"state": state, "step": step, "artifact": "checkpoint"}
        if state == "committed":
            record["last_committed_step"] = step
        elif "last_committed_step" in previous:
            record["last_committed_step"] = previous["last_committed_step"]
        if error:
            record["error"] = error
        metrics["checkpoint_publication"] = record
        self.job_manager.update_job(self.job_id, metrics=metrics)

    def on_save(self, trainer, path):
        # save_state writes each rank's RNG/optimizer state. Every rank installs
        # this callback and joins before rank 0 snapshots, then waits for the
        # bounded publication before any rank trains or rotates old files.
        accelerator = trainer.accelerator
        accelerator.wait_for_everyone()
        try:
            if not accelerator.is_main_process:
                return
            step = int(trainer.global_step)
            try:
                self._record("publishing", step)
                with tempfile.TemporaryDirectory(prefix="merlina-checkpoint-") as temp:
                    request = Path(temp) / "request.json"
                    request.write_text(json.dumps({"store": self.description, "metadata": self.metadata,
                                                   "path": str(Path(path).resolve()), "step": step}))
                    request.chmod(0o600)
                    # subprocess.run kills and reaps the uploader on timeout.
                    # Its scratch files live under our temp dir for cleanup.
                    # An in-flight final PUT may have committed despite a lost
                    # acknowledgement; only a verified success is announced.
                    self.runner([sys.executable, "-m", "src.remote.checkpoints", str(request)],
                                timeout=self.timeout, check=True,
                                env={**os.environ, "TMPDIR": temp},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._record("committed", step)
            except subprocess.TimeoutExpired:
                self._failure(step, "Checkpoint publication timed out; previous committed checkpoint remains available")
            except Exception:
                # Provider exceptions may contain credentials; expose a fixed
                # diagnostic and leave final output publication to the worker.
                self._failure(step, "Checkpoint publication failed; previous committed checkpoint remains available")
        finally:
            accelerator.wait_for_everyone()

    def _failure(self, step, message):
        logger.warning("Job %s: %s", self.job_id, message)
        try:
            self._record("failed", step, message)
        except Exception:
            logger.warning("Could not persist checkpoint publication status for %s", self.job_id)


def remote_checkpoint_callback(job_id, job_manager):
    """Enable only for the matching remote train payload; local jobs unchanged."""
    raw = os.environ.get("MERLINA_REMOTE_JOB_B64")
    if not raw:
        return None
    payload = json.loads(base64.b64decode(raw))
    if payload.get("job_id") != job_id or payload.get("stage") != "train":
        return None
    # Local shared-volume/HF contracts do not provide immutable verified
    # checkpoints. Periodic durability is enabled for the S3 storage contract.
    if payload.get("artifact_store", {}).get("kind") != "s3":
        return None
    from config import settings
    return DurableCheckpointCallback(job_id, job_manager, payload["artifact_store"],
                                     artifact_metadata(payload.get("config", {})),
                                     timeout=settings.merlina_checkpoint_timeout_seconds)


def publish_checkpoint(request, *, store=None):
    """Upload one already-complete checkpoint; used in the isolated child."""
    store = store or store_from_description(request["store"], token=os.environ.get("HF_TOKEN"))
    metadata = dict(request.get("metadata") or {})
    metadata["checkpoint"] = {"step": request["step"], "format": "accelerate.save_state",
                              "resume": "backend-specific; restore before loading state"}
    store.metadata = metadata
    store.push_dir(Path(request["path"]), "checkpoint")


if __name__ == "__main__":
    publish_checkpoint(json.loads(Path(sys.argv[1]).read_text()))
