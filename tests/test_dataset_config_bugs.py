"""
Regression tests for the dataset-config / job-lifecycle bug batch:

- M-1: dataset_name silently ignored by non-diffusion modes
- M-2: pre-flight passed configs whose dataset never resolves
- M-3: API restart orphaned the worker (worker_pid persistence + reattach)
- M-4: worker error handler could itself raise (covered structurally)
- M-5: GET /jobs/{id}/config returned hf_token + wandb_key in plaintext
- M-6: default dataset was a hardcoded real repo
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from merlina import TrainingConfig, DatasetConfig, DatasetSource  # noqa: E402
from src.job_manager import JobManager  # noqa: E402
from src.preflight_checks import PreflightValidator  # noqa: E402
from src.worker_reattach import (  # noqa: E402
    _is_our_worker,
    _pid_alive,
    reattach_orphaned_workers,
)


# ============================================================================
# M-6: no hardcoded default dataset
# ============================================================================

class TestNoDefaultDataset:
    def test_default_dataset_source_has_no_repo(self):
        config = TrainingConfig(output_name="test")
        assert config.dataset.source.repo_id is None
        assert config.dataset.source.source_type == "huggingface"

    def test_default_dataset_config_has_no_repo(self):
        config = DatasetConfig()
        assert config.source.repo_id is None

    def test_dataset_source_defaults_are_not_shared(self):
        a = TrainingConfig(output_name="a", dataset_name="user/set-a", training_mode="sft")
        b = TrainingConfig(output_name="b")
        assert a.dataset.source.repo_id == "user/set-a"
        assert b.dataset.source.repo_id is None

    def test_preflight_errors_on_missing_repo_id(self):
        validator = PreflightValidator()
        config = TrainingConfig(output_name="test")
        validator._check_dataset_config(config)
        assert any("No dataset configured" in e for e in validator.errors)


# ============================================================================
# M-1: dataset_name honored (or loudly rejected) for text modes
# ============================================================================

class TestDatasetNameForTextModes:
    def test_dataset_name_maps_into_source_for_sft(self):
        config = TrainingConfig(
            output_name="test",
            training_mode="sft",
            dataset_name="user/my-corpus",
            dataset_split="validation",
        )
        assert config.dataset.source.source_type == "huggingface"
        assert config.dataset.source.repo_id == "user/my-corpus"
        assert config.dataset.source.split == "validation"

    def test_dataset_name_maps_for_orpo_default_split(self):
        config = TrainingConfig(
            output_name="test",
            training_mode="orpo",
            dataset_name="user/my-corpus",
        )
        assert config.dataset.source.repo_id == "user/my-corpus"
        assert config.dataset.source.split == "train"

    def test_dataset_name_fills_empty_explicit_dataset(self):
        """An explicit dataset block that only sets format still gets the
        dataset_name mapped into its (unset) source."""
        config = TrainingConfig(
            output_name="test",
            training_mode="sft",
            dataset_name="user/my-corpus",
            dataset={"format": {"format_type": "chatml"}},
        )
        assert config.dataset.source.repo_id == "user/my-corpus"
        assert config.dataset.format.format_type == "chatml"

    def test_conflicting_dataset_name_and_source_rejected(self):
        with pytest.raises(ValueError, match="ambiguous"):
            TrainingConfig(
                output_name="test",
                training_mode="sft",
                dataset_name="user/corpus-a",
                dataset={
                    "source": {
                        "source_type": "huggingface",
                        "repo_id": "user/corpus-b",
                    }
                },
            )

    def test_matching_dataset_name_and_source_accepted(self):
        config = TrainingConfig(
            output_name="test",
            training_mode="sft",
            dataset_name="user/corpus",
            dataset={
                "source": {
                    "source_type": "huggingface",
                    "repo_id": "user/corpus",
                }
            },
        )
        assert config.dataset.source.repo_id == "user/corpus"

    def test_dataset_name_untouched_for_diffusion(self):
        config = TrainingConfig(
            output_name="test",
            training_mode="diffusion_sdxl",
            dataset_name="user/image-corpus",
        )
        # Diffusion runner reads dataset_name directly; dataset.source must
        # not be rewritten (and no conflict error raised).
        assert config.dataset_name == "user/image-corpus"
        assert config.dataset.source.repo_id is None

    def test_dataset_jsonl_path_rejected_for_text_modes(self):
        with pytest.raises(ValueError, match="diffusion"):
            TrainingConfig(
                output_name="test",
                training_mode="sft",
                dataset_jsonl_path="/data/foo.jsonl",
            )

    def test_dataset_jsonl_path_allowed_for_diffusion(self):
        config = TrainingConfig(
            output_name="test",
            training_mode="diffusion_sdxl",
            dataset_jsonl_path="/data/foo.jsonl",
        )
        assert config.dataset_jsonl_path == "/data/foo.jsonl"

    def test_retry_roundtrip_keeps_mapped_source(self):
        """model_dump() -> re-validate (the /jobs/{id}/retry path) must not
        raise a conflict for a config whose dataset_name was auto-mapped."""
        config = TrainingConfig(
            output_name="test",
            training_mode="sft",
            dataset_name="user/my-corpus",
        )
        again = TrainingConfig(**config.model_dump())
        assert again.dataset.source.repo_id == "user/my-corpus"


# ============================================================================
# M-2: pre-flight fails datasets that don't resolve
# ============================================================================

class TestPreflightDatasetResolution:
    def _config(self, repo="user/real-corpus"):
        return TrainingConfig(
            output_name="test",
            training_mode="sft",
            dataset={
                "source": {"source_type": "huggingface", "repo_id": repo}
            },
        )

    def test_unresolvable_repo_is_an_error(self):
        from huggingface_hub.utils import RepositoryNotFoundError

        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.dataset_info.side_effect = RepositoryNotFoundError(
                "nope", response=MagicMock()
            )
            validator = PreflightValidator()
            validator._check_dataset_config(self._config())
        assert any("does not resolve" in e for e in validator.errors)

    def test_gated_repo_is_an_error(self):
        from huggingface_hub.utils import GatedRepoError

        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.dataset_info.side_effect = GatedRepoError(
                "gated", response=MagicMock()
            )
            validator = PreflightValidator()
            validator._check_dataset_config(self._config())
        assert any("gated" in e for e in validator.errors)

    def test_network_failure_only_warns(self):
        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.dataset_info.side_effect = ConnectionError("offline")
            validator = PreflightValidator()
            validator._check_dataset_config(self._config())
        assert not validator.errors
        assert any("Could not verify" in w for w in validator.warnings)

    def test_resolvable_repo_passes(self):
        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.dataset_info.return_value = MagicMock()
            validator = PreflightValidator()
            validator._check_dataset_config(self._config())
        assert not validator.errors

    def test_additional_sources_are_checked(self):
        from huggingface_hub.utils import RepositoryNotFoundError

        config = self._config()
        config.dataset.additional_sources = [
            DatasetSource(source_type="huggingface", repo_id="user/extra-corpus")
        ]

        def _info(repo_id, **kwargs):
            if repo_id == "user/extra-corpus":
                raise RepositoryNotFoundError("nope", response=MagicMock())
            return MagicMock()

        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.dataset_info.side_effect = _info
            validator = PreflightValidator()
            validator._check_dataset_config(config)
        assert any("user/extra-corpus" in e for e in validator.errors)

    def test_stray_dataset_name_mismatch_is_an_error(self):
        """Defense in depth: a config object whose dataset_name disagrees with
        dataset.source (bypassing Pydantic) must not pass pre-flight."""
        config = self._config()
        object.__setattr__(config, "dataset_name", "user/other-corpus")
        with patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.dataset_info.return_value = MagicMock()
            validator = PreflightValidator()
            validator._check_dataset_config(config)
        assert any("dataset_name" in e for e in validator.errors)


# ============================================================================
# M-5: secrets never leave via the job-config endpoint
# ============================================================================

class TestJobConfigSecretRedaction:
    def test_get_job_config_strips_secrets(self, tmp_path):
        import asyncio
        import merlina

        jm = JobManager(db_path=str(tmp_path / "jobs.db"))
        jm.create_job(
            "job_secret",
            {
                "output_name": "m",
                "hf_token": "hf_supersecret",
                "wandb_key": "wb_supersecret",
            },
        )
        with patch.object(merlina, "job_manager", jm):
            result = asyncio.run(merlina.get_job_config("job_secret"))

        assert "hf_token" not in result["config"]
        assert "wandb_key" not in result["config"]
        assert result["config"]["output_name"] == "m"
        # The stored config keeps the token so /jobs/{id}/retry still works.
        assert jm.get_job("job_secret").config["hf_token"] == "hf_supersecret"


# ============================================================================
# M-3: worker pid persistence + restart reconciliation
# ============================================================================

class TestWorkerReattach:
    def test_worker_pid_roundtrip_and_clear(self, tmp_path):
        jm = JobManager(db_path=str(tmp_path / "jobs.db"))
        jm.create_job("job_a", {"output_name": "m"})

        jm.update_job("job_a", worker_pid=4242, progress_file="/data/p.jsonl")
        job = jm.get_job("job_a")
        assert job.worker_pid == 4242
        assert job.progress_file == "/data/p.jsonl"

        # Explicit None clears; omitting leaves unchanged.
        jm.update_job("job_a", status="completed")
        assert jm.get_job("job_a").worker_pid == 4242
        jm.update_job("job_a", worker_pid=None)
        assert jm.get_job("job_a").worker_pid is None

    def test_pid_alive(self):
        assert _pid_alive(os.getpid())
        # Spawn and reap a child so its pid is definitely gone.
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        assert not _pid_alive(proc.pid)

    def test_is_our_worker_checks_cmdline(self):
        job_id = "job_reattach_test_123"
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys, time; time.sleep(30)", "--job-id", job_id]
        )
        try:
            assert _is_our_worker(proc.pid, job_id)
            assert not _is_our_worker(proc.pid, "job_some_other")
        finally:
            proc.kill()
            proc.wait()

    def test_dead_worker_job_marked_failed(self, tmp_path):
        jm = JobManager(db_path=str(tmp_path / "jobs.db"))
        jm.create_job("job_dead", {"output_name": "m"})
        jm.update_job("job_dead", status="training", worker_pid=None)

        result = reattach_orphaned_workers(jm)
        assert "job_dead" in result["failed"]
        job = jm.get_job("job_dead")
        assert job.status == "failed"
        assert "restarted" in job.error

    def test_queued_job_marked_failed_after_restart(self, tmp_path):
        jm = JobManager(db_path=str(tmp_path / "jobs.db"))
        jm.create_job("job_q", {"output_name": "m"})
        jm.update_job("job_q", status="queued")

        result = reattach_orphaned_workers(jm)
        assert "job_q" in result["failed"]
        assert jm.get_job("job_q").status == "failed"

    def test_terminal_jobs_untouched(self, tmp_path):
        jm = JobManager(db_path=str(tmp_path / "jobs.db"))
        for job_id, status in [("job_c", "completed"), ("job_f", "failed"), ("job_s", "stopped")]:
            jm.create_job(job_id, {"output_name": "m"})
            jm.update_job(job_id, status=status)

        result = reattach_orphaned_workers(jm)
        assert result["failed"] == []
        assert result["reattached"] == []
        assert jm.get_job("job_c").status == "completed"

    def test_live_worker_is_reattached_and_finalized(self, tmp_path):
        """A job whose worker is alive gets a monitor instead of being failed;
        when the worker later dies without a terminal status, the monitor
        marks the job failed and clears the pid."""
        jm = JobManager(db_path=str(tmp_path / "jobs.db"))
        job_id = "job_live_worker"
        run_dir = tmp_path / "worker_runs" / job_id
        run_dir.mkdir(parents=True)
        progress_file = run_dir / "progress.jsonl"
        progress_file.write_text("")

        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys, time; time.sleep(60)", "--job-id", job_id]
        )
        try:
            jm.create_job(job_id, {"output_name": "m"})
            jm.update_job(
                job_id,
                status="training",
                worker_pid=proc.pid,
                progress_file=str(progress_file),
            )

            result = reattach_orphaned_workers(jm)
            assert job_id in result["reattached"]
            assert result["failed"] == []
            # Still training — monitor attached, job not failed.
            assert jm.get_job(job_id).status == "training"

            proc.kill()
            proc.wait()

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                job = jm.get_job(job_id)
                if job.status == "failed":
                    break
                time.sleep(0.25)
            job = jm.get_job(job_id)
            assert job.status == "failed"
            assert "exited without reporting" in job.error
            assert job.worker_pid is None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
