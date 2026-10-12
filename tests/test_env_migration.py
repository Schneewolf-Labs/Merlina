"""
`merlina config migrate-env`: legacy storage names -> role-named scheme.

Values are compared in memory only; the plan and CLI output must never
contain them. Comments, ordering and unrelated lines survive byte-for-byte,
--write keeps a 600 backup and a 600 file, and a second run has nothing to do.
"""

import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import env_migration as em  # noqa: E402

LEGACY_ENV = """\
# Merlina server settings
HOST=127.0.0.1
RUNPOD_API_KEY=rp_SECRET_RUNPOD
HF_TOKEN="hf_SECRET_TOKEN"

# internal tier
S3_ENDPOINT_URL=http://minio.lan:9000
S3_DATASET_BUCKET=datasets
S3_ACCESS_KEY=INT_AK_SECRET
S3_SECRET_KEY=INT_SK_SECRET
S3_REGION=auto
PREFER_INTERNAL=true

# external tier
export R2_ENDPOINT_URL=https://acct.r2.cloudflarestorage.com
R2_DATASET_BUCKET=shared-bucket
R2_ACCESS_KEY='EXT_AK_SECRET'
R2_SECRET_KEY=EXT_SK_SECRET   # inline comment
DATASET_STORE=internal

# artifacts (duplicate of R2)
MERLINA_ARTIFACT_STORE=s3
MERLINA_ARTIFACT_S3_BUCKET=shared-bucket
MERLINA_ARTIFACT_S3_PREFIX=runs
MERLINA_ARTIFACT_S3_ENDPOINT=https://acct.r2.cloudflarestorage.com/
MERLINA_ARTIFACT_S3_REGION=auto
MERLINA_ARTIFACT_ACCESS_KEY=EXT_AK_SECRET
MERLINA_ARTIFACT_SECRET_KEY=EXT_SK_SECRET
WANDB_API_KEY=wandb_SECRET
"""
VALUES = ("rp_SECRET_RUNPOD", "hf_SECRET_TOKEN", "INT_AK_SECRET", "INT_SK_SECRET",
          "EXT_AK_SECRET", "EXT_SK_SECRET", "wandb_SECRET", "minio.lan", "acct.r2", "shared-bucket")


def write(tmp_path, text, name=".env"):
    p = tmp_path / name
    p.write_text(text)
    os.chmod(p, 0o600)
    return p


def mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


class TestPlan:
    def test_full_mapping_and_collapse(self):
        plan = em.plan_migration(LEGACY_ENV)
        desc = [c.describe() for c in plan.changes]
        for old, new in [("S3_ENDPOINT_URL", "STORAGE_INTERNAL_ENDPOINT"), ("S3_REGION", "STORAGE_INTERNAL_REGION"),
                         ("R2_SECRET_KEY", "STORAGE_EXTERNAL_SECRET_KEY"), ("DATASET_STORE", "DATASET_TIER"),
                         ("PREFER_INTERNAL", "DATASET_MIRROR"), ("MERLINA_ARTIFACT_S3_PREFIX", "ARTIFACT_PREFIX")]:
            assert f"rename  {old} -> {new}" in desc
        assert any(d.startswith("replace MERLINA_ARTIFACT_STORE -> ARTIFACT_STORE") and "external tier" in d for d in desc)
        removed = {c.key for c in plan.changes if c.kind == "remove"}
        assert removed == {"MERLINA_ARTIFACT_S3_BUCKET", "MERLINA_ARTIFACT_S3_ENDPOINT",
                           "MERLINA_ARTIFACT_S3_REGION", "MERLINA_ARTIFACT_ACCESS_KEY",
                           "MERLINA_ARTIFACT_SECRET_KEY"}
        assert not plan.conflicts

    def test_rewrite_preserves_everything_else(self):
        out = em.apply_plan(LEGACY_ENV, em.plan_migration(LEGACY_ENV))
        lines = out.splitlines()
        # unrelated lines and comments, byte-for-byte and in order
        for keep in ("# Merlina server settings", "HOST=127.0.0.1", "RUNPOD_API_KEY=rp_SECRET_RUNPOD",
                     'HF_TOKEN="hf_SECRET_TOKEN"', "# internal tier", "# external tier",
                     "# artifacts (duplicate of R2)", "WANDB_API_KEY=wandb_SECRET"):
            assert keep in lines
        assert lines.index("HOST=127.0.0.1") < lines.index("# internal tier") < lines.index("WANDB_API_KEY=wandb_SECRET")
        # renamed in place with the value text untouched (quotes, export, comments)
        assert "STORAGE_INTERNAL_ENDPOINT=http://minio.lan:9000" in lines
        assert "export STORAGE_EXTERNAL_ENDPOINT=https://acct.r2.cloudflarestorage.com" in lines
        assert "STORAGE_EXTERNAL_ACCESS_KEY='EXT_AK_SECRET'" in lines
        assert "STORAGE_EXTERNAL_SECRET_KEY=EXT_SK_SECRET   # inline comment" in lines
        assert lines.index("STORAGE_INTERNAL_ENDPOINT=http://minio.lan:9000") == LEGACY_ENV.splitlines().index(
            "S3_ENDPOINT_URL=http://minio.lan:9000")
        assert "ARTIFACT_STORE=external" in lines and "ARTIFACT_PREFIX=runs" in lines
        assert not [ln for ln in lines if ln.startswith(("S3_", "R2_", "MERLINA_ARTIFACT_", "export R2_"))]
        assert out.endswith("\n")

    def test_migrated_file_resolves_like_the_original(self, tmp_path, monkeypatch):
        import config
        for k in list(os.environ):
            if k.startswith(("STORAGE_", "S3_", "R2_", "MERLINA_ARTIFACT_", "ARTIFACT_", "DATASET_", "PREFER_")):
                monkeypatch.delenv(k)
        old = write(tmp_path, LEGACY_ENV, "old.env")
        new = write(tmp_path, em.apply_plan(LEGACY_ENV, em.plan_migration(LEGACY_ENV)), "new.env")
        a, b = config.Settings(_env_file=str(old)), config.Settings(_env_file=str(new))
        for tier in ("internal", "external"):
            assert config.tier_settings(a, tier) == config.tier_settings(b, tier)
        assert (a.dataset_tier, a.dataset_mirror) == (b.dataset_tier, b.dataset_mirror) == ("internal", True)
        la, lb = a.artifact_location(), b.artifact_location()
        assert (la.kind, la.tier, la.bucket, la.prefix) == (lb.kind, lb.tier, lb.bucket, lb.prefix) == \
            ("s3", "external", "shared-bucket", "runs")
        assert config.artifact_credentials(a) == config.artifact_credentials(b)
        assert b._deprecated_storage_names == []

    def test_different_artifact_bucket_becomes_artifact_bucket(self):
        text = LEGACY_ENV.replace("MERLINA_ARTIFACT_S3_BUCKET=shared-bucket", "MERLINA_ARTIFACT_S3_BUCKET=artifacts-bucket")
        out = em.apply_plan(text, em.plan_migration(text))
        assert "ARTIFACT_BUCKET=artifacts-bucket" in out and "ARTIFACT_STORE=external" in out

    def test_collapse_onto_internal(self):
        text = (LEGACY_ENV.replace("MERLINA_ARTIFACT_S3_ENDPOINT=https://acct.r2.cloudflarestorage.com/",
                                   "MERLINA_ARTIFACT_S3_ENDPOINT=http://minio.lan:9000")
                .replace("MERLINA_ARTIFACT_ACCESS_KEY=EXT_AK_SECRET", "MERLINA_ARTIFACT_ACCESS_KEY=INT_AK_SECRET")
                .replace("MERLINA_ARTIFACT_SECRET_KEY=EXT_SK_SECRET", "MERLINA_ARTIFACT_SECRET_KEY=INT_SK_SECRET"))
        out = em.apply_plan(text, em.plan_migration(text))
        assert "ARTIFACT_STORE=internal" in out and "ARTIFACT_BUCKET=shared-bucket" in out

    @pytest.mark.parametrize("change", [
        ("MERLINA_ARTIFACT_ACCESS_KEY=EXT_AK_SECRET", "MERLINA_ARTIFACT_ACCESS_KEY=SCOPED_SECRET"),
        ("MERLINA_ARTIFACT_S3_ENDPOINT=https://acct.r2.cloudflarestorage.com/",
         "MERLINA_ARTIFACT_S3_ENDPOINT=https://other.example.com"),
        ("WANDB_API_KEY=wandb_SECRET", "WANDB_API_KEY=wandb_SECRET\nMERLINA_ARTIFACT_SESSION_TOKEN=TOK_SECRET"),
    ])
    def test_custom_location_is_kept(self, change):
        text = LEGACY_ENV.replace(*change)
        plan = em.plan_migration(text)
        out = em.apply_plan(text, plan)
        assert "MERLINA_ARTIFACT_STORE=s3" in out and "ARTIFACT_STORE=external" not in out
        assert "MERLINA_ARTIFACT_S3_BUCKET=shared-bucket" in out
        assert any("custom artifact location" in n for n in plan.notes)
        assert "ARTIFACT_PREFIX=runs" in out  # the prefix rename still applies

    def test_hf_hub_store_is_renamed(self):
        plan = em.plan_migration("MERLINA_ARTIFACT_STORE=hf_hub\n")
        assert em.apply_plan("MERLINA_ARTIFACT_STORE=hf_hub\n", plan) == "ARTIFACT_STORE=hf_hub\n"

    def test_bare_worker_credentials_left_alone(self):
        text = "MERLINA_ARTIFACT_ACCESS_KEY=W_SECRET\nMERLINA_ARTIFACT_SECRET_KEY=W2_SECRET\n"
        plan = em.plan_migration(text)
        assert not plan.has_changes and plan.notes

    def test_conflicts(self):
        text = "S3_DATASET_BUCKET=old-bucket\nSTORAGE_INTERNAL_BUCKET=new-bucket\nS3_REGION=auto\nSTORAGE_INTERNAL_REGION=auto\n"
        plan = em.plan_migration(text)
        assert len(plan.conflicts) == 1 and "S3_DATASET_BUCKET" in plan.conflicts[0]
        out = em.apply_plan(text, plan)
        assert "S3_DATASET_BUCKET=old-bucket" in out and "STORAGE_INTERNAL_BUCKET=new-bucket" in out
        assert "S3_REGION" not in out  # same value: the duplicate old name goes
        text2 = "ARTIFACT_STORE=external\nMERLINA_ARTIFACT_STORE=s3\n"
        assert em.plan_migration(text2).conflicts

    def test_admin_names_from_the_storage_branch(self):
        out = em.apply_plan("S3_ADMIN_ACCESS_KEY=a_SECRET\nR2_ADMIN_SECRET_KEY=b_SECRET\n",
                            em.plan_migration("S3_ADMIN_ACCESS_KEY=a_SECRET\nR2_ADMIN_SECRET_KEY=b_SECRET\n"))
        assert out == "STORAGE_INTERNAL_ADMIN_ACCESS_KEY=a_SECRET\nSTORAGE_EXTERNAL_ADMIN_SECRET_KEY=b_SECRET\n"

    def test_report_never_contains_values(self):
        plans = [em.plan_migration(LEGACY_ENV),
                 em.plan_migration(LEGACY_ENV.replace("MERLINA_ARTIFACT_ACCESS_KEY=EXT_AK_SECRET",
                                                      "MERLINA_ARTIFACT_ACCESS_KEY=SCOPED_SECRET")),
                 em.plan_migration("S3_DATASET_BUCKET=old-bucket\nSTORAGE_INTERNAL_BUCKET=new-bucket\n")]
        for plan in plans:
            for write_flag in (False, True):
                text = em.format_plan(plan, write=write_flag)
                assert not any(v in text for v in VALUES + ("SCOPED_SECRET", "old-bucket", "new-bucket"))


class TestWrite:
    def test_dry_run_writes_nothing(self, tmp_path):
        p = write(tmp_path, LEGACY_ENV)
        plan, backup = em.migrate_file(str(p))
        assert plan.has_changes and backup is None and p.read_text() == LEGACY_ENV
        assert not list(tmp_path.glob(".env.bak-*"))

    def test_write_backup_modes_and_idempotent(self, tmp_path):
        p = write(tmp_path, LEGACY_ENV)
        os.chmod(p, 0o644)  # even a too-open file ends up 600
        plan, backup = em.migrate_file(str(p), write=True, now=datetime(2026, 1, 2, 3, 4, 5))
        assert backup == f"{p}.bak-20260102-030405"
        assert Path(backup).read_text() == LEGACY_ENV
        assert mode(backup) == 0o600 and mode(p) == 0o600
        assert "ARTIFACT_STORE=external" in p.read_text()
        again, backup2 = em.migrate_file(str(p), write=True)
        assert not again.has_changes and backup2 is None
        assert "nothing to do" in em.format_plan(again, write=True)
        assert len(list(tmp_path.glob(".env.bak-*"))) == 1

    def test_cli(self, tmp_path, capsys):
        import merlina_cli
        p = write(tmp_path, LEGACY_ENV)
        assert merlina_cli.main(["config", "migrate-env", str(p)]) == 0
        out = capsys.readouterr().out
        assert "dry run" in out and "--write" in out and p.read_text() == LEGACY_ENV
        assert not any(v in out for v in VALUES)
        assert merlina_cli.main(["config", "migrate-env", str(p), "--write"]) == 0
        out = capsys.readouterr().out
        assert "applied" in out and "backup written" in out and not any(v in out for v in VALUES)
        assert merlina_cli.main(["config", "migrate-env", str(p)]) == 0
        assert "nothing to do" in capsys.readouterr().out
        assert merlina_cli.main(["config", "migrate-env", str(tmp_path / "missing.env")]) == 2

    def test_cli_conflict_exit_code(self, tmp_path, capsys):
        import merlina_cli
        p = write(tmp_path, "S3_DATASET_BUCKET=a-bucket\nSTORAGE_INTERNAL_BUCKET=b-bucket\n")
        assert merlina_cli.main(["config", "migrate-env", str(p)]) == 1
        assert "conflicts" in capsys.readouterr().out
