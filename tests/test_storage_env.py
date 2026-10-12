"""
Role-named storage settings (STORAGE_INTERNAL_* / STORAGE_EXTERNAL_* /
DATASET_TIER / DATASET_MIRROR / ARTIFACT_*) and their deprecated aliases.

The worker wire format must not change: the control plane still forwards
exactly MERLINA_ARTIFACT_ACCESS_KEY / _SECRET_KEY / _SESSION_TOKEN (plus the
checkpoint timeout), built from the new config, and a worker that only has
those names still finds its credentials.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402

ALL_STORAGE_ENV = [
    "STORAGE_INTERNAL_ENDPOINT", "STORAGE_INTERNAL_BUCKET", "STORAGE_INTERNAL_ACCESS_KEY",
    "STORAGE_INTERNAL_SECRET_KEY", "STORAGE_INTERNAL_REGION", "STORAGE_EXTERNAL_ENDPOINT",
    "STORAGE_EXTERNAL_BUCKET", "STORAGE_EXTERNAL_ACCESS_KEY", "STORAGE_EXTERNAL_SECRET_KEY",
    "STORAGE_EXTERNAL_REGION", "DATASET_TIER", "DATASET_MIRROR", "ARTIFACT_STORE", "ARTIFACT_BUCKET",
    "ARTIFACT_PREFIX", "S3_ENDPOINT_URL", "S3_DATASET_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY",
    "S3_REGION", "R2_ENDPOINT_URL", "R2_DATASET_BUCKET", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_REGION",
    "DATASET_STORE", "PREFER_INTERNAL", "MERLINA_ARTIFACT_STORE", "MERLINA_ARTIFACT_S3_BUCKET",
    "MERLINA_ARTIFACT_S3_PREFIX", "MERLINA_ARTIFACT_S3_ENDPOINT", "MERLINA_ARTIFACT_S3_REGION",
    "MERLINA_ARTIFACT_ACCESS_KEY", "MERLINA_ARTIFACT_SECRET_KEY", "MERLINA_ARTIFACT_SESSION_TOKEN",
]

EXT = "https://acct.r2.cloudflarestorage.com"
LEGACY = dict(
    s3_endpoint_url="http://minio.lan:9000", s3_dataset_bucket="datasets",
    s3_access_key="INT_AK_VALUE", s3_secret_key="INT_SK_VALUE",
    r2_endpoint_url=EXT, r2_dataset_bucket="shared", r2_access_key="EXT_AK_VALUE",
    r2_secret_key="EXT_SK_VALUE", dataset_store="external", prefer_internal=True,
    merlina_artifact_store="s3", merlina_artifact_s3_bucket="shared",
    merlina_artifact_s3_endpoint=EXT, merlina_artifact_s3_prefix="runs",
    merlina_artifact_access_key="EXT_AK_VALUE", merlina_artifact_secret_key="EXT_SK_VALUE",
)
NEW = dict(
    storage_internal_endpoint="http://minio.lan:9000", storage_internal_bucket="datasets",
    storage_internal_access_key="INT_AK_VALUE", storage_internal_secret_key="INT_SK_VALUE",
    storage_external_endpoint=EXT, storage_external_bucket="shared",
    storage_external_access_key="EXT_AK_VALUE", storage_external_secret_key="EXT_SK_VALUE",
    dataset_tier="external", dataset_mirror=True, artifact_store="external", artifact_prefix="runs",
)
SECRETS = ("INT_AK_VALUE", "INT_SK_VALUE", "EXT_AK_VALUE", "EXT_SK_VALUE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ALL_STORAGE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_DEPRECATION_WARNED", False)


def make(**kw):
    return config.Settings(_env_file=None, **kw)


class TestResolution:
    def test_new_names(self):
        s = make(**NEW)
        t = config.tier_settings(s, "external")
        assert (t["endpoint"], t["bucket"], t["region"]) == (EXT, "shared", "auto")
        loc = s.artifact_location()
        assert (loc.kind, loc.tier, loc.bucket, loc.prefix, loc.error) == ("s3", "external", "shared", "runs", None)
        assert s._deprecated_storage_names == []

    def test_legacy_names_resolve_identically(self):
        new, old = make(**NEW), make(**LEGACY)
        for tier in ("internal", "external"):
            assert config.tier_settings(new, tier) == config.tier_settings(old, tier)
        assert (old.dataset_tier, old.dataset_mirror) == ("external", True)
        a, b = new.artifact_location(), old.artifact_location()
        assert (a.kind, a.tier, a.bucket, a.prefix, a.endpoint_url) == (b.kind, b.tier, b.bucket, b.prefix, b.endpoint_url)
        assert config.artifact_credentials(new) == config.artifact_credentials(old)

    def test_defaults(self):
        s = make()
        assert s.artifact_location().kind == "hf_hub"
        assert s.artifact_location().prefix == "merlina/runs"
        assert s.dataset_tier is None and s.dataset_mirror is None

    def test_new_name_wins(self):
        s = make(storage_internal_bucket="new-bucket", s3_dataset_bucket="old-bucket")
        assert s.storage_internal_bucket == "new-bucket"
        assert s._conflicting_storage_names == ["s3_dataset_bucket"]

    def test_env_vars_are_read(self, monkeypatch):
        monkeypatch.setenv("S3_DATASET_BUCKET", "from-old-env")
        monkeypatch.setenv("ARTIFACT_STORE", "internal")
        s = make()
        assert s.storage_internal_bucket == "from-old-env" and s.artifact_store == "internal"

    def test_artifact_bucket_override(self):
        loc = make(**{**NEW, "artifact_bucket": "artifact-bucket"}).artifact_location()
        assert (loc.tier, loc.bucket) == ("external", "artifact-bucket")

    def test_internal_artifact_store(self):
        loc = make(**{**NEW, "artifact_store": "internal"}).artifact_location()
        assert (loc.tier, loc.bucket, loc.endpoint_url) == ("internal", "datasets", "http://minio.lan:9000")

    def test_invalid_and_unconfigured_artifact_store(self):
        assert "must be" in make(artifact_store="s3").artifact_location().error
        assert "needs the external tier" in make(artifact_store="external").artifact_location().error

    def test_repr_hides_credentials(self):
        assert not any(v in repr(make(**NEW).artifact_location()) for v in SECRETS)


class TestLegacyArtifactBlock:
    def test_matching_external_folds_onto_tier(self):
        loc = make(**LEGACY).artifact_location()
        assert (loc.tier, loc.bucket) == ("external", "shared")

    def test_different_bucket_becomes_artifact_bucket(self):
        loc = make(**{**LEGACY, "merlina_artifact_s3_bucket": "artifact-bucket"}).artifact_location()
        assert (loc.tier, loc.bucket) == ("external", "artifact-bucket")

    def test_no_keys_given_uses_tier_keys(self):
        legacy = {k: v for k, v in LEGACY.items() if k not in ("merlina_artifact_access_key", "merlina_artifact_secret_key")}
        assert make(**legacy).artifact_location().tier == "external"

    @pytest.mark.parametrize("override", [
        {"merlina_artifact_access_key": "SCOPED_AK"},
        {"merlina_artifact_session_token": "TOKEN"},
        {"merlina_artifact_s3_endpoint": "https://elsewhere.example.com"},
    ])
    def test_custom_location_kept(self, override):
        loc = make(**{**LEGACY, **override}).artifact_location()
        assert loc.kind == "s3" and loc.tier is None and loc.bucket == "shared"
        assert loc.access_key == override.get("merlina_artifact_access_key", "EXT_AK_VALUE")

    def test_legacy_hf_hub(self):
        assert make(merlina_artifact_store="hf_hub").artifact_location().kind == "hf_hub"


class TestDeprecationWarning:
    def test_single_warning_names_only(self, caplog):
        with caplog.at_level(logging.WARNING, logger="config"):
            make(**LEGACY)
            make(**LEGACY)
        msgs = [r.getMessage() for r in caplog.records if "Deprecated storage settings" in r.getMessage()]
        assert len(msgs) == 1
        msg = msgs[0]
        assert "S3_ENDPOINT_URL -> STORAGE_INTERNAL_ENDPOINT" in msg
        assert "R2_SECRET_KEY -> STORAGE_EXTERNAL_SECRET_KEY" in msg
        assert "DATASET_STORE -> DATASET_TIER" in msg and "PREFER_INTERNAL -> DATASET_MIRROR" in msg
        assert "MERLINA_ARTIFACT_STORE -> ARTIFACT_STORE" in msg
        assert "migrate-env" in msg
        assert not any(v in msg for v in SECRETS + ("minio.lan", "shared", "acct."))

    def test_conflict_reported_without_values(self, caplog):
        with caplog.at_level(logging.WARNING, logger="config"):
            make(storage_internal_bucket="new-bucket", s3_dataset_bucket="old-bucket")
        msg = caplog.records[-1].getMessage()
        assert "S3_DATASET_BUCKET" in msg and "new names win" in msg
        assert "new-bucket" not in msg and "old-bucket" not in msg

    def test_new_names_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="config"):
            make(**NEW)
        assert not [r for r in caplog.records if "Deprecated" in r.getMessage()]

    def test_worker_wire_names_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="config"):
            s = make(merlina_artifact_access_key="W_AK", merlina_artifact_secret_key="W_SK")
        assert not [r for r in caplog.records if "Deprecated" in r.getMessage()]
        # ...and a worker (no ARTIFACT_STORE) still gets its credentials from them
        assert config.artifact_credentials(s) == ("W_AK", "W_SK", None)


class TestWorkerWireFormat:
    """The control plane forwards the same MERLINA_ARTIFACT_* names and values."""

    def _worker_env(self, monkeypatch, settings):
        import src.remote.artifacts as artifacts
        monkeypatch.setattr(config, "settings", settings)
        cfg = SimpleNamespace(hf_token=None, model_dump=lambda: {"base_model": "org/m", "dataset": {}},
                              remote=SimpleNamespace(artifact_repo=None))
        store = artifacts.store_for_job(cfg, "job_1")
        return store.describe(), artifacts.artifact_worker_env(store.describe())

    def test_same_env_from_legacy_and_new(self, monkeypatch):
        d_new, env_new = self._worker_env(monkeypatch, make(**NEW))
        d_old, env_old = self._worker_env(monkeypatch, make(**LEGACY))
        assert d_new == d_old
        assert d_new["bucket"] == "shared" and d_new["prefix"] == "runs/job_1"
        assert env_new == env_old
        assert sorted(env_new) == ["MERLINA_ARTIFACT_ACCESS_KEY", "MERLINA_ARTIFACT_SECRET_KEY",
                                   "MERLINA_CHECKPOINT_TIMEOUT_SECONDS"]
        assert env_new["MERLINA_ARTIFACT_ACCESS_KEY"] == "EXT_AK_VALUE"

    def test_custom_location_forwards_its_own_keys(self, monkeypatch):
        s = make(**{**LEGACY, "merlina_artifact_access_key": "SCOPED_AK",
                    "merlina_artifact_secret_key": "SCOPED_SK", "merlina_artifact_session_token": "TOK"})
        _, env = self._worker_env(monkeypatch, s)
        assert env["MERLINA_ARTIFACT_ACCESS_KEY"] == "SCOPED_AK"
        assert env["MERLINA_ARTIFACT_SESSION_TOKEN"] == "TOK"


class TestRemoteRefusesInternalArtifacts:
    def test_plan_refuses_internal_tier(self, monkeypatch):
        from src.remote.plan import check_remote_compatible
        monkeypatch.setattr(config, "settings", make(**{**NEW, "artifact_store": "internal"}))
        cfg = SimpleNamespace(dataset=None, training_mode="orpo")
        problems = check_remote_compatible(cfg)
        assert any("ARTIFACT_STORE=internal is LAN-only" in p for p in problems)

    def test_plan_accepts_external(self, monkeypatch):
        from src.remote.plan import check_remote_compatible
        monkeypatch.setattr(config, "settings", make(**NEW))
        assert check_remote_compatible(SimpleNamespace(dataset=None, training_mode="orpo")) == []

    def test_local_use_of_internal_is_fine(self, monkeypatch):
        import src.remote.artifacts as artifacts
        monkeypatch.setattr(config, "settings", make(**{**NEW, "artifact_store": "internal"}))
        cfg = SimpleNamespace(hf_token=None, model_dump=lambda: {}, remote=SimpleNamespace(artifact_repo=None))
        assert artifacts.store_for_job(cfg, "j").describe()["bucket"] == "datasets"


class TestDatasetSettings:
    def test_internal_store_config_from_new_and_legacy(self):
        from dataset_handlers.internal_store import InternalStoreConfig
        for s in (make(**NEW), make(**LEGACY)):
            for tier, legacy in (("internal", "s3"), ("external", "r2")):
                a = InternalStoreConfig.from_settings(s, prefix=tier)
                b = InternalStoreConfig.from_settings(s, prefix=legacy)
                assert a == b and a is not None
        # plain objects with only legacy attributes still work
        plain = SimpleNamespace(s3_endpoint_url="http://h:9000", s3_dataset_bucket="b",
                                s3_access_key="k", s3_secret_key="s", s3_region=None)
        assert InternalStoreConfig.from_settings(plain).bucket == "b"

    def test_factory_reads_dataset_tier_and_mirror(self, monkeypatch):
        import dataset_handlers.factory as factory
        monkeypatch.setattr(config, "settings", make(**LEGACY))
        assert factory.active_store_name() == "external" and factory.default_prefer_internal() is True
        monkeypatch.setattr(config, "settings", make(dataset_tier="internal", dataset_mirror=False))
        assert factory.active_store_name() == "internal" and factory.default_prefer_internal() is False
