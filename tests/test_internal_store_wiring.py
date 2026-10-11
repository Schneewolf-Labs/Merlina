"""
Phase 1.5 wiring: the DatasetSource model accepts internal sources, the
TrainingConfig validator treats an swl:// uri as a specified source (rather than
clobbering it), and create_loader_from_config resolves the store from app
settings when none is passed explicitly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import dataset_handlers.factory as factory
from dataset_handlers.loaders import InternalStoreLoader
from dataset_handlers.internal_store import InternalStore, InternalStoreConfig


# ── DatasetSource model ──────────────────────────────────────────────────────
def test_dataset_source_accepts_internal_uri():
    from merlina import DatasetSource

    src = DatasetSource(source_type="internal", uri="swl://athanorlite-dpo@2026-10-08")
    assert src.source_type == "internal"
    assert src.uri == "swl://athanorlite-dpo@2026-10-08"


# ── TrainingConfig validator does not clobber an internal source ─────────────
def test_training_config_keeps_internal_source():
    """An internal source with no dataset_name is left intact (not overwritten
    with a huggingface source built from dataset_name)."""
    from merlina import TrainingConfig

    cfg = TrainingConfig.model_validate({
        "output_name": "m",
        "dataset": {
            "source": {"source_type": "internal", "uri": "swl://athanorlite-dpo"},
            "format": {"format_type": "chatml"},
        },
    })
    src = cfg.dataset.source
    assert src.source_type == "internal"
    assert src.uri == "swl://athanorlite-dpo"  # untouched


def test_training_config_internal_vs_dataset_name_is_ambiguous():
    """Setting both dataset_name and an internal source is rejected as ambiguous,
    the same guard that applies to local_file — proving uri now counts as a
    'specified' source rather than being silently clobbered."""
    from merlina import TrainingConfig

    with pytest.raises(Exception) as exc:
        TrainingConfig.model_validate({
            "output_name": "m",
            "dataset_name": "athanorlite-dpo",
            "dataset": {
                "source": {"source_type": "internal", "uri": "swl://athanorlite-dpo"},
                "format": {"format_type": "chatml"},
            },
        })
    assert "disagree" in str(exc.value)


# ── default_internal_store() from settings ───────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_store_cache():
    factory._reset_store_cache()
    yield
    factory._reset_store_cache()


def test_default_store_none_when_unconfigured(monkeypatch):
    import config
    for attr in ("s3_endpoint_url", "s3_dataset_bucket", "s3_access_key", "s3_secret_key"):
        monkeypatch.setattr(config.settings, attr, None, raising=False)
    assert factory.default_internal_store() is None


def test_default_store_built_when_configured(monkeypatch):
    import config
    monkeypatch.setattr(config.settings, "s3_endpoint_url", "http://127.0.0.1:9000", raising=False)
    monkeypatch.setattr(config.settings, "s3_dataset_bucket", "datasets", raising=False)
    monkeypatch.setattr(config.settings, "s3_access_key", "ak", raising=False)
    monkeypatch.setattr(config.settings, "s3_secret_key", "sk", raising=False)
    store = factory.default_internal_store()
    assert isinstance(store, InternalStore)
    assert store.config.bucket == "datasets"
    # cached: a second call returns the same instance
    assert factory.default_internal_store() is store


def test_factory_internal_uses_default_store(monkeypatch):
    import config
    monkeypatch.setattr(config.settings, "s3_endpoint_url", "http://127.0.0.1:9000", raising=False)
    monkeypatch.setattr(config.settings, "s3_dataset_bucket", "datasets", raising=False)
    monkeypatch.setattr(config.settings, "s3_access_key", "ak", raising=False)
    monkeypatch.setattr(config.settings, "s3_secret_key", "sk", raising=False)
    # no store= passed; it should be resolved from settings
    loader = factory.create_loader_from_config({"source_type": "internal", "uri": "swl://d@r1"})
    assert isinstance(loader, InternalStoreLoader)
    assert loader.rev == "r1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
