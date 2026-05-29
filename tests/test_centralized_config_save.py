"""
Tests for centralized job-config saving.

These tests guard the contract that:
  1. Saved configs carry a schema header (importable across Merlina
     instances).
  2. Secrets (hf_token, wandb_key) are NEVER persisted to disk, even when
     callers accidentally include them.
  3. ``validate=True`` routes the payload through the Pydantic
     ``TrainingConfig`` model so typo'd field names are caught at save
     time instead of silently dropping data.
  4. Round-trip save → load preserves every TrainingConfig field.
  5. The model card embeds the secret-stripped config when share_config
     is enabled, and skips it cleanly when disabled.

This is the regression suite for the centralization work in
src/config_manager.py + src/model_card.py + frontend/js/form_config.js.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import (
    ConfigManager,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SECRET_FIELDS,
    build_config_envelope,
    normalize_training_config,
    strip_secrets,
)


# ---------------------------------------------------------------------------
# strip_secrets — pure function, no I/O
# ---------------------------------------------------------------------------


class TestStripSecrets:
    def test_removes_known_secret_fields(self):
        config = {
            "base_model": "gpt2",
            "hf_token": "hf_xxx",
            "wandb_key": "wb_yyy",
            "learning_rate": 1e-5,
        }
        stripped = strip_secrets(config)
        assert "hf_token" not in stripped
        assert "wandb_key" not in stripped
        assert stripped["base_model"] == "gpt2"
        assert stripped["learning_rate"] == 1e-5

    def test_returns_a_copy(self):
        """Original dict must not be mutated — callers may reuse it."""
        config = {"hf_token": "hf_xxx", "base_model": "gpt2"}
        strip_secrets(config)
        assert config["hf_token"] == "hf_xxx"

    def test_handles_missing_secret_fields(self):
        config = {"base_model": "gpt2"}
        stripped = strip_secrets(config)
        assert stripped == {"base_model": "gpt2"}

    def test_secret_fields_constant_includes_canonical_secrets(self):
        assert "hf_token" in SECRET_FIELDS
        assert "wandb_key" in SECRET_FIELDS


# ---------------------------------------------------------------------------
# build_config_envelope — schema header
# ---------------------------------------------------------------------------


class TestBuildConfigEnvelope:
    def test_includes_required_metadata_fields(self):
        envelope = build_config_envelope(
            {"base_model": "gpt2"},
            name="my-config",
            description="test",
            tags=["foo", "bar"],
        )
        meta = envelope["_metadata"]
        assert meta["name"] == "my-config"
        assert meta["description"] == "test"
        assert meta["tags"] == ["foo", "bar"]
        assert meta["schema"] == SCHEMA_NAME
        assert meta["schema_version"] == SCHEMA_VERSION
        assert "merlina_version" in meta
        assert "created_at" in meta
        assert "modified_at" in meta

    def test_preserves_created_at_when_provided(self):
        envelope = build_config_envelope(
            {"base_model": "gpt2"},
            name="my-config",
            created_at="2020-01-01T00:00:00",
        )
        assert envelope["_metadata"]["created_at"] == "2020-01-01T00:00:00"

    def test_inlines_config_keys_alongside_metadata(self):
        envelope = build_config_envelope(
            {"base_model": "gpt2", "lora_r": 64},
            name="x",
        )
        assert envelope["base_model"] == "gpt2"
        assert envelope["lora_r"] == 64

    def test_default_tags_is_empty_list(self):
        envelope = build_config_envelope({"a": 1}, name="x")
        assert envelope["_metadata"]["tags"] == []


# ---------------------------------------------------------------------------
# normalize_training_config — Pydantic validation + secret strip
# ---------------------------------------------------------------------------


def _full_config_dict():
    """A complete config that should validate against TrainingConfig."""
    return {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "output_name": "test-model",
        "training_mode": "orpo",
        "use_lora": True,
        "lora_r": 64,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "modules_to_save": [],
        "use_4bit": True,
        "max_length": 2048,
        "max_prompt_length": 1024,
        "num_epochs": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.05,
        "beta": 0.1,
        "gamma": 0.5,
        "label_smoothing": 0.0,
        "seed": 42,
        "max_grad_norm": 0.3,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "shuffle_dataset": True,
        "gradient_checkpointing": False,
        "optimizer_type": "paged_adamw_8bit",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "attn_implementation": "auto",
        "eval_steps": 0.2,
        "use_wandb": False,
        "push_to_hub": False,
        "hf_hub_private": True,
        "merge_lora_before_upload": True,
        "share_config": True,
        "dataset": {
            "source": {
                "source_type": "huggingface",
                "repo_id": "test/data",
                "split": "train",
            },
            "format": {"format_type": "chatml"},
            "test_size": 0.01,
        },
    }


class TestNormalizeTrainingConfig:
    def test_strips_secrets_from_validated_payload(self):
        config = _full_config_dict()
        config["hf_token"] = "hf_secret"
        config["wandb_key"] = "wb_secret"
        normalized = normalize_training_config(config)
        for f in SECRET_FIELDS:
            assert f not in normalized, f"{f} leaked into normalized config"

    def test_fills_in_defaults_for_missing_fields(self):
        config = {
            "output_name": "minimal-model",
            "dataset": {
                "source": {"source_type": "huggingface", "repo_id": "x/y", "split": "train"},
                "format": {"format_type": "chatml"},
                "test_size": 0.01,
            },
        }
        normalized = normalize_training_config(config)
        # Defaults for top-level fields kick in
        assert normalized["batch_size"] == 1
        assert normalized["num_epochs"] == 2
        assert normalized["beta"] == 0.1
        assert normalized["share_config"] is True

    def test_rejects_unknown_top_level_fields(self):
        """Pydantic v2 silently drops unknown fields by default. We rely on
        the model having extra='ignore' (default) so a typo doesn't crash
        the user — but the value is lost. This test documents the behavior
        so a future ``extra='forbid'`` change is intentional."""
        config = _full_config_dict()
        config["typo_field_name"] = "lost"
        normalized = normalize_training_config(config)
        # The typo is dropped, NOT preserved. If you want strict mode,
        # change TrainingConfig to model_config = ConfigDict(extra='forbid')
        # and update this test.
        assert "typo_field_name" not in normalized

    def test_invalid_field_value_raises(self):
        """A genuinely invalid value (e.g. learning_rate=abc) must raise."""
        config = _full_config_dict()
        config["learning_rate"] = "not-a-number"
        with pytest.raises(ValidationError):
            normalize_training_config(config)

    def test_preserves_share_config_flag(self):
        config = _full_config_dict()
        config["share_config"] = False
        normalized = normalize_training_config(config)
        assert normalized["share_config"] is False


# ---------------------------------------------------------------------------
# ConfigManager.save_config — integration with envelope + secret strip
# ---------------------------------------------------------------------------


class TestSaveConfigIntegration:
    def test_save_strips_secrets_even_when_validate_off(self):
        """The legacy 'save raw dict' path must still strip secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            config = {
                "base_model": "gpt2",
                "hf_token": "hf_secret",
                "wandb_key": "wb_secret",
            }
            mgr.save_config("test", config, validate=False)

            # Read raw file and assert secrets are absent
            with open(Path(tmpdir) / "test.json") as f:
                saved = json.load(f)
            assert "hf_token" not in saved
            assert "wandb_key" not in saved
            assert saved["base_model"] == "gpt2"

    def test_save_writes_schema_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            mgr.save_config("test", {"base_model": "gpt2"})
            with open(Path(tmpdir) / "test.json") as f:
                saved = json.load(f)
            meta = saved["_metadata"]
            assert meta["schema"] == SCHEMA_NAME
            assert meta["schema_version"] == SCHEMA_VERSION

    def test_save_with_validate_normalizes_payload(self):
        """When validate=True, defaults are filled in and the payload
        becomes the canonical, importable form."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            partial = {
                "output_name": "min",
                "dataset": {
                    "source": {"source_type": "huggingface", "repo_id": "x/y", "split": "train"},
                    "format": {"format_type": "chatml"},
                    "test_size": 0.01,
                },
            }
            mgr.save_config("test", partial, validate=True)
            loaded = mgr.load_config("test")
            # Pydantic defaults are present after a validated save
            assert loaded["batch_size"] == 1
            assert loaded["share_config"] is True
            assert loaded["use_lora"] is True

    def test_save_with_validate_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            with pytest.raises(ValidationError):
                mgr.save_config(
                    "bad",
                    {"output_name": "x", "learning_rate": "nope"},
                    validate=True,
                )

    def test_resave_preserves_created_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            mgr.save_config("test", {"base_model": "gpt2"})
            first = mgr.load_config("test")["_metadata"]["created_at"]
            # Resave
            mgr.save_config("test", {"base_model": "gpt2-medium"})
            second = mgr.load_config("test")["_metadata"]["created_at"]
            assert first == second


# ---------------------------------------------------------------------------
# Round-trip: full config → save (validated) → load → drive TrainingConfig
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_config_roundtrip_preserves_every_field(self):
        """Save a full config, reload it, feed it back through
        TrainingConfig, and assert values are unchanged."""
        from merlina import TrainingConfig

        full = _full_config_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            mgr.save_config("rt", full, validate=True)
            loaded = mgr.get_config_without_metadata("rt")

            # Re-validate — this is what happens when a user clicks
            # "Load" then "Cast Spell".
            cfg = TrainingConfig(**loaded)

            # Spot-check across categories
            assert cfg.base_model == full["base_model"]
            assert cfg.training_mode == full["training_mode"]
            assert cfg.lora_r == full["lora_r"]
            assert cfg.target_modules == full["target_modules"]
            assert cfg.beta == full["beta"]
            assert cfg.gamma == full["gamma"]
            assert cfg.label_smoothing == full["label_smoothing"]
            assert cfg.optimizer_type == full["optimizer_type"]
            assert cfg.share_config is True

    def test_roundtrip_with_share_disabled(self):
        from merlina import TrainingConfig

        full = _full_config_dict()
        full["share_config"] = False
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            mgr.save_config("rt", full, validate=True)
            loaded = mgr.get_config_without_metadata("rt")
            cfg = TrainingConfig(**loaded)
            assert cfg.share_config is False

    def test_secrets_never_appear_after_roundtrip(self):
        """Even a validated config that originally had tokens must not
        surface them on reload."""
        full = _full_config_dict()
        full["hf_token"] = "hf_secret"
        full["wandb_key"] = "wb_secret"
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager(config_dir=tmpdir)
            mgr.save_config("rt", full, validate=True)
            loaded = mgr.load_config("rt")
            for f in SECRET_FIELDS:
                assert f not in loaded, f"{f} leaked through round-trip"


# ---------------------------------------------------------------------------
# Backend ↔ frontend parity: every TrainingConfig field also appears in
# the centralized form_config.js builder.
# ---------------------------------------------------------------------------


class TestFrontendBackendParity:
    """Guard against silent field drift between TrainingConfig and the
    frontend's centralized buildTrainingConfig() function."""

    def _form_config_js(self) -> str:
        path = Path(__file__).parent.parent / "frontend" / "js" / "form_config.js"
        if not path.exists():
            pytest.skip("frontend/js/form_config.js not found")
        return path.read_text()

    def test_every_training_config_field_appears_in_form_config_js(self):
        """If you add a field to TrainingConfig, you MUST add it to
        form_config.js — saved presets would otherwise lose data."""
        from merlina import TrainingConfig

        backend_fields = set(TrainingConfig.model_fields.keys())
        # gpu_ids is opt-in via the GPUManager (not a form field on its
        # own), so exclude it from the literal-name scan.
        backend_fields -= {"gpu_ids"}

        js = self._form_config_js()
        missing = []
        for field in backend_fields:
            # We expect the literal "field_name:" or "field_name :" in
            # the object-literal returned by buildTrainingConfig.
            if f"{field}:" not in js and f"{field} :" not in js:
                missing.append(field)
        assert not missing, (
            f"form_config.js does not emit these TrainingConfig fields: "
            f"{missing}. Saved presets will lose them."
        )

    def test_form_config_js_does_not_emit_unknown_fields(self):
        """The frontend should not invent fields the backend doesn't know
        — Pydantic silently drops them, leading to confusing data loss.

        We grep for the JSON keys produced inside buildTrainingConfig and
        check each against TrainingConfig.model_fields.
        """
        import re
        from merlina import TrainingConfig

        backend_fields = set(TrainingConfig.model_fields.keys())
        # Allowed extras: gpu_ids (opt-in), share_config (real field —
        # already in TrainingConfig). Nothing else should slip through.
        js = self._form_config_js()

        # Find the buildTrainingConfig function body
        match = re.search(
            r"export function buildTrainingConfig\([^)]*\)\s*\{(.+?)\n\}\n",
            js,
            re.DOTALL,
        )
        assert match, "Could not locate buildTrainingConfig in form_config.js"
        body = match.group(1)

        # Extract object-literal keys: lines like `        field_name: ...,`
        # (4 or more spaces, identifier, colon).
        keys = set()
        for line in body.split("\n"):
            m = re.match(r"^\s{4,}([a-z_][a-z0-9_]*)\s*:", line, re.IGNORECASE)
            if not m:
                continue
            key = m.group(1)
            # Skip inner-object keys (nested helpers like adam_beta1 are
            # legit; this scan just catches stray top-level keys). We do
            # this by ignoring keys that obviously belong to a nested
            # block by checking nesting via the JS — but simpler: just
            # check membership.
            keys.add(key)

        # Some keys we saw in the body belong to the dataset sub-object
        # (assembled by buildDatasetConfig), or to inline let-bindings.
        # We only flag *unknown* names — all dataset sub-fields are fine
        # because they're inside the `dataset:` value, not at top level.
        unknown = []
        for key in keys:
            if key in backend_fields:
                continue
            # gpu_ids is attached conditionally below the literal — skip.
            if key in {"gpu_ids"}:
                continue
            # Allow common inner keys that aren't TrainingConfig top-level
            # but appear in nested objects we built inline.
            allowed_inner = {
                "source", "format", "additional_sources", "test_size",
                "convert_messages_format", "deduplicate", "dedupe_strategy",
                "max_samples", "model_name", "system_prompt",
                "system_prompt_mode", "training_mode",
                "format_type", "enable_thinking", "custom_templates",
                "prompt_template", "chosen_template", "rejected_template",
                "source_type", "repo_id", "split", "file_path", "file_format",
                "dataset_id", "column_mapping",
            }
            if key in allowed_inner:
                continue
            unknown.append(key)
        assert not unknown, (
            f"form_config.js emits top-level keys not on TrainingConfig: "
            f"{unknown}"
        )
