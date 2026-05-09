"""
Tests for the "Reproduce this training run" model-card section.

Guards the share_config flag's contract:
  * share_config=True  → README embeds the validated, secret-stripped
    config inside a fenced ```json block with the standard schema header
    (matches what /configs/save writes to disk).
  * share_config=False → no embed, no leak.
  * Secrets (hf_token, wandb_key) are stripped unconditionally even when
    share_config=True.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from src.config_manager import SCHEMA_NAME, SCHEMA_VERSION, SECRET_FIELDS
from src.model_card import generate_model_readme


# ---------------------------------------------------------------------------
# Minimal config that matches what generate_model_readme looks at.
# ---------------------------------------------------------------------------


class _Source(BaseModel):
    source_type: str = "huggingface"
    repo_id: Optional[str] = None
    split: str = "train"
    file_path: Optional[str] = None


class _Dataset(BaseModel):
    source: _Source = Field(default_factory=_Source)
    additional_sources: List[_Source] = Field(default_factory=list)


class _Cfg(BaseModel):
    base_model: str = "meta-llama/Meta-Llama-3-8B"
    output_name: str = "test-model"
    training_mode: str = "orpo"
    learning_rate: float = 5e-6
    num_epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 2048
    max_prompt_length: int = 1024
    beta: float = 0.1
    gamma: float = 0.5
    label_smoothing: float = 0.0
    optimizer_type: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_4bit: bool = True
    attn_implementation: str = "auto"
    dataset: _Dataset = Field(default_factory=_Dataset)
    share_config: bool = True
    hf_token: Optional[str] = None
    wandb_key: Optional[str] = None


def _extract_json_block(readme: str) -> Optional[dict]:
    """Pull the first ```json fenced block out of a README, parse, return."""
    match = re.search(r"```json\s*\n(.*?)\n```", readme, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# share_config=True → embed
# ---------------------------------------------------------------------------


class TestShareConfigEnabled:
    def test_embeds_section_header(self):
        readme = generate_model_readme(_Cfg(share_config=True), "orpo")
        assert "## Reproduce this training run" in readme

    def test_embeds_json_block(self):
        readme = generate_model_readme(_Cfg(share_config=True), "orpo")
        payload = _extract_json_block(readme)
        assert payload is not None, "Expected a fenced ```json block"

    def test_embedded_payload_has_schema_header(self):
        readme = generate_model_readme(_Cfg(share_config=True), "orpo")
        payload = _extract_json_block(readme)
        meta = payload["_metadata"]
        assert meta["schema"] == SCHEMA_NAME
        assert meta["schema_version"] == SCHEMA_VERSION
        assert "merlina_version" in meta

    def test_embedded_payload_strips_secrets(self):
        cfg = _Cfg(share_config=True, hf_token="hf_secret", wandb_key="wb_secret")
        readme = generate_model_readme(cfg, "orpo")
        # Secrets should not appear anywhere in the README, period.
        for secret_value in ("hf_secret", "wb_secret"):
            assert secret_value not in readme, (
                f"Secret value {secret_value!r} leaked into README"
            )
        payload = _extract_json_block(readme)
        for f in SECRET_FIELDS:
            assert f not in payload, f"{f} present in shared payload"

    def test_embedded_payload_strips_share_config_itself(self):
        """share_config is a publishing toggle, not a hyperparameter.
        Including it would be confusing on import."""
        readme = generate_model_readme(_Cfg(share_config=True), "orpo")
        payload = _extract_json_block(readme)
        assert "share_config" not in payload

    def test_embedded_payload_includes_training_hyperparameters(self):
        cfg = _Cfg(
            share_config=True,
            learning_rate=2e-5,
            num_epochs=3,
            beta=0.2,
            lora_r=128,
        )
        readme = generate_model_readme(cfg, "orpo")
        payload = _extract_json_block(readme)
        assert payload["learning_rate"] == 2e-5
        assert payload["num_epochs"] == 3
        assert payload["beta"] == 0.2
        assert payload["lora_r"] == 128

    def test_embedded_payload_uses_output_name_in_metadata(self):
        cfg = _Cfg(share_config=True, output_name="cool-model-v2")
        readme = generate_model_readme(cfg, "orpo")
        payload = _extract_json_block(readme)
        assert payload["_metadata"]["name"] == "cool-model-v2"


# ---------------------------------------------------------------------------
# share_config=False → no embed
# ---------------------------------------------------------------------------


class TestShareConfigDisabled:
    def test_no_section_header_when_disabled(self):
        readme = generate_model_readme(_Cfg(share_config=False), "orpo")
        assert "## Reproduce this training run" not in readme

    def test_no_json_block_when_disabled(self):
        readme = generate_model_readme(_Cfg(share_config=False), "orpo")
        assert _extract_json_block(readme) is None

    def test_no_secret_leak_even_when_disabled(self):
        """If a user accidentally commits secrets but disables sharing,
        we should still not leak — defense in depth."""
        cfg = _Cfg(share_config=False, hf_token="hf_secret", wandb_key="wb_secret")
        readme = generate_model_readme(cfg, "orpo")
        for secret_value in ("hf_secret", "wb_secret"):
            assert secret_value not in readme

    def test_legacy_config_without_share_config_attribute(self):
        """A pre-v1.6 config object (no share_config field) should be
        treated as 'don't share' — opt-in, not opt-out."""

        class LegacyCfg(BaseModel):
            base_model: str = "gpt2"
            output_name: str = "legacy"
            training_mode: str = "sft"
            learning_rate: float = 1e-5
            num_epochs: int = 1
            batch_size: int = 1
            gradient_accumulation_steps: int = 1
            max_length: int = 512
            max_prompt_length: int = 256
            beta: float = 0.1
            gamma: float = 0.5
            label_smoothing: float = 0.0
            optimizer_type: str = "adamw_torch"
            lr_scheduler_type: str = "cosine"
            warmup_ratio: float = 0.05
            weight_decay: float = 0.01
            max_grad_norm: float = 1.0
            seed: int = 42
            use_lora: bool = False
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.0
            target_modules: List[str] = Field(default_factory=list)
            use_4bit: bool = False
            attn_implementation: str = "auto"
            dataset: _Dataset = Field(default_factory=_Dataset)
            # No share_config attribute at all.

        readme = generate_model_readme(LegacyCfg(), "sft")
        assert "## Reproduce this training run" not in readme


# ---------------------------------------------------------------------------
# Importability: the JSON block in a README is structurally identical to
# what /configs/save writes to disk. Confirm we can load it back through
# ConfigManager.import_config and it round-trips through TrainingConfig.
# ---------------------------------------------------------------------------


class TestEmbeddedConfigIsImportable:
    def test_embedded_block_is_loadable_into_config_manager(self, tmp_path):
        from src.config_manager import ConfigManager

        # Use the real TrainingConfig — that's what an imported config
        # has to satisfy on a fresh Merlina instance.
        from merlina import TrainingConfig

        # Build a real TrainingConfig and render its README. The embedded
        # block must be importable straight back into a Merlina install.
        full = TrainingConfig(
            output_name="reproducible",
            base_model="gpt2",
            training_mode="sft",
            share_config=True,
            dataset={
                "source": {
                    "source_type": "huggingface",
                    "repo_id": "x/y",
                    "split": "train",
                },
                "format": {"format_type": "chatml"},
                "test_size": 0.01,
            },
        )
        readme = generate_model_readme(full, "sft")
        payload = _extract_json_block(readme)
        assert payload is not None

        # Persist the embed via the same import path the UI uses.
        external_file = tmp_path / "from_readme.json"
        external_file.write_text(json.dumps(payload))

        mgr = ConfigManager(config_dir=str(tmp_path / "store"))
        mgr.import_config(str(external_file), name="from-readme")

        loaded = mgr.get_config_without_metadata("from-readme")
        # Re-validate — this is what /train does when the user clicks
        # "Cast Spell" after loading the imported preset.
        rebuilt = TrainingConfig(**loaded)
        assert rebuilt.output_name == "reproducible"
        assert rebuilt.training_mode == "sft"
