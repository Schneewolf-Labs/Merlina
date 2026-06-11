"""
Tests for the shareable config image (merlina_config.png).

Covers the round-trip contract of ``src/config_image.py``:
  * build_config_image_bytes(envelope) -> PNG bytes carrying the config in
    BOTH a QR code and a tEXt metadata chunk.
  * extract_config_envelope_from_png(bytes) -> the same envelope back, via
    the lossless metadata channel (Pillow only, no QR decoding needed).

Also guards the README image section + the build_shareable_config_envelope
helper in ``src/model_card.py``:
  * share_config_image=True  → README references merlina_config.png.
  * share_config_image=False → no reference.
  * Secrets and the share_* toggles are stripped from the shared envelope.

These tests need Pillow + qrcode to exercise the image path; when those
optional deps are missing the image round-trip is skipped (the metadata
extraction is Pillow-only, the build needs qrcode too). The model-card
section tests run without any image deps.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from src.config_image import (
    PNG_TEXT_KEY,
    build_config_image_bytes,
    extract_config_envelope_from_png,
    _compress_envelope,
    _decompress_payload,
)
from src.config_manager import SCHEMA_NAME, SCHEMA_VERSION
from src.model_card import (
    CONFIG_IMAGE_FILENAME,
    build_shareable_config_envelope,
    generate_model_readme,
)


def _image_deps_available() -> bool:
    """True when Pillow + qrcode are importable (needed to render a PNG)."""
    try:
        import qrcode  # noqa: F401
        import PIL  # noqa: F401
        return True
    except Exception:
        return False


_SAMPLE_ENVELOPE = {
    "_metadata": {
        "name": "test-model",
        "description": "Training configuration shared from a Merlina-trained model.",
        "tags": [],
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "merlina_version": "9.9.9",
    },
    "base_model": "meta-llama/Meta-Llama-3-8B",
    "output_name": "test-model",
    "training_mode": "orpo",
    "learning_rate": 5e-6,
    "num_epochs": 2,
    "target_modules": ["q_proj", "v_proj"],
    "use_lora": True,
}


class TestCompressionRoundTrip(unittest.TestCase):
    """The QR wire format (gzip+base64, prefixed) must round-trip losslessly."""

    def test_compress_decompress(self):
        payload = _compress_envelope(_SAMPLE_ENVELOPE)
        self.assertTrue(payload.startswith("merlina-config-v1:"))
        restored = _decompress_payload(payload)
        self.assertEqual(restored, _SAMPLE_ENVELOPE)

    def test_decompress_rejects_foreign_payload(self):
        self.assertIsNone(_decompress_payload("not-a-merlina-blob"))
        self.assertIsNone(_decompress_payload("merlina-config-v1:###notbase64###"))


@unittest.skipUnless(_image_deps_available(), "Pillow + qrcode required")
class TestImageRoundTrip(unittest.TestCase):
    """build -> extract must return the original envelope (metadata channel)."""

    def test_build_returns_png_bytes(self):
        png = build_config_image_bytes(_SAMPLE_ENVELOPE)
        self.assertIsInstance(png, (bytes, bytearray))
        self.assertTrue(png[:8] == b"\x89PNG\r\n\x1a\n", "should be a PNG")

    def test_metadata_channel_round_trips(self):
        png = build_config_image_bytes(_SAMPLE_ENVELOPE)
        restored = extract_config_envelope_from_png(png)
        self.assertEqual(restored, _SAMPLE_ENVELOPE)

    def test_metadata_key_present(self):
        from io import BytesIO
        from PIL import Image

        png = build_config_image_bytes(_SAMPLE_ENVELOPE)
        img = Image.open(BytesIO(png))
        texts = dict(getattr(img, "text", {}) or {})
        self.assertIn(PNG_TEXT_KEY, texts)
        self.assertEqual(json.loads(texts[PNG_TEXT_KEY]), _SAMPLE_ENVELOPE)

    def test_extract_returns_none_for_non_merlina_png(self):
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
        self.assertIsNone(extract_config_envelope_from_png(buf.getvalue()))


# ---------------------------------------------------------------------------
# Model-card section + envelope helper (no image deps required).
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
    use_lora: bool = True
    lora_r: int = 64
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_4bit: bool = True
    dataset: _Dataset = Field(default_factory=_Dataset)
    share_config: bool = False
    share_config_image: bool = False
    hf_token: Optional[str] = None
    wandb_key: Optional[str] = None


class TestShareableEnvelope(unittest.TestCase):
    def test_strips_secrets_and_toggles(self):
        env = build_shareable_config_envelope(
            _Cfg(share_config=True, share_config_image=True,
                 hf_token="hf_secret", wandb_key="wb_secret")
        )
        self.assertIsNotNone(env)
        self.assertNotIn("hf_token", env)
        self.assertNotIn("wandb_key", env)
        self.assertNotIn("share_config", env)
        self.assertNotIn("share_config_image", env)
        # Real hyperparameters survive.
        self.assertEqual(env["learning_rate"], 5e-6)
        self.assertEqual(env["_metadata"]["schema"], SCHEMA_NAME)

    def test_returns_none_for_unserializable(self):
        self.assertIsNone(build_shareable_config_envelope(object()))


class TestReadmeImageSection(unittest.TestCase):
    def test_image_referenced_when_enabled(self):
        readme = generate_model_readme(_Cfg(share_config_image=True), "orpo")
        self.assertIn(CONFIG_IMAGE_FILENAME, readme)
        self.assertIn("QR", readme)

    def test_no_image_reference_when_disabled(self):
        readme = generate_model_readme(_Cfg(share_config_image=False), "orpo")
        self.assertNotIn(CONFIG_IMAGE_FILENAME, readme)

    def test_image_and_json_are_independent(self):
        # JSON off, image on: README has the picture, not the giant JSON block.
        readme = generate_model_readme(
            _Cfg(share_config=False, share_config_image=True), "orpo"
        )
        self.assertIn(CONFIG_IMAGE_FILENAME, readme)
        self.assertNotIn("## Reproduce this training run\n", readme)


if __name__ == "__main__":
    unittest.main()
