"""Lightweight tests for the diffusion backend dispatch + Pydantic schema.

These tests don't load any model weights — they exercise the dispatch
glue, the Pydantic TrainingConfig surface, and the diffusion mode
registry. Heavy end-to-end tests (real GPU + real model) live elsewhere
and are skipped under pytest.
"""
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------
# _resolve_sibling_runner: text/vlm/diffusion routing
# ---------------------------------------------------------------------

def _cfg(**fields):
    """Build a config-like object with only the dispatch-relevant fields."""
    return SimpleNamespace(model_type=fields.get("model_type", "auto"),
                           training_mode=fields.get("training_mode", "orpo"))


class TestResolveSiblingRunner:
    def test_text_mode_returns_none(self):
        from src.training_runner import _resolve_sibling_runner
        assert _resolve_sibling_runner(_cfg(training_mode="orpo")) is None
        assert _resolve_sibling_runner(_cfg(training_mode="sft")) is None
        assert _resolve_sibling_runner(_cfg(training_mode="dpo")) is None

    def test_vlm_prefix_dispatches_to_vlm_runner(self):
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(training_mode="vlm_stage1"))
        assert fn is not None
        assert fn.__name__ == "run_vlm_training_sync"

    def test_diffusion_prefix_dispatches_to_diffusion_runner(self):
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(training_mode="diffusion_qwen_image"))
        assert fn is not None
        assert fn.__name__ == "run_diffusion_training_sync"

    def test_explicit_model_type_vlm_wins(self):
        """model_type='vlm' dispatches to VLM even with a non-vlm_ training_mode."""
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(model_type="vlm", training_mode="sft"))
        assert fn is not None
        assert fn.__name__ == "run_vlm_training_sync"

    def test_explicit_model_type_diffusion_wins(self):
        from src.training_runner import _resolve_sibling_runner
        fn = _resolve_sibling_runner(_cfg(model_type="diffusion", training_mode="sft"))
        assert fn is not None
        assert fn.__name__ == "run_diffusion_training_sync"

    def test_auto_model_type_falls_back_to_training_mode(self):
        from src.training_runner import _resolve_sibling_runner
        assert _resolve_sibling_runner(_cfg(model_type="auto", training_mode="sft")) is None
        assert _resolve_sibling_runner(
            _cfg(model_type="auto", training_mode="vlm_stage1")
        ).__name__ == "run_vlm_training_sync"

    def test_missing_attrs_doesnt_crash(self):
        """Bare object without model_type / training_mode falls through to text."""
        from src.training_runner import _resolve_sibling_runner
        empty = SimpleNamespace()
        assert _resolve_sibling_runner(empty) is None


# ---------------------------------------------------------------------
# diffusion runner registry
# ---------------------------------------------------------------------

class TestDiffusionModeRegistry:
    def test_known_modes(self):
        from src.training_runner_diffusion import _DIFFUSION_MODES
        assert "diffusion_qwen_image" in _DIFFUSION_MODES
        assert "diffusion_qwen_edit" in _DIFFUSION_MODES
        assert "diffusion_sdxl" in _DIFFUSION_MODES

    def test_resolve_unknown_mode_raises(self):
        from src.training_runner_diffusion import _resolve_adapter_and_loss
        with pytest.raises(ValueError, match="Unknown diffusion training_mode"):
            _resolve_adapter_and_loss("diffusion_not_real")

    def test_mode_tuple_shape(self):
        """Each registry entry is (adapter_name, loss_name, default_targets)."""
        from src.training_runner_diffusion import _DIFFUSION_MODES
        for mode, spec in _DIFFUSION_MODES.items():
            assert len(spec) == 3, f"{mode}: expected 3-tuple, got {spec}"
            adapter_name, loss_name, targets = spec
            assert isinstance(adapter_name, str)
            assert isinstance(loss_name, str)
            assert isinstance(targets, list) and len(targets) > 0


# ---------------------------------------------------------------------
# Pydantic TrainingConfig — diffusion fields accepted
# ---------------------------------------------------------------------

class TestTrainingConfigDiffusionFields:
    def test_diffusion_fields_optional(self):
        """A text-mode request should not need any diffusion field set."""
        from merlina import TrainingConfig
        cfg = TrainingConfig(output_name="text-run")
        assert cfg.model_name is None
        assert cfg.image_resolution is None
        assert cfg.lora_rank is None
        assert cfg.lora_use_dora is None
        assert cfg.dataset_jsonl_path is None

    def test_diffusion_fields_round_trip(self):
        from merlina import TrainingConfig
        cfg = TrainingConfig(
            output_name="diffusion-run",
            model_type="diffusion",
            training_mode="diffusion_qwen_image",
            model_name="Qwen/Qwen-Image",
            image_resolution=1024,
            lora_rank=32,
            lora_target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_use_dora=True,
            dataset_jsonl_path="/tmp/train.jsonl",
        )
        assert cfg.model_name == "Qwen/Qwen-Image"
        assert cfg.image_resolution == 1024
        assert cfg.lora_rank == 32
        assert cfg.lora_target_modules == ["to_k", "to_q", "to_v", "to_out.0"]
        assert cfg.lora_use_dora is True

    def test_image_resolution_bounds(self):
        from merlina import TrainingConfig
        with pytest.raises(Exception):  # Pydantic ValidationError
            TrainingConfig(output_name="x", image_resolution=100)  # below min 256
        with pytest.raises(Exception):
            TrainingConfig(output_name="x", image_resolution=4096)  # above max 2048

    def test_lora_rank_bounds(self):
        from merlina import TrainingConfig
        with pytest.raises(Exception):
            TrainingConfig(output_name="x", lora_rank=2)  # below min 4

    def test_diffusion_mode_forces_use_4bit_off(self):
        """When model_type='diffusion' the Pydantic validator should
        suppress use_4bit so the LLM-only quantization flag doesn't
        leak into the runner / produce misleading preflight warnings."""
        from merlina import TrainingConfig
        cfg = TrainingConfig(
            output_name="diffusion-run",
            model_type="diffusion",
            training_mode="diffusion_qwen_image",
            use_4bit=True,           # user sent it; should get forced off
            export_gguf=True,        # ditto — no GGUF path for diffusion LoRAs
            gguf_quant_types=["Q4_K_M"],
        )
        assert cfg.use_4bit is False
        assert cfg.export_gguf is False

    def test_text_mode_preserves_use_4bit(self):
        """Text-mode jobs must NOT have their use_4bit flag mutated."""
        from merlina import TrainingConfig
        cfg = TrainingConfig(
            output_name="text-run",
            training_mode="orpo",
            use_4bit=True,
        )
        assert cfg.use_4bit is True

    def test_diffusion_via_prefix_also_suppressed(self):
        """The fallback (training_mode prefix sniffing) should also
        trigger the suppression — backward-compat with stored configs."""
        from merlina import TrainingConfig
        cfg = TrainingConfig(
            output_name="x",
            model_type="auto",
            training_mode="diffusion_sdxl",
            use_4bit=True,
        )
        assert cfg.use_4bit is False


class TestPreflightDiffusionGating:
    """The LLM-centric preflight checks (VRAM math, model access,
    dataset columns, training config nags) should be skipped when the
    job is a diffusion run; a small diffusion-specific check runs instead."""

    def test_is_diffusion_detection(self):
        from types import SimpleNamespace
        from src.preflight_checks import PreflightValidator
        v = PreflightValidator()
        assert v._is_diffusion(SimpleNamespace(model_type="diffusion", training_mode="diffusion_qwen_image"))
        assert v._is_diffusion(SimpleNamespace(model_type="auto", training_mode="diffusion_sdxl"))
        assert not v._is_diffusion(SimpleNamespace(model_type="auto", training_mode="orpo"))
        assert not v._is_diffusion(SimpleNamespace(model_type="causal_lm", training_mode="sft"))
        assert not v._is_diffusion(SimpleNamespace())  # missing both attrs

    def test_check_diffusion_config_missing_dataset_warns(self):
        from types import SimpleNamespace
        from src.preflight_checks import PreflightValidator
        v = PreflightValidator()
        cfg = SimpleNamespace(
            model_type="diffusion",
            training_mode="diffusion_qwen_image",
            dataset_jsonl_path=None,
            dataset_name=None,
            lora_rank=64,
            lora_r=None,
        )
        v._check_diffusion_config(cfg)
        assert any("without dataset_jsonl_path or dataset_name" in w for w in v.warnings)

    def test_check_diffusion_config_nonexistent_jsonl_errors(self):
        from types import SimpleNamespace
        from src.preflight_checks import PreflightValidator
        v = PreflightValidator()
        cfg = SimpleNamespace(
            model_type="diffusion",
            training_mode="diffusion_qwen_image",
            dataset_jsonl_path="/tmp/does-not-exist-xyz.jsonl",
            dataset_name=None,
            lora_rank=64,
            lora_r=None,
        )
        v._check_diffusion_config(cfg)
        assert any("does not exist" in e for e in v.errors)

    def test_check_diffusion_config_high_rank_warns(self):
        from types import SimpleNamespace
        from src.preflight_checks import PreflightValidator
        v = PreflightValidator()
        cfg = SimpleNamespace(
            model_type="diffusion",
            training_mode="diffusion_qwen_image",
            dataset_jsonl_path=None,
            dataset_name="user/x",
            lora_rank=1024,
            lora_r=None,
        )
        v._check_diffusion_config(cfg)
        assert any("very high" in w for w in v.warnings)


class TestJsonlPreviewEndpoints:
    """The /dataset/preview-images + /dataset/save-jsonl + /dataset/image-content
    endpoints enable in-Merlina inspection + editing of any local image
    dataset. The tests use a synthetic JSONL + on-disk PNGs in a tempdir
    so no model/GPU is involved."""

    def _make_dataset(self, tmp, n=3):
        """Create n tiny PNGs + a JSONL referencing them via relative paths."""
        import json
        from pathlib import Path
        # Minimal valid 1x1 PNG bytes (raw — Pillow not required for the test)
        png_1x1 = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63000100000005000167a4d0a30000000049454e44ae426082"
        )
        images_dir = Path(tmp) / "images"
        images_dir.mkdir()
        for i in range(n):
            (images_dir / f"img{i}.png").write_bytes(png_1x1)
        jsonl = Path(tmp) / "train.jsonl"
        with open(jsonl, "w") as f:
            for i in range(n):
                f.write(json.dumps({"prompt": f"caption {i}", "image": f"images/img{i}.png"}) + "\n")
        return jsonl

    def test_preview_returns_rows_with_resolved_urls(self):
        import tempfile
        from fastapi.testclient import TestClient
        from merlina import app
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = self._make_dataset(tmp, n=5)
            client = TestClient(app)
            r = client.get(f"/dataset/preview-images?jsonl_path={jsonl}&limit=3")
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["total"] == 5
            assert data["returned"] == 3
            assert data["offset"] == 0
            assert len(data["rows"]) == 3
            for i, row in enumerate(data["rows"]):
                assert row["prompt"] == f"caption {i}"
                assert row["image_path"].endswith(f"img{i}.png")
                assert "image_url" in row
                assert row["row_index"] == i

    def test_preview_missing_jsonl_404(self):
        from fastapi.testclient import TestClient
        from merlina import app
        client = TestClient(app)
        r = client.get("/dataset/preview-images?jsonl_path=/tmp/does-not-exist.jsonl")
        assert r.status_code == 404

    def test_preview_pagination_offset(self):
        import tempfile
        from fastapi.testclient import TestClient
        from merlina import app
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = self._make_dataset(tmp, n=10)
            client = TestClient(app)
            r = client.get(f"/dataset/preview-images?jsonl_path={jsonl}&limit=3&offset=5")
            assert r.status_code == 200
            data = r.json()
            assert data["total"] == 10
            assert data["returned"] == 3
            assert data["offset"] == 5
            assert data["rows"][0]["row_index"] == 5
            assert data["rows"][0]["prompt"] == "caption 5"

    def test_image_content_serves_file_under_jsonl_dir(self):
        import tempfile
        from fastapi.testclient import TestClient
        from merlina import app
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = self._make_dataset(tmp, n=1)
            client = TestClient(app)
            img_path = jsonl.parent / "images" / "img0.png"
            r = client.get(f"/dataset/image-content?path={img_path}&jsonl_path={jsonl}")
            assert r.status_code == 200
            assert r.headers["content-type"] == "image/png"
            assert len(r.content) > 0

    def test_image_content_blocks_path_outside_jsonl_dir(self):
        """Path-safety: requesting /etc/passwd or anything outside the
        JSONL's parent must 403 — the endpoint must not become an
        arbitrary file reader."""
        import tempfile
        from fastapi.testclient import TestClient
        from merlina import app
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = self._make_dataset(tmp, n=1)
            client = TestClient(app)
            r = client.get(f"/dataset/image-content?path=/etc/hostname&jsonl_path={jsonl}")
            assert r.status_code == 403

    def test_save_jsonl_edits_captions_and_keeps_backup(self):
        import json, tempfile
        from pathlib import Path
        from fastapi.testclient import TestClient
        from merlina import app
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = self._make_dataset(tmp, n=4)
            client = TestClient(app)
            r = client.post("/dataset/save-jsonl", json={
                "jsonl_path": str(jsonl),
                "edits":   [{"row_index": 1, "prompt": "EDITED caption"}],
                "deletes": [3],
            })
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["edited"] == 1
            assert data["deleted"] == 1
            assert data["total_after"] == 3
            # Backup file should exist
            assert Path(data["backup"]).exists()
            # Re-read the JSONL and assert edits + deletion stuck
            with open(jsonl) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            assert len(rows) == 3
            prompts = [r["prompt"] for r in rows]
            assert "EDITED caption" in prompts
            assert "caption 3" not in prompts  # the deleted row's caption

    def test_save_jsonl_missing_file_404(self):
        from fastapi.testclient import TestClient
        from merlina import app
        client = TestClient(app)
        r = client.post("/dataset/save-jsonl", json={
            "jsonl_path": "/tmp/does-not-exist.jsonl",
            "edits": [], "deletes": [],
        })
        assert r.status_code == 404

    def _make_vlm_dataset(self, tmp, n=3):
        """VLM-shape JSONL — {image, caption} instead of {image, prompt}."""
        import json
        from pathlib import Path
        png_1x1 = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63000100000005000167a4d0a30000000049454e44ae426082"
        )
        images_dir = Path(tmp) / "images"
        images_dir.mkdir()
        for i in range(n):
            (images_dir / f"img{i}.png").write_bytes(png_1x1)
        jsonl = Path(tmp) / "train.jsonl"
        with open(jsonl, "w") as f:
            for i in range(n):
                f.write(json.dumps({"caption": f"vlm caption {i}", "image": f"images/img{i}.png"}) + "\n")
        return jsonl

    def test_preview_vlm_shape_falls_back_to_caption_column(self):
        """A {image, caption} JSONL (Artemis Stage 1 shape) should
        preview correctly without needing the caller to pass
        caption_column — the endpoint falls back to 'caption' when
        'prompt' is absent."""
        import tempfile
        from fastapi.testclient import TestClient
        from merlina import app
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = self._make_vlm_dataset(tmp, n=3)
            client = TestClient(app)
            r = client.get(f"/dataset/preview-images?jsonl_path={jsonl}")
            assert r.status_code == 200
            data = r.json()
            assert data["returned"] == 3
            assert data["rows"][0]["prompt"] == "vlm caption 0"
            assert data["rows"][0]["caption_field"] == "caption"

    def test_preview_explicit_column_overrides(self):
        """Caller can override which keys are inspected — useful for
        datasets with non-standard column names (image_url, label, etc.)."""
        import json, tempfile
        from pathlib import Path
        from fastapi.testclient import TestClient
        from merlina import app
        png_1x1 = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63000100000005000167a4d0a30000000049454e44ae426082"
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "images").mkdir()
            (Path(tmp) / "images" / "x.png").write_bytes(png_1x1)
            jsonl = Path(tmp) / "train.jsonl"
            with open(jsonl, "w") as f:
                f.write(json.dumps({"label": "weird label", "image_url": "images/x.png"}) + "\n")
            client = TestClient(app)
            r = client.get(
                f"/dataset/preview-images?jsonl_path={jsonl}"
                "&image_column=image_url&caption_column=label"
            )
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["rows"][0]["prompt"] == "weird label"
            assert data["rows"][0]["caption_field"] == "label"
            assert data["rows"][0]["image_field"] == "image_url"

    def test_save_jsonl_writes_to_specified_caption_column(self):
        """When caption_column='caption' is passed (VLM dataset), the
        edit should write to the caption field, not prompt."""
        import json, tempfile
        from fastapi.testclient import TestClient
        from merlina import app
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = self._make_vlm_dataset(tmp, n=2)
            client = TestClient(app)
            r = client.post("/dataset/save-jsonl", json={
                "jsonl_path":     str(jsonl),
                "edits":          [{"row_index": 0, "prompt": "rewritten"}],
                "deletes":        [],
                "caption_column": "caption",
            })
            assert r.status_code == 200, r.text
            # Re-read and assert the 'caption' key was updated, not 'prompt'
            with open(jsonl) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            assert rows[0].get("caption") == "rewritten"
            assert "prompt" not in rows[0]
