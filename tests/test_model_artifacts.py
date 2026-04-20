"""
Tests for the model artifact inventory + path-traversal safety.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_artifacts import (
    PROTECTED_FILES,
    inventory_model,
    is_protected,
    resolve_artifact_path,
)


def _touch(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _mk_lora_model(tmp_path: Path) -> Path:
    d = tmp_path / "my-lora"
    _touch(d / "adapter_config.json")
    _touch(d / "adapter_model.safetensors", size=1024)
    _touch(d / "tokenizer.json")
    _touch(d / "tokenizer_config.json")
    _touch(d / "special_tokens_map.json")
    _touch(d / "config.json")
    _touch(d / "README.md")
    return d


def _mk_lora_with_gguf(tmp_path: Path) -> Path:
    d = _mk_lora_model(tmp_path)
    _touch(d / "gguf" / "my-lora.q4_k_m.gguf", size=4096)
    _touch(d / "gguf" / "my-lora.q8_0.gguf", size=8192)
    _touch(d / "gguf" / "manifest.json")
    return d


def _mk_vlm_model(tmp_path: Path) -> Path:
    d = _mk_lora_model(tmp_path)
    _touch(d / "preprocessor_config.json")
    _touch(d / "processor_config.json")
    return d


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

def test_categorizes_lora_files(tmp_path):
    inv = inventory_model(_mk_lora_model(tmp_path))

    assert inv.is_lora is True
    assert inv.has_merged is False
    assert inv.has_gguf is False

    by_cat = {cat: [f.filename for f in files] for cat, files in inv.categories.items()}
    assert "adapter_config.json" in by_cat["adapter"]
    assert "adapter_model.safetensors" in by_cat["adapter"]
    assert set(by_cat["tokenizer"]) >= {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    assert "config.json" in by_cat["config"]
    assert "README.md" in by_cat["readme"]


def test_categorizes_gguf_artifacts(tmp_path):
    inv = inventory_model(_mk_lora_with_gguf(tmp_path))

    assert inv.has_gguf is True
    gguf_files = {f.filename: f for f in inv.categories["gguf"]}
    assert "my-lora.q4_k_m.gguf" in gguf_files
    assert "my-lora.q8_0.gguf" in gguf_files
    assert "manifest.json" in gguf_files

    assert gguf_files["my-lora.q4_k_m.gguf"].quant_type == "Q4_K_M"
    assert gguf_files["my-lora.q8_0.gguf"].quant_type == "Q8_0"


def test_recognizes_vlm_processor(tmp_path):
    inv = inventory_model(_mk_vlm_model(tmp_path))
    processor_files = {f.filename for f in inv.categories.get("processor", [])}
    assert {"preprocessor_config.json", "processor_config.json"} <= processor_files


def test_total_bytes_sums_all_files(tmp_path):
    d = _mk_lora_model(tmp_path)
    inv = inventory_model(d)
    expected = sum(
        p.stat().st_size for p in d.rglob("*") if p.is_file()
    )
    assert inv.total_bytes == expected


def test_full_model_without_adapter_is_not_lora(tmp_path):
    d = tmp_path / "full-model"
    _touch(d / "config.json")
    _touch(d / "model.safetensors", size=4096)
    _touch(d / "tokenizer.json")

    inv = inventory_model(d)
    assert inv.is_lora is False
    assert inv.has_merged is True
    merged = {f.filename for f in inv.categories["merged"]}
    assert "model.safetensors" in merged


def test_protected_flag_set_on_core_files(tmp_path):
    inv = inventory_model(_mk_lora_model(tmp_path))
    protected = {
        f.filename for files in inv.categories.values()
        for f in files if f.protected
    }
    assert "adapter_config.json" in protected
    assert "config.json" in protected
    assert "tokenizer_config.json" in protected


def test_missing_model_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inventory_model(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def test_resolve_artifact_path_rejects_traversal(tmp_path):
    d = _mk_lora_model(tmp_path)
    for bad in ("../escape", "../../etc/passwd", "/etc/passwd"):
        with pytest.raises(ValueError):
            resolve_artifact_path(d, bad)


def test_resolve_artifact_path_rejects_empty(tmp_path):
    d = _mk_lora_model(tmp_path)
    with pytest.raises(ValueError):
        resolve_artifact_path(d, "")


def test_resolve_artifact_path_allows_subdir(tmp_path):
    d = _mk_lora_with_gguf(tmp_path)
    resolved = resolve_artifact_path(d, "gguf/my-lora.q4_k_m.gguf")
    assert resolved.is_file()
    assert resolved.name == "my-lora.q4_k_m.gguf"


def test_is_protected_matches_tokenizer_and_config(tmp_path):
    d = _mk_lora_model(tmp_path)
    assert is_protected(d, "adapter_config.json") is True
    assert is_protected(d, "config.json") is True
    assert is_protected(d, "tokenizer.json") is True
    assert is_protected(d, "gguf/my-lora.q4_k_m.gguf") is False
    assert is_protected(d, "README.md") is False


def test_protected_set_is_non_empty():
    assert len(PROTECTED_FILES) >= 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
