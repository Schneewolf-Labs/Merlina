"""Routing LoRA merges for natively quantized checkpoints (src/quantized_merge.py).

torch and transformers are mocked in this suite; scripts/smoke_quantized_merge.py runs the real merge on a
tiny DeepSeek-V4-shaped FP8 checkpoint and checks the real conversion mapping."""

import json
from pathlib import Path

import src.quantized_merge as qm
from src import gguf_exporter


def _cfg(path, cfg):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg))
    return str(path)


def test_is_native_quantized(tmp_path):
    fp8 = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    assert qm.is_native_quantized(_cfg(tmp_path / "dsv4", {"model_type": "deepseek_v4", "quantization_config": fp8}))
    assert not qm.is_native_quantized(_cfg(tmp_path / "plain", {"model_type": "llama"}))
    # FP8 without block scales (per-tensor, e.g. quantized on load) isn't a native block-quantized checkpoint
    assert not qm.is_native_quantized(_cfg(tmp_path / "pt", {"quantization_config": {"quant_method": "fp8"}}))


def test_merge_lora_routes_quantized_bases_through_kessel(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(qm, "is_native_quantized", lambda base: base == "q")
    monkeypatch.setattr(qm, "merge_quantized", lambda base, ad, out: calls.append(("kessel", base)))
    monkeypatch.setattr(gguf_exporter, "_merge_with_peft", lambda base, ad, out, mt: calls.append(("peft", base)))
    monkeypatch.setattr("src.model_refs.resolve_model_ref", lambda ref: ref)
    for base in ("q", "plain"):
        gguf_exporter.merge_lora_to_directory(base, tmp_path / "adapter", tmp_path / f"out-{base}")
    assert calls == [("kessel", "q"), ("peft", "plain")]
    assert Path(tmp_path / "out-q").is_dir()
