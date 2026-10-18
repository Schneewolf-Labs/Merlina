#!/usr/bin/env python3
"""Smoke test for src/quantized_merge.py with the real stack: transformers >= 5.20.1, torch, Kessel.

The unit tests mock torch and transformers. This checks that the real DeepSeek-V4 conversion mapping inverts
to the checkpoint's tensor names, then merges an adapter keyed by transformers names into a tiny DeepSeek-V4-
shaped FP8 checkpoint: touched weights become bf16 holding exactly W + delta, untouched ones keep their bytes,
and config.json lists the promoted modules. With a real checkpoint path it also checks the name map against
that checkpoint's index:

    python scripts/smoke_quantized_merge.py [/path/to/Smaug-Flash]
"""
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kessel.io as kessel_io
import kessel.quant as kessel_quant

from src.quantized_merge import (
    checkpoint_name_map,
    is_native_quantized,
    merge_quantized,
)

EXPECT = {"layers.3.self_attn.q_a_proj.weight": "layers.3.attn.wq_a.weight",
          "layers.3.self_attn.q_b_proj.weight": "layers.3.attn.wq_b.weight",
          "layers.3.self_attn.kv_proj.weight": "layers.3.attn.wkv.weight",
          "layers.3.self_attn.o_b_proj.weight": "layers.3.attn.wo_b.weight",
          "layers.3.mlp.shared_experts.gate_proj.weight": "layers.3.ffn.shared_experts.w1.weight",
          "layers.3.mlp.shared_experts.up_proj.weight": "layers.3.ffn.shared_experts.w3.weight",
          "layers.3.mlp.shared_experts.down_proj.weight": "layers.3.ffn.shared_experts.w2.weight"}


def check_name_map(names):
    m = checkpoint_name_map(names, "deepseek_v4")
    for hf, ck in EXPECT.items():
        assert m.get(hf) == ck, (hf, m.get(hf))
    assert not any(".experts." in v.replace("shared_experts", "") for v in m.values())
    print(f"name map ok ({len(m)} weights)")


QCFG = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128]}


def _write_ckpt(path, layers=2):
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for i in range(layers):
        for name, shape in (("attn.wq_b", (256, 128)), ("attn.wkv", (128, 256)), ("ffn.shared_experts.w2", (256, 128))):
            w, s, _ = kessel_quant.quantize(torch.randn(*shape, generator=g) * 0.02, "fp8")
            tensors[f"layers.{i}.{name}.weight"] = w
            tensors[f"layers.{i}.{name}.scale"] = s
        tensors[f"layers.{i}.attn_norm.weight"] = torch.ones(256).bfloat16()
    tensors["embed.weight"] = (torch.randn(64, 256, generator=g) * 0.02).bfloat16()
    path.mkdir(parents=True)
    items = []
    for n in sorted(tensors):
        data, dt = kessel_io.tensor_to_bytes(tensors[n])
        items.append((n, dt, tuple(tensors[n].shape), data))
    kessel_io.write_safetensors(path / "model.safetensors", items)
    (path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "quantization_config": QCFG}))
    return tensors


def _write_adapter(path, modules, shapes, scale=1e-5):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(1)
    t = {}
    for m, (out_f, in_f) in zip(modules, shapes):
        t[f"base_model.model.{m}.lora_A.weight"] = torch.randn(4, in_f, generator=g)
        t[f"base_model.model.{m}.lora_B.weight"] = torch.randn(out_f, 4, generator=g) * scale
    path.mkdir(parents=True)
    save_file(t, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "target_modules": ["q_b_proj"]}))
    return t


def check_merge_keeps_delta_and_marks_modules(tmp_path):
    base = _write_ckpt(tmp_path / "base")
    hf = ["model.layers.0.self_attn.q_b_proj", "model.layers.1.mlp.shared_experts.down_proj"]
    ad = _write_adapter(tmp_path / "adapter", hf, [(256, 128), (256, 128)])
    out = merge_quantized(str(tmp_path / "base"), tmp_path / "adapter", tmp_path / "out")

    mf = kessel_io.ModelFiles(out)
    for ck in ("layers.0.attn.wq_b", "layers.1.ffn.shared_experts.w2"):
        assert mf.info[ck + ".weight"].dtype == "BF16"
        assert ck + ".scale" not in mf.info
    # untouched quantized tensors keep their bytes
    assert mf.info["layers.0.attn.wkv.weight"].dtype == "F8_E4M3"
    assert mf.tensor("layers.1.attn.wq_b.weight").view(torch.uint8).equal(base["layers.1.attn.wq_b.weight"].view(torch.uint8))

    spec = kessel_quant.QuantSpec("fp8", "", "F8_E8M0", (128, 128))
    w = kessel_quant.dequantize(base["layers.0.attn.wq_b.weight"], base["layers.0.attn.wq_b.scale"], spec)
    delta = 2.0 * (ad[f"base_model.model.{hf[0]}.lora_B.weight"] @ ad[f"base_model.model.{hf[0]}.lora_A.weight"])
    assert torch.equal(mf.tensor("layers.0.attn.wq_b.weight").float(), (w + delta).bfloat16().float())

    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert set(qc["modules_to_not_convert"]) >= set(hf)  # transformers names
    assert set(qc["ignored_layers"]) == {"layers.0.attn.wq_b", "layers.1.ffn.shared_experts.w2"}  # checkpoint names


def check_rejects_unmapped_modules(tmp_path):
    _write_ckpt(tmp_path / "base")
    _write_adapter(tmp_path / "adapter", ["model.layers.0.self_attn.nonexistent"], [(256, 128)])
    try:
        merge_quantized(str(tmp_path / "base"), tmp_path / "adapter", tmp_path / "out")
    except ValueError as exc:
        assert "no checkpoint tensor" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def main():
    if len(sys.argv) > 1:
        ckpt = Path(sys.argv[1])
        assert is_native_quantized(str(ckpt)), "not a block-FP8 checkpoint"
        check_name_map(list(json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]))
        return
    check_name_map(list(EXPECT.values()) + ["layers.3.attn.wq_b.scale", "layers.3.ffn.experts.0.w1.weight"])
    for check in (check_merge_keeps_delta_and_marks_modules, check_rejects_unmapped_modules):
        with tempfile.TemporaryDirectory() as d:
            check(Path(d))
        print(check.__name__, "ok")


if __name__ == "__main__":
    main()
