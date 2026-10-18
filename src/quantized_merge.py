"""Merging a LoRA adapter into a natively quantized checkpoint (block-FP8 / FP4, e.g. DeepSeek-V4).

The usual merge (load in bf16, ``merge_and_unload``, ``save_pretrained``) doesn't work for these:
transformers loads them quantized, and even a successful requantization erases the adapter. A LoRA
delta is ~0.1% of a weight while FP8 e4m3 steps are ~6%; measured on Smaug-Flash, requantizing kept
almost none of it. So the merge goes through Kessel instead:

- the adapter's keys (transformers module names, ``model.layers.3.self_attn.q_b_proj``) are mapped
  to the checkpoint's own tensor names (``layers.3.attn.wq_b``) with transformers' conversion
  mapping for the model type, run forward over the checkpoint's names and inverted;
- ``kessel.merge.run(..., promote_changed=True)`` applies the delta and writes the touched weights as
  bf16 (dropping their scales), leaving every other tensor in its original quantized bytes;
- ``config.json`` lists the promoted modules as unquantized: Kessel adds the checkpoint names
  (``ignored_layers``, which vLLM maps itself) and this adds the transformers names to
  ``modules_to_not_convert``.

Kessel is an optional dependency (``pip install git+https://github.com/Schneewolf-Labs/kessel``).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_LORA_KEY = re.compile(r"^base_model\.model\.(?P<module>.+)\.(?P<kind>lora_[AB]|lora_embedding_[AB])\.weight$")


def _config(base_model: str) -> dict | None:
    p = Path(base_model)
    if p.is_dir():
        f = p / "config.json"
        return json.loads(f.read_text()) if f.exists() else None
    try:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(base_model, "config.json")).read_text())
    except Exception as exc:  # noqa: BLE001 — offline / gated: fall back to the regular merge path
        logger.debug("Could not read config.json for %s: %s", base_model, exc)
        return None


def is_native_quantized(base_model: str) -> bool:
    """True for checkpoints stored block-quantized (fine-grained FP8, e.g. DeepSeek-V3/V4)."""
    qc = (_config(base_model) or {}).get("quantization_config") or {}
    return qc.get("quant_method") == "fp8" and bool(qc.get("weight_block_size"))


def checkpoint_name_map(checkpoint_names: list[str], model_type: str) -> dict[str, str]:
    """transformers parameter name (no leading ``model.``) -> checkpoint tensor name, for ``.weight`` tensors."""
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import (
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )

    transforms = get_checkpoint_conversion_mapping(model_type) or []
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    out: dict[str, str] = {}
    for name in checkpoint_names:
        if not name.endswith(".weight"):
            continue
        hf, converted = rename_source_key(name, renamings, converters)
        if converted is None:  # converters fuse several tensors (e.g. routed experts); LoRA can't target those
            out[hf.removeprefix("model.")] = name
    return out


def rename_adapter(adapter_dir: Path, name_map: dict[str, str], out_dir: Path) -> list[str]:
    """Write a copy of the adapter keyed by checkpoint names. Returns the adapter's transformers module names."""
    from safetensors.torch import load_file, save_file

    tensors = {}
    for f in sorted(Path(adapter_dir).glob("adapter_model*.safetensors")):
        tensors.update(load_file(str(f)))
    if not tensors:
        raise FileNotFoundError(f"{adapter_dir}: no adapter_model.safetensors")
    renamed, modules, missing = {}, set(), set()
    for key, t in tensors.items():
        m = _LORA_KEY.match(key)
        if not m:
            raise ValueError(f"{adapter_dir}: {key} is not a LoRA weight (modules_to_save isn't supported here)")
        module = m["module"]
        target = name_map.get(module.removeprefix("model.") + ".weight")
        if target is None:
            missing.add(module)
            continue
        renamed[f"base_model.model.{target[:-len('.weight')]}.{m['kind']}.weight"] = t
        modules.add(module)
    if missing:
        raise ValueError(f"adapter modules with no checkpoint tensor: {sorted(missing)[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    cfg["target_modules"] = sorted({k.split(".")[-3] for k in renamed})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return sorted(modules)


def merge_quantized(base_model: str, adapter_dir: Path, output_dir: Path) -> Path:
    """Merge ``adapter_dir`` into the quantized ``base_model``; writes a full checkpoint to ``output_dir``."""
    try:
        from kessel.io import ModelFiles, resolve
        from kessel.merge import run
    except ImportError as exc:
        raise RuntimeError(
            "Merging into a natively quantized (FP8/FP4) checkpoint needs Kessel: "
            "pip install git+https://github.com/Schneewolf-Labs/kessel") from exc

    base_path = resolve(base_model)  # local directory, or a (cached) Hub download
    model_type = json.loads((Path(base_path) / "config.json").read_text())["model_type"]
    name_map = checkpoint_name_map(list(ModelFiles(Path(base_path)).info), model_type)

    output_dir = Path(output_dir)
    tmp = Path(tempfile.mkdtemp(prefix="merlina_ckpt_adapter_"))
    try:
        modules = rename_adapter(Path(adapter_dir), name_map, tmp)
        cfg = {"models": [{"model": f"{base_path}+{tmp}", "parameters": {"weight": 1.0}}],
               "merge_method": "task_arithmetic", "base_model": str(base_path), "dtype": "bfloat16"}
        logger.info("Merging %d adapter modules into quantized %s with Kessel", len(modules), base_model)
        rep = run(cfg, output_dir, promote_changed=True, progress=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cfg_path = output_dir / "config.json"
    mcfg = json.loads(cfg_path.read_text())
    qc = mcfg.setdefault("quantization_config", {})
    promoted_ckpt = {n[:-len(".weight")] for n in rep.promoted}
    hf_modules = [m for m in modules if name_map[m.removeprefix("model.") + ".weight"][:-len(".weight")]
                  in promoted_ckpt]
    qc["modules_to_not_convert"] = sorted(set(qc.get("modules_to_not_convert") or []) | set(hf_modules))
    cfg_path.write_text(json.dumps(mcfg, indent=2) + "\n")
    logger.info("Quantized merge: %d tensors promoted to bf16, %d copied", len(rep.promoted), rep.copied)
    return output_dir
