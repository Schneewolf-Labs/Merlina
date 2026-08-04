"""Recover a usable LoRA adapter from a crashed run's checkpoint.

When training dies part-way — OOM, memory guard, power cut — the periodic checkpoint on disk is
usually intact and represents real GPU hours. Until now none of it was reachable: the job was
marked ``failed`` without recording ``output_dir``, so the API had no pointer to the artifacts,
``/jobs/{id}/upload`` refused anything that was not ``completed`` or ``stopped``, and even given
the path there was nothing named ``adapter_model.safetensors`` to upload.

That last part is the awkward one. ``accelerate.save_state`` does not know the model is
PEFT-wrapped, so a checkpoint from a LoRA run contains the *entire* model — a 27B run writes a
55GB ``model.safetensors`` with the 640MB adapter buried inside it. Recovering by hand means
extracting 512 tensors out of 1696 and synthesising an ``adapter_config.json`` from the job config.

This module does that automatically:

    find_resumable_adapter(output_dir)   -> newest checkpoint holding LoRA weights, or None
    extract_adapter(ckpt, dest, config)  -> writes a normal PEFT adapter directory

The extraction reads safetensors directly rather than loading the checkpoint, because
materialising 55GB of frozen base weights to keep 1% of them would defeat the purpose on exactly
the machines where this matters most — unified-memory boxes that just died of memory pressure.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

# accelerate writes the raw state dict here; peft writes the adapter here
_STATE_NAMES = ("adapter_model.safetensors", "model.safetensors")


def _checkpoints(job_dir: Path) -> list[Path]:
    """Checkpoint subdirs of ``job_dir``, newest first."""
    if not job_dir.is_dir():
        return []
    cks = [d for d in job_dir.iterdir() if d.is_dir() and _CKPT_RE.match(d.name)]
    return sorted(cks, key=lambda d: int(_CKPT_RE.match(d.name).group(1)), reverse=True)


def _read_header(path: Path) -> tuple[dict, int]:
    """Return a safetensors file's JSON header and the byte offset of its data section."""
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    return header, 8 + n


def _lora_keys(header: dict) -> list[str]:
    return sorted(k for k in header
                  if k != "__metadata__" and ("lora_A" in k or "lora_B" in k))


def find_resumable_adapter(output_dir: str | Path) -> Optional[Path]:
    """Newest checkpoint under ``output_dir`` that actually contains LoRA weights.

    Checks the directory itself first — a run that got as far as ``save_model`` needs no
    extraction — then walks checkpoints newest to oldest. Returns None if nothing usable is found,
    which is the normal answer for a run that died before its first checkpoint.
    """
    root = Path(output_dir)
    for cand in [root, *_checkpoints(root)]:
        for name in _STATE_NAMES:
            p = cand / name
            if not p.is_file():
                continue
            try:
                header, _ = _read_header(p)
            except Exception as exc:                     # truncated by the same crash
                logger.warning("Unreadable safetensors at %s: %s", p, exc)
                continue
            if _lora_keys(header):
                return cand
    return None


def _normalise(key: str) -> str:
    """Map a checkpoint tensor name onto peft's expected layout.

    accelerate may keep peft's ``base_model.model.`` prefix, drop it, or carry the adapter name
    as ``.default.`` — normalise all three so the result loads with PeftModel.from_pretrained.
    """
    k = key.replace(".default.weight", ".weight").replace(".default.", ".")
    if not k.startswith("base_model.model."):
        k = "base_model.model." + k.lstrip(".")
    if not k.endswith(".weight"):
        k += ".weight"
    return k


def extract_adapter(ckpt_dir: str | Path, dest_dir: str | Path,
                    config: Any = None) -> Path:
    """Write a loadable PEFT adapter directory from a checkpoint.

    ``config`` is the job's TrainingConfig, used for the LoRA hyperparameters and base model
    name. Anything missing falls back to peft's own defaults, which are correct far more often
    than they are wrong, but a mismatched ``r``/``lora_alpha`` silently rescales every delta —
    so the values are logged rather than applied quietly.
    """
    ckpt, dest = Path(ckpt_dir), Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    src = next((ckpt / n for n in _STATE_NAMES if (ckpt / n).is_file()), None)
    if src is None:
        raise FileNotFoundError(f"No safetensors state file in {ckpt}")

    header, data_start = _read_header(src)
    keys = _lora_keys(header)
    if not keys:
        raise ValueError(f"No LoRA tensors in {src}")

    # Rebuild a contiguous header covering only the adapter tensors, then stream their
    # byte ranges across. Never loads the frozen base weights.
    new_header: dict[str, dict] = {}
    plan: list[tuple[int, int]] = []
    off = 0
    for k in keys:
        e = header[k]
        s0, s1 = e["data_offsets"]
        size = s1 - s0
        new_header[_normalise(k)] = {
            "dtype": e["dtype"], "shape": e["shape"],
            "data_offsets": [off, off + size],
        }
        plan.append((data_start + s0, size))
        off += size

    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)          # keep the data section 8-byte aligned

    out_path = dest / "adapter_model.safetensors"
    with src.open("rb") as fh, out_path.open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for start, size in plan:
            fh.seek(start)
            left = size
            while left:
                chunk = fh.read(min(left, 8 << 20))
                if not chunk:
                    raise EOFError(f"Truncated checkpoint at {src}")
                out.write(chunk)
                left -= len(chunk)

    r = int(getattr(config, "lora_r", None) or getattr(config, "lora_rank", None) or 16)
    alpha = int(getattr(config, "lora_alpha", None) or 32)
    dropout = float(getattr(config, "lora_dropout", None) or 0.0)
    base = getattr(config, "base_model", "") or ""
    targets = getattr(config, "lora_target_modules", None) or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]

    adapter_config = {
        "peft_type": "LORA",
        "task_type": getattr(config, "lora_task_type", None) or "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": r, "lora_alpha": alpha, "lora_dropout": dropout,
        "target_modules": list(targets),
        "bias": "none", "inference_mode": False, "fan_in_fan_out": False,
        "init_lora_weights": True, "modules_to_save": None,
        "layers_to_transform": None, "layers_pattern": None,
        "rank_pattern": {}, "alpha_pattern": {},
        "use_rslora": bool(getattr(config, "lora_use_rslora", False)),
        "use_dora": bool(getattr(config, "lora_use_dora", False)),
        "megatron_config": None, "megatron_core": "megatron.core", "loftq_config": {},
    }
    (dest / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

    logger.info(
        "Rescued adapter from %s: %d tensors, %.1f MB, r=%d alpha=%d base=%s",
        ckpt.name, len(keys), off / 1e6, r, alpha, base or "(unknown)",
    )
    return dest
