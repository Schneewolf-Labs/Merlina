"""Artemis piece #14 test: staged-freeze control + multimodal generate().

Validates the training capstone — Stage-1/Stage-2 trainability toggles and
that `generate()` works with an image (image injected only at step 0, then
text-only autoregression). Run: python tests/test_artemis_stage_gen.py
"""
import sys

if "pytest" in sys.modules:
    import pytest
    pytest.skip(
        "Hardware smoke script — requires real A2 checkpoint + ML stack; "
        "run via 'python tests/test_artemis_stage_gen.py'",
        allow_module_level=True,
    )

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/jupyter/Merlina")
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from src.artemis_vlm import (
    ArtemisVLMForConditionalGeneration, ArtemisVLMProcessor, artemis_loss_fn,
)

A2 = "/home/jupyter/schneewolf-a1/A2-ep1-loadable"

print("=== assemble ===", flush=True)
vcfg = Qwen3VLVisionConfig(depth=2)
model = ArtemisVLMForConditionalGeneration.from_a2_and_vision(
    A2, vision_model=Qwen3VLVisionModel(vcfg).eval(), image_token_id=22,
    torch_dtype=torch.bfloat16,
).eval()
dev = model.language_model.device
tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
proc = ArtemisVLMProcessor(tokenizer=tok, vision_config=vcfg)

print("=== staged-freeze toggles ===", flush=True)
tr1, tot = model.set_training_stage("stage1")
proj = sum(p.numel() for p in model.multi_modal_projector.parameters())
lm_tr_1 = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
print(f"  stage1: trainable={tr1/1e6:.1f}M / {tot/1e9:.1f}B | projector={proj/1e6:.1f}M | LM_trainable={lm_tr_1}", flush=True)
assert lm_tr_1 == 0, "stage1 must freeze the decoder"
assert abs(tr1 - proj) / proj < 0.01, "stage1 trainable should be ~the projector only"

tr2, _ = model.set_training_stage("stage2")
lm_tr_2 = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
vis_tr_2 = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)
print(f"  stage2: trainable={tr2/1e9:.2f}B | LM_trainable>0={lm_tr_2>0} | vision_frozen={vis_tr_2==0}", flush=True)
assert lm_tr_2 > 0 and tr2 > tr1 * 100, "stage2 must unfreeze the decoder"
assert vis_tr_2 == 0, "vision frozen by default in stage2"

print("=== multimodal generate() (image only at step 0) ===", flush=True)
img = Image.fromarray((np.random.rand(56, 84, 3) * 255).astype("uint8"))
msgs = [{"role": "user", "content": [{"type": "image"},
        {"type": "text", "text": "Describe the image in one word."}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
b = proc(text=text, images=[img], return_tensors="pt")
b = {k: v.to(dev) for k, v in b.items()}
in_len = b["input_ids"].shape[1]
with torch.no_grad():
    gen = model.generate(
        input_ids=b["input_ids"], attention_mask=b["attention_mask"],
        pixel_values=b["pixel_values"], image_grid_thw=b["image_grid_thw"],
        max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id,
    )
new = gen.shape[1] - in_len
print(f"  prompt_len={in_len}  generated={new} new tokens  ok={new>0}", flush=True)
print(f"  sample: {tok.decode(gen[0][in_len:], skip_special_tokens=False)[:120]!r}", flush=True)
assert new > 0, "generate produced no tokens"

print("=== grimoire loss adapter ===", flush=True)
model.set_training_stage("stage1")
labels = b["input_ids"].clone(); labels[b["input_ids"] == 22] = -100
loss, metrics = artemis_loss_fn(model, {**b, "labels": labels})
print(f"  artemis_loss_fn -> loss={loss.item():.4f} metrics={metrics}", flush=True)
assert torch.isfinite(loss)

print("\nARTEMIS_STAGE_GEN: PASS — staged-freeze + multimodal generate + loss adapter all work",
      flush=True)
