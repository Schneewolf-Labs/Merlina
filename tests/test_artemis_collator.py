"""ArtemisDataCollator test (Project Artemis piece #13).

The win: a *batch* of variable-size images + captions collates correctly
(flat pixel_values, per-image grids, padded ids, masked labels) and a real
ArtemisVLM training step (forward WITH labels -> finite scalar loss) runs.

Run: python tests/test_artemis_collator.py
"""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/jupyter/Merlina")
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from src.artemis_vlm import (
    ArtemisVLMForConditionalGeneration,
    ArtemisVLMProcessor,
    ArtemisDataCollator,
)

A2 = "/home/jupyter/schneewolf-a1/A2-ep1-loadable"
IMG_PAD = 22

print("=== assemble model + processor + collator ===", flush=True)
vcfg = Qwen3VLVisionConfig(depth=2)
model = ArtemisVLMForConditionalGeneration.from_a2_and_vision(
    A2, vision_model=Qwen3VLVisionModel(vcfg).eval(),
    image_token_id=IMG_PAD, torch_dtype=torch.bfloat16,
).eval()
dev = model.language_model.device
tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
proc = ArtemisVLMProcessor(tokenizer=tok, vision_config=vcfg)
collator = ArtemisDataCollator(proc)

def ex(hw, caption):
    h, w = hw
    img = Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"))
    return {
        "images": [img],
        "messages": [
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "Describe the image briefly."}]},
            {"role": "assistant", "content": caption},
        ],
    }

feats = [ex((56, 84), "A red square on white."),
         ex((84, 56), "Two blue circles."),
         ex((70, 70), "A green triangle in the corner of the frame.")]

print("=== collate (varying image sizes -> flat pixel_values, per-image grids) ===", flush=True)
batch = collator(feats)
B = len(feats)
S = batch["input_ids"].shape[1]
print("  input_ids", tuple(batch["input_ids"].shape),
      "attention_mask", tuple(batch["attention_mask"].shape),
      "labels", tuple(batch["labels"].shape), flush=True)
print("  pixel_values", tuple(batch["pixel_values"].shape),
      "image_grid_thw", batch["image_grid_thw"].tolist(), flush=True)
assert batch["input_ids"].shape == (B, S)
assert batch["labels"].shape == (B, S)
assert batch["image_grid_thw"].shape == (B, 3)
# flat pixel_values rows == sum of patches across the 3 images
exp_patches = int(sum(int(g.prod()) for g in batch["image_grid_thw"]))
assert batch["pixel_values"].shape[0] == exp_patches, (batch["pixel_values"].shape[0], exp_patches)
assert batch["pixel_values"].shape[1] == 3 * vcfg.temporal_patch_size * vcfg.patch_size ** 2

print("=== label masking sanity ===", flush=True)
for i in range(B):
    ids, lab = batch["input_ids"][i], batch["labels"][i]
    learnable = (lab != -100).sum().item()
    pad_masked = ((ids == IMG_PAD) & (lab == -100)).sum().item()
    n_pad = (ids == IMG_PAD).sum().item()
    print(f"  ex{i}: learnable_tokens={learnable}  image_pads={n_pad} (all masked: {pad_masked==n_pad})", flush=True)
    assert learnable > 0, "caption must be learnable"
    assert pad_masked == n_pad, "every <|image_pad|> must be label-masked"

print("=== real training step: forward WITH labels -> finite loss ===", flush=True)
batch = {k: v.to(dev) for k, v in batch.items()}
with torch.no_grad():
    out = model(**batch)
print(f"  loss={out.loss.item():.4f}  logits={tuple(out.logits.shape)}", flush=True)
assert torch.isfinite(out.loss), "loss not finite"
assert out.logits.shape == (B, S, model.config.text_config.vocab_size)

print("\nARTEMIS_COLLATOR: PASS — batched multimodal collate + masked-label "
      "training step runs end-to-end with finite loss", flush=True)
