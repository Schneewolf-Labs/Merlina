"""Standalone smoke test for ArtemisVLM (Project Artemis piece #1).

Real A2 decoder + tiny random Qwen3-VL ViT. Validates the assembled class:
the MERGED vision path (pooler_output), the projector dim bridge, the
<|image_pad|> splice contract, and an end-to-end forward through the
unmodified A2 decoder.

Run: python tests/test_artemis_vlm.py
"""
import sys
import torch

sys.path.insert(0, "/home/jupyter/Merlina")
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from src.artemis_vlm import ArtemisVLMForConditionalGeneration

A2 = "/home/jupyter/schneewolf-a1/A2-ep1-loadable"
IMG_PAD, VIS_START, VIS_END = 22, 20, 21

print("=== assemble ArtemisVLM (real A2 + tiny random Qwen3-VL ViT) ===", flush=True)
vcfg = Qwen3VLVisionConfig(depth=2)                       # tiny for speed; arch preserved
vis = Qwen3VLVisionModel(vcfg).eval()
model = ArtemisVLMForConditionalGeneration.from_a2_and_vision(
    A2, vision_model=vis, image_token_id=IMG_PAD, torch_dtype=torch.bfloat16
)
model.eval()
dev = model.language_model.device
tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
th = model.config.text_config.hidden_size
print(f"  text_hidden={th}  vision_out={vcfg.out_hidden_size}  "
      f"patch={vcfg.patch_size} merge={vcfg.spatial_merge_size} "
      f"image_token_id={model.config.image_token_id}", flush=True)

# synthetic single image: grid (t=1, h=4, w=4) -> 16 patches; merged -> 16//merge^2
t, h, w = 1, 4, 4
ppd = 3 * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
pixel_values = torch.randn(t * h * w, ppd, dtype=torch.bfloat16, device=dev)
grid = torch.tensor([[t, h, w]], device=dev)
exp_tokens = (t * h * w) // (vcfg.spatial_merge_size ** 2)

print("=== get_image_features uses MERGED pooler_output ===", flush=True)
with torch.no_grad():
    feats = model.get_image_features(pixel_values, grid)
print(f"  image features: {tuple(feats.shape)}  (expect ({exp_tokens}, {th}))", flush=True)
assert feats.shape == (exp_tokens, th), f"merged-path wrong: {tuple(feats.shape)}"

print("=== end-to-end forward (splice @ <|image_pad|> -> A2 decoder) ===", flush=True)
pre = tok("Describe the image:", add_special_tokens=False)["input_ids"]
post = tok("\nWhat is shown?", add_special_tokens=False)["input_ids"]
ids = pre + [VIS_START] + [IMG_PAD] * exp_tokens + [VIS_END] + post
input_ids = torch.tensor([ids], device=dev)
with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=pixel_values,
        image_grid_thw=grid,
    )
S, V = input_ids.shape[1], model.config.text_config.vocab_size
print(f"  logits: {tuple(out.logits.shape)}  (expect (1, {S}, {V}))", flush=True)
assert out.logits.shape == (1, S, V)

# negative check: the splice contract must raise on a count mismatch
print("=== splice-contract guard fires on mismatch ===", flush=True)
bad = torch.tensor([pre + [IMG_PAD] * (exp_tokens + 1) + post], device=dev)
try:
    with torch.no_grad():
        model(input_ids=bad, attention_mask=torch.ones_like(bad),
              pixel_values=pixel_values, image_grid_thw=grid)
    raise AssertionError("expected ValueError on token/feature mismatch")
except ValueError as e:
    print("  raised as expected:", str(e)[:80], flush=True)

print("\nARTEMIS_VLM SMOKE: PASS — class assembles, merged vision path correct, "
      "splice + A2 forward + contract guard all work", flush=True)
