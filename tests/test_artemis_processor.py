"""ArtemisVLMProcessor integration test (Project Artemis piece #12).

The real win: a synthetic image + a chat message goes through the Qwen3
chat template -> ArtemisVLMProcessor expansion -> ArtemisVLM.forward with
NO count-mismatch ValueError. I.e. processor's <|image_pad|> expansion
exactly equals the model's merged feature count, end to end on a real image.

Run: python tests/test_artemis_processor.py
"""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/jupyter/Merlina")
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from src.artemis_vlm import ArtemisVLMForConditionalGeneration, ArtemisVLMProcessor

A2 = "/home/jupyter/schneewolf-a1/A2-ep1-loadable"
IMG_PAD = 22

print("=== assemble model + processor ===", flush=True)
vcfg = Qwen3VLVisionConfig(depth=2)
model = ArtemisVLMForConditionalGeneration.from_a2_and_vision(
    A2, vision_model=Qwen3VLVisionModel(vcfg).eval(),
    image_token_id=IMG_PAD, torch_dtype=torch.bfloat16,
).eval()
dev = model.language_model.device
tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
proc = ArtemisVLMProcessor(tokenizer=tok, vision_config=vcfg)
print(f"  image proc patch={proc.image_processor.patch_size} "
      f"merge_len={proc.merge_length} image_token_id={proc.image_token_id}", flush=True)

print("=== chat template emits one <|image_pad|> for image content ===", flush=True)
img = Image.fromarray((np.random.rand(56, 84, 3) * 255).astype("uint8"))
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"},
]}]
text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
n_pad_pre = text.count("<|image_pad|>")
print(f"  <|image_pad|> in templated text (pre-expansion): {n_pad_pre} "
      f"(expect 1) | has vision_start/end: "
      f"{'<|vision_start|>' in text and '<|vision_end|>' in text}", flush=True)
assert n_pad_pre == 1, f"template should emit exactly one image pad, got {n_pad_pre}"

print("=== processor expands + tokenizes; counts must match the model ===", flush=True)
batch = proc(text=text, images=[img], return_tensors="pt")
batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
g = batch["image_grid_thw"][0]
expected = int(g.prod()) // proc.merge_length
n_in_ids = int((batch["input_ids"] == IMG_PAD).sum())
print(f"  grid={g.tolist()}  expected_merged={expected}  "
      f"<|image_pad|> in input_ids={n_in_ids}  pixel_values={tuple(batch['pixel_values'].shape)}", flush=True)
assert n_in_ids == expected, f"expansion {n_in_ids} != merged count {expected}"

print("=== end-to-end forward (no contract ValueError = processor<->model agree) ===", flush=True)
with torch.no_grad():
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
    )
S, V = batch["input_ids"].shape[1], model.config.text_config.vocab_size
print(f"  logits: {tuple(out.logits.shape)}  (expect (1, {S}, {V}))", flush=True)
assert out.logits.shape == (1, S, V)

print("\nARTEMIS_PROCESSOR: PASS — chat template -> processor expansion -> "
      "ArtemisVLM forward agrees end-to-end on a real image", flush=True)
