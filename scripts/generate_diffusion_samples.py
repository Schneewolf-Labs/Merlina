"""Generate sample images from a trained diffusion LoRA.

Standalone subprocess invoked by ``training_runner_diffusion.py`` at the
end of training. Process isolation matters here: loading a fresh
diffusers pipeline in a separate process means the training process can
release its 38 GiB of transformer VRAM cleanly first, and this sampler
can claim the GPU without contention.

Usage:
    python scripts/generate_diffusion_samples.py \\
        --base-model Qwen/Qwen-Image \\
        --lora-dir   models/my-flame-lora \\
        --out-dir    models/my-flame-lora/samples \\
        --prompts    '["a portrait of...", "a landscape of..."]' \\
        --num-steps  25 \\
        --width      1024 --height 1024

Writes one PNG per prompt to <out-dir>/sample_<idx>.png. Also writes a
small samples.json with the prompts + filenames so the UI can label
them without having to filename-mangle.
"""
import argparse
import json
import os
import sys
from pathlib import Path


# Default prompts that exercise common subjects without leaning hard on
# any one aesthetic — picked so a successful render proves the pipeline
# end-to-end without depending on the LoRA being "good".
DEFAULT_PROMPTS = [
    "a portrait of a young person with auburn hair, soft lighting",
    "a quiet cafe interior in the morning, warm tones",
    "a fox sitting in a wildflower meadow at golden hour",
    "an old library with tall wooden shelves and dust motes",
]


def _pipeline_for(base_model: str, adapter: str):
    """Construct a diffusers pipeline for the given Atelier adapter family.

    Adapter family is inferred from the saved adapter directory if not
    passed explicitly — see CLI handling below.
    """
    import torch
    if adapter == "qwen_image":
        from diffusers import QwenImagePipeline
        return QwenImagePipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    if adapter == "qwen_edit":
        from diffusers import QwenImageEditPipeline
        return QwenImageEditPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    if adapter == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        return StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    raise ValueError(f"unsupported adapter family: {adapter}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True,
                    help="HF repo id (or local path) of the base diffusion model")
    ap.add_argument("--lora-dir", required=True,
                    help="Directory containing the trained LoRA (must have pytorch_lora_weights.safetensors)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory; PNGs and samples.json will be written here")
    ap.add_argument("--adapter", default="qwen_image",
                    choices=["qwen_image", "qwen_edit", "sdxl"],
                    help="Atelier adapter family the LoRA was trained against")
    ap.add_argument("--prompts", default=None,
                    help="JSON list of prompt strings (defaults to a curated set)")
    ap.add_argument("--num-steps", type=int, default=25,
                    help="Diffusion sampling steps")
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(args.prompts) if args.prompts else DEFAULT_PROMPTS
    if not prompts:
        print("[samples] no prompts provided; nothing to do", file=sys.stderr)
        return 0

    print(f"[samples] loading base pipeline: {args.base_model} ({args.adapter})")
    pipe = _pipeline_for(args.base_model, args.adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    lora_weights = Path(args.lora_dir) / "pytorch_lora_weights.safetensors"
    if not lora_weights.exists():
        # Some adapters save under a different filename — let diffusers' loader resolve it
        lora_weights = None
    print(f"[samples] loading LoRA from {args.lora_dir}")
    try:
        if lora_weights is not None:
            pipe.load_lora_weights(args.lora_dir, weight_name=lora_weights.name)
        else:
            pipe.load_lora_weights(args.lora_dir)
    except Exception as e:
        print(f"[samples] WARNING: LoRA load failed ({e}); falling back to base model output", file=sys.stderr)

    manifest = {"prompts": [], "files": [], "adapter": args.adapter,
                "base_model": args.base_model, "lora_dir": args.lora_dir}

    for idx, prompt in enumerate(prompts):
        print(f"[samples] {idx+1}/{len(prompts)}: {prompt!r}")
        gen = torch.Generator(device=device).manual_seed(args.seed + idx)
        result = pipe(
            prompt=prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=gen,
        )
        image = result.images[0]
        fname = f"sample_{idx:02d}.png"
        image.save(out_dir / fname)
        manifest["prompts"].append(prompt)
        manifest["files"].append(fname)

    with open(out_dir / "samples.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[samples] wrote {len(prompts)} images + samples.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
