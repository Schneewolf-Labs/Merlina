"""
Model card (README.md) generation, W&B run naming, and HuggingFace upload.

Kept separate from training_runner to avoid heavy imports (torch, grimoire)
in tests and other lightweight consumers.
"""

import json
import os
import logging
from typing import Any, Optional

from huggingface_hub import HfApi

from src.preflight_checks import is_local_model_path
from src.utils import calculate_effective_batch_size
from src.config_manager import (
    SECRET_FIELDS as _CONFIG_SECRET_FIELDS,
    SCHEMA_NAME as _CONFIG_SCHEMA_NAME,
    SCHEMA_VERSION as _CONFIG_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


def generate_model_readme(
    config: Any,
    training_mode: str,
    is_vlm: bool = False,
    is_diffusion: bool = False,
) -> str:
    """
    Generate a README.md for the HuggingFace model card.

    Creates a model card with YAML frontmatter containing metadata about the
    training configuration, base model, and dataset used.

    Args:
        config: Training configuration object
        training_mode: 'sft' or 'orpo' for LLM/VLM, 'diffusion_qwen_image' etc.
            for diffusion
        is_vlm: Whether the model is a vision-language model
        is_diffusion: Whether the model is a diffusion LoRA (Qwen-Image,
            Qwen-Image-Edit, SDXL). Switches library / pipeline_tag, picks
            base_model from config.model_name, and embeds a sample preview.

    Returns:
        README content as a string
    """
    # Diffusion runs put the base model in `model_name`; LLM/VLM runs use
    # `base_model`. Fall back the other way for safety.
    if is_diffusion:
        base_model = getattr(config, "model_name", None) or getattr(config, "base_model", "")
    else:
        base_model = config.base_model
    if is_local_model_path(base_model):
        # For local models, just use the directory name
        base_model_display = os.path.basename(base_model.rstrip('/'))
    else:
        base_model_display = base_model

    # Build YAML frontmatter — library/pipeline differ across the three
    # model families so HF classifies the repo correctly.
    frontmatter_lines = ["---"]
    if is_diffusion:
        frontmatter_lines.append("library_name: diffusers")
        # Edit adapter is image-conditioned; Qwen-Image and SDXL are pure T2I.
        if "edit" in training_mode.lower():
            frontmatter_lines.append("pipeline_tag: image-to-image")
        else:
            frontmatter_lines.append("pipeline_tag: text-to-image")
    else:
        frontmatter_lines.append("library_name: transformers")
        if is_vlm:
            frontmatter_lines.append("pipeline_tag: image-text-to-text")
        else:
            frontmatter_lines.append("pipeline_tag: text-generation")

    # Add keyword tags for discoverability
    frontmatter_lines.append("tags:")
    frontmatter_lines.append("- merlina")
    if is_diffusion:
        frontmatter_lines.append("- atelier")
        frontmatter_lines.append("- lora")
        frontmatter_lines.append("- diffusion")
        if "edit" in training_mode.lower():
            frontmatter_lines.append("- image-to-image")
        elif "sdxl" in training_mode.lower():
            frontmatter_lines.append("- stable-diffusion-xl")
            frontmatter_lines.append("- text-to-image")
        else:
            frontmatter_lines.append("- text-to-image")
    else:
        frontmatter_lines.append("- grimoire")
        if is_vlm:
            frontmatter_lines.append("- image-text-to-text")
            frontmatter_lines.append("- vision-language-model")
        else:
            frontmatter_lines.append("- text-generation")
    frontmatter_lines.append(f"- {training_mode.lower()}")

    # Add dataset info if using HuggingFace dataset(s). Includes every
    # HuggingFace source the user concatenated — primary + additional.
    hf_repo_ids = []
    if hasattr(config, 'dataset') and hasattr(config.dataset, 'source'):
        primary = config.dataset.source
        if primary.source_type == "huggingface" and primary.repo_id:
            hf_repo_ids.append(primary.repo_id)
        for extra in getattr(config.dataset, 'additional_sources', None) or []:
            if getattr(extra, 'source_type', None) == "huggingface" and getattr(extra, 'repo_id', None):
                hf_repo_ids.append(extra.repo_id)

    if hf_repo_ids:
        frontmatter_lines.append("datasets:")
        for repo_id in hf_repo_ids:
            frontmatter_lines.append(f"- {repo_id}")

    # Add base model
    frontmatter_lines.append(f"base_model:")
    frontmatter_lines.append(f"- {base_model}")

    frontmatter_lines.append("---")

    # Build configuration section. Each field is read via getattr so post-hoc
    # uploads (which build a minimal SimpleNamespace) produce a partial table
    # rather than crashing.
    _MISSING = object()

    def _row(label: str, value: Any) -> str | None:
        if value is _MISSING or value is None:
            return None
        return f"| {label} | {value} |"

    batch_size = getattr(config, "batch_size", _MISSING)
    grad_accum = getattr(config, "gradient_accumulation_steps", _MISSING)
    if batch_size is not _MISSING and grad_accum is not _MISSING:
        effective_batch: Any = calculate_effective_batch_size(batch_size, grad_accum)
    else:
        effective_batch = _MISSING

    config_lines = [
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Training Mode | {training_mode.upper()} |",
        f"| Base Model | `{base_model_display}` |",
    ]
    optional_rows = [
        _row("Learning Rate", getattr(config, "learning_rate", _MISSING)),
        _row("Epochs", getattr(config, "num_epochs", _MISSING)),
        _row("Batch Size", batch_size),
        _row("Gradient Accumulation", grad_accum),
        _row("Effective Batch Size", effective_batch),
        _row("Max Sequence Length", getattr(config, "max_length", _MISSING)),
        _row("Optimizer", getattr(config, "optimizer_type", _MISSING)),
        _row("LR Scheduler", getattr(config, "lr_scheduler_type", _MISSING)),
        _row("Warmup Ratio", getattr(config, "warmup_ratio", _MISSING)),
        _row("Weight Decay", getattr(config, "weight_decay", _MISSING)),
        _row("Max Grad Norm", getattr(config, "max_grad_norm", _MISSING)),
        _row("Seed", getattr(config, "seed", _MISSING)),
    ]
    config_lines.extend(r for r in optional_rows if r is not None)

    # Add preference-specific params (best-effort)
    preference_modes = {"orpo", "dpo", "simpo", "cpo", "ipo", "kto"}
    if training_mode.lower() in preference_modes:
        for label, attr in (("Beta", "beta"), ("Max Prompt Length", "max_prompt_length")):
            row = _row(label, getattr(config, attr, _MISSING))
            if row:
                config_lines.append(row)
    if training_mode.lower() == "simpo":
        row = _row("SimPO Gamma", getattr(config, "gamma", _MISSING))
        if row:
            config_lines.append(row)
    if training_mode.lower() in ("dpo", "cpo"):
        row = _row("Label Smoothing", getattr(config, "label_smoothing", _MISSING))
        if row:
            config_lines.append(row)

    # Add LoRA params. Diffusion runs always use LoRA and write to
    # `lora_rank` / `lora_target_modules`; LLM/VLM runs gate on use_lora
    # and write to `lora_r` / `target_modules`.
    if is_diffusion or getattr(config, "use_lora", False):
        rank_attr = "lora_rank" if is_diffusion else "lora_r"
        rank_label = "LoRA Rank" if is_diffusion else "LoRA Rank (r)"
        for label, attr in (
            (rank_label, rank_attr),
            ("LoRA Alpha", "lora_alpha"),
            ("LoRA Dropout", "lora_dropout"),
        ):
            row = _row(label, getattr(config, attr, _MISSING))
            if row:
                config_lines.append(row)
        if is_diffusion and getattr(config, "lora_use_dora", False):
            config_lines.append("| DoRA | true (Weight-Decomposed LoRA) |")
        targets_attr = "lora_target_modules" if is_diffusion else "target_modules"
        target_modules = getattr(config, targets_attr, _MISSING)
        if target_modules is not _MISSING and target_modules:
            config_lines.append(f"| Target Modules | {', '.join(target_modules)} |")

    # Add quantization info
    if getattr(config, "use_4bit", False):
        config_lines.append("| Quantization | 4-bit (NF4) |")

    # Add GPU info (torch may not be available in test/CI)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                config_lines.append(f"| GPU | {gpu_name} (x{gpu_count}) |")
            else:
                config_lines.append(f"| GPU | {gpu_name} |")
    except ImportError:
        pass

    # Build complete README
    readme_parts = [
        '\n'.join(frontmatter_lines),
        "",
        f"# {config.output_name}",
        "",
        "## Training Configuration",
        "",
        '\n'.join(config_lines),
    ]

    # Diffusion: usage snippet + sample preview so the model card actually
    # tells someone how to load it. The preview image references the LoRA's
    # samples/sample_00.png (the post-training renderer always writes this).
    if is_diffusion:
        usage = _build_diffusion_usage_section(config, training_mode, base_model)
        if usage:
            readme_parts.extend(["", usage])

    # If multiple datasets were concatenated, surface them in a body section
    # so readers see more than just the YAML frontmatter.
    dataset_section = _build_dataset_section(config)
    if dataset_section:
        readme_parts.extend(["", dataset_section])

    # Optionally embed the full training config so a reader can drop it
    # into their own Merlina instance and reproduce the run with one click.
    # Gated by ``share_config`` (defaults to True). Secrets are always
    # stripped — see _build_share_config_section for the contract.
    share_section = _build_share_config_section(config)
    if share_section:
        readme_parts.extend(["", share_section])

    # Optionally reference the shareable config image (merlina_config.png) —
    # a scannable QR code that also carries the full config in its PNG
    # metadata. Gated by ``share_config_image`` (defaults to False). Works
    # independently of the JSON block above so a user can publish the image
    # *instead of* the giant JSON, both, or neither.
    image_section = _build_config_image_section(config)
    if image_section:
        readme_parts.extend(["", image_section])

    readme_parts.extend([
        "",
        "---",
        "",
        "![Trained with Merlina](https://raw.githubusercontent.com/Schneewolf-Labs/Merlina/refs/heads/main/frontend/madewithmerlina_smol.png)",
        "",
        "[Merlina on GitHub](https://github.com/Schneewolf-Labs/Merlina)",
        "",
    ])

    return '\n'.join(readme_parts)


def _build_diffusion_usage_section(config: Any, training_mode: str, base_model: str) -> str:
    """Render a "How to use this LoRA" block for diffusion model cards.

    Includes a sample preview image (samples/sample_00.png, written by the
    post-training renderer) and a minimal diffusers usage snippet keyed to
    the adapter family. Returns "" if anything required is missing.
    """
    if not base_model:
        return ""

    mode_lc = training_mode.lower()
    # Pick the right pipeline class for the snippet. Match the same
    # adapter-family mapping the runner uses.
    if "qwen_edit" in mode_lc or mode_lc == "diffusion_qwen_edit":
        pipeline_cls = "QwenImageEditPipeline"
    elif "qwen_image" in mode_lc or mode_lc == "diffusion_qwen_image":
        pipeline_cls = "QwenImagePipeline"
    elif "sdxl" in mode_lc:
        pipeline_cls = "StableDiffusionXLPipeline"
    else:
        pipeline_cls = "DiffusionPipeline"

    output_name = getattr(config, "output_name", "")
    parts = ["## Sample", ""]
    parts.append("![preview](samples/sample_00.png)")
    parts.extend([
        "",
        "More samples — including mid-training snapshots if `mid_training_samples` "
        "was enabled — are in the `samples/` directory of this repo.",
        "",
        "## Usage",
        "",
        "```python",
        "import torch",
        f"from diffusers import {pipeline_cls}",
        "",
        f'pipe = {pipeline_cls}.from_pretrained("{base_model}", torch_dtype=torch.bfloat16)',
        'pipe.enable_model_cpu_offload()  # fits Qwen-Image on a 24GB card; drop for SDXL',
        "",
        f'pipe.load_lora_weights("{output_name}")',
        "",
        'image = pipe(prompt="…your prompt here…", num_inference_steps=25).images[0]',
        'image.save("out.png")',
        "```",
    ])
    return "\n".join(parts)


# Publishing toggles that gate *how* the config is shared — not training
# hyperparameters, so they're stripped from the shared payload itself.
_SHARE_TOGGLE_FIELDS = ("share_config", "share_config_image")


def build_shareable_config_envelope(config: Any) -> Optional[dict]:
    """
    Build the secret-stripped, importable config envelope for ``config``.

    This is the single source of truth for the shareable payload — used by
    both the README JSON block (:func:`_build_share_config_section`) and the
    QR/metadata image (:func:`upload_config_image`). It does **not** consult
    any ``share_*`` toggle; callers gate on their own flag first.

    Returns ``None`` when the config can't be serialized (e.g. a Mock in
    tests, or a plain object without ``model_dump``).
    """
    # Pydantic models expose model_dump(); plain dicts/Mocks don't, in
    # which case we silently skip rather than crash.
    try:
        payload = config.model_dump(mode="json")
    except (AttributeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    # Drop secrets and the publishing toggles themselves (they control how
    # the config is shared, not how the model was trained).
    payload = {
        k: v for k, v in payload.items()
        if k not in _CONFIG_SECRET_FIELDS and k not in _SHARE_TOGGLE_FIELDS
    }

    # Wrap with the same schema header as data/configs/*.json so the blob is
    # importable through /configs/import (or the image loader) without ceremony.
    try:
        from version import __version__ as _merlina_version
    except Exception:  # pragma: no cover
        _merlina_version = "unknown"

    return {
        "_metadata": {
            "name": getattr(config, "output_name", ""),
            "description": (
                "Training configuration shared from a Merlina-trained model."
            ),
            "tags": [],
            "schema": _CONFIG_SCHEMA_NAME,
            "schema_version": _CONFIG_SCHEMA_VERSION,
            "merlina_version": _merlina_version,
        },
        **payload,
    }


def _build_share_config_section(config: Any) -> str:
    """
    Render the optional "Reproduce this training run" block.

    Returns an empty string when ``share_config`` is False (or absent on a
    legacy/test config that pre-dates the field), or when the config can't
    be serialized (e.g. a Mock in tests). Otherwise returns markdown with:

      * A short note explaining how to import the config
      * A fenced JSON block containing the validated, secret-stripped
        TrainingConfig payload (same shape /configs/save produces)

    Secrets (``hf_token``, ``wandb_key``) are stripped unconditionally.
    """
    if not getattr(config, "share_config", False):
        return ""

    envelope = build_shareable_config_envelope(config)
    if envelope is None:
        return ""

    try:
        rendered = json.dumps(envelope, indent=2, sort_keys=False)
    except (TypeError, ValueError):
        return ""

    lines = [
        "## Reproduce this training run",
        "",
        "This model was trained with [Merlina](https://github.com/Schneewolf-Labs/Merlina). "
        "Save the JSON below to `data/configs/<name>.json` (or import it via the "
        "*Load Configuration* dialog) to reproduce the exact training setup. "
        "Credentials are not included — Merlina will use your own `HF_TOKEN` "
        "and `WANDB_API_KEY` from `.env` or the form.",
        "",
        "```json",
        rendered,
        "```",
    ]
    return "\n".join(lines)


# Filename used for the shareable config image, both in the repo and in the
# README reference. Kept here so the README section and the uploader agree.
CONFIG_IMAGE_FILENAME = "merlina_config.png"


def _build_config_image_section(config: Any) -> str:
    """
    Render the optional "Reproduce this run — scan or load" image block.

    Returns markdown referencing :data:`CONFIG_IMAGE_FILENAME` when
    ``share_config_image`` is True, else "". The image itself is uploaded
    separately by :func:`upload_config_image`; here we just point at it.
    """
    if not getattr(config, "share_config_image", False):
        return ""

    return "\n".join([
        "## Reproduce this training run (QR)",
        "",
        f"![Merlina training config]({CONFIG_IMAGE_FILENAME})",
        "",
        "Scan the QR code above, or download "
        f"`{CONFIG_IMAGE_FILENAME}` and load it via Merlina's *Load "
        "Configuration → From Image* button. The full (secret-stripped) "
        "training config is embedded in the image's metadata, so Merlina "
        "can reproduce this exact run from the PNG alone.",
    ])


def upload_config_image(repo_id: str, config: Any, token: str) -> None:
    """
    Generate and upload the shareable config image to a HuggingFace repo.

    Best-effort and gated by ``share_config_image``: does nothing when the
    flag is off, when the optional ``qrcode``/Pillow deps are missing, or on
    any upload error (logged, never raised) — a failure here must not abort
    the surrounding model upload.
    """
    if not getattr(config, "share_config_image", False):
        return

    try:
        from src.config_image import build_config_image_bytes

        envelope = build_shareable_config_envelope(config)
        if envelope is None:
            logger.debug("config image: envelope unavailable; skipping upload")
            return

        png_bytes = build_config_image_bytes(envelope)
        if not png_bytes:
            return  # build_config_image_bytes already logged the reason

        api = HfApi()
        full_repo_id = repo_id
        if "/" not in full_repo_id:
            user_info = api.whoami(token=token)
            username = user_info.get("name") or user_info.get("username")
            full_repo_id = f"{username}/{full_repo_id}"

        api.upload_file(
            path_or_fileobj=png_bytes,
            path_in_repo=CONFIG_IMAGE_FILENAME,
            repo_id=full_repo_id,
            token=token,
            commit_message="Add shareable training config (QR + PNG metadata)",
        )
        logger.info("%s uploaded to %s", CONFIG_IMAGE_FILENAME, full_repo_id)
    except Exception as exc:
        logger.warning("Could not upload config image: %s", exc)


def _build_dataset_section(config: Any) -> str:
    """
    Return a markdown section listing every dataset used for training when
    more than one source was configured. Returns '' when there's a single
    source (the YAML frontmatter alone is enough in that case).
    """
    if not hasattr(config, 'dataset') or not hasattr(config.dataset, 'source'):
        return ""
    sources = [config.dataset.source]
    sources.extend(getattr(config.dataset, 'additional_sources', None) or [])
    if len(sources) < 2:
        return ""

    def describe(src):
        t = getattr(src, 'source_type', None)
        if t == "huggingface" and getattr(src, 'repo_id', None):
            split = getattr(src, 'split', None) or 'train'
            return f"[`{src.repo_id}`](https://huggingface.co/datasets/{src.repo_id}) (split: `{split}`)"
        if t == "local_file" and getattr(src, 'file_path', None):
            return f"local file: `{src.file_path}`"
        if t == "upload":
            return "uploaded file"
        return f"source: `{t or 'unknown'}`"

    lines = ["## Datasets", "", f"Trained on {len(sources)} concatenated datasets:", ""]
    for i, src in enumerate(sources, start=1):
        lines.append(f"{i}. {describe(src)}")
    return '\n'.join(lines)


def upload_model_readme(repo_id: str, readme_content: str, token: str) -> None:
    """
    Upload a README.md to an existing HuggingFace repository.

    Args:
        repo_id: The repository ID (e.g., 'model-name' or 'username/model-name')
        readme_content: The README.md content to upload
        token: HuggingFace API token
    """
    api = HfApi()

    # If repo_id doesn't contain a slash, prepend the username
    if '/' not in repo_id:
        user_info = api.whoami(token=token)
        username = user_info.get('name') or user_info.get('username')
        repo_id = f"{username}/{repo_id}"
        logger.info(f"Full repository path: {repo_id}")

    # Upload README.md
    api.upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        commit_message="Add model card with training configuration"
    )

    logger.info(f"README.md uploaded to {repo_id}")


def generate_wandb_run_name(config: Any) -> str:
    """
    Generate a descriptive W&B run name from training configuration.

    Format: [model_name]-[training_mode]-[lora_r]-[lr]-[batch]-[epochs]ep-[optimizer]-[attention]
    Example: llama3-8b-orpo-r64-5e-6LR-256B-2ep-adamw8bit-flash2
    Example (no LoRA): llama3-8b-sft-full-5e-6LR-256B-2ep-adamw8bit-flash2

    Args:
        config: Training configuration object

    Returns:
        Generated run name string
    """
    # Extract model name from path (e.g., "meta-llama/Llama-3-8B" -> "llama3-8b")
    model_name = config.base_model.split('/')[-1].lower()
    # Simplify common model names
    model_name = model_name.replace('-instruct', '').replace('-base', '').replace('meta-', '')

    # Format learning rate (e.g., 0.000005 -> "5e-6")
    lr = config.learning_rate
    if lr >= 1e-3:
        lr_str = f"{lr:.0e}".replace('e-0', 'e-')
    else:
        lr_str = f"{lr:.0e}".replace('e-0', 'e-')

    # Calculate effective batch size
    effective_batch = calculate_effective_batch_size(config.batch_size, config.gradient_accumulation_steps)

    # Simplify optimizer name
    opt = config.optimizer_type.replace('paged_', '').replace('_', '')

    # Simplify attention
    attn_map = {
        'flash_attention_2': 'flash2',
        'sdpa': 'sdpa',
        'eager': 'eager',
        'auto': 'auto'
    }
    attn = attn_map.get(config.attn_implementation, config.attn_implementation)

    # Build run name - include training mode and LoRA rank or "full" if not using LoRA
    parts = [
        model_name,
        config.training_mode.lower(),
        f"r{config.lora_r}" if config.use_lora else "full",
        f"{lr_str}LR",
        f"{effective_batch}B",
        f"{config.num_epochs}ep",
        opt,
        attn
    ]

    run_name = "-".join(parts)

    # Add optional suffix for special settings
    suffixes = []
    if config.use_4bit:
        suffixes.append("4bit")
    if config.gradient_checkpointing:
        suffixes.append("gc")
    if getattr(config, "use_liger", False):
        suffixes.append("liger")
    if getattr(config, "torch_compile", False):
        suffixes.append("compile")
    # Only add beta for preference modes (SFT doesn't use beta)
    preference_modes = {"orpo", "dpo", "simpo", "cpo", "ipo", "kto"}
    if config.beta != 0.1 and config.training_mode in preference_modes:
        suffixes.append(f"beta{config.beta}")

    if suffixes:
        run_name += f"-{'-'.join(suffixes)}"

    return run_name
