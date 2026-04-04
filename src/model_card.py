"""
Model card (README.md) generation, W&B run naming, and HuggingFace upload.

Kept separate from training_runner to avoid heavy imports (torch, grimoire)
in tests and other lightweight consumers.
"""

import os
import logging
from typing import Any

from huggingface_hub import HfApi

from src.preflight_checks import is_local_model_path
from src.utils import calculate_effective_batch_size

logger = logging.getLogger(__name__)


def generate_model_readme(config: Any, training_mode: str, is_vlm: bool = False) -> str:
    """
    Generate a README.md for the HuggingFace model card.

    Creates a model card with YAML frontmatter containing metadata about the
    training configuration, base model, and dataset used.

    Args:
        config: Training configuration object
        training_mode: 'sft' or 'orpo'
        is_vlm: Whether the model is a vision-language model

    Returns:
        README content as a string
    """
    # Extract base model name (handle both HF IDs and local paths)
    base_model = config.base_model
    if is_local_model_path(base_model):
        # For local models, just use the directory name
        base_model_display = os.path.basename(base_model.rstrip('/'))
    else:
        base_model_display = base_model

    # Build YAML frontmatter
    frontmatter_lines = [
        "---",
        "library_name: transformers",
    ]

    # Set pipeline_tag so HuggingFace correctly classifies the model
    if is_vlm:
        frontmatter_lines.append("pipeline_tag: image-text-to-text")
    else:
        frontmatter_lines.append("pipeline_tag: text-generation")

    # Add keyword tags for discoverability
    frontmatter_lines.append("tags:")
    frontmatter_lines.append("- merlina")
    frontmatter_lines.append("- grimoire")
    if is_vlm:
        frontmatter_lines.append("- image-text-to-text")
        frontmatter_lines.append("- vision-language-model")
    else:
        frontmatter_lines.append("- text-generation")
    frontmatter_lines.append(f"- {training_mode.lower()}")  # sft or orpo

    # Add dataset info if using HuggingFace dataset
    if hasattr(config, 'dataset') and hasattr(config.dataset, 'source'):
        source = config.dataset.source
        if source.source_type == "huggingface" and source.repo_id:
            frontmatter_lines.append(f"datasets:")
            frontmatter_lines.append(f"- {source.repo_id}")

    # Add base model
    frontmatter_lines.append(f"base_model:")
    frontmatter_lines.append(f"- {base_model}")

    frontmatter_lines.append("---")

    # Build configuration section
    effective_batch = calculate_effective_batch_size(config.batch_size, config.gradient_accumulation_steps)

    config_lines = [
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Training Mode | {training_mode.upper()} |",
        f"| Base Model | `{base_model_display}` |",
        f"| Learning Rate | {config.learning_rate} |",
        f"| Epochs | {config.num_epochs} |",
        f"| Batch Size | {config.batch_size} |",
        f"| Gradient Accumulation | {config.gradient_accumulation_steps} |",
        f"| Effective Batch Size | {effective_batch} |",
        f"| Max Sequence Length | {config.max_length} |",
        f"| Optimizer | {config.optimizer_type} |",
        f"| LR Scheduler | {config.lr_scheduler_type} |",
        f"| Warmup Ratio | {config.warmup_ratio} |",
        f"| Weight Decay | {config.weight_decay} |",
        f"| Max Grad Norm | {config.max_grad_norm} |",
        f"| Seed | {config.seed} |",
    ]

    # Add preference-specific params
    preference_modes = {"orpo", "dpo", "simpo", "cpo", "ipo", "kto"}
    if training_mode.lower() in preference_modes:
        config_lines.append(f"| Beta | {config.beta} |")
        config_lines.append(f"| Max Prompt Length | {config.max_prompt_length} |")
    if training_mode.lower() == "simpo":
        config_lines.append(f"| SimPO Gamma | {config.gamma} |")
    if training_mode.lower() in ("dpo", "cpo"):
        config_lines.append(f"| Label Smoothing | {config.label_smoothing} |")

    # Add LoRA params if enabled
    if config.use_lora:
        config_lines.append(f"| LoRA Rank (r) | {config.lora_r} |")
        config_lines.append(f"| LoRA Alpha | {config.lora_alpha} |")
        config_lines.append(f"| LoRA Dropout | {config.lora_dropout} |")
        config_lines.append(f"| Target Modules | {', '.join(config.target_modules)} |")

    # Add quantization info
    if config.use_4bit:
        config_lines.append(f"| Quantization | 4-bit (NF4) |")

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
        "",
        "---",
        "",
        "![Trained with Merlina](https://raw.githubusercontent.com/Schneewolf-Labs/Merlina/refs/heads/main/frontend/madewithmerlina_smol.png)",
        "",
        "[Merlina on GitHub](https://github.com/Schneewolf-Labs/Merlina)",
        "",
    ]

    return '\n'.join(readme_parts)


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
    # Only add beta for preference modes (SFT doesn't use beta)
    preference_modes = {"orpo", "dpo", "simpo", "cpo", "ipo", "kto"}
    if config.beta != 0.1 and config.training_mode in preference_modes:
        suffixes.append(f"beta{config.beta}")

    if suffixes:
        run_name += f"-{'-'.join(suffixes)}"

    return run_name
