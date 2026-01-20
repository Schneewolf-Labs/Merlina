"""
Pre-flight Validation for Training Jobs
Validates configuration, resources, and dependencies before starting training
"""

import torch
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


def get_num_gpus() -> int:
    """Get the number of GPUs available for training."""
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def calculate_effective_batch_size(batch_size: int, gradient_accumulation_steps: int) -> int:
    """
    Calculate the effective batch size accounting for gradient accumulation and multiple GPUs.

    Effective batch size = per_device_batch_size × gradient_accumulation_steps × num_gpus
    """
    num_gpus = get_num_gpus()
    return batch_size * gradient_accumulation_steps * num_gpus


def estimate_training_vram(
    model_size_billions: float,
    batch_size: int,
    max_length: int,
    use_4bit: bool = True,
    use_lora: bool = True,
    gradient_checkpointing: bool = False
) -> float:
    """
    Estimate VRAM requirements for training based on configuration.

    This is an approximation based on common patterns. Actual usage varies
    based on model architecture, optimizer states, and other factors.

    Args:
        model_size_billions: Model size in billions of parameters (e.g., 7 for 7B)
        batch_size: Per-device batch size
        max_length: Maximum sequence length
        use_4bit: Whether 4-bit quantization is enabled
        use_lora: Whether LoRA is enabled (reduces trainable params)
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        Estimated VRAM in GB
    """
    # Base model memory
    if use_4bit:
        # 4-bit: ~0.5 bytes per parameter + overhead
        model_mem = model_size_billions * 0.6
    else:
        # FP16: ~2 bytes per parameter
        model_mem = model_size_billions * 2.0

    # Optimizer states (AdamW: 2 states per trainable param)
    if use_lora:
        # LoRA typically trains ~0.1-1% of parameters
        trainable_ratio = 0.01
    else:
        trainable_ratio = 1.0

    # Optimizer states in FP32 (8 bytes per trainable param for AdamW)
    optimizer_mem = model_size_billions * trainable_ratio * 8

    # Activation memory (rough estimate)
    # Scales with batch_size * max_length * hidden_size
    # Using approximation: hidden_size ≈ model_size_billions * 512
    hidden_size_approx = model_size_billions * 512
    activation_mem = (batch_size * max_length * hidden_size_approx * 2) / (1024**3)

    if gradient_checkpointing:
        activation_mem *= 0.3  # Checkpointing reduces activation memory significantly

    # Total estimate with some buffer
    total = model_mem + optimizer_mem + activation_mem
    buffer = total * 0.15  # 15% buffer for fragmentation and overhead

    return total + buffer


def get_model_size_from_name(model_name: str) -> Optional[float]:
    """
    Extract model size in billions from model name.

    Args:
        model_name: Model name or path

    Returns:
        Model size in billions, or None if not detected
    """
    import re

    model_lower = model_name.lower()

    # Common patterns: "7b", "7B", "7-b", "7_b", "7billion"
    patterns = [
        r'(\d+(?:\.\d+)?)\s*b(?:illion)?(?:\b|$)',  # 7b, 7B, 7billion
        r'(\d+(?:\.\d+)?)-b(?:\b|$)',  # 7-b
        r'(\d+(?:\.\d+)?)_b(?:\b|$)',  # 7_b
    ]

    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            return float(match.group(1))

    return None


def is_local_model_path(model_path: str) -> bool:
    """
    Determine if model_path is a local directory or a HuggingFace model ID.

    Args:
        model_path: Model identifier (local path or HF repo ID)

    Returns:
        True if it's a local path, False if it's a HuggingFace model ID
    """
    # Check for path indicators
    path = Path(model_path)

    # If it starts with ./ or ../ or /, it's definitely a path
    if model_path.startswith('./') or model_path.startswith('../') or model_path.startswith('/'):
        return True

    # If it contains backslashes (Windows paths)
    if '\\' in model_path:
        return True

    # If it contains path separators beyond a single slash (org/model format)
    # and has more than one slash, likely a path
    if model_path.count('/') > 1:
        return True

    # If it exists as a directory AND contains model files, it's a local path
    # This handles cases like "models" which could be either a directory or HF username
    if path.exists() and path.is_dir():
        # Check for typical model files
        has_model_files = (
            (path / "config.json").exists() or
            (path / "pytorch_model.bin").exists() or
            (path / "model.safetensors").exists() or
            any(path.glob("*.safetensors")) or
            any(path.glob("pytorch_model*.bin"))
        )
        if has_model_files:
            return True

    # Otherwise, assume it's a HuggingFace model ID (org/model or just model)
    return False


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class PreflightValidator:
    """
    Validates training configuration and system resources before starting training.
    Provides detailed error messages and warnings.
    """

    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate_all(self, config: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all validation checks.

        Args:
            config: TrainingConfig instance

        Returns:
            Tuple of (is_valid, results_dict)
        """
        self.warnings = []
        self.errors = []

        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "checks": {}
        }

        # Run all checks
        checks = [
            ("GPU", self._check_gpu),
            ("VRAM", lambda: self._check_vram(config)),
            ("Disk Space", lambda: self._check_disk_space(config)),
            ("Model Access", lambda: self._check_model_access(config)),
            ("Dataset Config", lambda: self._check_dataset_config(config)),
            ("Training Config", lambda: self._check_training_config(config)),
            ("Tokens", lambda: self._check_tokens(config)),
            ("Dependencies", self._check_dependencies),
        ]

        for check_name, check_func in checks:
            try:
                check_result = check_func()
                results["checks"][check_name] = {
                    "status": "pass" if check_result else "fail",
                    "details": check_result if isinstance(check_result, dict) else {}
                }
            except Exception as e:
                self.errors.append(f"{check_name} check failed: {str(e)}")
                results["checks"][check_name] = {
                    "status": "error",
                    "error": str(e)
                }

        # Compile results
        results["warnings"] = self.warnings
        results["errors"] = self.errors
        results["valid"] = len(self.errors) == 0

        return results["valid"], results

    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability and capabilities"""
        if not torch.cuda.is_available():
            self.errors.append("CUDA is not available. GPU training requires CUDA.")
            return {"available": False}

        gpu_count = torch.cuda.device_count()
        gpu_info = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024**3)
            })

            # Check compute capability for flash attention
            if props.major < 8:
                self.warnings.append(
                    f"GPU {i} ({props.name}) has compute capability {props.major}.{props.minor}. "
                    "Flash Attention 2 requires 8.0+ (Ampere or newer). Will fall back to eager attention."
                )

        logger.info(f"Found {gpu_count} GPU(s): {[g['name'] for g in gpu_info]}")

        return {
            "available": True,
            "count": gpu_count,
            "devices": gpu_info
        }

    def _check_vram(self, config: Any) -> Dict[str, Any]:
        """Check if available VRAM is sufficient with detailed estimation"""
        if not torch.cuda.is_available():
            return {"skipped": "No GPU available"}

        # Get model size from name
        model_size = get_model_size_from_name(config.base_model)

        if model_size is None:
            self.warnings.append(
                f"Could not detect model size from '{config.base_model}'. "
                "Please ensure you have sufficient GPU memory."
            )
            return {"estimate_unavailable": True, "model_size_detected": False}

        # Check available VRAM
        gpu = torch.cuda.get_device_properties(0)
        available_vram = gpu.total_memory / (1024**3)

        # Use enhanced VRAM estimation
        estimated_vram = estimate_training_vram(
            model_size_billions=model_size,
            batch_size=config.batch_size,
            max_length=config.max_length,
            use_4bit=config.use_4bit,
            use_lora=config.use_lora,
            gradient_checkpointing=getattr(config, 'gradient_checkpointing', False)
        )

        # Check batch_size × max_length combination
        batch_seq_product = config.batch_size * config.max_length
        if batch_seq_product > 8192:  # Warning threshold
            self.warnings.append(
                f"Large batch×sequence product ({config.batch_size}×{config.max_length}={batch_seq_product}). "
                f"This may cause OOM. Consider reducing batch_size to 1 or max_length to {4096 // config.batch_size}."
            )

        if batch_seq_product > 16384:  # Error threshold for smaller GPUs
            if available_vram < 24:  # Less than 24GB
                self.errors.append(
                    f"Batch×sequence product ({batch_seq_product}) is very large for {available_vram:.0f}GB GPU. "
                    f"Reduce batch_size (currently {config.batch_size}) or max_length (currently {config.max_length})."
                )

        # Adjust for quantization
        if not config.use_4bit:
            self.warnings.append(
                f"Training without 4-bit quantization will require significantly more VRAM (~{estimated_vram:.1f}GB estimated). "
                "Consider enabling 4-bit quantization to reduce memory usage."
            )

        # Check if sufficient
        if available_vram < estimated_vram:
            suggestions = []
            if not config.use_4bit:
                suggestions.append("enable 4-bit quantization")
            if config.batch_size > 1:
                suggestions.append(f"reduce batch_size (currently {config.batch_size})")
            if config.max_length > 1024:
                suggestions.append(f"reduce max_length (currently {config.max_length})")
            if not getattr(config, 'gradient_checkpointing', False):
                suggestions.append("enable gradient_checkpointing")

            suggestion_text = ", ".join(suggestions) if suggestions else "use a smaller model"

            self.errors.append(
                f"Insufficient VRAM: Training ~{model_size}B model requires ~{estimated_vram:.1f}GB, "
                f"but only {available_vram:.1f}GB available. Try: {suggestion_text}."
            )
        elif available_vram < estimated_vram * 1.2:
            self.warnings.append(
                f"VRAM is tight: ~{estimated_vram:.1f}GB estimated, {available_vram:.1f}GB available. "
                "Training may fail. Consider enabling gradient_checkpointing or reducing batch size."
            )

        return {
            "available_gb": round(available_vram, 2),
            "estimated_required_gb": round(estimated_vram, 2),
            "model_size_billions": model_size,
            "batch_seq_product": batch_seq_product,
            "sufficient": available_vram >= estimated_vram
        }

    def _check_disk_space(self, config: Any) -> Dict[str, Any]:
        """Check available disk space for model checkpoints"""
        output_dir = Path(f"./models/{config.output_name}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Get disk usage
        disk_usage = psutil.disk_usage(str(output_dir.parent))
        available_gb = disk_usage.free / (1024**3)

        # Estimate space needed (model + checkpoints)
        # Rough estimate: 20GB for 7B model, scales linearly
        model_name = config.base_model.lower()
        if "70b" in model_name:
            estimated_space = 140
        elif "34b" in model_name:
            estimated_space = 70
        elif "13b" in model_name or "14b" in model_name:
            estimated_space = 30
        elif "7b" in model_name or "8b" in model_name:
            estimated_space = 20
        elif "3b" in model_name:
            estimated_space = 10
        else:
            estimated_space = 25  # Default estimate

        if available_gb < estimated_space:
            self.errors.append(
                f"Insufficient disk space: Need ~{estimated_space}GB, "
                f"but only {available_gb:.1f}GB available at {output_dir.parent}"
            )
        elif available_gb < estimated_space * 1.5:
            self.warnings.append(
                f"Low disk space: {available_gb:.1f}GB available. "
                f"Training may require ~{estimated_space}GB for checkpoints and final model."
            )

        return {
            "available_gb": available_gb,
            "estimated_required_gb": estimated_space,
            "sufficient": available_gb >= estimated_space
        }

    def _check_model_access(self, config: Any) -> Dict[str, Any]:
        """Check if model is accessible (local path or HuggingFace)"""
        model_path = config.base_model
        is_local = is_local_model_path(model_path)

        if is_local:
            # Validate local model path
            model_dir = Path(model_path)

            if not model_dir.exists():
                self.errors.append(
                    f"Local model path '{model_path}' does not exist. "
                    "Please provide a valid directory path."
                )
                return {
                    "model": model_path,
                    "is_local": True,
                    "exists": False,
                    "has_config": False
                }

            if not model_dir.is_dir():
                self.errors.append(
                    f"Local model path '{model_path}' is not a directory."
                )
                return {
                    "model": model_path,
                    "is_local": True,
                    "exists": True,
                    "has_config": False
                }

            # Check for required model files
            config_file = model_dir / "config.json"
            if not config_file.exists():
                self.warnings.append(
                    f"Local model path '{model_path}' does not contain config.json. "
                    "This may cause loading issues."
                )

            return {
                "model": model_path,
                "is_local": True,
                "exists": True,
                "has_config": config_file.exists()
            }
        else:
            # HuggingFace model - check for gated models
            gated_models = [
                "meta-llama/Llama-2",
                "meta-llama/Meta-Llama-3",
                "mistralai/Mixtral",
            ]

            is_gated = any(gated in model_path for gated in gated_models)

            if is_gated and not config.hf_token:
                self.errors.append(
                    f"Model '{model_path}' is gated and requires a HuggingFace token. "
                    "Please provide hf_token in the configuration or set HF_TOKEN environment variable."
                )

            return {
                "model": model_path,
                "is_local": False,
                "is_gated": is_gated,
                "has_token": bool(config.hf_token or os.getenv("HF_TOKEN"))
            }

    def _check_dataset_config(self, config: Any) -> Dict[str, Any]:
        """Validate dataset configuration"""
        dataset_source = config.dataset.source

        if dataset_source.source_type == "huggingface":
            if not dataset_source.repo_id:
                self.errors.append("HuggingFace dataset requires repo_id")

        elif dataset_source.source_type == "local_file":
            if not dataset_source.file_path:
                self.errors.append("Local file dataset requires file_path")
            elif not Path(dataset_source.file_path).exists():
                self.errors.append(f"Dataset file not found: {dataset_source.file_path}")

        elif dataset_source.source_type == "upload":
            if not dataset_source.dataset_id:
                self.errors.append("Upload dataset requires dataset_id")

        else:
            self.errors.append(f"Invalid dataset source_type: {dataset_source.source_type}")

        # Check format
        dataset_format = config.dataset.format

        if dataset_format.format_type == "custom":
            if not dataset_format.custom_templates:
                self.errors.append("Custom format requires custom_templates")

        return {
            "source_type": dataset_source.source_type,
            "format_type": dataset_format.format_type
        }

    def _check_training_config(self, config: Any) -> Dict[str, Any]:
        """Validate training hyperparameters"""
        issues = []

        # ===== LoRA Configuration Checks =====
        if config.use_lora:
            # Check LoRA rank bounds
            if config.lora_r > 256:
                self.warnings.append(
                    f"LoRA rank ({config.lora_r}) is very high. "
                    "Consider using 64-128 for most cases. Higher ranks increase memory usage with diminishing returns."
                )

            if config.lora_r < 8:
                self.warnings.append(
                    f"LoRA rank ({config.lora_r}) is very low. "
                    "This may limit model capacity. Consider using at least 16-32 for meaningful adaptation."
                )

            # Check LoRA alpha ratio
            if config.lora_alpha < config.lora_r:
                self.warnings.append(
                    f"LoRA alpha ({config.lora_alpha}) is less than rank ({config.lora_r}). "
                    "Typically alpha should be 1-2x the rank for stable training."
                )

            if config.lora_alpha > config.lora_r * 4:
                self.warnings.append(
                    f"LoRA alpha ({config.lora_alpha}) is much larger than rank ({config.lora_r}). "
                    "This creates a very large scaling factor which may cause instability."
                )

            # Check LoRA rank vs model size (approximate hidden size)
            model_size = get_model_size_from_name(config.base_model)
            if model_size:
                # Typical hidden sizes: 7B~4096, 13B~5120, 70B~8192
                approx_hidden = int(model_size * 512 + 2048)  # Rough approximation
                if config.lora_r > approx_hidden // 4:
                    self.warnings.append(
                        f"LoRA rank ({config.lora_r}) is unusually high relative to estimated model hidden size (~{approx_hidden}). "
                        f"Ranks above {approx_hidden // 8} rarely provide benefits."
                    )

            # Check target modules
            if not config.target_modules:
                self.warnings.append(
                    "No LoRA target_modules specified. "
                    "Common targets include: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj"
                )

        # ===== Batch Size and Memory Checks =====
        effective_batch_size = calculate_effective_batch_size(config.batch_size, config.gradient_accumulation_steps)

        if effective_batch_size < 4:
            self.warnings.append(
                f"Effective batch size ({effective_batch_size}) is very small. "
                "Consider increasing gradient_accumulation_steps for more stable training."
            )

        if effective_batch_size > 128:
            self.warnings.append(
                f"Effective batch size ({effective_batch_size}) is very large. "
                "This may require significant memory and could lead to worse generalization."
            )

        # ===== Learning Rate Checks =====
        if config.learning_rate > 1e-4:
            self.warnings.append(
                f"Learning rate ({config.learning_rate}) is high for fine-tuning. "
                "Consider using 1e-5 to 1e-4 for LoRA fine-tuning."
            )

        if config.learning_rate < 1e-7:
            self.warnings.append(
                f"Learning rate ({config.learning_rate}) is very low. "
                "Training may be extremely slow or not converge."
            )

        # ===== Sequence Length Checks =====
        if config.max_prompt_length >= config.max_length:
            self.errors.append(
                f"max_prompt_length ({config.max_prompt_length}) must be less than "
                f"max_length ({config.max_length})"
            )

        completion_length = config.max_length - config.max_prompt_length
        if completion_length < 128:
            self.warnings.append(
                f"Completion length is only {completion_length} tokens "
                f"(max_length {config.max_length} - max_prompt_length {config.max_prompt_length}). "
                "This may truncate responses during training."
            )

        # ===== Output Name Validation =====
        output_name = config.output_name
        if output_name:
            import re
            # Check for invalid characters
            if not re.match(r'^[a-zA-Z0-9_-]+$', output_name):
                self.errors.append(
                    f"Output name '{output_name}' contains invalid characters. "
                    "Use only letters, numbers, underscores, and hyphens."
                )

            # Check length
            if len(output_name) > 100:
                self.warnings.append(
                    f"Output name is very long ({len(output_name)} chars). "
                    "Consider using a shorter name to avoid path length issues."
                )

            if len(output_name) < 3:
                self.warnings.append(
                    f"Output name '{output_name}' is very short. "
                    "Consider using a more descriptive name."
                )

        return {
            "effective_batch_size": effective_batch_size,
            "completion_length": config.max_length - config.max_prompt_length,
            "lora_enabled": config.use_lora,
            "issues": issues
        }

    def _check_tokens(self, config: Any) -> Dict[str, Any]:
        """Check for required API tokens"""
        checks = {}

        # W&B token
        if config.use_wandb:
            wandb_key = config.wandb_key or os.getenv("WANDB_API_KEY")
            if not wandb_key:
                self.warnings.append(
                    "W&B logging enabled but no WANDB_API_KEY found. "
                    "Training will proceed but metrics won't be logged to W&B."
                )
            checks["wandb"] = bool(wandb_key)

        # HF token
        if config.push_to_hub:
            hf_token = config.hf_token or os.getenv("HF_TOKEN")
            if not hf_token:
                self.errors.append(
                    "Push to HuggingFace Hub enabled but no hf_token provided. "
                    "Provide hf_token in config or set HF_TOKEN environment variable."
                )
            checks["huggingface"] = bool(hf_token)

        return checks

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check for required dependencies"""
        issues = []

        # Check for flash attention (optional but recommended)
        try:
            import flash_attn
            flash_available = True
        except ImportError:
            flash_available = False
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                if props.major >= 8:
                    self.warnings.append(
                        "Flash Attention not installed but GPU supports it. "
                        "Install with: pip install flash-attn --no-build-isolation"
                    )

        # Check for xformers (optional)
        try:
            import xformers
            xformers_available = True
        except ImportError:
            xformers_available = False

        return {
            "flash_attention": flash_available,
            "xformers": xformers_available,
            "issues": issues
        }


def validate_config(config: Any) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate a training config.

    Args:
        config: TrainingConfig instance

    Returns:
        Tuple of (is_valid, results_dict)
    """
    validator = PreflightValidator()
    return validator.validate_all(config)
