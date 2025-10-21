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
        """Check if available VRAM is sufficient"""
        if not torch.cuda.is_available():
            return {"skipped": "No GPU available"}

        # Estimate VRAM requirements based on model size
        model_name = config.base_model.lower()

        # Model size estimates (in GB of VRAM with 4-bit quantization)
        size_estimates = {
            "3b": 6,
            "7b": 10,
            "8b": 10,
            "13b": 16,
            "14b": 18,
            "34b": 30,
            "70b": 60,
        }

        estimated_vram = None
        for size_key, vram_needed in size_estimates.items():
            if size_key in model_name:
                estimated_vram = vram_needed
                break

        if estimated_vram is None:
            self.warnings.append(
                f"Could not estimate VRAM requirements for model '{config.base_model}'. "
                "Please ensure you have sufficient GPU memory."
            )
            return {"estimate_unavailable": True}

        # Adjust for quantization
        if not config.use_4bit:
            estimated_vram *= 3.5  # Roughly 3.5x more VRAM for full precision
            self.warnings.append(
                f"Training without 4-bit quantization will require ~{estimated_vram:.1f}GB VRAM. "
                "Consider enabling 4-bit quantization to reduce memory usage."
            )

        # Check available VRAM
        gpu = torch.cuda.get_device_properties(0)
        available_vram = gpu.total_memory / (1024**3)

        if available_vram < estimated_vram:
            self.errors.append(
                f"Insufficient VRAM: Model requires ~{estimated_vram}GB, "
                f"but only {available_vram:.1f}GB available. "
                f"{'Enable 4-bit quantization to reduce memory usage.' if not config.use_4bit else 'Consider using a smaller model or reducing batch size/sequence length.'}"
            )

        return {
            "available_gb": available_vram,
            "estimated_required_gb": estimated_vram,
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

        # Check LoRA config
        if config.lora_r > 256:
            self.warnings.append(f"LoRA rank ({config.lora_r}) is very high. Consider using 64-128 for most cases.")

        if config.lora_alpha < config.lora_r:
            self.warnings.append(
                f"LoRA alpha ({config.lora_alpha}) is less than rank ({config.lora_r}). "
                "Typically alpha should be 1-2x the rank."
            )

        # Check batch size and gradient accumulation
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
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

        # Check learning rate
        if config.learning_rate > 1e-4:
            self.warnings.append(
                f"Learning rate ({config.learning_rate}) is high for fine-tuning. "
                "Consider using 1e-5 to 1e-4 for LoRA fine-tuning."
            )

        # Check sequence lengths
        if config.max_prompt_length >= config.max_length:
            self.errors.append(
                f"max_prompt_length ({config.max_prompt_length}) must be less than "
                f"max_length ({config.max_length})"
            )

        return {
            "effective_batch_size": effective_batch_size,
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
