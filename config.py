"""
Merlina Configuration Management

Centralized configuration using Pydantic BaseSettings.
Configuration can be set via:
1. .env file (recommended for secrets)
2. Environment variables
3. Defaults in this file
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Settings priority (highest to lowest):
    1. Environment variables
    2. .env file
    3. Default values
    """

    # ==========================================
    # Server Configuration
    # ==========================================
    app_name: str = "Merlina"
    host: str = "0.0.0.0"
    port: int = 8000
    domain: Optional[str] = None  # e.g., "https://merlina.example.com" for production

    # ==========================================
    # Directory Paths
    # ==========================================
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    results_dir: Path = Path("./results")
    uploads_dir: Path = Path("./data/uploads")
    frontend_dir: Path = Path("./frontend")

    # ==========================================
    # Database
    # ==========================================
    database_path: str = "./data/jobs.db"

    # ==========================================
    # External Services (API Keys)
    # ==========================================
    wandb_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    # ==========================================
    # Training Defaults
    # ==========================================
    default_lora_r: int = 64
    default_lora_alpha: int = 32
    default_learning_rate: float = 5e-6
    default_batch_size: int = 1
    default_gradient_accumulation_steps: int = 16
    default_num_epochs: int = 2

    # ==========================================
    # Job Queue Configuration
    # ==========================================
    max_concurrent_jobs: int = 1  # Maximum number of training jobs to run simultaneously

    # ==========================================
    # System Configuration
    # ==========================================
    cuda_visible_devices: Optional[str] = None
    log_level: str = "INFO"

    # ==========================================
    # Security & Limits
    # ==========================================
    cors_origins: List[str] = ["*"]
    max_upload_size_mb: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore unknown environment variables

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Apply and validate CUDA_VISIBLE_DEVICES if set
        if self.cuda_visible_devices:
            validation_result = validate_cuda_visible_devices(self.cuda_visible_devices)
            if validation_result["valid"]:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_visible_devices
                if validation_result.get("warnings"):
                    for warning in validation_result["warnings"]:
                        logger.warning(warning)
            else:
                # Log error but don't crash - let training fail later with better context
                for error in validation_result.get("errors", []):
                    logger.error(error)


def validate_cuda_visible_devices(cuda_str: str) -> dict:
    """
    Validate CUDA_VISIBLE_DEVICES string before applying it.

    Args:
        cuda_str: The CUDA_VISIBLE_DEVICES value (e.g., "0", "0,1", "GPU-uuid")

    Returns:
        dict with "valid" bool, and optional "errors" and "warnings" lists
    """
    result = {"valid": True, "errors": [], "warnings": []}

    if not cuda_str or cuda_str.strip() == "":
        return result

    # Check if CUDA is even available before validating device IDs
    try:
        import torch
        if not torch.cuda.is_available():
            result["warnings"].append(
                f"CUDA_VISIBLE_DEVICES='{cuda_str}' is set but CUDA is not available. "
                "This setting will have no effect."
            )
            return result

        actual_gpu_count = torch.cuda.device_count()
    except ImportError:
        result["warnings"].append(
            "PyTorch not available for CUDA validation. Skipping GPU index validation."
        )
        return result
    except Exception as e:
        result["warnings"].append(f"Could not check CUDA availability: {e}")
        return result

    # Parse the CUDA_VISIBLE_DEVICES value
    # It can be: "0", "0,1,2", "GPU-<uuid>", etc.
    devices = [d.strip() for d in cuda_str.split(",")]

    for device in devices:
        # Skip UUID-based device specifications
        if device.startswith("GPU-") or device.startswith("MIG-"):
            continue

        # Try to parse as integer
        try:
            device_id = int(device)

            # Note: We validate against actual_gpu_count BEFORE setting CUDA_VISIBLE_DEVICES
            # because once it's set, torch.cuda.device_count() returns the filtered count
            if device_id < 0:
                result["valid"] = False
                result["errors"].append(
                    f"Invalid GPU index '{device_id}': GPU indices must be non-negative."
                )
            elif device_id >= actual_gpu_count:
                result["valid"] = False
                result["errors"].append(
                    f"Invalid GPU index '{device_id}': Only {actual_gpu_count} GPU(s) available (indices 0-{actual_gpu_count - 1})."
                )

        except ValueError:
            result["valid"] = False
            result["errors"].append(
                f"Invalid CUDA_VISIBLE_DEVICES value '{device}': "
                "Expected integer GPU index or GPU-<uuid> format."
            )

    return result


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings
