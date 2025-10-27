"""
Merlina Configuration Management

Centralized configuration using Pydantic BaseSettings.
Configuration can be set via:
1. .env file (recommended for secrets)
2. Environment variables
3. Defaults in this file
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings


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
        # Apply CUDA_VISIBLE_DEVICES if set
        if self.cuda_visible_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_visible_devices


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings
