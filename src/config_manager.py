"""
Configuration Manager for Merlina Training Configs

Handles saving, loading, and managing training configuration presets as JSON files.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ConfigMetadata:
    """Metadata for a saved configuration"""
    name: str
    description: str
    created_at: str
    modified_at: str
    tags: List[str]


class ConfigManager:
    """Manages training configuration presets stored as JSON files"""

    def __init__(self, config_dir: str = "./data/configs"):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_config(
        self,
        name: str,
        config: dict,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a training configuration to a JSON file.

        Args:
            name: Name for the configuration (used as filename)
            config: Training configuration dictionary
            description: Optional description of the configuration
            tags: Optional list of tags for categorization

        Returns:
            Path to the saved configuration file

        Raises:
            ValueError: If name contains invalid characters
        """
        # Sanitize filename
        safe_name = self._sanitize_filename(name)
        if not safe_name:
            raise ValueError(f"Invalid configuration name: {name}")

        filepath = self.config_dir / f"{safe_name}.json"

        # Check if config exists
        existing_metadata = None
        if filepath.exists():
            try:
                existing = self.load_config(safe_name)
                existing_metadata = existing.get("_metadata", {})
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

        # Create metadata
        now = datetime.utcnow().isoformat()
        metadata = {
            "name": name,
            "description": description,
            "created_at": existing_metadata.get("created_at", now) if existing_metadata else now,
            "modified_at": now,
            "tags": tags or []
        }

        # Combine config and metadata
        config_with_metadata = {
            "_metadata": metadata,
            **config
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)

        return str(filepath)

    def load_config(self, name: str) -> dict:
        """
        Load a training configuration from a JSON file.

        Args:
            name: Name of the configuration to load

        Returns:
            Configuration dictionary with metadata

        Raises:
            FileNotFoundError: If configuration doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        safe_name = self._sanitize_filename(name)
        filepath = self.config_dir / f"{safe_name}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found")

        with open(filepath, 'r') as f:
            return json.load(f)

    def list_configs(self, tag: Optional[str] = None) -> List[Dict]:
        """
        List all saved configurations with their metadata.

        Args:
            tag: Optional tag to filter configurations

        Returns:
            List of configuration metadata dictionaries
        """
        configs = []

        for filepath in self.config_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    config = json.load(f)
                    metadata = config.get("_metadata", {})

                    # Filter by tag if specified
                    if tag and tag not in metadata.get("tags", []):
                        continue

                    configs.append({
                        "filename": filepath.stem,
                        **metadata
                    })
            except (json.JSONDecodeError, KeyError):
                # Skip invalid config files
                continue

        # Sort by modified date (newest first)
        configs.sort(key=lambda x: x.get("modified_at", ""), reverse=True)

        return configs

    def delete_config(self, name: str) -> bool:
        """
        Delete a saved configuration.

        Args:
            name: Name of the configuration to delete

        Returns:
            True if deleted, False if not found
        """
        safe_name = self._sanitize_filename(name)
        filepath = self.config_dir / f"{safe_name}.json"

        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def get_config_without_metadata(self, name: str) -> dict:
        """
        Load a configuration without the metadata section.

        Args:
            name: Name of the configuration to load

        Returns:
            Configuration dictionary without _metadata field
        """
        config = self.load_config(name)
        config_copy = config.copy()
        config_copy.pop("_metadata", None)
        return config_copy

    def export_config(self, name: str, output_path: str) -> str:
        """
        Export a configuration to a specific path.

        Args:
            name: Name of the configuration to export
            output_path: Path to export the configuration to

        Returns:
            Path to the exported file
        """
        config = self.load_config(name)

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        return output_path

    def import_config(self, filepath: str, name: Optional[str] = None) -> str:
        """
        Import a configuration from an external file.

        Args:
            filepath: Path to the configuration file to import
            name: Optional name for the imported config (defaults to filename)

        Returns:
            Path to the saved configuration file
        """
        with open(filepath, 'r') as f:
            config = json.load(f)

        # Remove metadata if present (will be recreated)
        config.pop("_metadata", None)

        # Use provided name or derive from filename
        if name is None:
            name = Path(filepath).stem

        return self.save_config(name, config, description=f"Imported from {filepath}")

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a configuration name for use as a filename.

        Args:
            name: Configuration name

        Returns:
            Sanitized filename (without extension)
        """
        # Remove invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        safe_name = ''.join(c if c not in invalid_chars else '_' for c in name)

        # Remove leading/trailing whitespace and dots
        safe_name = safe_name.strip('. ')

        # Limit length
        safe_name = safe_name[:200]

        return safe_name
