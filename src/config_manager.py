"""
Configuration Manager for Merlina Training Configs

Handles saving, loading, and managing training configuration presets as
JSON files. Centralizes the rules for what a "saved config" looks like so
both /configs/save (preset library) and the model-card embedder produce
identical, importable JSON.

The serialized format is intentionally simple:

    {
      "_metadata": {
        "name": "<config name>",
        "description": "...",
        "created_at": "<iso>",
        "modified_at": "<iso>",
        "tags": [...],
        "schema": "merlina/training-config",
        "schema_version": 1,
        "merlina_version": "X.Y.Z"
      },
      "<TrainingConfig field>": ...,
      ...
    }

Anyone with a Merlina instance running can drop one of these JSONs into
``data/configs/`` and load it from the UI. We strip credentials before
persisting so a saved config is safe to commit, share in a model card,
or paste into a HuggingFace README.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# Fields that must NEVER be persisted to a saved config or shared in a
# model README. /train endpoints can still receive these via the request
# body or fall back to ``.env``.
SECRET_FIELDS: Tuple[str, ...] = ("hf_token", "wandb_key")

# Schema identifier embedded in every saved config. Bump SCHEMA_VERSION on
# breaking changes to the on-disk shape so importers can detect mismatches.
SCHEMA_NAME = "merlina/training-config"
SCHEMA_VERSION = 1


@dataclass
class ConfigMetadata:
    """Metadata for a saved configuration"""
    name: str
    description: str
    created_at: str
    modified_at: str
    tags: List[str]


def strip_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a shallow copy of ``config`` with credential fields removed.

    Centralizes the secret-stripping rule so callers don't each maintain
    their own list. Used by /configs/save (persistence) and the model-card
    embedder (sharing).
    """
    return {k: v for k, v in config.items() if k not in SECRET_FIELDS}


def normalize_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a TrainingConfig-shaped dict.

    Routes the config through the Pydantic ``TrainingConfig`` model so:
      * Unknown / typo'd fields raise instead of silently dropping
      * Defaults are filled in for absent fields
      * Lists/numbers come back in their canonical form

    Always strips secrets — this is the form that's safe to write to disk.
    Importable by any Merlina instance running the same major version.
    """
    # Local import — TrainingConfig lives in merlina.py and pulls heavy
    # ML deps at import time. Keep this lazy so unit tests of ConfigManager
    # don't pay the cost when they don't need it.
    from merlina import TrainingConfig
    from pydantic import TypeAdapter

    validated = TypeAdapter(TrainingConfig).validate_python(config)
    dumped = validated.model_dump(mode="json")
    return strip_secrets(dumped)


def build_config_envelope(
    config: Dict[str, Any],
    *,
    name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrap a TrainingConfig payload with the standard ``_metadata`` block.

    The envelope is the single, importable serialization format produced
    by /configs/save and (optionally) by the model-card embedder. Keeping
    one builder means the two paths can never drift.
    """
    # Lazy import to avoid pulling version on hot path of empty test envs.
    try:
        from version import __version__ as _merlina_version
    except Exception:  # pragma: no cover - version.py always exists in repo
        _merlina_version = "unknown"

    now = datetime.utcnow().isoformat()
    metadata = {
        "name": name,
        "description": description,
        "created_at": created_at or now,
        "modified_at": now,
        "tags": list(tags or []),
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "merlina_version": _merlina_version,
    }
    return {"_metadata": metadata, **config}


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
        tags: Optional[List[str]] = None,
        validate: bool = False,
    ) -> str:
        """
        Save a training configuration to a JSON file.

        Args:
            name: Name for the configuration (used as filename)
            config: Training configuration dictionary
            description: Optional description of the configuration
            tags: Optional list of tags for categorization
            validate: When True, route the config through the Pydantic
                ``TrainingConfig`` schema before saving. Catches typo'd
                or missing fields at save time and produces a normalized
                payload that's safe to import elsewhere.

        Returns:
            Path to the saved configuration file

        Raises:
            ValueError: If name contains invalid characters
            pydantic.ValidationError: If validate=True and the config
                fails Pydantic validation
        """
        # Sanitize filename
        safe_name = self._sanitize_filename(name)
        if not safe_name:
            raise ValueError(f"Invalid configuration name: {name}")

        filepath = self.config_dir / f"{safe_name}.json"

        # Check if config exists — preserve created_at across resaves
        existing_created_at: Optional[str] = None
        if filepath.exists():
            try:
                existing = self.load_config(safe_name)
                existing_created_at = existing.get("_metadata", {}).get("created_at")
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

        # Normalize through Pydantic when requested. Otherwise just strip
        # secrets — the legacy contract is "save whatever I give you" and
        # we don't want to break callers that pass partial dicts in tests.
        if validate:
            payload = normalize_training_config(config)
        else:
            payload = strip_secrets(config)

        envelope = build_config_envelope(
            payload,
            name=name,
            description=description,
            tags=tags,
            created_at=existing_created_at,
        )

        with open(filepath, "w") as f:
            json.dump(envelope, f, indent=2)

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

        with open(filepath, "r") as f:
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
                with open(filepath, "r") as f:
                    config = json.load(f)
                    metadata = config.get("_metadata", {})

                    # Filter by tag if specified
                    if tag and tag not in metadata.get("tags", []):
                        continue

                    configs.append({
                        "filename": filepath.stem,
                        **metadata,
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

        with open(output_path, "w") as f:
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
        with open(filepath, "r") as f:
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
        safe_name = "".join(c if c not in invalid_chars else "_" for c in name)

        # Remove leading/trailing whitespace and dots
        safe_name = safe_name.strip(". ")

        # Limit length
        safe_name = safe_name[:200]

        return safe_name
