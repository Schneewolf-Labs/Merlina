# Configuration Management

Merlina now supports saving and loading training configurations as JSON files, making it easy to reuse configurations across different training runs.

## Features

- **Save Configurations**: Save your current training settings with name, description, and tags
- **Load Configurations**: Load previously saved configurations into the UI
- **Manage Configurations**: View, load, and delete saved configurations
- **Export/Import**: Export configurations to specific paths or import from external files
- **Metadata Tracking**: Automatically tracks creation time, modification time, tags, and descriptions

## Storage Location

Configurations are stored as JSON files in: `./data/configs/`

Each configuration file includes:
- Training parameters (model, LoRA settings, hyperparameters)
- Dataset configuration (source, format, options)
- Metadata (name, description, tags, timestamps)

## Using the Web UI

### Saving a Configuration

1. Configure your training parameters in Step 3
2. Click the **💾 Save Config** button
3. Enter a name for your configuration
4. Optionally add a description and tags (comma-separated)
5. Click "Save Configuration"

### Loading a Configuration

1. Click the **📂 Load Config** button
2. Browse the list of saved configurations
3. Click on a configuration to load it into the form

### Managing Configurations

1. Click the **🗂️ Manage** button
2. View all saved configurations with details
3. Click **📂 Load** to load a configuration
4. Click **🗑️ Delete** to remove a configuration

## Using the API

### Save a Configuration

```bash
POST /configs/save
Content-Type: application/json

{
  "name": "my-config",
  "config": {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "output_name": "my-model",
    "use_lora": true,
    "lora_r": 16,
    ...
  },
  "description": "My training configuration",
  "tags": ["llama3", "orpo", "4bit"]
}
```

### List Configurations

```bash
GET /configs/list?tag=llama3
```

### Load a Configuration

```bash
GET /configs/my-config
```

### Delete a Configuration

```bash
DELETE /configs/my-config
```

### Export a Configuration

```bash
POST /configs/export
Content-Type: application/json

{
  "name": "my-config",
  "output_path": "/path/to/export.json"
}
```

### Import a Configuration

```bash
POST /configs/import
Content-Type: application/json

{
  "filepath": "/path/to/config.json",
  "name": "imported-config"
}
```

## Using the Python API

```python
from src.config_manager import ConfigManager

# Initialize
manager = ConfigManager(config_dir="./data/configs")

# Save a configuration
config = {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "use_lora": True,
    "lora_r": 16
}

manager.save_config(
    name="my-config",
    config=config,
    description="My training setup",
    tags=["llama3", "test"]
)

# List configurations
configs = manager.list_configs()
for cfg in configs:
    print(f"{cfg['name']}: {cfg['description']}")

# Load a configuration
loaded_config = manager.load_config("my-config")

# Get config without metadata
clean_config = manager.get_config_without_metadata("my-config")

# Delete a configuration
manager.delete_config("my-config")

# Export and import
manager.export_config("my-config", "./backup.json")
manager.import_config("./backup.json", "restored-config")
```

## Example Script

Use the provided example script to manage configurations:

```bash
# Save a configuration (interactive demo)
python examples/manage_configs.py save

# List all configurations
python examples/manage_configs.py list

# Load a specific configuration
python examples/manage_configs.py load my-config

# Delete a configuration
python examples/manage_configs.py delete my-config

# Export a configuration
python examples/manage_configs.py export my-config ./backup.json

# Import a configuration
python examples/manage_configs.py import ./backup.json restored-config

# Load and use for training
python examples/manage_configs.py use my-config
```

## Configuration File Format

Saved configurations are JSON files with this structure:

```json
{
  "_metadata": {
    "name": "my-config",
    "description": "My training configuration",
    "created_at": "2025-10-27T10:30:00.000000",
    "modified_at": "2025-10-27T10:30:00.000000",
    "tags": ["llama3", "orpo"]
  },
  "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "output_name": "my-model",
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "use_4bit": true,
  "dataset": {
    "source": {
      "source_type": "huggingface",
      "repo_id": "schneewolflabs/Athanor-DPO"
    },
    "format": {
      "format_type": "tokenizer"
    }
  }
}
```

## Best Practices

1. **Use Descriptive Names**: Choose clear, meaningful names for your configurations
2. **Add Tags**: Tag configurations by model type, use case, or training stage for easy filtering
3. **Write Descriptions**: Document what makes each configuration special
4. **Version Control**: Consider committing configurations to git for team collaboration
5. **Regular Backups**: Export important configurations as backups
6. **Organize with Tags**: Use consistent tagging schemes (e.g., "production", "experimental", "model-name")

## Tips

- Configurations are automatically updated if you save with the same name
- The modification timestamp is updated, but creation timestamp is preserved
- Invalid filename characters are automatically sanitized
- Configurations can be shared by copying the JSON files
- The UI automatically loads all available configurations on demand

## Testing

Run the test suite to verify the configuration manager:

```bash
python tests/test_config_manager.py
```
