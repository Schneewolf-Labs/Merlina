#!/usr/bin/env python3
"""
Example script demonstrating config management in Merlina

Shows how to:
- Save training configurations
- List saved configurations
- Load configurations
- Delete configurations
- Export/import configurations
"""

import sys
import json
import requests
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# API base URL
API_URL = "http://localhost:8000"


def save_config_example():
    """Example: Save a training configuration"""
    print("\n=== Saving Configuration ===")

    # Example training config
    config = {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "output_name": "my-llama3-orpo",
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "use_4bit": True,
        "max_seq_length": 2048,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "beta": 0.1,
        "dataset": {
            "source": {
                "source_type": "huggingface",
                "repo_id": "schneewolflabs/Athanor-DPO",
                "split": "train"
            },
            "format": {
                "format_type": "tokenizer"
            },
            "test_size": 0.01
        }
    }

    # Save the config
    response = requests.post(
        f"{API_URL}/configs/save",
        json={
            "name": "llama3-base-config",
            "config": config,
            "description": "Basic Llama 3 ORPO training configuration with LoRA",
            "tags": ["llama3", "orpo", "4bit"]
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Saved: {result['message']}")
        print(f"  File: {result['filepath']}")
    else:
        print(f"✗ Error: {response.json()}")


def list_configs_example():
    """Example: List all saved configurations"""
    print("\n=== Listing Configurations ===")

    response = requests.get(f"{API_URL}/configs/list")

    if response.status_code == 200:
        result = response.json()
        configs = result['configs']

        if not configs:
            print("No saved configurations found.")
            return

        print(f"Found {result['count']} configuration(s):\n")

        for config in configs:
            print(f"Name: {config['name']}")
            print(f"  File: {config['filename']}.json")
            print(f"  Description: {config.get('description', 'N/A')}")
            print(f"  Tags: {', '.join(config.get('tags', []))}")
            print(f"  Modified: {config.get('modified_at', 'N/A')}")
            print()
    else:
        print(f"✗ Error: {response.json()}")


def load_config_example(name: str = "llama3-base-config"):
    """Example: Load a saved configuration"""
    print(f"\n=== Loading Configuration: {name} ===")

    response = requests.get(f"{API_URL}/configs/{name}")

    if response.status_code == 200:
        result = response.json()
        config = result['config']

        print("✓ Configuration loaded successfully")
        print(json.dumps(config, indent=2))

        return config
    else:
        print(f"✗ Error: {response.json()}")
        return None


def delete_config_example(name: str):
    """Example: Delete a saved configuration"""
    print(f"\n=== Deleting Configuration: {name} ===")

    response = requests.delete(f"{API_URL}/configs/{name}")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
    else:
        print(f"✗ Error: {response.json()}")


def export_config_example(name: str, output_path: str):
    """Example: Export a configuration to a file"""
    print(f"\n=== Exporting Configuration: {name} ===")

    response = requests.post(
        f"{API_URL}/configs/export",
        json={
            "name": name,
            "output_path": output_path
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
    else:
        print(f"✗ Error: {response.json()}")


def import_config_example(filepath: str, name: str = None):
    """Example: Import a configuration from a file"""
    print(f"\n=== Importing Configuration from: {filepath} ===")

    response = requests.post(
        f"{API_URL}/configs/import",
        json={
            "filepath": filepath,
            "name": name
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
    else:
        print(f"✗ Error: {response.json()}")


def use_config_for_training(config_name: str):
    """Example: Load a config and use it to start training"""
    print(f"\n=== Using Config for Training: {config_name} ===")

    # Load the config
    response = requests.get(f"{API_URL}/configs/{config_name}")

    if response.status_code != 200:
        print(f"✗ Failed to load config: {response.json()}")
        return

    config = response.json()['config']

    # You can now modify the config if needed
    config['output_name'] = f"{config['output_name']}-{int(datetime.now().timestamp())}"

    print("✓ Config loaded. Ready to start training with:")
    print(f"  Base model: {config['base_model']}")
    print(f"  Output name: {config['output_name']}")
    print(f"  LoRA: {config['use_lora']}")
    print(f"  4-bit: {config['use_4bit']}")

    # Uncomment to actually start training:
    # response = requests.post(f"{API_URL}/train", json=config)
    # if response.status_code == 200:
    #     job = response.json()
    #     print(f"✓ Training started! Job ID: {job['job_id']}")
    # else:
    #     print(f"✗ Training failed: {response.json()}")


def main():
    """Run all examples"""
    import sys
    from datetime import datetime

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "save":
            save_config_example()

        elif command == "list":
            list_configs_example()

        elif command == "load":
            name = sys.argv[2] if len(sys.argv) > 2 else "llama3-base-config"
            load_config_example(name)

        elif command == "delete":
            if len(sys.argv) < 3:
                print("Usage: python manage_configs.py delete <name>")
                return
            delete_config_example(sys.argv[2])

        elif command == "export":
            if len(sys.argv) < 4:
                print("Usage: python manage_configs.py export <name> <output_path>")
                return
            export_config_example(sys.argv[2], sys.argv[3])

        elif command == "import":
            if len(sys.argv) < 3:
                print("Usage: python manage_configs.py import <filepath> [name]")
                return
            name = sys.argv[3] if len(sys.argv) > 3 else None
            import_config_example(sys.argv[2], name)

        elif command == "use":
            if len(sys.argv) < 3:
                print("Usage: python manage_configs.py use <name>")
                return
            use_config_for_training(sys.argv[2])

        else:
            print(f"Unknown command: {command}")
            print("Available commands: save, list, load, delete, export, import, use")

    else:
        # Run full demo
        print("=" * 60)
        print("Merlina Config Management Demo")
        print("=" * 60)

        # Save a config
        save_config_example()

        # List all configs
        list_configs_example()

        # Load a config
        config = load_config_example("llama3-base-config")

        # Export example (commented to avoid file creation)
        # export_config_example("llama3-base-config", "./exported-config.json")

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("\nTo use individual commands:")
        print("  python manage_configs.py save")
        print("  python manage_configs.py list")
        print("  python manage_configs.py load <name>")
        print("  python manage_configs.py delete <name>")
        print("  python manage_configs.py export <name> <output_path>")
        print("  python manage_configs.py import <filepath> [name]")
        print("  python manage_configs.py use <name>")
        print("=" * 60)


if __name__ == "__main__":
    main()
