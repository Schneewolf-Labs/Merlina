"""
Example: Validate configuration before training
Demonstrates pre-flight validation to catch errors early
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from pprint import pprint

# Merlina API endpoint
API_URL = "http://localhost:8000"

# Training configuration
config = {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "output_name": "llama3-wizard",
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"],
    "learning_rate": 5e-6,
    "num_epochs": 2,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_length": 2048,
    "max_prompt_length": 1024,
    "beta": 0.1,
    "dataset": {
        "source": {
            "source_type": "huggingface",
            "repo_id": "schneewolflabs/Athanor-DPO",
            "split": "train"
        },
        "format": {
            "format_type": "tokenizer"  # Use model's native chat template
        },
        "test_size": 0.01,
        "max_samples": 1000  # Limit for testing
    },
    "warmup_ratio": 0.05,
    "eval_steps": 0.2,
    "use_4bit": True,
    "use_wandb": False,  # Set to True if you have W&B
    "push_to_hub": False,
    "hf_token": None,  # Add your token if needed
    "wandb_key": None
}


def validate_config(config):
    """Validate configuration before training"""
    print("🔍 Validating configuration...")
    print("=" * 60)

    response = requests.post(f"{API_URL}/validate", json=config)

    if response.status_code != 200:
        print(f"❌ Validation request failed: {response.status_code}")
        print(response.text)
        return False

    result = response.json()

    # Print validation results
    print("\n📊 Validation Results:")
    print("-" * 60)

    # Show check results
    for check_name, check_result in result['results']['checks'].items():
        status = check_result['status']
        emoji = "✅" if status == "pass" else "⚠️" if status == "warning" else "❌"
        print(f"{emoji} {check_name}: {status}")

        # Show details for important checks
        if check_name == "GPU" and 'details' in check_result:
            devices = check_result['details'].get('devices', [])
            for gpu in devices:
                print(f"   - {gpu['name']} ({gpu['total_memory_gb']:.1f}GB)")

        if check_name == "VRAM" and 'details' in check_result:
            details = check_result['details']
            print(f"   - Available: {details.get('available_gb', 0):.1f}GB")
            print(f"   - Required: {details.get('estimated_required_gb', 0):.1f}GB")

    # Show warnings
    if result['results']['warnings']:
        print("\n⚠️  Warnings:")
        for warning in result['results']['warnings']:
            print(f"   - {warning}")

    # Show errors
    if result['results']['errors']:
        print("\n❌ Errors:")
        for error in result['results']['errors']:
            print(f"   - {error}")

    print("\n" + "=" * 60)

    return result['valid']


def start_training(config):
    """Start training job"""
    print("\n🚀 Starting training...")

    response = requests.post(f"{API_URL}/train", json=config)

    if response.status_code != 200:
        print(f"❌ Training request failed: {response.status_code}")
        print(response.text)
        return None

    result = response.json()
    print(f"✅ {result['message']}")
    print(f"📝 Job ID: {result['job_id']}")

    return result['job_id']


def main():
    """Main execution"""
    print("🧙‍♀️ Merlina - Validate and Train Example")
    print("=" * 60)

    # Step 1: Validate configuration
    is_valid = validate_config(config)

    if not is_valid:
        print("\n❌ Configuration is invalid. Please fix errors and try again.")
        return

    # Step 2: Ask user to confirm
    print("\n✅ Configuration is valid!")
    response = input("\nProceed with training? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("Training cancelled.")
        return

    # Step 3: Start training
    job_id = start_training(config)

    if job_id:
        print(f"\n✨ Training started successfully!")
        print(f"\nMonitor progress:")
        print(f"  - REST API: GET {API_URL}/status/{job_id}")
        print(f"  - WebSocket: ws://localhost:8000/ws/{job_id}")
        print(f"  - Web UI: http://localhost:8000")


if __name__ == "__main__":
    main()
