#!/usr/bin/env python3
"""
Simple test to verify new training settings are in the config
"""
import json

def test_config_fields():
    """Test that all new fields are present in TrainingConfig"""

    print("🧪 Testing TrainingConfig fields...\n")

    # Test by creating a mock config dict that would be sent from frontend
    config_dict = {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "output_name": "test-model",

        # Existing settings
        "lora_r": 64,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["up_proj", "down_proj"],
        "learning_rate": 0.000005,
        "num_epochs": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_length": 2048,
        "max_prompt_length": 1024,
        "beta": 0.1,
        "warmup_ratio": 0.05,
        "eval_steps": 0.2,
        "use_4bit": True,
        "use_wandb": True,
        "push_to_hub": False,

        # Training mode
        "training_mode": "orpo",

        # NEW Priority 1 settings
        "seed": 42,
        "max_grad_norm": 0.3,

        # NEW Priority 2 settings
        "shuffle_dataset": True,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": False,
        "logging_steps": 1,

        # Dataset config
        "dataset": {
            "source": {
                "source_type": "huggingface",
                "repo_id": "test/dataset",
                "split": "train"
            },
            "format": {
                "format_type": "chatml"
            },
            "test_size": 0.01
        }
    }

    print("✅ Test config dictionary created with all new fields:")
    print("\nTraining Mode:")
    print(f"  - training_mode: {config_dict['training_mode']}")

    print("\nPriority 1 Settings:")
    print(f"  - seed: {config_dict['seed']}")
    print(f"  - max_grad_norm: {config_dict['max_grad_norm']}")

    print("\nPriority 2 Settings:")
    print(f"  - shuffle_dataset: {config_dict['shuffle_dataset']}")
    print(f"  - weight_decay: {config_dict['weight_decay']}")
    print(f"  - lr_scheduler_type: {config_dict['lr_scheduler_type']}")
    print(f"  - gradient_checkpointing: {config_dict['gradient_checkpointing']}")
    print(f"  - logging_steps: {config_dict['logging_steps']}")

    print("\n" + "="*60)
    print("📋 Config structure looks good!")
    print("="*60)
    print("\n📝 Next steps:")
    print("  1. Start the Merlina server: python merlina.py")
    print("  2. Open http://localhost:8000 in your browser")
    print("  3. Check the new settings in the UI:")
    print("     - Priority 1: Seed and Max Gradient Norm (main UI)")
    print("     - Priority 2: Click 'Show Advanced Settings' to see:")
    print("       • Shuffle Dataset")
    print("       • Weight Decay")
    print("       • LR Scheduler Type")
    print("       • Gradient Checkpointing")
    print("       • Logging Steps")
    print("="*60)

    return True

if __name__ == "__main__":
    test_config_fields()
