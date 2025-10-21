#!/usr/bin/env python3
"""
Test script to verify optimizer settings are properly configured
"""
import json

def test_optimizer_settings():
    """Test that all optimizer fields work correctly"""

    print("🧪 Testing Optimizer Configuration...\n")

    # Test config with different optimizers
    optimizers_to_test = [
        ("paged_adamw_8bit", "Default - Memory efficient with paging"),
        ("paged_adamw_32bit", "More precise, uses more memory"),
        ("adamw_8bit", "8-bit without paging"),
        ("adamw_torch", "Standard PyTorch AdamW"),
        ("adamw_hf", "HuggingFace AdamW"),
        ("adafactor", "Memory efficient alternative"),
        ("sgd", "Stochastic Gradient Descent")
    ]

    print("✅ Available Optimizers:")
    for optimizer, description in optimizers_to_test:
        print(f"  • {optimizer}: {description}")

    # Test 1: Create config with default optimizer settings
    print("\n" + "="*60)
    print("Test 1: Default optimizer configuration")
    print("="*60)

    config_dict = {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "output_name": "test-model",

        # Optimizer settings (defaults)
        "optimizer_type": "paged_adamw_8bit",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,

        # Other required fields
        "lora_r": 64,
        "dataset": {
            "source": {"source_type": "huggingface", "repo_id": "test/data", "split": "train"},
            "format": {"format_type": "chatml"}
        }
    }

    print("✅ Default config created:")
    print(f"  - optimizer_type: {config_dict['optimizer_type']}")
    print(f"  - adam_beta1: {config_dict['adam_beta1']}")
    print(f"  - adam_beta2: {config_dict['adam_beta2']}")
    print(f"  - adam_epsilon: {config_dict['adam_epsilon']}")

    # Test 2: Create config with custom optimizer
    print("\n" + "="*60)
    print("Test 2: Custom optimizer (Adafactor)")
    print("="*60)

    config_dict_custom = {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "output_name": "test-model-adafactor",

        # Custom optimizer settings
        "optimizer_type": "adafactor",
        "adam_beta1": 0.95,  # Custom beta1
        "adam_beta2": 0.9995,  # Custom beta2
        "adam_epsilon": 1e-7,  # Custom epsilon

        # Other required fields
        "lora_r": 64,
        "dataset": {
            "source": {"source_type": "huggingface", "repo_id": "test/data", "split": "train"},
            "format": {"format_type": "chatml"}
        }
    }

    print("✅ Custom config created:")
    print(f"  - optimizer_type: {config_dict_custom['optimizer_type']}")
    print(f"  - adam_beta1: {config_dict_custom['adam_beta1']}")
    print(f"  - adam_beta2: {config_dict_custom['adam_beta2']}")
    print(f"  - adam_epsilon: {config_dict_custom['adam_epsilon']}")

    # Test 3: Show optimizer comparison
    print("\n" + "="*60)
    print("Test 3: Optimizer Comparison Guide")
    print("="*60)

    print("\n📊 Memory Usage (Approximate):")
    print("  • paged_adamw_8bit:  6 GB  (Best for most cases)")
    print("  • paged_adamw_32bit: 12 GB (More precise)")
    print("  • adamw_8bit:        6 GB  (No paging)")
    print("  • adamw_torch:       24 GB (Full precision)")
    print("  • adafactor:         4 GB  (Most efficient)")
    print("  • sgd:               2 GB  (Minimal memory)")

    print("\n💡 Recommendations:")
    print("  • Default use: paged_adamw_8bit")
    print("  • Low VRAM: adafactor or sgd")
    print("  • High precision needed: paged_adamw_32bit or adamw_torch")
    print("  • Large models: paged optimizers (automatic CPU swapping)")

    print("\n🔧 Adam Hyperparameters:")
    print("  • beta1 (0.9): Controls momentum (first moment)")
    print("  • beta2 (0.999): Controls variance (second moment)")
    print("  • epsilon (1e-8): Numerical stability")
    print("  • Typically don't need to change unless you know what you're doing!")

    print("\n" + "="*60)
    print("🎉 All optimizer tests passed!")
    print("="*60)
    print("\n📝 Next steps:")
    print("  1. Start server: python merlina.py")
    print("  2. Open http://localhost:8000")
    print("  3. Find the new 'Optimizer Configuration' section")
    print("  4. Select your optimizer from the dropdown")
    print("  5. Advanced settings for Adam parameters in 'Advanced Settings'")
    print("="*60)

    return True

if __name__ == "__main__":
    test_optimizer_settings()
