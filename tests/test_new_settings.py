#!/usr/bin/env python3
"""
Test script to verify new training settings are properly configured
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merlina import TrainingConfig, DatasetConfig, DatasetSource, DatasetFormat
from pydantic import ValidationError

def test_new_settings():
    """Test that all new Priority 1 and Priority 2 settings work"""

    print("🧪 Testing new training settings...\n")

    # Test 1: Create config with default values
    print("Test 1: Creating config with default values...")
    try:
        config = TrainingConfig(
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            output_name="test-model"
        )
        print("✅ Default config created successfully")
        print(f"  - seed: {config.seed}")
        print(f"  - max_grad_norm: {config.max_grad_norm}")
        print(f"  - shuffle_dataset: {config.shuffle_dataset}")
        print(f"  - weight_decay: {config.weight_decay}")
        print(f"  - lr_scheduler_type: {config.lr_scheduler_type}")
        print(f"  - gradient_checkpointing: {config.gradient_checkpointing}")
        print(f"  - logging_steps: {config.logging_steps}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    # Test 2: Create config with custom values
    print("\nTest 2: Creating config with custom Priority 1 settings...")
    try:
        config = TrainingConfig(
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            output_name="test-model",
            seed=12345,
            max_grad_norm=1.0
        )
        print("✅ Custom Priority 1 config created successfully")
        print(f"  - seed: {config.seed} (expected: 12345)")
        print(f"  - max_grad_norm: {config.max_grad_norm} (expected: 1.0)")
        assert config.seed == 12345
        assert config.max_grad_norm == 1.0
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    # Test 3: Create config with custom Priority 2 settings
    print("\nTest 3: Creating config with custom Priority 2 settings...")
    try:
        config = TrainingConfig(
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            output_name="test-model",
            shuffle_dataset=False,
            weight_decay=0.05,
            lr_scheduler_type="linear",
            gradient_checkpointing=True,
            logging_steps=10
        )
        print("✅ Custom Priority 2 config created successfully")
        print(f"  - shuffle_dataset: {config.shuffle_dataset} (expected: False)")
        print(f"  - weight_decay: {config.weight_decay} (expected: 0.05)")
        print(f"  - lr_scheduler_type: {config.lr_scheduler_type} (expected: linear)")
        print(f"  - gradient_checkpointing: {config.gradient_checkpointing} (expected: True)")
        print(f"  - logging_steps: {config.logging_steps} (expected: 10)")
        assert config.shuffle_dataset == False
        assert config.weight_decay == 0.05
        assert config.lr_scheduler_type == "linear"
        assert config.gradient_checkpointing == True
        assert config.logging_steps == 10
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    # Test 4: Test validation boundaries
    print("\nTest 4: Testing validation boundaries...")
    try:
        # Test seed boundary
        config = TrainingConfig(
            base_model="test",
            output_name="test",
            seed=99999
        )
        print("✅ Max seed value (99999) accepted")

        # Test max_grad_norm boundary
        config = TrainingConfig(
            base_model="test",
            output_name="test",
            max_grad_norm=5.0
        )
        print("✅ Max grad norm value (5.0) accepted")

        # Test weight_decay boundary
        config = TrainingConfig(
            base_model="test",
            output_name="test",
            weight_decay=0.5
        )
        print("✅ Max weight decay value (0.5) accepted")

        # Test logging_steps boundary
        config = TrainingConfig(
            base_model="test",
            output_name="test",
            logging_steps=100
        )
        print("✅ Max logging steps value (100) accepted")

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    # Test 5: Test invalid values should fail
    print("\nTest 5: Testing that invalid values are rejected...")
    try:
        config = TrainingConfig(
            base_model="test",
            output_name="test",
            seed=100000  # Too high, max is 99999
        )
        print("❌ Should have rejected seed=100000")
        return False
    except ValidationError:
        print("✅ Correctly rejected invalid seed value")

    try:
        config = TrainingConfig(
            base_model="test",
            output_name="test",
            max_grad_norm=6.0  # Too high, max is 5.0
        )
        print("❌ Should have rejected max_grad_norm=6.0")
        return False
    except ValidationError:
        print("✅ Correctly rejected invalid max_grad_norm value")

    print("\n" + "="*60)
    print("🎉 All tests passed! New settings are properly configured.")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_new_settings()
    sys.exit(0 if success else 1)
