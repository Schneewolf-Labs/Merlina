#!/usr/bin/env python3
"""
Integration test for model loading support (HuggingFace and local paths)
Tests the preflight validation with both types of model sources
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preflight_checks import PreflightValidator, is_local_model_path
from pydantic import BaseModel, Field
from typing import Optional


# Mock minimal config for testing
class MockDatasetSource(BaseModel):
    source_type: str = "huggingface"
    repo_id: Optional[str] = "test/dataset"

class MockDatasetFormat(BaseModel):
    format_type: str = "tokenizer"

class MockDatasetConfig(BaseModel):
    source: MockDatasetSource = MockDatasetSource()
    format: MockDatasetFormat = MockDatasetFormat()

class MockTrainingConfig(BaseModel):
    base_model: str
    output_name: str = "test-output"
    lora_r: int = 64
    lora_alpha: int = 32
    learning_rate: float = 5e-6
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 2048
    max_prompt_length: int = 1024
    beta: float = 0.1
    use_4bit: bool = True
    hf_token: Optional[str] = None
    wandb_api_key: Optional[str] = None
    dataset: MockDatasetConfig = MockDatasetConfig()


def test_huggingface_model():
    """Test validation with HuggingFace model ID"""
    print("\n=== Test 1: HuggingFace Model ===")

    config = MockTrainingConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct"
    )

    validator = PreflightValidator()
    is_valid, result = validator.validate_all(config)

    model_check = result["checks"].get("Model Access", {}).get("details", {})

    print(f"Model: {config.base_model}")
    print(f"Detected as local: {is_local_model_path(config.base_model)}")
    print(f"Model check result: {model_check}")

    assert not is_local_model_path(config.base_model), "Should detect as HuggingFace model"
    assert model_check.get("is_local") == False, "Model check should mark as non-local"

    print("✓ Test passed: HuggingFace model correctly identified")


def test_local_model_nonexistent():
    """Test validation with non-existent local path"""
    print("\n=== Test 2: Non-existent Local Path ===")

    config = MockTrainingConfig(
        base_model="/path/to/nonexistent/model"
    )

    validator = PreflightValidator()
    is_valid, result = validator.validate_all(config)

    model_check = result["checks"].get("Model Access", {}).get("details", {})

    print(f"Model: {config.base_model}")
    print(f"Detected as local: {is_local_model_path(config.base_model)}")
    print(f"Model check result: {model_check}")
    print(f"Errors: {result['errors']}")

    assert is_local_model_path(config.base_model), "Should detect as local path"
    assert model_check.get("is_local") == True, "Model check should mark as local"
    assert not is_valid, "Validation should fail for non-existent path"
    assert any("does not exist" in err for err in result["errors"]), "Should have existence error"

    print("✓ Test passed: Non-existent local path correctly rejected")


def test_local_model_relative_path():
    """Test validation with relative path"""
    print("\n=== Test 3: Relative Path ===")

    config = MockTrainingConfig(
        base_model="./models/my-model"
    )

    validator = PreflightValidator()
    is_local = is_local_model_path(config.base_model)

    print(f"Model: {config.base_model}")
    print(f"Detected as local: {is_local}")

    assert is_local, "Should detect relative path as local"

    print("✓ Test passed: Relative path correctly identified as local")


def test_gated_model_without_token():
    """Test validation with gated HuggingFace model"""
    print("\n=== Test 4: Gated HuggingFace Model Without Token ===")

    config = MockTrainingConfig(
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        hf_token=None
    )

    validator = PreflightValidator()
    is_valid, result = validator.validate_all(config)

    model_check = result["checks"].get("Model Access", {}).get("details", {})

    print(f"Model: {config.base_model}")
    print(f"Detected as local: {is_local_model_path(config.base_model)}")
    print(f"Model check result: {model_check}")
    print(f"Errors: {result['errors']}")

    assert not is_local_model_path(config.base_model), "Should detect as HuggingFace model"
    assert model_check.get("is_gated") == True, "Should detect as gated model"
    assert not is_valid, "Validation should fail without token"
    assert any("gated" in err.lower() for err in result["errors"]), "Should have gating error"

    print("✓ Test passed: Gated model without token correctly rejected")


def test_model_path_edge_cases():
    """Test edge cases for path detection"""
    print("\n=== Test 5: Edge Cases ===")

    test_cases = [
        ("meta-llama/Meta-Llama-3-8B", False, "Standard HF format"),
        ("./local/model", True, "Relative path with ./"),
        ("../models/llama", True, "Relative path with ../"),
        ("/absolute/path/model", True, "Absolute path"),
        ("models/sub/folder/model", True, "Multiple slashes (likely path)"),
        ("username/modelname", False, "HF org/model format"),
    ]

    all_passed = True
    for model_path, expected_local, description in test_cases:
        result = is_local_model_path(model_path)
        status = "✓" if result == expected_local else "✗"
        print(f"{status} {description}: {model_path} -> {'Local' if result else 'HF'}")

        if result != expected_local:
            all_passed = False

    assert all_passed, "Some edge cases failed"
    print("✓ Test passed: All edge cases handled correctly")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Model Loading Integration Tests")
    print("Testing support for both HuggingFace and local model paths")
    print("=" * 70)

    try:
        test_huggingface_model()
        test_local_model_nonexistent()
        test_local_model_relative_path()
        test_gated_model_without_token()
        test_model_path_edge_cases()

        print("\n" + "=" * 70)
        print("✓ All integration tests passed!")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"✗ Test failed: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
