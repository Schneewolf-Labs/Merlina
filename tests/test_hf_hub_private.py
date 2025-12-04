#!/usr/bin/env python3
"""
Test script for HuggingFace Hub private repository feature
Validates that the new hf_hub_private parameter is properly configured
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import ValidationError
from merlina import TrainingConfig


def test_hf_hub_private_parameter():
    """Test the hf_hub_private parameter in TrainingConfig"""
    print("=" * 70)
    print("Testing HuggingFace Hub Private Repository Feature")
    print("=" * 70)
    print()

    # Test 1: Default value should be True (private by default)
    print("Test 1: Default value")
    config = TrainingConfig(
        output_name="test-model",
        dataset={
            "source": {"source_type": "huggingface", "repo_id": "test/dataset"},
            "format": {"format_type": "tokenizer"}
        }
    )
    assert config.hf_hub_private == True, "Default should be True (private)"
    print(f"  ✓ Default hf_hub_private: {config.hf_hub_private} (private)")
    print()

    # Test 2: Can be set to False (public)
    print("Test 2: Set to False (public repository)")
    config = TrainingConfig(
        output_name="test-model",
        hf_hub_private=False,
        dataset={
            "source": {"source_type": "huggingface", "repo_id": "test/dataset"},
            "format": {"format_type": "tokenizer"}
        }
    )
    assert config.hf_hub_private == False, "Should be False when explicitly set"
    print(f"  ✓ hf_hub_private: {config.hf_hub_private} (public)")
    print()

    # Test 3: Can be set to True (explicit private)
    print("Test 3: Set to True (explicit private)")
    config = TrainingConfig(
        output_name="test-model",
        hf_hub_private=True,
        dataset={
            "source": {"source_type": "huggingface", "repo_id": "test/dataset"},
            "format": {"format_type": "tokenizer"}
        }
    )
    assert config.hf_hub_private == True, "Should be True when explicitly set"
    print(f"  ✓ hf_hub_private: {config.hf_hub_private} (private)")
    print()

    # Test 4: Full configuration with push_to_hub enabled
    print("Test 4: Full configuration with push_to_hub enabled")
    config = TrainingConfig(
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        output_name="my-private-model",
        push_to_hub=True,
        hf_hub_private=True,
        hf_token="hf_example_token",
        dataset={
            "source": {"source_type": "huggingface", "repo_id": "test/dataset"},
            "format": {"format_type": "tokenizer"}
        }
    )

    print(f"  Model: {config.base_model}")
    print(f"  Output: {config.output_name}")
    print(f"  Push to Hub: {config.push_to_hub}")
    print(f"  Private Repository: {config.hf_hub_private}")
    print(f"  Has HF Token: {config.hf_token is not None}")
    print()

    assert config.push_to_hub == True
    assert config.hf_hub_private == True
    assert config.hf_token == "hf_example_token"
    print("  ✓ All settings configured correctly")
    print()

    # Test 5: Configuration with public repository
    print("Test 5: Configuration with public repository")
    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        output_name="my-public-model",
        push_to_hub=True,
        hf_hub_private=False,
        hf_token="hf_example_token",
        dataset={
            "source": {"source_type": "huggingface", "repo_id": "test/dataset"},
            "format": {"format_type": "tokenizer"}
        }
    )

    print(f"  Model: {config.base_model}")
    print(f"  Output: {config.output_name}")
    print(f"  Push to Hub: {config.push_to_hub}")
    print(f"  Private Repository: {config.hf_hub_private}")
    print()

    assert config.push_to_hub == True
    assert config.hf_hub_private == False
    print("  ✓ Public repository configured correctly")
    print()

    # Test 6: JSON serialization/deserialization
    print("Test 6: JSON serialization")
    config_dict = config.dict()
    assert "hf_hub_private" in config_dict
    print(f"  ✓ hf_hub_private in serialized config: {config_dict['hf_hub_private']}")
    print()

    print("=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  - hf_hub_private parameter is working correctly")
    print("  - Default value is True (private repositories)")
    print("  - Can be set to False for public repositories")
    print("  - Properly serialized in configuration")
    print()


if __name__ == "__main__":
    try:
        test_hf_hub_private_parameter()
        exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
