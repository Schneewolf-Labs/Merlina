#!/usr/bin/env python3
"""
Test script for model size detection
Tests the get_model_size_from_name() function used by the pre-flight VRAM check
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preflight_checks import get_model_size_from_name


def test_model_size_detection():
    """Test size detection for various model names"""

    test_cases = [
        # (input, expected_size_billions, description)
        # Explicit size tokens
        ("meta-llama/Meta-Llama-3-8B", 8.0, "Uppercase B"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", 8.0, "Size mid-name"),
        ("Qwen/Qwen2.5-7B-Instruct", 7.0, "HF ID with dot"),
        ("mistralai/Mistral-7B-v0.1", 7.0, "Mistral 7B"),
        ("microsoft/Phi-3-mini-4k-instruct", None, "No size token, unknown family"),
        ("google/gemma-2-27b-it", 27.0, "27B"),
        ("Qwen/Qwen2.5-0.5B", 0.5, "Fractional size"),
        ("some-org/model-1.5b", 1.5, "Fractional lowercase"),

        # Mixture-of-experts: total params, not per-expert size
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 56.0, "MoE 8x7B"),
        ("mistralai/Mixtral-8x22B-v0.1", 176.0, "MoE 8x22B"),

        # Known families with no size token in the name
        ("mistralai/Mistral-Nemo-Instruct-2407", 12.0, "Mistral-Nemo (12B)"),
        ("mistralai/Mistral-Nemo-Base-2407", 12.0, "Mistral-Nemo base"),
        ("mistral_nemo_finetune", 12.0, "Underscore separators"),
        ("mistralai/Mistral-Small-Instruct-2409", 22.0, "Mistral-Small (22B)"),
        ("mistralai/Mistral-Large-Instruct-2407", 123.0, "Mistral-Large (123B)"),
        ("mistralai/Codestral-22B-v0.1", 22.0, "Codestral with explicit size"),
        ("mistralai/Mamba-Codestral-7B-v0.1", 7.0, "Codestral-Mamba explicit size"),

        # Explicit size token wins over family fallback
        ("nvidia/Mistral-NeMo-Minitron-8B-Base", 8.0, "Nemo-derived 8B pruned model"),

        # Genuinely unknown
        ("my-org/my-custom-model", None, "No size info at all"),
        ("/home/user/models/my-model", None, "Local path without size"),
    ]

    print("Testing model size detection...\n")
    print(f"{'Input':<45} {'Expected':<10} {'Actual':<10} {'Status'}")
    print("=" * 85)

    all_passed = True

    for input_name, expected, description in test_cases:
        result = get_model_size_from_name(input_name)
        passed = result == expected
        status = "✓ PASS" if passed else "✗ FAIL"

        if not passed:
            all_passed = False

        print(f"{input_name:<45} {str(expected):<10} {str(result):<10} {status}")

    print("=" * 85)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(test_model_size_detection())
