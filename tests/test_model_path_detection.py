#!/usr/bin/env python3
"""
Test script for model path detection
Tests the is_local_model_path() function with various inputs
"""

import sys
sys.path.insert(0, '/home/python/AI/merlina')

from src.preflight_checks import is_local_model_path


def test_model_path_detection():
    """Test various model path formats"""

    test_cases = [
        # (input, expected_is_local, description)
        ("meta-llama/Meta-Llama-3-8B", False, "HuggingFace model ID"),
        ("Qwen/Qwen2.5-7B-Instruct", False, "HuggingFace model ID with dot"),
        ("mistralai/Mistral-7B-v0.1", False, "HuggingFace model ID with version"),

        ("/home/python/models/llama3", True, "Absolute path"),
        ("./models/my-model", True, "Relative path with ./"),
        ("../models/my-model", True, "Relative path with ../"),

        ("/path/to/models/llama3-8b", True, "Absolute path with multiple slashes"),
        ("models/subfolder/my-model", True, "Relative path with multiple slashes"),

        ("C:\\Users\\models\\llama", True, "Windows path"),

        # Edge cases
        ("models", False, "Single directory name (could be HF username)"),
        ("my-org/my-model", False, "Standard HF format"),
    ]

    print("Testing model path detection...\n")
    print(f"{'Input':<40} {'Expected':<12} {'Actual':<12} {'Status'}")
    print("=" * 80)

    all_passed = True

    for input_path, expected_is_local, description in test_cases:
        result = is_local_model_path(input_path)
        status = "✓ PASS" if result == expected_is_local else "✗ FAIL"

        if result != expected_is_local:
            all_passed = False

        expected_str = "Local" if expected_is_local else "HuggingFace"
        actual_str = "Local" if result else "HuggingFace"

        print(f"{input_path:<40} {expected_str:<12} {actual_str:<12} {status}")

    print("=" * 80)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(test_model_path_detection())
