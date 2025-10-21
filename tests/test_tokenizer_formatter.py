"""
Test script for the TokenizerFormatter feature
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from transformers import AutoTokenizer
from dataset_handlers import get_formatter

def test_tokenizer_formatter():
    """Test the tokenizer-based formatter"""

    print("=" * 60)
    print("Testing TokenizerFormatter")
    print("=" * 60)

    # Test data
    test_row = {
        "system": "You are a helpful assistant.",
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "I don't know."
    }

    # Test with different tokenizers
    models_to_test = [
        "meta-llama/Llama-3.2-1B-Instruct",  # Has chat template
        "gpt2"  # Doesn't have chat template (fallback test)
    ]

    for model_name in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing with: {model_name}")
        print(f"{'=' * 60}")

        try:
            # Load tokenizer
            print(f"Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Check if it has chat template
            has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
            print(f"Has chat template: {has_template}")

            if has_template:
                print(f"Chat template preview (first 200 chars):")
                print(f"  {str(tokenizer.chat_template)[:200]}...")

            # Create formatter
            formatter = get_formatter(
                format_type='tokenizer',
                tokenizer=tokenizer
            )

            # Format the test row
            formatted = formatter.format(test_row)

            # Display results
            print(f"\nFormatted output:")
            print(f"  Prompt:")
            print(f"    {repr(formatted['prompt'][:200])}...")
            print(f"\n  Chosen:")
            print(f"    {repr(formatted['chosen'][:100])}...")
            print(f"\n  Rejected:")
            print(f"    {repr(formatted['rejected'][:100])}...")

            # Get format info
            info = formatter.get_format_info()
            print(f"\nFormat info:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            print(f"\n{'=' * 60}")
            print(f"✓ Test passed for {model_name}")

        except Exception as e:
            print(f"\n✗ Test failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Test error handling
    print(f"\n{'=' * 60}")
    print("Testing error handling")
    print(f"{'=' * 60}")

    try:
        # Try to create tokenizer formatter without tokenizer
        formatter = get_formatter(format_type='tokenizer')
        print("✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print(f"\n{'=' * 60}")
    print("All tests completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_tokenizer_formatter()
