"""
Test script for the TokenizerFormatter feature
"""

import sys
import os

# Get the directory containing this test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TEST_DIR, '..'))

from dataset_handlers import get_formatter


class MockTokenizerWithChatTemplate:
    """Mock tokenizer with chat template support"""

    def __init__(self):
        self.name_or_path = "mock-llama3-tokenizer"
        # Simple ChatML-style template
        self.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Simple implementation of apply_chat_template"""
        result = ""
        for message in messages:
            role = message['role']
            content = message['content']
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        return result


class MockTokenizerWithoutChatTemplate:
    """Mock tokenizer without chat template support (fallback test)"""

    def __init__(self):
        self.name_or_path = "mock-basic-tokenizer"
        self.chat_template = None


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

    # Test 1: Tokenizer with chat template
    print(f"\n{'=' * 60}")
    print("Test 1: Tokenizer with chat template")
    print(f"{'=' * 60}")

    try:
        tokenizer = MockTokenizerWithChatTemplate()
        print(f"Using mock tokenizer: {tokenizer.name_or_path}")
        print(f"Has chat template: True")

        formatter = get_formatter(
            format_type='tokenizer',
            tokenizer=tokenizer
        )

        formatted = formatter.format(test_row)

        print(f"\nFormatted output:")
        print(f"  Prompt:")
        print(f"    {repr(formatted['prompt'][:200])}")
        print(f"\n  Chosen:")
        print(f"    {repr(formatted['chosen'][:100])}")
        print(f"\n  Rejected:")
        print(f"    {repr(formatted['rejected'][:100])}")

        # Verify output contains expected patterns
        assert "<|im_start|>system" in formatted['prompt'], "Missing system tag in prompt"
        assert "<|im_start|>user" in formatted['prompt'], "Missing user tag in prompt"
        assert "<|im_start|>assistant" in formatted['prompt'], "Missing assistant tag in prompt"
        assert "Paris" in formatted['chosen'], "Missing expected content in chosen"

        info = formatter.get_format_info()
        print(f"\nFormat info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print(f"\n✓ Test 1 passed!")

    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test 2: Tokenizer without chat template (fallback)
    print(f"\n{'=' * 60}")
    print("Test 2: Tokenizer without chat template (fallback)")
    print(f"{'=' * 60}")

    try:
        tokenizer = MockTokenizerWithoutChatTemplate()
        print(f"Using mock tokenizer: {tokenizer.name_or_path}")
        print(f"Has chat template: False")

        formatter = get_formatter(
            format_type='tokenizer',
            tokenizer=tokenizer
        )

        formatted = formatter.format(test_row)

        print(f"\nFormatted output (fallback):")
        print(f"  Prompt:")
        print(f"    {repr(formatted['prompt'][:200])}")
        print(f"\n  Chosen:")
        print(f"    {repr(formatted['chosen'][:100])}")

        # Verify fallback works
        assert test_row['prompt'] in formatted['prompt'], "Missing prompt content"
        assert formatted['chosen'] == test_row['chosen'], "Chosen should be unchanged in fallback"

        info = formatter.get_format_info()
        print(f"\nFormat info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        assert info['has_chat_template'] == False, "Should indicate no chat template"

        print(f"\n✓ Test 2 passed!")

    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test 3: Error handling - no tokenizer provided
    print(f"\n{'=' * 60}")
    print("Test 3: Error handling - no tokenizer")
    print(f"{'=' * 60}")

    try:
        formatter = get_formatter(format_type='tokenizer')
        print("✗ Should have raised ValueError!")
        raise AssertionError("Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 4: Test with no system message
    print(f"\n{'=' * 60}")
    print("Test 4: Format row without system message")
    print(f"{'=' * 60}")

    try:
        tokenizer = MockTokenizerWithChatTemplate()
        formatter = get_formatter(
            format_type='tokenizer',
            tokenizer=tokenizer
        )

        test_row_no_system = {
            "prompt": "Hello!",
            "chosen": "Hi there!",
            "rejected": "Go away."
        }

        formatted = formatter.format(test_row_no_system)

        print(f"  Prompt: {repr(formatted['prompt'])}")

        # Should not have system tag since system is empty
        assert "<|im_start|>user" in formatted['prompt'], "Missing user tag"

        print(f"\n✓ Test 4 passed!")

    except Exception as e:
        print(f"\n✗ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n{'=' * 60}")
    print("✓ All TokenizerFormatter tests passed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_tokenizer_formatter()
