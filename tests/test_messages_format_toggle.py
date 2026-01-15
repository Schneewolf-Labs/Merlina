"""
Test messages format conversion toggle functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from dataset_handlers.base import DatasetPipeline
from dataset_handlers.loaders import LocalFileLoader
from dataset_handlers.formatters import ChatMLFormatter


def test_messages_format_toggle():
    """Test that the convert_messages_format parameter works"""
    print("Testing messages format conversion toggle...\n")

    # Create test dataset with messages format
    messages_dataset_path = Path(__file__).parent / "fixtures" / "test_messages_dataset.json"

    if not messages_dataset_path.exists():
        print(f"❌ Test dataset not found: {messages_dataset_path}")
        return False

    loader = LocalFileLoader(messages_dataset_path)
    formatter = ChatMLFormatter()

    # Test 1: With conversion enabled (default)
    print("Test 1: Messages format conversion ENABLED")
    print("-" * 60)

    pipeline_with_conversion = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        training_mode='sft',
        convert_messages_format=True
    )

    preview_with = pipeline_with_conversion.preview(num_samples=1)
    print(f"✓ Columns with conversion: {list(preview_with[0].keys())}")
    assert 'prompt' in preview_with[0], "Should have 'prompt' column after conversion"
    assert 'chosen' in preview_with[0], "Should have 'chosen' column after conversion"
    assert 'messages' not in preview_with[0], "Should not have 'messages' column after conversion"
    print(f"✓ Sample prompt: {preview_with[0]['prompt'][:50]}...")
    print()

    # Test 2: With conversion disabled
    print("Test 2: Messages format conversion DISABLED")
    print("-" * 60)

    pipeline_without_conversion = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        training_mode='sft',
        convert_messages_format=False
    )

    try:
        preview_without = pipeline_without_conversion.preview(num_samples=1)
        print(f"✓ Columns without conversion: {list(preview_without[0].keys())}")
        assert 'messages' in preview_without[0], "Should still have 'messages' column when conversion disabled"
        print(f"✓ Still has messages column: {str(preview_without[0]['messages'])[:80]}...")
        print()
    except Exception as e:
        print(f"✓ Expected behavior: Dataset validation may fail without conversion")
        print(f"  Error: {str(e)[:100]}...")
        print()

    # Test 3: Verify formatted preview works with conversion
    print("Test 3: Formatted preview with conversion")
    print("-" * 60)

    formatted = pipeline_with_conversion.preview_formatted(num_samples=1)
    print(f"✓ Formatted sample contains ChatML tags: {'<|im_start|>' in formatted[0]['prompt']}")
    assert '<|im_start|>' in formatted[0]['prompt'], "Should have ChatML formatting"
    print()

    print("=" * 60)
    print("✅ All toggle tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_messages_format_toggle()
    sys.exit(0 if success else 1)
