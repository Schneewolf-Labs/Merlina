"""
Example: Using Messages Format Datasets with Merlina

This example demonstrates how to use datasets in the common "messages" format
with Merlina. The messages format is automatically detected and converted.

Messages format structure:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_handlers.base import DatasetPipeline
from dataset_handlers.loaders import LocalFileLoader
from dataset_handlers.formatters import ChatMLFormatter


def main():
    print("=" * 60)
    print("Merlina Messages Format Example")
    print("=" * 60)
    print()

    # Path to example messages format dataset
    dataset_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_messages_dataset.json"

    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please ensure tests/fixtures/test_messages_dataset.json exists")
        return

    print(f"Loading dataset from: {dataset_path}")
    print()

    # Create loader and formatter
    loader = LocalFileLoader(dataset_path)
    formatter = ChatMLFormatter()

    # Create pipeline
    pipeline = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        training_mode='sft'  # Messages format is best suited for SFT mode
    )

    # Preview raw data (after conversion from messages format)
    print("📋 Preview of converted dataset (raw):")
    print("-" * 60)
    raw_samples = pipeline.preview(num_samples=2)

    for i, sample in enumerate(raw_samples, 1):
        print(f"\nSample {i}:")
        if sample.get('system'):
            print(f"  System: {sample['system'][:100]}...")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Chosen: {sample['chosen'][:100]}...")

    print()
    print()

    # Preview formatted data
    print("✨ Preview of formatted dataset (ChatML):")
    print("-" * 60)
    formatted_samples = pipeline.preview_formatted(num_samples=2)

    for i, sample in enumerate(formatted_samples, 1):
        print(f"\nSample {i}:")
        print(f"  Prompt: {sample['prompt'][:150]}...")
        print(f"  Chosen: {sample['chosen'][:100]}...")

    print()
    print()

    print("✅ Messages format automatic conversion works!")
    print()
    print("Key Points:")
    print("  • Messages format is automatically detected")
    print("  • System messages are extracted to 'system' field")
    print("  • User messages are combined into 'prompt' field")
    print("  • Assistant messages are combined into 'chosen' field")
    print("  • Multi-turn conversations are separated by double newlines")
    print("  • Best suited for SFT training mode")
    print()
    print("You can now use this dataset for training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
