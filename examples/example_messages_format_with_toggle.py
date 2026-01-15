"""
Example: Using Messages Format with Toggle Control

This example demonstrates how to control messages format conversion
using the convert_messages_format parameter.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_handlers.base import DatasetPipeline
from dataset_handlers.loaders import LocalFileLoader
from dataset_handlers.formatters import ChatMLFormatter


def main():
    print("=" * 70)
    print("Merlina Messages Format with Toggle Control")
    print("=" * 70)
    print()

    # Path to example messages format dataset
    dataset_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_messages_dataset.json"

    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return

    print(f"Dataset: {dataset_path}")
    print()

    # Example 1: With conversion enabled (default behavior)
    print("📝 Example 1: Auto-Conversion ENABLED (Default)")
    print("-" * 70)

    loader = LocalFileLoader(dataset_path)
    formatter = ChatMLFormatter()

    pipeline_enabled = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        training_mode='sft',
        convert_messages_format=True  # Explicitly enable (this is the default)
    )

    preview = pipeline_enabled.preview(num_samples=1)
    print(f"✓ Columns: {list(preview[0].keys())}")
    print(f"✓ Prompt: {preview[0]['prompt']}")
    print(f"✓ Chosen: {preview[0]['chosen'][:100]}...")
    if preview[0].get('system'):
        print(f"✓ System: {preview[0]['system']}")
    print()

    # Example 2: With conversion disabled
    print("📝 Example 2: Auto-Conversion DISABLED")
    print("-" * 70)

    pipeline_disabled = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        training_mode='sft',
        convert_messages_format=False  # Disable conversion
    )

    preview_disabled = pipeline_disabled.preview(num_samples=1)
    print(f"✓ Columns: {list(preview_disabled[0].keys())}")
    print(f"✓ Raw messages format preserved:")
    print(f"  {preview_disabled[0]['messages'][:2]}")  # Show first 2 messages
    print()

    # Example 3: UI Integration
    print("📝 Example 3: How it works in the UI")
    print("-" * 70)
    print("""
When you use the Merlina web interface:

1. Click "🔍 Inspect Dataset Columns" button

2. If a "messages" column is detected, you'll see a notice:

   ╔════════════════════════════════════════════════════════╗
   ║  ✨ Messages Format Detected!                          ║
   ║                                                         ║
   ║  This dataset uses the common "messages" format.       ║
   ║  Enable auto-convert to transform it into standard     ║
   ║  format (prompt, chosen, system).                      ║
   ║                                                         ║
   ║  ☑ Auto-convert messages format                        ║
   ║                                                         ║
   ║  When enabled: user → prompt, assistant → chosen       ║
   ╚════════════════════════════════════════════════════════╝

3. The checkbox is checked by default (auto-conversion enabled)

4. Uncheck it if you want to keep the raw messages format

5. Preview and training will respect your choice
    """)

    # Example 4: API Usage
    print("📝 Example 4: API Configuration")
    print("-" * 70)
    print("""
When using the API directly:

POST /train
{
    "dataset": {
        "source": {
            "source_type": "local_file",
            "file_path": "/path/to/messages_dataset.json"
        },
        "format": {
            "format_type": "chatml"
        },
        "convert_messages_format": true,  // <- Toggle here!
        "training_mode": "sft"
    },
    ...
}

Set to `false` to disable automatic conversion.
    """)

    print("=" * 70)
    print("✅ Examples complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  • Default behavior: Auto-conversion is ENABLED")
    print("  • UI: Toggle appears automatically when messages column detected")
    print("  • API: Use convert_messages_format parameter in DatasetConfig")
    print("  • Flexibility: Choose auto-conversion or keep raw format")


if __name__ == "__main__":
    main()
