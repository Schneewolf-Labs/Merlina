#!/usr/bin/env python3
"""
Test formatted preview functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset_handlers import (
    DatasetPipeline,
    LocalFileLoader,
    get_formatter
)

print("=" * 60)
print("Testing Formatted Preview")
print("=" * 60)

# Test with different formatters
formatters_to_test = ["chatml", "llama3", "mistral"]

for format_type in formatters_to_test:
    print(f"\n{'='*60}")
    print(f"Format: {format_type.upper()}")
    print('='*60)

    try:
        loader = LocalFileLoader("test_dataset.json")
        formatter = get_formatter(format_type)

        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            test_size=0.25
        )

        # Get formatted preview
        formatted = pipeline.preview_formatted(num_samples=1)

        if formatted:
            sample = formatted[0]

            print("\n📝 PROMPT:")
            print("-" * 60)
            print(sample['prompt'])

            print("\n✅ CHOSEN:")
            print("-" * 60)
            print(sample['chosen'])

            print("\n❌ REJECTED:")
            print("-" * 60)
            print(sample['rejected'])

        print("\n✅ Test passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# Test custom formatter
print(f"\n{'='*60}")
print("Format: CUSTOM")
print('='*60)

try:
    loader = LocalFileLoader("test_dataset.json")
    formatter = get_formatter(
        "custom",
        custom_templates={
            "prompt_template": "### Human: {prompt}\n### Assistant: ",
            "chosen_template": "{chosen}",
            "rejected_template": "{rejected}"
        }
    )

    pipeline = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        test_size=0.25
    )

    formatted = pipeline.preview_formatted(num_samples=1)

    if formatted:
        sample = formatted[0]

        print("\n📝 PROMPT:")
        print("-" * 60)
        print(sample['prompt'])

        print("\n✅ CHOSEN:")
        print("-" * 60)
        print(sample['chosen'])

    print("\n✅ Custom format test passed!")

except Exception as e:
    print(f"\n❌ Test failed: {e}")

print("\n" + "=" * 60)
print("✅ All formatted preview tests complete!")
print("=" * 60)
