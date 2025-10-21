#!/usr/bin/env python3
"""
Quick test script for the dataset module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset_handlers import (
    DatasetPipeline,
    HuggingFaceLoader,
    get_formatter
)

print("=" * 60)
print("Testing Merlina Dataset Module")
print("=" * 60)

# Test 1: HuggingFace Loader with ChatML Formatter
print("\n1. Testing HuggingFace Loader...")
try:
    loader = HuggingFaceLoader(
        repo_id="schneewolflabs/Athanor-DPO",
        split="train"
    )
    formatter = get_formatter("chatml")

    pipeline = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        max_samples=5,  # Just load 5 samples for testing
        test_size=0.2
    )

    # Preview raw data
    print("\n   Raw dataset preview:")
    preview = pipeline.preview(num_samples=2)
    for i, sample in enumerate(preview[:1]):
        print(f"   Sample {i+1} keys: {list(sample.keys())}")

    # Prepare formatted data
    print("\n   Preparing formatted dataset...")
    train_ds, eval_ds = pipeline.prepare()
    print(f"   ✅ Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    # Show one formatted example
    print("\n   Formatted example:")
    example = train_ds[0]
    print(f"   Prompt (first 100 chars): {example['prompt'][:100]}...")
    print(f"   Chosen (first 50 chars): {example['chosen'][:50]}...")

except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Formatter Factory
print("\n2. Testing Formatter Factory...")
try:
    formatters = ["chatml", "llama3", "mistral"]
    for fmt in formatters:
        formatter = get_formatter(fmt)
        info = formatter.get_format_info()
        print(f"   ✅ {info['format_type']}: {info['description'][:50]}...")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Custom Formatter
print("\n3. Testing Custom Formatter...")
try:
    custom_formatter = get_formatter(
        "custom",
        custom_templates={
            "prompt_template": "USER: {prompt}\nASSISTANT: ",
            "chosen_template": "{chosen}",
            "rejected_template": "{rejected}"
        }
    )

    test_row = {
        "system": "You are helpful",
        "prompt": "Hello!",
        "chosen": "Hi there!",
        "rejected": "Go away"
    }

    result = custom_formatter.format(test_row)
    print(f"   ✅ Custom format result: {result['prompt'][:50]}...")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 60)
print("✅ Dataset module tests complete!")
print("=" * 60)
