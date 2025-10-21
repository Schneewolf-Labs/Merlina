#!/usr/bin/env python3
"""
Test local file loading
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
print("Testing Local File Dataset Loading")
print("=" * 60)

try:
    # Load from local JSON file
    loader = LocalFileLoader("test_dataset.json")
    formatter = get_formatter("chatml")

    pipeline = DatasetPipeline(
        loader=loader,
        formatter=formatter,
        test_size=0.25
    )

    # Preview raw data
    print("\n1. Raw dataset preview:")
    preview = pipeline.preview(num_samples=2)
    for i, sample in enumerate(preview):
        print(f"\n   Sample {i+1}:")
        print(f"   System: {sample.get('system', 'N/A')}")
        print(f"   Prompt: {sample['prompt'][:50]}...")
        print(f"   Chosen: {sample['chosen'][:50]}...")

    # Prepare formatted dataset
    print("\n2. Preparing formatted dataset...")
    train_ds, eval_ds = pipeline.prepare()
    print(f"   ✅ Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    # Show formatted example
    print("\n3. Formatted training example:")
    example = train_ds[0]
    print(f"   Prompt:\n{example['prompt']}")
    print(f"\n   Chosen:\n{example['chosen']}")
    print(f"\n   Rejected:\n{example['rejected']}")

    # Test with different formatter
    print("\n4. Testing Llama3 formatter:")
    formatter_llama = get_formatter("llama3")
    pipeline_llama = DatasetPipeline(
        loader=loader,
        formatter=formatter_llama,
        test_size=0.25
    )
    formatted = pipeline_llama.preview_formatted(num_samples=1)
    print(f"   Llama3 prompt:\n{formatted[0]['prompt'][:150]}...")

    print("\n" + "=" * 60)
    print("✅ All local file tests passed!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
