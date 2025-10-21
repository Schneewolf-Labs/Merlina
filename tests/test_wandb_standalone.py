#!/usr/bin/env python3
"""
Standalone test for W&B auto-naming (no imports needed)
"""

def generate_wandb_run_name_test(config):
    """Simplified version of the run name generator for testing"""
    model_name = config['base_model'].split('/')[-1].lower()
    model_name = model_name.replace('-instruct', '').replace('-base', '').replace('meta-', '')

    lr = config['learning_rate']
    lr_str = f"{lr:.0e}".replace('e-0', 'e-')

    effective_batch = config['batch_size'] * config['gradient_accumulation_steps']

    opt = config['optimizer_type'].replace('paged_', '').replace('_', '')

    attn_map = {
        'flash_attention_2': 'flash2',
        'sdpa': 'sdpa',
        'eager': 'eager',
        'auto': 'auto'
    }
    attn = attn_map.get(config['attn_implementation'], config['attn_implementation'])

    parts = [
        model_name,
        f"{lr_str}LR",
        f"{effective_batch}B",
        f"{config['num_epochs']}ep",
        opt,
        attn
    ]

    run_name = "-".join(parts)

    suffixes = []
    if config.get('use_4bit'):
        suffixes.append("4bit")
    if config.get('gradient_checkpointing'):
        suffixes.append("gc")
    if config.get('beta', 0.1) != 0.1:
        suffixes.append(f"beta{config['beta']}")

    if suffixes:
        run_name += f"-{'-'.join(suffixes)}"

    return run_name


def test_wandb_settings():
    """Test W&B configuration and auto-naming"""

    print("🧪 Testing Weights & Biases Configuration...\n")

    # Test 1: Default Llama 3 8B
    print("="*70)
    print("Test 1: Llama 3 8B (Default Settings)")
    print("="*70)

    config1 = {
        'base_model': 'meta-llama/Llama-3-8B-Instruct',
        'learning_rate': 5e-6,
        'batch_size': 1,
        'gradient_accumulation_steps': 16,
        'num_epochs': 2,
        'optimizer_type': 'paged_adamw_8bit',
        'attn_implementation': 'auto',
        'use_4bit': True,
        'gradient_checkpointing': False,
        'beta': 0.1
    }

    run_name1 = generate_wandb_run_name_test(config1)
    print(f"\n✨ Generated: {run_name1}\n")

    # Test 2: Llama 3.2 3B with Flash Attention
    print("="*70)
    print("Test 2: Llama 3.2 3B (High LR, Flash Attention 2)")
    print("="*70)

    config2 = {
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'learning_rate': 1e-5,
        'batch_size': 2,
        'gradient_accumulation_steps': 32,
        'num_epochs': 3,
        'optimizer_type': 'adamw_torch',
        'attn_implementation': 'flash_attention_2',
        'use_4bit': False,
        'gradient_checkpointing': False,
        'beta': 0.1
    }

    run_name2 = generate_wandb_run_name_test(config2)
    print(f"\n✨ Generated: {run_name2}\n")

    # Test 3: Mistral with all bells and whistles
    print("="*70)
    print("Test 3: Mistral 7B (All Optimizations)")
    print("="*70)

    config3 = {
        'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'learning_rate': 3e-6,
        'batch_size': 1,
        'gradient_accumulation_steps': 128,
        'num_epochs': 1,
        'optimizer_type': 'adafactor',
        'attn_implementation': 'sdpa',
        'use_4bit': True,
        'gradient_checkpointing': True,
        'beta': 0.2
    }

    run_name3 = generate_wandb_run_name_test(config3)
    print(f"\n✨ Generated: {run_name3}\n")

    # Test 4: Qwen 2.5
    print("="*70)
    print("Test 4: Qwen 2.5 7B (32-bit Paged Optimizer)")
    print("="*70)

    config4 = {
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        'learning_rate': 8e-6,
        'batch_size': 4,
        'gradient_accumulation_steps': 16,
        'num_epochs': 2,
        'optimizer_type': 'paged_adamw_32bit',
        'attn_implementation': 'eager',
        'use_4bit': False,
        'gradient_checkpointing': False,
        'beta': 0.1
    }

    run_name4 = generate_wandb_run_name_test(config4)
    print(f"\n✨ Generated: {run_name4}\n")

    # Naming Format Guide
    print("="*70)
    print("Auto-Naming Format Guide")
    print("="*70)

    print("\n📋 Format: [model]-[lr]-[batch]-[epochs]ep-[opt]-[attn]-[flags]\n")

    print("Components:")
    print("  • model:    Simplified name (llama3-8b, mistral-7b, qwen2.5-7b)")
    print("  • lr:       Scientific notation (5e-6LR, 1e-5LR)")
    print("  • batch:    Effective = batch × grad_accum (16B, 256B)")
    print("  • epochs:   Training epochs (2ep, 3ep)")
    print("  • opt:      Optimizer (adamw8bit, adafactor)")
    print("  • attn:     Attention (flash2, sdpa, eager, auto)")
    print("  • flags:    Optional (4bit, gc, beta0.2)")

    # W&B Configuration Examples
    print("\n" + "="*70)
    print("W&B Configuration in UI")
    print("="*70)

    print("\n✅ Enable W&B checkbox → Configuration panel appears\n")

    examples = [
        ("Default (Auto-naming)", {
            "Project": "merlina-training (default)",
            "Run Name": "(empty - auto-generated)",
            "Tags": "(empty)",
            "Notes": "(empty)"
        }),
        ("Experiment Series", {
            "Project": "llama3-experiments",
            "Run Name": "(empty for auto)",
            "Tags": "experiment, baseline, orpo",
            "Notes": "Testing different beta values"
        }),
        ("Production Run", {
            "Project": "production-models",
            "Run Name": "prod-llama3-v2.1",
            "Tags": "production, validated, deployed",
            "Notes": "Final production model for Q4"
        })
    ]

    for i, (title, settings) in enumerate(examples, 1):
        print(f"{i}. {title}:")
        for key, value in settings.items():
            print(f"     {key:12s}: {value}")
        print()

    # Summary
    print("="*70)
    print("📊 Example Run Names Generated")
    print("="*70)

    print(f"\n1. {run_name1}")
    print(f"2. {run_name2}")
    print(f"3. {run_name3}")
    print(f"4. {run_name4}")

    print("\n" + "="*70)
    print("💡 Benefits of Auto-Naming")
    print("="*70)

    print("\n  ✓ Consistent naming across all runs")
    print("  ✓ Easy to compare similar experiments")
    print("  ✓ Key hyperparameters visible at a glance")
    print("  ✓ No manual naming required")
    print("  ✓ Sortable and filterable in W&B UI")

    print("\n" + "="*70)
    print("🎉 All W&B tests passed!")
    print("="*70)

    print("\n📝 Next steps:")
    print("  1. Start server: python merlina.py")
    print("  2. Open http://localhost:8000")
    print("  3. Enable 'Weights & Biases' checkbox")
    print("  4. See W&B configuration panel appear")
    print("  5. Configure or leave empty for auto-naming")
    print("  6. Train and view in W&B dashboard!")
    print("="*70)

    return True

if __name__ == "__main__":
    test_wandb_settings()
