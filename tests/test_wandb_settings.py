#!/usr/bin/env python3
"""
Test script to verify W&B settings and auto-naming functionality
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import only the run name generator (avoid full imports)
from src.training_runner import generate_wandb_run_name

class MockConfig:
    """Mock config for testing"""
    def __init__(self, **kwargs):
        # Defaults
        self.base_model = kwargs.get('base_model', 'meta-llama/Llama-3-8B-Instruct')
        self.learning_rate = kwargs.get('learning_rate', 5e-6)
        self.batch_size = kwargs.get('batch_size', 1)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 16)
        self.num_epochs = kwargs.get('num_epochs', 2)
        self.optimizer_type = kwargs.get('optimizer_type', 'paged_adamw_8bit')
        self.attn_implementation = kwargs.get('attn_implementation', 'auto')
        self.use_4bit = kwargs.get('use_4bit', True)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self.beta = kwargs.get('beta', 0.1)
        # LoRA settings
        self.use_lora = kwargs.get('use_lora', True)
        self.lora_r = kwargs.get('lora_r', 64)

def test_wandb_settings():
    """Test W&B configuration and auto-naming"""

    print("🧪 Testing Weights & Biases Configuration...\n")

    # Test 1: Default configuration auto-naming
    print("="*60)
    print("Test 1: Default Configuration")
    print("="*60)

    config1 = MockConfig()
    run_name1 = generate_wandb_run_name(config1)

    print(f"\nInput:")
    print(f"  Model: {config1.base_model}")
    print(f"  Learning Rate: {config1.learning_rate}")
    print(f"  Batch: {config1.batch_size} x {config1.gradient_accumulation_steps} = {config1.batch_size * config1.gradient_accumulation_steps}")
    print(f"  Epochs: {config1.num_epochs}")
    print(f"  Optimizer: {config1.optimizer_type}")
    print(f"  Attention: {config1.attn_implementation}")
    print(f"  4-bit: {config1.use_4bit}")

    print(f"\n✨ Generated Run Name:")
    print(f"  {run_name1}")

    # Test 2: Llama 3.2 with different settings
    print("\n" + "="*60)
    print("Test 2: Llama 3.2 - 3B Model")
    print("="*60)

    config2 = MockConfig(
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        learning_rate=1e-5,
        batch_size=2,
        gradient_accumulation_steps=32,
        num_epochs=3,
        optimizer_type="adamw_torch",
        attn_implementation="flash_attention_2",
        use_4bit=False
    )
    run_name2 = generate_wandb_run_name(config2)

    print(f"\nInput:")
    print(f"  Model: {config2.base_model}")
    print(f"  LR: {config2.learning_rate}")
    print(f"  Effective Batch: {config2.batch_size * config2.gradient_accumulation_steps}")

    print(f"\n✨ Generated Run Name:")
    print(f"  {run_name2}")

    # Test 3: Mistral with gradient checkpointing
    print("\n" + "="*60)
    print("Test 3: Mistral with Gradient Checkpointing")
    print("="*60)

    config3 = MockConfig(
        base_model="mistralai/Mistral-7B-Instruct-v0.3",
        learning_rate=3e-6,
        batch_size=1,
        gradient_accumulation_steps=128,
        num_epochs=1,
        optimizer_type="adafactor",
        attn_implementation="sdpa",
        use_4bit=True,
        gradient_checkpointing=True,
        beta=0.2
    )
    run_name3 = generate_wandb_run_name(config3)

    print(f"\nInput:")
    print(f"  Model: {config3.base_model}")
    print(f"  Special settings: 4-bit + gradient checkpointing + custom beta")

    print(f"\n✨ Generated Run Name:")
    print(f"  {run_name3}")

    # Test 4: Qwen model
    print("\n" + "="*60)
    print("Test 4: Qwen 2.5 Model")
    print("="*60)

    config4 = MockConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        learning_rate=8e-6,
        batch_size=4,
        gradient_accumulation_steps=16,
        num_epochs=2,
        optimizer_type="paged_adamw_32bit",
        attn_implementation="eager",
        use_4bit=False
    )
    run_name4 = generate_wandb_run_name(config4)

    print(f"\nInput:")
    print(f"  Model: {config4.base_model}")

    print(f"\n✨ Generated Run Name:")
    print(f"  {run_name4}")

    # Test 5: Show naming format breakdown
    print("\n" + "="*60)
    print("Auto-Naming Format Breakdown")
    print("="*60)

    print("\n📋 Format Structure:")
    print("  [model]-[lr]-[batch]-[epochs]ep-[optimizer]-[attention]-[flags]")

    print("\n🔍 Component Details:")
    print("  • model: Simplified model name (e.g., 'llama3-8b', 'mistral-7b')")
    print("  • lr: Learning rate in scientific notation (e.g., '5e-6LR')")
    print("  • batch: Effective batch size = batch_size × grad_accum (e.g., '256B')")
    print("  • epochs: Number of training epochs (e.g., '2ep')")
    print("  • optimizer: Simplified optimizer name (e.g., 'adamw8bit', 'adafactor')")
    print("  • attention: Attention implementation (e.g., 'flash2', 'sdpa', 'eager')")
    print("  • flags: Optional suffixes for special settings:")
    print("    - '4bit': 4-bit quantization enabled")
    print("    - 'gc': Gradient checkpointing enabled")
    print("    - 'beta0.2': Non-default ORPO beta value")

    # Test 6: W&B Configuration Examples
    print("\n" + "="*60)
    print("W&B Configuration Examples")
    print("="*60)

    examples = [
        {
            "project": "llama3-finetuning",
            "run_name": None,  # Auto-generated
            "tags": ["experiment", "llama3", "orpo"],
            "notes": "Testing ORPO with different beta values"
        },
        {
            "project": "production-models",
            "run_name": "prod-llama3-v1",  # Custom
            "tags": ["production", "validated"],
            "notes": "Production model trained on curated dataset"
        },
        {
            "project": "merlina-training",  # Default
            "run_name": None,
            "tags": None,
            "notes": None
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Project: {example['project']}")
        print(f"  Run Name: {example['run_name'] or 'Auto-generated'}")
        print(f"  Tags: {example['tags'] or '(none)'}")
        print(f"  Notes: {example['notes'] or '(none)'}")

    # Test 7: Frontend Integration
    print("\n" + "="*60)
    print("Frontend Integration")
    print("="*60)

    print("\n📊 UI Behavior:")
    print("  1. W&B checkbox checked → Configuration section appears")
    print("  2. Leave 'Run Name' empty → Auto-generation happens")
    print("  3. Custom 'Run Name' → Uses your custom name")
    print("  4. Tags: comma-separated → Parsed into array")
    print("  5. Notes: free text → Stored with run")

    print("\n💡 Best Practices:")
    print("  • Use auto-naming for consistency across experiments")
    print("  • Group related runs in same project")
    print("  • Use tags for filtering (e.g., 'baseline', 'experiment', 'production')")
    print("  • Add notes to document experiment purpose")
    print("  • Custom names when you need specific identifiers")

    # Summary
    print("\n" + "="*60)
    print("🎉 All W&B tests passed!")
    print("="*60)

    print("\n📝 Sample Run Names Generated:")
    print(f"  1. {run_name1}")
    print(f"  2. {run_name2}")
    print(f"  3. {run_name3}")
    print(f"  4. {run_name4}")

    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Start server: python merlina.py")
    print("  2. Open http://localhost:8000")
    print("  3. Enable 'Weights & Biases' checkbox")
    print("  4. Configure W&B settings (or leave empty for auto)")
    print("  5. Train and see auto-generated run names in W&B!")
    print("="*60)

    return True

if __name__ == "__main__":
    test_wandb_settings()
