#!/usr/bin/env python3
"""
Tests for model README generation functionality.
Verifies the README.md content and YAML frontmatter generation for HuggingFace model cards.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from typing import Optional


# Minimal mock classes for testing
class MockDatasetSource(BaseModel):
    source_type: str = "huggingface"
    repo_id: Optional[str] = None


class MockDatasetConfig(BaseModel):
    source: MockDatasetSource = Field(default_factory=MockDatasetSource)


class MockTrainingConfig(BaseModel):
    """Minimal TrainingConfig for testing README generation"""
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    output_name: str = "test-model"
    training_mode: str = "orpo"

    # Training params
    learning_rate: float = 5e-6
    num_epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 2048
    max_prompt_length: int = 1024

    # ORPO specific
    beta: float = 0.1

    # Optimizer
    optimizer_type: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    seed: int = 42

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']

    # Quantization
    use_4bit: bool = True

    # Dataset
    dataset: MockDatasetConfig = Field(default_factory=MockDatasetConfig)


# Import the function to test
from src.training_runner import generate_model_readme


def test_readme_basic_structure():
    """Test that README has correct basic structure"""
    print("=" * 70)
    print("Test 1: Basic README structure")
    print("=" * 70)

    config = MockTrainingConfig()
    readme = generate_model_readme(config, "orpo")

    # Check it starts with YAML frontmatter
    assert readme.startswith("---"), "README should start with YAML frontmatter"
    assert readme.count("---") >= 2, "README should have opening and closing YAML delimiters"

    # Check for key sections
    assert "# test-model" in readme, "README should have model title"
    assert "## Training Configuration" in readme, "README should have configuration section"
    assert "library_name: transformers" in readme, "README should specify library_name"

    print("  ✓ README starts with YAML frontmatter")
    print("  ✓ Has model title")
    print("  ✓ Has configuration section")
    print()


def test_readme_base_model():
    """Test that base_model is correctly included"""
    print("=" * 70)
    print("Test 2: Base model in frontmatter")
    print("=" * 70)

    config = MockTrainingConfig(base_model="nbeerbower/Schreiber-mistral-nemo-12B")
    readme = generate_model_readme(config, "orpo")

    assert "base_model:" in readme, "README should have base_model field"
    assert "- nbeerbower/Schreiber-mistral-nemo-12B" in readme, "README should list the base model"

    print("  ✓ Base model included in frontmatter")
    print()


def test_readme_huggingface_dataset():
    """Test that HuggingFace dataset is included in frontmatter"""
    print("=" * 70)
    print("Test 3: HuggingFace dataset in frontmatter")
    print("=" * 70)

    config = MockTrainingConfig()
    config.dataset = MockDatasetConfig(
        source=MockDatasetSource(
            source_type="huggingface",
            repo_id="schneewolflabs/Athanorlite-DPO"
        )
    )
    readme = generate_model_readme(config, "orpo")

    assert "datasets:" in readme, "README should have datasets field"
    assert "- schneewolflabs/Athanorlite-DPO" in readme, "README should list the dataset"

    print("  ✓ HuggingFace dataset included in frontmatter")
    print()


def test_readme_local_dataset_excluded():
    """Test that local datasets don't add datasets field"""
    print("=" * 70)
    print("Test 4: Local dataset excluded from frontmatter")
    print("=" * 70)

    config = MockTrainingConfig()
    config.dataset = MockDatasetConfig(
        source=MockDatasetSource(
            source_type="local_file",
            repo_id=None
        )
    )
    readme = generate_model_readme(config, "orpo")

    # Should not have datasets field for local files
    lines = readme.split('\n')
    frontmatter_started = False
    frontmatter_ended = False
    has_datasets_field = False

    for line in lines:
        if line == '---' and not frontmatter_started:
            frontmatter_started = True
            continue
        if line == '---' and frontmatter_started:
            frontmatter_ended = True
            break
        if frontmatter_started and line.startswith('datasets:'):
            has_datasets_field = True

    assert not has_datasets_field, "Local datasets should not add datasets field"
    print("  ✓ Local dataset correctly excluded from frontmatter")
    print()


def test_readme_training_params():
    """Test that training parameters are in the configuration table"""
    print("=" * 70)
    print("Test 5: Training parameters in configuration table")
    print("=" * 70)

    config = MockTrainingConfig(
        learning_rate=1e-5,
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=8,
        max_length=4096,
        beta=0.15
    )
    readme = generate_model_readme(config, "orpo")

    # Check for config table entries
    assert "| Learning Rate | 1e-05 |" in readme, "Should have learning rate"
    assert "| Epochs | 3 |" in readme, "Should have epochs"
    assert "| Batch Size | 2 |" in readme, "Should have batch size"
    assert "| Gradient Accumulation | 8 |" in readme, "Should have gradient accumulation"
    assert "| Max Sequence Length | 4096 |" in readme, "Should have max length"
    assert "| ORPO Beta | 0.15 |" in readme, "Should have ORPO beta"

    print("  ✓ All training parameters present in configuration table")
    print()


def test_readme_lora_params():
    """Test that LoRA parameters are included when enabled"""
    print("=" * 70)
    print("Test 6: LoRA parameters when enabled")
    print("=" * 70)

    config = MockTrainingConfig(
        use_lora=True,
        lora_r=128,
        lora_alpha=64,
        lora_dropout=0.1
    )
    readme = generate_model_readme(config, "orpo")

    assert "| LoRA Rank (r) | 128 |" in readme, "Should have LoRA rank"
    assert "| LoRA Alpha | 64 |" in readme, "Should have LoRA alpha"
    assert "| LoRA Dropout | 0.1 |" in readme, "Should have LoRA dropout"
    assert "| Target Modules |" in readme, "Should have target modules"

    print("  ✓ LoRA parameters present when enabled")
    print()


def test_readme_lora_disabled():
    """Test that LoRA parameters are excluded when disabled"""
    print("=" * 70)
    print("Test 7: LoRA parameters excluded when disabled")
    print("=" * 70)

    config = MockTrainingConfig(use_lora=False)
    readme = generate_model_readme(config, "orpo")

    assert "| LoRA Rank" not in readme, "Should not have LoRA params when disabled"
    assert "| LoRA Alpha" not in readme, "Should not have LoRA params when disabled"

    print("  ✓ LoRA parameters correctly excluded when disabled")
    print()


def test_readme_sft_mode():
    """Test README generation for SFT mode"""
    print("=" * 70)
    print("Test 8: SFT mode README")
    print("=" * 70)

    config = MockTrainingConfig()
    readme = generate_model_readme(config, "sft")

    assert "| Training Mode | SFT |" in readme, "Should show SFT training mode"
    assert "| ORPO Beta |" not in readme, "Should not have ORPO beta in SFT mode"
    assert "| Max Prompt Length |" not in readme, "Should not have max prompt length in SFT mode"

    print("  ✓ SFT mode correctly formatted")
    print()


def test_readme_orpo_mode():
    """Test README generation for ORPO mode includes ORPO-specific params"""
    print("=" * 70)
    print("Test 9: ORPO mode includes ORPO-specific params")
    print("=" * 70)

    config = MockTrainingConfig(beta=0.2, max_prompt_length=512)
    readme = generate_model_readme(config, "orpo")

    assert "| Training Mode | ORPO |" in readme, "Should show ORPO training mode"
    assert "| ORPO Beta | 0.2 |" in readme, "Should have ORPO beta"
    assert "| Max Prompt Length | 512 |" in readme, "Should have max prompt length"

    print("  ✓ ORPO mode correctly includes ORPO-specific params")
    print()


def test_readme_merlina_badge():
    """Test that README includes Merlina badge"""
    print("=" * 70)
    print("Test 10: Merlina badge")
    print("=" * 70)

    config = MockTrainingConfig()
    readme = generate_model_readme(config, "orpo")

    expected_badge = "![Trained with Merlina](https://raw.githubusercontent.com/Schneewolf-Labs/Merlina/refs/heads/main/frontend/madewithmerlina_smol.png)"
    assert expected_badge in readme, "Should include Merlina badge"

    print("  ✓ Merlina badge present")
    print()


def test_readme_quantization():
    """Test that quantization info is included when enabled"""
    print("=" * 70)
    print("Test 11: Quantization info")
    print("=" * 70)

    config = MockTrainingConfig(use_4bit=True)
    readme = generate_model_readme(config, "orpo")

    assert "| Quantization | 4-bit (NF4) |" in readme, "Should show 4-bit quantization"

    # Test with 4-bit disabled
    config = MockTrainingConfig(use_4bit=False)
    readme = generate_model_readme(config, "orpo")

    assert "| Quantization |" not in readme, "Should not show quantization when disabled"

    print("  ✓ Quantization info correctly shown/hidden")
    print()


def print_example_readme():
    """Print an example README for visual inspection"""
    print("=" * 70)
    print("Example Generated README")
    print("=" * 70)
    print()

    config = MockTrainingConfig(
        base_model="nbeerbower/Schreiber-mistral-nemo-12B",
        output_name="A0l-12B"
    )
    config.dataset = MockDatasetConfig(
        source=MockDatasetSource(
            source_type="huggingface",
            repo_id="schneewolflabs/Athanorlite-DPO"
        )
    )

    readme = generate_model_readme(config, "orpo")
    print(readme)
    print()


if __name__ == "__main__":
    try:
        test_readme_basic_structure()
        test_readme_base_model()
        test_readme_huggingface_dataset()
        test_readme_local_dataset_excluded()
        test_readme_training_params()
        test_readme_lora_params()
        test_readme_lora_disabled()
        test_readme_sft_mode()
        test_readme_orpo_mode()
        test_readme_merlina_badge()
        test_readme_quantization()

        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print()

        print_example_readme()

        exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
