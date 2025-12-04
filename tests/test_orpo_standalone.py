"""
Test suite for standalone ORPO implementation

Run with: python tests/test_orpo_standalone.py
Or with pytest: pytest tests/test_orpo_standalone.py -v
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import torch - skip entire module if not available
try:
    import torch
    from src.orpo_standalone import (
        ORPOConfig,
        MerlinaORPOTrainer,
        pad_to_length,
        selective_log_softmax,
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason=f"PyTorch not available: {e}")


class TestUtilityFunctions:
    """Test utility functions"""

    def test_pad_to_length(self):
        """Test pad_to_length function"""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # Pad to longer length
        padded = pad_to_length(tensor, length=5, pad_value=0, dim=-1)
        assert padded.shape == (2, 5)
        assert padded[0, 3].item() == 0
        assert padded[0, 4].item() == 0

        # No padding needed
        same = pad_to_length(tensor, length=3, pad_value=0, dim=-1)
        assert torch.equal(same, tensor)

        # Already longer
        longer = pad_to_length(tensor, length=2, pad_value=0, dim=-1)
        assert torch.equal(longer, tensor)

    def test_selective_log_softmax(self):
        """Test selective_log_softmax function"""
        # Create simple logits (batch=2, seq_len=3, vocab=10)
        torch.manual_seed(42)
        logits = torch.randn(2, 3, 10)
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # Compute selective log softmax
        result = selective_log_softmax(logits, labels)

        # Check shape
        assert result.shape == (2, 3)

        # Verify it matches full log_softmax + gather
        full_log_probs = torch.log_softmax(logits, dim=-1)
        expected = torch.gather(full_log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        assert torch.allclose(result, expected, atol=1e-6)


class TestORPOConfig:
    """Test ORPO configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ORPOConfig(output_dir="./test")

        assert config.max_length == 2048
        assert config.max_prompt_length == 1024
        assert config.beta == 0.1
        assert config.disable_dropout == True
        assert config.label_pad_token_id == -100

    def test_max_completion_length_calculation(self):
        """Test automatic max_completion_length calculation"""
        config = ORPOConfig(
            output_dir="./test",
            max_length=1000,
            max_prompt_length=300
        )

        assert config.max_completion_length == 700

    def test_validation(self):
        """Test configuration validation"""
        # Should raise error if max_prompt_length >= max_length
        with pytest.raises(ValueError):
            config = ORPOConfig(
                output_dir="./test",
                max_length=100,
                max_prompt_length=100
            )


class TestMerlinaORPOTrainer:
    """Test ORPO Trainer"""

    def test_get_batch_logps(self):
        """Test log probability computation"""
        torch.manual_seed(42)

        # Create mock logits and labels
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute log probs (average)
        logps_avg = MerlinaORPOTrainer.get_batch_logps(
            logits,
            labels,
            average_log_prob=True,
            label_pad_token_id=-100,
            is_encoder_decoder=False
        )

        # Check shape
        assert logps_avg.shape == (batch_size,)

        # Compute log probs (sum)
        logps_sum = MerlinaORPOTrainer.get_batch_logps(
            logits,
            labels,
            average_log_prob=False,
            label_pad_token_id=-100,
            is_encoder_decoder=False
        )

        # Check shape
        assert logps_sum.shape == (batch_size,)

        # Average should be less than or equal to sum in magnitude
        assert logps_avg.abs().mean() <= logps_sum.abs().mean()

    def test_get_batch_logps_with_padding(self):
        """Test log probability with padding tokens"""
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 5
        vocab_size = 100
        pad_token_id = -100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Add padding to second sequence
        labels[1, 3:] = pad_token_id

        # Compute log probs
        logps = MerlinaORPOTrainer.get_batch_logps(
            logits,
            labels,
            average_log_prob=False,
            label_pad_token_id=pad_token_id,
            is_encoder_decoder=False
        )

        # Second sequence should have contribution only from non-padded tokens
        assert logps.shape == (batch_size,)

        # Log prob should be finite
        assert torch.isfinite(logps).all()

    def test_concatenated_inputs(self):
        """Test concatenation of chosen and rejected inputs"""
        batch_size = 4
        chosen_len = 10
        rejected_len = 12

        # Create mock batch
        batch = {
            "chosen_input_ids": torch.randint(0, 100, (batch_size, chosen_len)),
            "chosen_attention_mask": torch.ones(batch_size, chosen_len),
            "chosen_labels": torch.randint(0, 100, (batch_size, chosen_len)),
            "rejected_input_ids": torch.randint(0, 100, (batch_size, rejected_len)),
            "rejected_attention_mask": torch.ones(batch_size, rejected_len),
            "rejected_labels": torch.randint(0, 100, (batch_size, rejected_len)),
        }

        # Concatenate
        concatenated = MerlinaORPOTrainer.concatenated_inputs(
            batch,
            is_encoder_decoder=False,
            label_pad_token_id=-100,
            padding_value=0,
            device=None
        )

        # Check shapes
        max_len = max(chosen_len, rejected_len)
        assert concatenated["concatenated_input_ids"].shape == (batch_size * 2, max_len)
        assert concatenated["concatenated_attention_mask"].shape == (batch_size * 2, max_len)
        assert concatenated["concatenated_labels"].shape == (batch_size * 2, max_len)

        # First half should be chosen, second half rejected
        # Check that chosen part is padded correctly
        assert (concatenated["concatenated_input_ids"][:batch_size, :chosen_len] ==
                batch["chosen_input_ids"]).all()

    def test_odds_ratio_loss_shape(self):
        """Test odds ratio loss computation shape"""
        from src.orpo_standalone import MerlinaORPOTrainer, ORPOConfig

        # Create mock trainer
        config = ORPOConfig(output_dir="./test", beta=0.1)

        # We need a model for initialization, but we'll just test the loss function
        batch_size = 4
        chosen_logps = torch.randn(batch_size) * -2  # Negative log probs
        rejected_logps = torch.randn(batch_size) * -3  # Worse log probs

        # Create a minimal trainer instance (without model)
        # We'll manually set beta for testing
        class MockTrainer:
            def __init__(self):
                self.beta = 0.1

        mock = MockTrainer()

        # Compute loss using the static-like method
        losses, chosen_rewards, rejected_rewards, ratio_mean, log_odds_mean = (
            MerlinaORPOTrainer.odds_ratio_loss(
                mock,
                chosen_logps,
                rejected_logps
            )
        )

        # Check shapes
        assert losses.shape == (batch_size,)
        assert chosen_rewards.shape == (batch_size,)
        assert rejected_rewards.shape == (batch_size,)
        assert ratio_mean.shape == ()  # Scalar
        assert log_odds_mean.shape == ()  # Scalar

        # Check that values are finite
        assert torch.isfinite(losses).all()
        assert torch.isfinite(chosen_rewards).all()
        assert torch.isfinite(rejected_rewards).all()

    def test_odds_ratio_loss_preference(self):
        """Test that ORPO loss favors chosen over rejected"""
        from src.orpo_standalone import MerlinaORPOTrainer

        class MockTrainer:
            def __init__(self):
                self.beta = 0.1

        mock = MockTrainer()

        # Chosen has higher probability (less negative log prob)
        chosen_logps = torch.tensor([-1.0, -1.5, -2.0])
        rejected_logps = torch.tensor([-3.0, -3.5, -4.0])

        losses, chosen_rewards, rejected_rewards, _, _ = (
            MerlinaORPOTrainer.odds_ratio_loss(
                mock,
                chosen_logps,
                rejected_logps
            )
        )

        # Chosen rewards should be higher (less negative)
        assert (chosen_rewards > rejected_rewards).all()

        # Losses should be negative (we want to maximize ratio)
        assert (losses < 0).all()


def test_import():
    """Test that all exports work"""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from src.orpo_standalone import (
        ORPOConfig,
        MerlinaORPOTrainer,
        create_orpo_trainer,
        pad_to_length,
        selective_log_softmax,
    )

    print("✅ All imports successful")


def test_config_creation():
    """Test creating a basic config"""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    config = ORPOConfig(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        beta=0.1,
        max_length=512,
    )

    assert config.output_dir == "./test_output"
    assert config.beta == 0.1
    assert config.max_length == 512

    print("✅ Config creation successful")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - skipping ORPO standalone tests")
        print("Install PyTorch to run these tests: pip install torch")
        sys.exit(1)

    print("Running standalone ORPO tests...")
    print("=" * 50)

    # Run basic tests
    test_import()
    test_config_creation()

    # Run test classes
    print("\n📋 Testing utility functions...")
    utils = TestUtilityFunctions()
    utils.test_pad_to_length()
    utils.test_selective_log_softmax()
    print("✅ Utility functions passed")

    print("\n📋 Testing ORPO config...")
    config_tests = TestORPOConfig()
    config_tests.test_default_config()
    config_tests.test_max_completion_length_calculation()
    print("✅ Config tests passed")

    print("\n📋 Testing ORPO trainer...")
    trainer_tests = TestMerlinaORPOTrainer()
    trainer_tests.test_get_batch_logps()
    trainer_tests.test_get_batch_logps_with_padding()
    trainer_tests.test_concatenated_inputs()
    trainer_tests.test_odds_ratio_loss_shape()
    trainer_tests.test_odds_ratio_loss_preference()
    print("✅ Trainer tests passed")

    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
    print("\nNext steps:")
    print("1. Review docs/ORPO_MIGRATION_GUIDE.md")
    print("2. Update imports in src/training_runner.py")
    print("3. Run a test training job")
    print("4. Compare results with TRL baseline")
