"""
Pytest configuration and shared fixtures for Merlina tests

This file is automatically discovered by pytest and provides
shared fixtures and configuration for all test modules.
"""

import sys
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Mock GPU-dependent imports for CI environments without GPUs
# ============================================================================

# Mock torch and CUDA
if 'torch' not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False  # Default to no CUDA for safety
    mock_torch.cuda.device_count.return_value = 0
    mock_torch.cuda.empty_cache = Mock()
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.float16 = "float16"
    sys.modules['torch'] = mock_torch

# Mock other ML libraries if not already imported
for module in ['transformers', 'trl', 'peft', 'accelerate', 'bitsandbytes', 'wandb']:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()


# ============================================================================
# Session-level fixtures (run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_dataset_path(test_data_dir):
    """Path to sample test dataset"""
    return test_data_dir / "test_dataset.json"


# ============================================================================
# Function-level fixtures (run for each test function)
# ============================================================================

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test"""
    # Store original env vars
    original_env = os.environ.copy()

    yield

    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database path for testing"""
    db_path = tmp_path / "test_jobs.db"
    return str(db_path)


# ============================================================================
# Mock fixtures
# ============================================================================

@pytest.fixture
def mock_torch():
    """Mock torch module to avoid GPU requirements in tests"""
    mock = MagicMock()
    mock.cuda.is_available.return_value = True
    mock.cuda.device_count.return_value = 1
    mock.cuda.get_device_capability.return_value = (8, 6)
    mock.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3090"
    mock.cuda.empty_cache = Mock()
    mock.bfloat16 = "bfloat16"
    mock.float16 = "float16"

    return mock


@pytest.fixture
def mock_transformers():
    """Mock transformers module"""
    mock = MagicMock()

    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.vocab_size = 50257
    mock_tokenizer.model_max_length = 1024
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.bos_token = "<bos>"
    mock_tokenizer.chat_template = None

    mock.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock.AutoModelForCausalLM.from_pretrained.return_value = Mock()

    return mock


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing"""
    return {
        "base_model": "gpt2",
        "output_name": "test_model",
        "use_lora": True,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["c_attn"],
        "learning_rate": 5e-6,
        "num_epochs": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_length": 512,
        "max_prompt_length": 256,
        "beta": 0.1,
        "dataset": {
            "source": {
                "source_type": "huggingface",
                "repo_id": "test/dataset",
                "split": "train"
            },
            "format": {
                "format_type": "chatml"
            },
            "test_size": 0.01
        },
        "warmup_ratio": 0.05,
        "eval_steps": 0.2,
        "use_4bit": True,
        "use_wandb": False,
        "push_to_hub": False
    }


# ============================================================================
# Pytest hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "websocket: WebSocket tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add 'api' marker to all tests in test_api_*.py files
        if "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.api)

        # Add 'unit' marker to all tests in test_*_unit.py files
        if "_unit.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add 'integration' marker to all tests in test_*_integration.py files
        if "_integration.py" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
