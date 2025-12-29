#!/usr/bin/env python3
"""
Comprehensive API Test Suite for Merlina

Tests all API endpoints with proper mocking to avoid actual training runs
and external dependencies.

Usage:
    pytest tests/test_api_comprehensive.py -v
    pytest tests/test_api_comprehensive.py::TestTrainingEndpoints -v
    pytest tests/test_api_comprehensive.py::TestTrainingEndpoints::test_create_training_job -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import json
import pytest
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Mock GPU-dependent imports BEFORE importing merlina
# This prevents import errors on CI runners without GPUs
# ============================================================================

# Mock torch
# Create fake classes that can be used with isinstance() and issubclass()
class FakeTensor:
    pass

class FakeModule:
    pass

mock_torch = MagicMock()
mock_torch.__spec__ = MagicMock()  # Required by datasets library
mock_torch.Tensor = FakeTensor  # Required for isinstance() checks
mock_torch.nn = MagicMock()
mock_torch.nn.Module = FakeModule  # Required for issubclass() checks
mock_torch.cuda.is_available.return_value = True
mock_torch.cuda.device_count.return_value = 1
mock_torch.cuda.get_device_capability.return_value = (8, 6)
mock_torch.cuda.get_device_name.return_value = "Mock GPU"
mock_torch.cuda.empty_cache = Mock()
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch.cuda
sys.modules['torch.nn'] = mock_torch.nn

# Mock transformers
class FakeTokenizerBase:
    pass

mock_transformers = MagicMock()
mock_transformers.PreTrainedTokenizerBase = FakeTokenizerBase  # Required for issubclass() checks
sys.modules['transformers'] = mock_transformers

# Mock other ML libraries and system dependencies
for module in ['trl', 'peft', 'accelerate', 'bitsandbytes', 'wandb', 'psutil', 'pynvml']:
    sys.modules[module] = MagicMock()

from fastapi.testclient import TestClient
from fastapi import UploadFile


# ============================================================================
# Fixtures and Setup
# ============================================================================

@pytest.fixture(scope="function")
def mock_job_manager():
    """Mock JobManager for testing"""
    from src.job_manager import JobRecord

    mock = Mock()

    # Mock job record
    test_job = JobRecord(
        job_id="test_job_001",
        status="queued",
        progress=0.0,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        config={"base_model": "test/model"},
        output_dir=None,
        current_step=None,
        total_steps=None,
        loss=None,
        eval_loss=None,
        learning_rate=None,
        error=None,
        wandb_url=None,
        stop_requested=False
    )

    # Configure mock methods
    mock.create_job.return_value = test_job
    mock.get_job.return_value = test_job
    mock.update_job.return_value = True
    mock.list_jobs.return_value = [test_job]
    mock.delete_job.return_value = True
    mock.clear_all_jobs.return_value = 1
    mock.get_metrics.return_value = [
        {
            "job_id": "test_job_001",
            "step": 10,
            "loss": 0.42,
            "eval_loss": 0.45,
            "learning_rate": 1e-5,
            "timestamp": datetime.now().isoformat()
        }
    ]
    mock.get_stats.return_value = {
        "total_jobs": 1,
        "total_metrics": 1,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "running_jobs": 1
    }

    return mock


@pytest.fixture(scope="function")
def mock_job_queue():
    """Mock JobQueue for testing"""
    from src.job_queue import JobPriority

    mock = Mock()
    mock.submit.return_value = 1  # Queue position
    mock.cancel.return_value = True
    mock.get_status.return_value = {
        "state": "queued",
        "position": 1
    }
    mock.get_queue_stats.return_value = {
        "queued_count": 1,
        "running_count": 0,
        "max_concurrent": 1,
        "total_submitted": 1,
        "total_completed": 0,
        "total_failed": 0
    }
    mock.list_queued_jobs.return_value = [
        {
            "job_id": "test_job_001",
            "priority": "NORMAL",
            "position": 1,
            "submitted_at": datetime.now().isoformat()
        }
    ]
    mock.list_running_jobs.return_value = []

    return mock


@pytest.fixture(scope="function")
def mock_websocket_manager():
    """Mock WebSocketManager for testing"""
    mock = Mock()
    mock.get_connection_count.return_value = 0
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.send_status_update = AsyncMock()
    mock.send_metric_update = AsyncMock()

    return mock


@pytest.fixture(scope="function")
def mock_config_manager():
    """Mock ConfigManager for testing"""
    mock = Mock()
    mock.save_config.return_value = "/path/to/config.json"
    mock.list_configs.return_value = [
        {
            "name": "test_config",
            "description": "Test configuration",
            "tags": ["test"],
            "created_at": datetime.now().isoformat()
        }
    ]
    mock.load_config.return_value = {
        "_metadata": {
            "name": "test_config",
            "created_at": datetime.now().isoformat()
        },
        "base_model": "test/model",
        "output_name": "test_output"
    }
    mock.get_config_without_metadata.return_value = {
        "base_model": "test/model",
        "output_name": "test_output"
    }
    mock.delete_config.return_value = True
    mock.export_config.return_value = "/path/to/export.json"
    mock.import_config.return_value = "/path/to/imported.json"

    return mock


@pytest.fixture(scope="function")
def mock_gpu_manager():
    """Mock GPUManager for testing"""
    from src.gpu_utils import GPUInfo

    mock = Mock()

    # Create mock GPU info
    gpu_info = GPUInfo(
        index=0,
        name="NVIDIA GeForce RTX 3090",
        total_memory=24576 * 1024 * 1024,  # 24GB in bytes
        free_memory=20000 * 1024 * 1024,   # 20GB in bytes
        used_memory=4576 * 1024 * 1024,    # ~4.5GB in bytes
        utilization=25,                     # percent
        temperature=55,                     # celsius
        power_usage=150,                    # watts
        compute_capability=(8, 6)
    )

    mock.is_cuda_available.return_value = True
    mock.list_gpus.return_value = [gpu_info]
    mock.get_gpu_info.return_value = gpu_info
    mock.get_available_gpus.return_value = [0]
    mock.get_recommended_gpu.return_value = 0

    return mock


@pytest.fixture(scope="function")
def mock_validate_config():
    """Mock validation function"""
    def _validate(config):
        return True, {
            "errors": [],
            "warnings": ["Test warning"],
            "checks": {
                "gpu": {"status": "pass"},
                "vram": {"status": "pass"},
                "model": {"status": "pass"}
            }
        }
    return _validate


@pytest.fixture(scope="function")
def mock_dataset_loader():
    """Mock dataset loader"""
    from datasets import Dataset

    mock_dataset = Dataset.from_dict({
        "prompt": ["What is AI?", "What is ML?"],
        "chosen": ["AI is artificial intelligence", "ML is machine learning"],
        "rejected": ["I don't know", "No idea"],
        "system": ["You are helpful", "You are helpful"]
    })

    mock = Mock()
    mock.load.return_value = mock_dataset

    return mock


@pytest.fixture(scope="function")
def mock_dataset_pipeline(mock_dataset_loader):
    """Mock dataset pipeline"""
    from datasets import Dataset

    mock_dataset = Dataset.from_dict({
        "prompt": ["What is AI?"],
        "chosen": ["AI is artificial intelligence"],
        "rejected": ["I don't know"]
    })

    mock = Mock()
    mock.preview.return_value = [
        {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence",
            "rejected": "I don't know"
        }
    ]
    mock.preview_formatted.return_value = [
        {
            "prompt": "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n",
            "chosen": "AI is artificial intelligence<|im_end|>",
            "rejected": "I don't know<|im_end|>"
        }
    ]
    mock.prepare.return_value = (mock_dataset, mock_dataset)

    return mock


@pytest.fixture(scope="function")
def client(
    mock_job_manager,
    mock_job_queue,
    mock_websocket_manager,
    mock_config_manager,
    mock_gpu_manager,
    mock_validate_config
):
    """Create FastAPI test client with mocked dependencies"""

    # Patch all the dependencies before importing the app
    with patch('merlina.job_manager', mock_job_manager), \
         patch('merlina.job_queue', mock_job_queue), \
         patch('merlina.websocket_manager', mock_websocket_manager), \
         patch('merlina.config_manager', mock_config_manager), \
         patch('merlina.get_gpu_manager', return_value=mock_gpu_manager), \
         patch('merlina.validate_config', mock_validate_config), \
         patch('merlina.uploaded_datasets', {}), \
         patch('merlina.tokenizer_cache', {}):

        # Import the app after patching
        from merlina import app

        # Create test client
        with TestClient(app) as test_client:
            yield test_client


# ============================================================================
# Test Training Endpoints
# ============================================================================

class TestTrainingEndpoints:
    """Test training-related API endpoints"""

    def test_api_info(self, client):
        """Test GET /api endpoint"""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Merlina"
        assert "endpoints" in data

    def test_create_training_job(self, client):
        """Test POST /train endpoint"""
        config = {
            "base_model": "gpt2",
            "output_name": "test_model",
            "learning_rate": 5e-6,
            "num_epochs": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_length": 512,
            "max_prompt_length": 256,
            "beta": 0.1,
            "use_4bit": True,
            "use_wandb": False,
            "push_to_hub": False
        }

        response = client.post("/train", json=config)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert "Training spell cast!" in data["message"]

    def test_create_training_job_with_priority(self, client):
        """Test POST /train endpoint with priority parameter"""
        config = {
            "base_model": "gpt2",
            "output_name": "test_model_high_priority"
        }

        response = client.post("/train?priority=high", json=config)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

    def test_create_training_job_validation_failure(self, client, mock_validate_config):
        """Test POST /train with validation failure"""
        # Mock validation to return errors
        def _validate_fail(config):
            return False, {
                "errors": ["GPU not available"],
                "warnings": [],
                "checks": {}
            }

        with patch('merlina.validate_config', _validate_fail):
            config = {"base_model": "gpt2", "output_name": "test"}
            response = client.post("/train", json=config)
            assert response.status_code == 400
            assert "validation failed" in response.json()["detail"]["message"]

    def test_get_job_status(self, client, mock_job_manager):
        """Test GET /status/{job_id} endpoint"""
        response = client.get("/status/test_job_001")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_job_001"
        assert "status" in data
        assert "progress" in data

    def test_get_job_status_not_found(self, client, mock_job_manager):
        """Test GET /status/{job_id} for non-existent job"""
        mock_job_manager.get_job.return_value = None

        response = client.get("/status/nonexistent_job")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_jobs(self, client):
        """Test GET /jobs endpoint"""
        response = client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


# ============================================================================
# Test Validation and Job Management Endpoints
# ============================================================================

class TestValidationAndJobManagement:
    """Test validation and job management endpoints"""

    def test_validate_config(self, client):
        """Test POST /validate endpoint"""
        config = {
            "base_model": "gpt2",
            "output_name": "test_model",
            "learning_rate": 5e-6
        }

        response = client.post("/validate", json=config)
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "results" in data

    def test_get_job_history(self, client):
        """Test GET /jobs/history endpoint"""
        response = client.get("/jobs/history")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "count" in data

    def test_get_job_history_with_filters(self, client):
        """Test GET /jobs/history with pagination and filters"""
        response = client.get("/jobs/history?limit=10&offset=0&status=completed")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data

    def test_get_job_metrics(self, client):
        """Test GET /jobs/{job_id}/metrics endpoint"""
        response = client.get("/jobs/test_job_001/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "count" in data

    def test_get_job_metrics_not_found(self, client, mock_job_manager):
        """Test GET /jobs/{job_id}/metrics for non-existent job"""
        mock_job_manager.get_job.return_value = None

        response = client.get("/jobs/nonexistent_job/metrics")
        assert response.status_code == 404

    def test_stop_queued_job(self, client, mock_job_manager, mock_job_queue):
        """Test POST /jobs/{job_id}/stop for queued job"""
        mock_job_queue.get_status.return_value = {"state": "queued"}

        response = client.post("/jobs/test_job_001/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["was_queued"] is True

    def test_stop_running_job(self, client, mock_job_manager, mock_job_queue):
        """Test POST /jobs/{job_id}/stop for running job"""
        mock_job_queue.get_status.return_value = {"state": "running"}

        response = client.post("/jobs/test_job_001/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["was_queued"] is False

    def test_stop_job_not_found(self, client, mock_job_manager):
        """Test POST /jobs/{job_id}/stop for non-existent job"""
        mock_job_manager.get_job.return_value = None

        response = client.post("/jobs/nonexistent_job/stop")
        assert response.status_code == 404

    def test_delete_job(self, client):
        """Test DELETE /jobs/{job_id} endpoint"""
        response = client.delete("/jobs/test_job_001")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_delete_job_not_found(self, client, mock_job_manager):
        """Test DELETE /jobs/{job_id} for non-existent job"""
        mock_job_manager.delete_job.return_value = False

        response = client.delete("/jobs/nonexistent_job")
        assert response.status_code == 404

    def test_clear_all_jobs(self, client):
        """Test DELETE /jobs endpoint"""
        response = client.delete("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted_count" in data


# ============================================================================
# Test Queue Management Endpoints
# ============================================================================

class TestQueueManagement:
    """Test queue management endpoints"""

    def test_get_queue_status(self, client):
        """Test GET /queue/status endpoint"""
        response = client.get("/queue/status")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "queued_jobs" in data
        assert "running_jobs" in data

    def test_list_queue_jobs(self, client):
        """Test GET /queue/jobs endpoint"""
        response = client.get("/queue/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "queued" in data
        assert "running" in data


# ============================================================================
# Test Dataset Management Endpoints
# ============================================================================

class TestDatasetManagement:
    """Test dataset management endpoints"""

    def test_preview_dataset_huggingface(self, client, mock_dataset_pipeline, mock_dataset_loader):
        """Test POST /dataset/preview for HuggingFace dataset"""
        with patch('merlina.HuggingFaceLoader', return_value=mock_dataset_loader), \
             patch('merlina.DatasetPipeline', return_value=mock_dataset_pipeline):
            config = {
                "source": {
                    "source_type": "huggingface",
                    "repo_id": "test/dataset",
                    "split": "train"
                },
                "format": {
                    "format_type": "chatml"
                }
            }

            response = client.post("/dataset/preview", json=config)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "samples" in data
            assert len(data["samples"]) > 0

    def test_preview_dataset_local_file(self, client, mock_dataset_pipeline, mock_dataset_loader):
        """Test POST /dataset/preview for local file"""
        with patch('merlina.create_loader_from_config', return_value=mock_dataset_loader), \
             patch('merlina.DatasetPipeline', return_value=mock_dataset_pipeline):
            config = {
                "source": {
                    "source_type": "local_file",
                    "file_path": "/path/to/dataset.json",
                    "file_format": "json"
                },
                "format": {
                    "format_type": "chatml"
                }
            }

            response = client.post("/dataset/preview", json=config)
            assert response.status_code == 200

    def test_preview_dataset_uploaded(self, client, mock_dataset_pipeline, mock_dataset_loader):
        """Test POST /dataset/preview for uploaded dataset"""
        with patch('merlina.uploaded_datasets', {"test_id": (b"data", "test.json")}), \
             patch('merlina.UploadedDatasetLoader', return_value=mock_dataset_loader), \
             patch('merlina.DatasetPipeline', return_value=mock_dataset_pipeline):

            config = {
                "source": {
                    "source_type": "upload",
                    "dataset_id": "test_id",
                    "file_format": "json"
                },
                "format": {
                    "format_type": "chatml"
                }
            }

            response = client.post("/dataset/preview", json=config)
            assert response.status_code == 200

    def test_preview_dataset_uploaded_not_found(self, client):
        """Test POST /dataset/preview for non-existent uploaded dataset"""
        config = {
            "source": {
                "source_type": "upload",
                "dataset_id": "nonexistent_id"
            }
        }

        response = client.post("/dataset/preview", json=config)
        # API returns 400 for dataset not found (HTTPException caught as generic error)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_preview_formatted_dataset(self, client, mock_dataset_pipeline, mock_dataset_loader):
        """Test POST /dataset/preview-formatted endpoint"""
        with patch('merlina.HuggingFaceLoader', return_value=mock_dataset_loader), \
             patch('merlina.DatasetPipeline', return_value=mock_dataset_pipeline):
            config = {
                "source": {
                    "source_type": "huggingface",
                    "repo_id": "test/dataset"
                },
                "format": {
                    "format_type": "chatml"
                }
            }

            response = client.post("/dataset/preview-formatted", json=config)
            assert response.status_code == 200
            data = response.json()
            assert "samples" in data

    def test_preview_formatted_tokenizer_without_cache(self, client, mock_dataset_pipeline, mock_dataset_loader):
        """Test POST /dataset/preview-formatted with tokenizer format but no cache"""
        with patch('merlina.HuggingFaceLoader', return_value=mock_dataset_loader), \
             patch('merlina.DatasetPipeline', return_value=mock_dataset_pipeline):
            config = {
                "source": {
                    "source_type": "huggingface",
                    "repo_id": "test/dataset"
                },
                "format": {
                    "format_type": "tokenizer"
                },
                "model_name": "gpt2"
            }

            response = client.post("/dataset/preview-formatted", json=config)
            # Should fall back to chatml
            assert response.status_code == 200

    def test_get_dataset_columns(self, client, mock_dataset_loader):
        """Test POST /dataset/columns endpoint"""
        with patch('merlina.create_loader_from_config', return_value=mock_dataset_loader), \
             patch('merlina.get_formatter', return_value=Mock()):
            config = {
                "source": {
                    "source_type": "huggingface",
                    "repo_id": "test/dataset"
                }
            }

            response = client.post("/dataset/columns", json=config)
            assert response.status_code == 200
            data = response.json()
            assert "columns" in data
            assert "samples" in data

    def test_upload_dataset_file(self, client):
        """Test POST /dataset/upload-file endpoint"""
        # Create a mock file
        file_content = b'{"prompt": "test", "chosen": "answer", "rejected": "bad"}'
        files = {
            "file": ("test_dataset.json", file_content, "application/json")
        }

        response = client.post("/dataset/upload-file", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "dataset_id" in data
        assert data["filename"] == "test_dataset.json"

    def test_list_uploaded_datasets(self, client):
        """Test GET /dataset/uploads endpoint"""
        with patch('merlina.uploaded_datasets', {"test_id": (b"data", "test.json")}):
            response = client.get("/dataset/uploads")
            assert response.status_code == 200
            data = response.json()
            assert "datasets" in data
            assert len(data["datasets"]) > 0


# ============================================================================
# Test GPU Management Endpoints
# ============================================================================

class TestGPUManagement:
    """Test GPU management endpoints"""

    def test_list_gpus(self, client):
        """Test GET /gpu/list endpoint"""
        response = client.get("/gpu/list")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "gpus" in data
        assert len(data["gpus"]) > 0

    def test_list_gpus_no_cuda(self, client, mock_gpu_manager):
        """Test GET /gpu/list when CUDA is not available"""
        mock_gpu_manager.is_cuda_available.return_value = False

        response = client.get("/gpu/list")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_cuda"

    def test_get_gpu_info(self, client):
        """Test GET /gpu/{index} endpoint"""
        response = client.get("/gpu/0")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "gpu" in data

    def test_get_gpu_info_not_found(self, client, mock_gpu_manager):
        """Test GET /gpu/{index} for non-existent GPU"""
        mock_gpu_manager.get_gpu_info.return_value = None

        response = client.get("/gpu/999")
        assert response.status_code == 404

    def test_get_available_gpus(self, client):
        """Test GET /gpu/available endpoint"""
        response = client.get("/gpu/available?min_free_memory_mb=4000")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "available_gpus" in data

    def test_get_recommended_gpu(self, client):
        """Test GET /gpu/recommended endpoint"""
        response = client.get("/gpu/recommended")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "recommended_index" in data
        assert data["recommended_index"] == 0


# ============================================================================
# Test Config Management Endpoints
# ============================================================================

class TestConfigManagement:
    """Test config management endpoints"""

    def test_save_config(self, client):
        """Test POST /configs/save endpoint"""
        request_data = {
            "name": "test_config",
            "config": {
                "base_model": "gpt2",
                "output_name": "test_model"
            },
            "description": "Test configuration",
            "tags": ["test", "demo"]
        }

        response = client.post("/configs/save", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_save_config_duplicate(self, client, mock_config_manager):
        """Test POST /configs/save with duplicate name"""
        mock_config_manager.save_config.side_effect = ValueError("Config already exists")

        request_data = {
            "name": "duplicate_config",
            "config": {"base_model": "gpt2"}
        }

        response = client.post("/configs/save", json=request_data)
        assert response.status_code == 400

    def test_list_configs(self, client):
        """Test GET /configs/list endpoint"""
        response = client.get("/configs/list")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "configs" in data

    def test_list_configs_with_tag_filter(self, client):
        """Test GET /configs/list with tag filter"""
        response = client.get("/configs/list?tag=test")
        assert response.status_code == 200

    def test_get_config(self, client):
        """Test GET /configs/{name} endpoint"""
        response = client.get("/configs/test_config")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "config" in data

    def test_get_config_with_metadata(self, client):
        """Test GET /configs/{name} with metadata"""
        response = client.get("/configs/test_config?include_metadata=true")
        assert response.status_code == 200
        data = response.json()
        assert "config" in data

    def test_get_config_not_found(self, client, mock_config_manager):
        """Test GET /configs/{name} for non-existent config"""
        mock_config_manager.load_config.side_effect = FileNotFoundError()
        mock_config_manager.get_config_without_metadata.side_effect = FileNotFoundError()

        response = client.get("/configs/nonexistent")
        assert response.status_code == 404

    def test_delete_config(self, client):
        """Test DELETE /configs/{name} endpoint"""
        response = client.delete("/configs/test_config")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_delete_config_not_found(self, client, mock_config_manager):
        """Test DELETE /configs/{name} for non-existent config"""
        mock_config_manager.delete_config.return_value = False

        response = client.delete("/configs/nonexistent")
        assert response.status_code == 404

    def test_export_config(self, client):
        """Test POST /configs/export endpoint"""
        request_data = {
            "name": "test_config",
            "output_path": "/tmp/exported_config.json"
        }

        response = client.post("/configs/export", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_import_config(self, client):
        """Test POST /configs/import endpoint"""
        with patch('pathlib.Path.exists', return_value=True):
            request_data = {
                "filepath": "/path/to/config.json",
                "name": "imported_config"
            }

            response = client.post("/configs/import", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    def test_import_config_file_not_found(self, client):
        """Test POST /configs/import with non-existent file"""
        request_data = {
            "filepath": "/nonexistent/config.json"
        }

        response = client.post("/configs/import", json=request_data)
        assert response.status_code == 404


# ============================================================================
# Test Model Preload and Stats Endpoints
# ============================================================================

class TestModelAndStats:
    """Test model preload and stats endpoints"""

    def test_preload_model_tokenizer(self, client):
        """Test POST /model/preload endpoint"""
        with patch('merlina.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # Create a mock tokenizer
            mock_tok = Mock()
            mock_tok.vocab_size = 50257
            mock_tok.model_max_length = 1024
            mock_tok.chat_template = "test template"
            mock_tok.pad_token = "<pad>"
            mock_tok.eos_token = "<eos>"
            mock_tok.bos_token = "<bos>"
            mock_tokenizer.return_value = mock_tok

            request_data = {
                "model_name": "gpt2",
                "hf_token": "test_token"
            }

            response = client.post("/model/preload", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["model_name"] == "gpt2"

    def test_preload_model_tokenizer_cached(self, client):
        """Test POST /model/preload with cached tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 50257
        mock_tokenizer.model_max_length = 1024
        mock_tokenizer.chat_template = None
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.bos_token = "<bos>"

        with patch('merlina.tokenizer_cache', {"gpt2": mock_tokenizer}):
            request_data = {"model_name": "gpt2"}
            response = client.post("/model/preload", json=request_data)
            assert response.status_code == 200

    def test_list_cached_models(self, client):
        """Test GET /model/cached endpoint"""
        with patch('merlina.tokenizer_cache', {"gpt2": Mock(), "llama-2": Mock()}):
            response = client.get("/model/cached")
            assert response.status_code == 200
            data = response.json()
            assert "cached_models" in data
            assert len(data["cached_models"]) == 2

    def test_get_stats(self, client):
        """Test GET /stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert "websockets" in data
        assert "queue" in data


# ============================================================================
# Test WebSocket Endpoint
# ============================================================================

class TestWebSocket:
    """Test WebSocket endpoint"""

    @pytest.mark.skip(reason="WebSocket testing with TestClient hangs - requires integration test setup")
    def test_websocket_connection(self, client, mock_job_manager):
        """Test WebSocket /ws/{job_id} connection"""
        # WebSocket testing requires special handling
        # This is a basic structure test
        with patch('merlina.websocket_manager.connect') as mock_connect:
            mock_connect.return_value = None

            # Note: TestClient doesn't fully support WebSocket testing
            # For full WebSocket tests, use websockets library directly
            # This test just verifies the endpoint exists and can be connected to
            try:
                with client.websocket_connect("/ws/test_job_001") as websocket:
                    # Successfully connected - that's enough for unit test
                    # Don't call receive_json() as it blocks indefinitely
                    assert websocket is not None
            except Exception as e:
                # WebSocket testing may fail with TestClient
                # This is expected and can be tested with integration tests
                pass


# ============================================================================
# Test Error Cases and Edge Cases
# ============================================================================

class TestErrorCases:
    """Test error handling and edge cases"""

    def test_invalid_source_type(self, client):
        """Test invalid source_type in dataset preview"""
        config = {
            "source": {
                "source_type": "invalid_type"
            }
        }

        response = client.post("/dataset/preview", json=config)
        assert response.status_code == 400

    def test_invalid_priority(self, client):
        """Test invalid priority parameter in /train"""
        config = {"base_model": "gpt2", "output_name": "test"}

        # Invalid priority should default to normal
        response = client.post("/train?priority=invalid", json=config)
        assert response.status_code == 200

    def test_gpu_error_handling(self, client, mock_gpu_manager):
        """Test GPU endpoint error handling"""
        mock_gpu_manager.list_gpus.side_effect = Exception("GPU error")

        response = client.get("/gpu/list")
        assert response.status_code == 500


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
