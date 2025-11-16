# Merlina Test Suite

Comprehensive test suite for the Merlina API with full endpoint coverage, proper mocking, and isolation.

## 🚀 Quick Start

### Install Test Dependencies

```bash
# Install test requirements
pip install -r tests/requirements-test.txt

# Or install with main requirements
pip install -r requirements.txt -r tests/requirements-test.txt
```

### Run Tests

```bash
# Run all tests
pytest

# Run comprehensive API tests
pytest tests/test_api_comprehensive.py -v

# Run specific test class
pytest tests/test_api_comprehensive.py::TestTrainingEndpoints -v

# Run with coverage
make test-cov
```

## 📁 Test Structure

```
tests/
├── README.md                      # This file - comprehensive testing guide
├── TEST_SUMMARY.md                # Detailed test summary and coverage report
├── conftest.py                    # Pytest configuration and shared fixtures
├── requirements-test.txt          # Test dependencies
├── test_api_comprehensive.py      # ⭐ Complete API endpoint test suite (NEW)
├── test_v1.1_features.py          # v1.1 features tests
├── test_dataset_loaders.py        # Dataset loader tests
├── test_tokenizer_formatter.py    # Tokenizer formatting tests
├── test_pipeline.py                # Dataset pipeline tests
└── fixtures/                       # Test data files
    └── test_dataset.json           # Sample test dataset
```

## ⭐ New: Comprehensive API Test Suite

### Coverage: 100% of API Endpoints (29/29)

The new `test_api_comprehensive.py` provides complete coverage of all Merlina API endpoints:

#### 1. Training Endpoints (6 tests)
- ✅ POST /train - Create training job
- ✅ POST /train?priority=high - Create with priority
- ✅ GET /status/{job_id} - Get job status
- ✅ GET /jobs - List all jobs

#### 2. Job Management (9 tests)
- ✅ POST /validate - Validate configuration
- ✅ GET /jobs/history - Job history with pagination
- ✅ GET /jobs/{job_id}/metrics - Get metrics
- ✅ POST /jobs/{job_id}/stop - Stop/cancel job
- ✅ DELETE /jobs/{job_id} - Delete job
- ✅ DELETE /jobs - Clear all jobs

#### 3. Queue Management (2 tests)
- ✅ GET /queue/status - Queue statistics
- ✅ GET /queue/jobs - List queued jobs

#### 4. Dataset Management (8 tests)
- ✅ POST /dataset/preview - Preview datasets (HF, local, uploaded)
- ✅ POST /dataset/preview-formatted - Formatted preview
- ✅ POST /dataset/columns - Get dataset columns
- ✅ POST /dataset/upload-file - Upload dataset
- ✅ GET /dataset/uploads - List uploads

#### 5. GPU Management (6 tests)
- ✅ GET /gpu/list - List all GPUs
- ✅ GET /gpu/{index} - Get GPU info
- ✅ GET /gpu/available - Available GPUs
- ✅ GET /gpu/recommended - Recommended GPU

#### 6. Config Management (10 tests)
- ✅ POST /configs/save - Save configuration
- ✅ GET /configs/list - List configurations
- ✅ GET /configs/{name} - Load configuration
- ✅ DELETE /configs/{name} - Delete configuration
- ✅ POST /configs/export - Export configuration
- ✅ POST /configs/import - Import configuration

#### 7. Model & Stats (4 tests)
- ✅ POST /model/preload - Preload tokenizer
- ✅ GET /model/cached - List cached models
- ✅ GET /stats - Get statistics

#### 8. WebSocket (1 test)
- ✅ WebSocket /ws/{job_id} - Real-time updates

### Test Features

- **Comprehensive Mocking**: All external dependencies mocked (JobManager, JobQueue, GPU, etc.)
- **Fast Execution**: Pure unit tests, no I/O operations
- **Isolated**: No database writes, no actual training, no GPU required
- **Well Organized**: Tests grouped into logical classes
- **Error Handling**: Tests for success and error cases
- **CI/CD Ready**: pytest framework with coverage reporting

## 🧪 Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api_comprehensive.py

# Run specific test class
pytest tests/test_api_comprehensive.py::TestTrainingEndpoints

# Run specific test
pytest tests/test_api_comprehensive.py::TestTrainingEndpoints::test_create_training_job
```

### Using Makefile

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run in parallel (faster)
make test-parallel

# Run only API tests
make test-api

# Clean test artifacts
make clean-test
```

### Advanced Options

```bash
# Run with coverage
pytest --cov=src --cov=merlina --cov-report=html

# Run in parallel
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run only API tests
pytest -m api

# Generate HTML report
pytest --html=report.html
```

## 🔍 Legacy Test Coverage

### V1.1 Features (`test_v1.1_features.py`)
- ✅ Job Manager (SQLite persistence, CRUD operations)
- ✅ Metrics tracking (time-series storage)
- ✅ WebSocket Manager (connection management, broadcasts)
- ✅ Pre-flight Validation (8 validation checks)
- ✅ Module imports

### Dataset Loaders (`test_dataset_loaders.py`)
- ✅ Loading local JSON files
- ✅ File format detection
- ✅ Error handling

### Tokenizer Formatter (`test_tokenizer_formatter.py`)
- ✅ Formatting with models that have chat templates
- ✅ Fallback for models without chat templates
- ✅ Error handling

### Pipeline (`test_pipeline.py`)
- ✅ End-to-end dataset preparation
- ✅ Column mapping
- ✅ Train/test splitting

## 📊 Test Coverage Report

```bash
# Generate coverage report
pytest --cov=src --cov=merlina --cov-report=html --cov-report=term

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 📝 Writing New Tests

### Using the Comprehensive Test Suite

Add new tests to `test_api_comprehensive.py`:

```python
class TestYourEndpoints:
    """Test your new endpoints"""

    def test_your_endpoint(self, client, mock_dependency):
        """Test description"""
        # Arrange
        test_data = {"key": "value"}

        # Act
        response = client.post("/your-endpoint", json=test_data)

        # Assert
        assert response.status_code == 200
        assert response.json()["status"] == "success"
```

### Legacy Test Pattern

For standalone tests:

```python
import sys
sys.path.insert(0, '/path/to/merlina')

from your_module import YourClass

def test_your_feature():
    # Setup
    obj = YourClass()

    # Execute
    result = obj.method()

    # Assert
    assert result == expected
    print("✅ Test passed")

if __name__ == "__main__":
    test_your_feature()
```

## 🔗 More Information

- **Test Summary**: [TEST_SUMMARY.md](TEST_SUMMARY.md) - Detailed coverage report
- **Project Documentation**: [../CLAUDE.md](../CLAUDE.md) - Full project guide
- **API Documentation**: http://localhost:8000/api/docs (when server running)

## 🎯 Test Philosophy

1. **Comprehensive Coverage**: Test all endpoints and edge cases
2. **Fast Execution**: Use mocking to avoid slow operations
3. **Isolation**: Tests don't affect each other or external systems
4. **Maintainability**: Clear organization and naming
5. **CI/CD Ready**: Automated testing in pipelines

## 📦 Dependencies

See `requirements-test.txt` for full list:

- pytest - Testing framework
- pytest-asyncio - Async test support
- httpx - FastAPI TestClient dependency
- pytest-cov - Coverage reporting
- pytest-mock - Enhanced mocking

## 🚀 CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run tests
  run: pytest tests/test_api_comprehensive.py --cov=src --cov-report=xml
```

See full examples in [TEST_SUMMARY.md](TEST_SUMMARY.md).

---

**Last Updated**: 2025-11-16
**Test Framework**: pytest
**Coverage**: 100% of API endpoints
