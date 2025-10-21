# Merlina Tests

Test suite for the Merlina training system.

## 🧪 Running Tests

### Run all tests
```bash
# From project root
python -m pytest tests/

# Or run individual test files
python tests/test_tokenizer_formatter.py
```

### Run specific tests
```bash
# Test dataset loaders
python tests/test_dataset_loaders.py

# Test formatters
python tests/test_tokenizer_formatter.py

# Test pipeline
python tests/test_pipeline.py

# Test API endpoints
python tests/test_api_endpoints.py
```

## 📁 Test Structure

```
tests/
├── test_dataset_loaders.py       # Test HuggingFace, local, upload loaders
├── test_tokenizer_formatter.py   # Test tokenizer-based formatting
├── test_pipeline.py               # Test full dataset pipeline
├── test_api_endpoints.py          # Test preview and upload endpoints
└── fixtures/                      # Test data files
    └── test_dataset.json          # Sample test dataset
```

## 🔍 Test Coverage

### Dataset Loaders (`test_dataset_loaders.py`)
- ✅ Loading local JSON files
- ✅ File format detection
- ✅ Error handling

### Tokenizer Formatter (`test_tokenizer_formatter.py`)
- ✅ Formatting with models that have chat templates (Llama 3)
- ✅ Fallback for models without chat templates (GPT-2)
- ✅ Error handling (missing tokenizer parameter)
- ✅ Format output validation

### Pipeline (`test_pipeline.py`)
- ✅ Module imports
- ✅ Basic functionality
- ✅ End-to-end dataset preparation

### API Endpoints (`test_api_endpoints.py`)
- ✅ Preview endpoints (raw and formatted)
- ✅ Dataset formatting
- ✅ Response validation

## 📊 Test Fixtures

Test data files are stored in `fixtures/`:

- **test_dataset.json** - Sample DPO dataset with system/prompt/chosen/rejected

## ✅ Test Requirements

Tests require these packages:
```bash
pip install pytest transformers datasets
```

## 🐛 Debugging Tests

### Verbose output
```bash
python tests/test_tokenizer_formatter.py -v
```

### Run with logging
```bash
PYTHONPATH=. python tests/test_tokenizer_formatter.py
```

## 📝 Writing New Tests

Follow this pattern:
```python
"""
Test description
"""
import sys
sys.path.insert(0, '/path/to/merlina')

from dataset_handlers import YourModule

def test_your_feature():
    # Setup
    # Execute
    # Assert
    print("✅ Test passed")

if __name__ == "__main__":
    test_your_feature()
```

## 🔗 More Information

- **Main README**: [../README.md](../README.md)
- **Documentation**: [../docs/](../docs/)
