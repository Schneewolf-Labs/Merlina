# Merlina Dataset Guide

## Overview

Merlina now supports flexible dataset loading and formatting! You can use datasets from HuggingFace, upload your own files, or use local files.

## Dataset Format Requirements

Your dataset must have these columns:
- `prompt` (required): The user's question or input
- `chosen` (required): The preferred/correct response
- `rejected` (required): The non-preferred/incorrect response
- `system` (optional): System prompt for the conversation

## Dataset Sources

### 1. HuggingFace Datasets

Load datasets directly from the HuggingFace Hub:

```json
{
  "source": {
    "source_type": "huggingface",
    "repo_id": "schneewolflabs/Athanor-DPO",
    "split": "train"
  }
}
```

### 2. Upload Files

Upload JSON, JSONL, CSV, or Parquet files through the UI:

1. Select "Upload File" as source type
2. Choose your file
3. Click "Upload Dataset"
4. The uploaded dataset will be ready for training

**Supported Formats:**
- `.json` - JSON array of objects
- `.jsonl` - JSON Lines (one object per line)
- `.csv` - CSV with headers
- `.parquet` - Apache Parquet files

**Example JSON:**
```json
[
  {
    "system": "You are a helpful assistant.",
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "I don't know."
  },
  {
    "prompt": "Tell me a joke.",
    "chosen": "Why don't scientists trust atoms? Because they make up everything!",
    "rejected": "No."
  }
]
```

### 3. Local File Path

If running Merlina locally, you can specify a file path:

```json
{
  "source": {
    "source_type": "local_file",
    "file_path": "/path/to/your/dataset.json",
    "file_format": "json"  // optional, auto-detected from extension
  }
}
```

## Format Types

Merlina supports multiple chat template formats:

### Tokenizer (Recommended)
**NEW!** Automatically uses the model's built-in chat template from `tokenizer_config.json`.

```json
{
  "format": {
    "format_type": "tokenizer"
  }
}
```

This is the **recommended format** for instruction-tuned models because:
- Automatically matches the model's native format
- No manual template configuration needed
- Works with Llama 3, Qwen, Mistral, and other models
- Always uses the correct special tokens

The output format depends on your model. For example, Llama 3 will use:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>
```

See [TOKENIZER_FORMATTER_GUIDE.md](TOKENIZER_FORMATTER_GUIDE.md) for detailed documentation.

### ChatML (Default)
Used by many models including Qwen and some fine-tuned models.

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

### Llama 3
Meta's Llama 3 format with special tokens (manual override).

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>
```

### Mistral Instruct
Mistral's instruction format.

```
[INST] You are a helpful assistant.

Hello! [/INST] Hi there!
```

### Custom Template
Define your own format using placeholders:

```json
{
  "format": {
    "format_type": "custom",
    "custom_templates": {
      "prompt_template": "{system}\n\nUser: {prompt}\nAssistant: ",
      "chosen_template": "{chosen}",
      "rejected_template": "{rejected}"
    }
  }
}
```

## Column Mapping

If your dataset uses different column names, you can map them:

```json
{
  "column_mapping": {
    "instruction": "prompt",
    "good_response": "chosen",
    "bad_response": "rejected",
    "system_message": "system"
  }
}
```

## Advanced Options

### Test Split Size
Fraction of data to use for evaluation (default: 0.01 = 1%)

```json
{
  "test_size": 0.05  // Use 5% for evaluation
}
```

### Max Samples
Limit dataset size for testing:

```json
{
  "max_samples": 100  // Only use first 100 samples
}
```

## Using the API

### Preview Dataset (Raw)
```bash
curl -X POST http://localhost:8000/dataset/preview \
  -H "Content-Type: application/json" \
  -d '{
    "source": {
      "source_type": "huggingface",
      "repo_id": "schneewolflabs/Athanor-DPO"
    },
    "format": {
      "format_type": "chatml"
    }
  }'
```

### Preview Formatted Dataset
```bash
curl -X POST http://localhost:8000/dataset/preview-formatted \
  -H "Content-Type: application/json" \
  -d '{
    "source": {
      "source_type": "local_file",
      "file_path": "/path/to/dataset.json"
    },
    "format": {
      "format_type": "llama3"
    }
  }'
```

### Upload Dataset
```bash
curl -X POST http://localhost:8000/dataset/upload-file \
  -F "file=@my_dataset.json"
```

Returns:
```json
{
  "status": "success",
  "dataset_id": "upload_20231120_143022_a1b2c3d4",
  "filename": "my_dataset.json",
  "size": 12345
}
```

## Complete Training Example

### Using Tokenizer Format (Recommended)

```json
{
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "output_name": "my-custom-model",

  "dataset": {
    "source": {
      "source_type": "upload",
      "dataset_id": "upload_20231120_143022_a1b2c3d4"
    },
    "format": {
      "format_type": "tokenizer"
    },
    "test_size": 0.02,
    "max_samples": 1000
  },

  "lora_r": 64,
  "lora_alpha": 32,
  "num_epochs": 2,
  "learning_rate": 5e-6,
  "use_4bit": true
}
```

### Using Manual Format (Alternative)

```json
{
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "output_name": "my-custom-model",

  "dataset": {
    "source": {
      "source_type": "upload",
      "dataset_id": "upload_20231120_143022_a1b2c3d4"
    },
    "format": {
      "format_type": "llama3"
    },
    "test_size": 0.02,
    "max_samples": 1000
  },

  "lora_r": 64,
  "lora_alpha": 32,
  "num_epochs": 2,
  "learning_rate": 5e-6,
  "use_4bit": true
}
```

## Troubleshooting

### "Dataset missing required columns"
Make sure your dataset has `prompt`, `chosen`, and `rejected` columns. Use `column_mapping` if needed.

### "Failed to load dataset"
- Check file format is supported (JSON, JSONL, CSV, Parquet)
- Verify file path is correct (for local files)
- Ensure HuggingFace repo exists and is accessible

### "Please upload a dataset first"
When using `source_type: "upload"`, you must upload the file before starting training.

## Module Structure

The dataset handling is organized in the `dataset_handlers/` module:

```
dataset_handlers/
├── __init__.py         # Module exports
├── base.py            # Abstract base classes and DatasetPipeline
├── loaders.py         # HuggingFace, LocalFile, Upload loaders
├── formatters.py      # ChatML, Llama3, Mistral, Custom formatters
└── validators.py      # Dataset validation utilities
```

This modular design makes it easy to add new loaders and formatters in the future!
