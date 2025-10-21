# Merlina Examples

Ready-to-use training configurations for different scenarios.

## 🚀 Quick Start

All examples can be run with:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d @examples/basic-training.json
```

## 📋 Available Examples

### Basic Examples

- **[basic-training.json](basic-training.json)** ⭐ Start here!
  - Simplest possible training setup
  - Uses HuggingFace dataset
  - Tokenizer format (recommended)
  - Small dataset for quick testing

- **[huggingface-dataset.json](huggingface-dataset.json)**
  - Load datasets from HuggingFace Hub
  - Full configuration example
  - W&B logging enabled

- **[local-dataset.json](local-dataset.json)**
  - Use datasets from your local filesystem
  - JSON, JSONL, CSV, or Parquet files

### Format Examples

- **[tokenizer-format.json](tokenizer-format.json)** ⭐ Recommended
  - Automatic chat formatting
  - Uses model's native chat template
  - No manual template configuration needed

- **[custom-format.json](custom-format.json)**
  - Define your own chat templates
  - Full control over formatting
  - Useful for special requirements

## 📊 Sample Datasets

- **[datasets/sample-dpo-dataset.json](datasets/sample-dpo-dataset.json)**
  - Example DPO dataset structure
  - Shows required columns: system, prompt, chosen, rejected
  - Use as template for your own data

## 🎯 Common Scenarios

### Training with your own data

1. **Upload a file** (easiest):
   ```bash
   # Upload your dataset
   curl -X POST http://localhost:8000/dataset/upload-file \
     -F "file=@your_dataset.json"

   # Use the returned dataset_id in your config
   ```

2. **Use local file** (if running locally):
   - Modify `local-dataset.json`
   - Change `file_path` to your dataset location

3. **Use HuggingFace dataset**:
   - Modify `huggingface-dataset.json`
   - Change `repo_id` to your dataset

### Choosing a format

- ✅ **Use `tokenizer` format** (recommended for most cases)
  - Automatic compatibility with your model
  - No configuration needed
  - Works with Llama 3, Qwen, Mistral, etc.

- Use `chatml`, `llama3`, `mistral` if:
  - You need a specific format override
  - Your tokenizer doesn't have a chat template

- Use `custom` if:
  - You have special formatting requirements
  - You want full control

## 📖 More Information

- **Dataset Guide**: [../docs/user/dataset-guide.md](../docs/user/dataset-guide.md)
- **Tokenizer Format**: [../docs/user/tokenizer-format.md](../docs/user/tokenizer-format.md)
- **Main README**: [../README.md](../README.md)
