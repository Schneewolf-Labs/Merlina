# Quick Start: Custom Datasets

## 3 Ways to Use Your Own Data

### Option 1: Upload a File (Easiest!)

1. Open Merlina web interface
2. In the "Dataset" section, select **"Upload File"**
3. Click "Choose File" and select your JSON, CSV, or Parquet file
4. Click **"Upload Dataset"**
5. Click **"Preview Raw Data"** to see your original data
6. Click **"Preview Formatted"** to see how it will look with your chosen format
7. Proceed to training configuration!

**Your file format:**
```json
[
  {
    "prompt": "What is Python?",
    "chosen": "Python is a high-level programming language...",
    "rejected": "It's a snake."
  }
]
```

### Option 2: Use HuggingFace Dataset

1. Find a DPO/preference dataset on HuggingFace
2. In "Dataset Source", keep **"HuggingFace Dataset"** selected
3. Enter the repository ID (e.g., `username/dataset-name`)
4. Click **"Preview Raw Data"** or **"Preview Formatted"** to verify
5. Done!

**Popular datasets:**
- `schneewolflabs/Athanor-DPO` (default)
- `Intel/orca_dpo_pairs`
- `argilla/ultrafeedback-binarized-preferences`

### Option 3: Local File Path

1. Save your dataset file somewhere on your computer
2. Select **"Local File Path"** as source
3. Enter the full path (e.g., `/home/user/my_data.json`)
4. Click **"Preview Raw Data"** or **"Preview Formatted"**
5. Start training!

## Required Columns

Your dataset **must** have these columns:
- `prompt` - The question or instruction
- `chosen` - The good/preferred response
- `rejected` - The bad/non-preferred response

Optional:
- `system` - System prompt (e.g., "You are a helpful assistant")

## Supported File Formats

| Format | Extension | Best For |
|--------|-----------|----------|
| JSON | `.json` | Small datasets, human-readable |
| JSON Lines | `.jsonl` | Large datasets, streaming |
| CSV | `.csv` | Spreadsheet exports |
| Parquet | `.parquet` | Very large datasets, efficient |

## Preview Your Dataset

Before training, you can preview your data in two ways:

### 🔍 Preview Raw Data
Shows the first 3 samples of your dataset exactly as they are, in JSON format. Use this to:
- Verify the dataset loaded correctly
- Check column names and structure
- Inspect the original text

### ✨ Preview Formatted
Shows how the first sample will look after applying your chosen format. Displays:
- **Prompt** - The formatted conversation context
- **Chosen Response** - The preferred completion (highlighted in green)
- **Rejected Response** - The non-preferred completion (highlighted in red)
- **Format Type** - Which chat template is being used

**Tip:** Try different format types and preview them to see which looks best for your model!

## Choosing a Format Type

The format type determines how conversations are structured:

| Format | Use For | Example Models |
|--------|---------|----------------|
| **ChatML** | Default, most compatible | Qwen, Phi, many fine-tunes |
| **Llama 3** | Meta Llama 3 models | Llama-3.x, Llama-3.2 |
| **Mistral** | Mistral models | Mistral-7B, Mixtral |
| **Custom** | Your own template | Any model |

**Not sure?** Use **ChatML** (the default) - it works with most models!

## Example Datasets

### Minimal Example (3 samples)
```json
[
  {
    "prompt": "What is 2+2?",
    "chosen": "2+2 equals 4",
    "rejected": "5"
  },
  {
    "prompt": "Explain recursion",
    "chosen": "Recursion is when a function calls itself...",
    "rejected": "I don't know"
  },
  {
    "prompt": "Best programming language?",
    "chosen": "It depends on your use case. Python is great for...",
    "rejected": "Python is the only good language."
  }
]
```

### With System Prompts
```json
[
  {
    "system": "You are a helpful coding assistant.",
    "prompt": "How do I reverse a list in Python?",
    "chosen": "You can reverse a list using: my_list.reverse() or reversed_list = my_list[::-1]",
    "rejected": "Use a for loop"
  }
]
```

### CSV Format
```csv
prompt,chosen,rejected,system
"What is AI?","AI stands for Artificial Intelligence...","It's computers","You are helpful"
"Hello!","Hi! How can I help you today?","What?",""
```

## Tips

1. **Start Small**: Test with 10-50 samples first using "Max Samples" option
2. **Preview First**: Always click "Preview Dataset" before training
3. **Quality Matters**: 100 high-quality samples > 1000 low-quality ones
4. **Balance**: Try to have similar length chosen/rejected responses
5. **Diversity**: Include various types of prompts and responses

## Troubleshooting

**"Dataset missing required columns"**
→ Make sure you have `prompt`, `chosen`, and `rejected` columns

**"Failed to load dataset"**
→ Check your file format is valid JSON/CSV
→ Verify the file path is correct (for local files)

**"Upload failed"**
→ Check file size (keep under 100MB for uploads)
→ Verify file format is supported

**Preview shows weird text**
→ Click "Preview Dataset" to see raw data
→ Check your column names match expected format

## Need Help?

- 📖 Read [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed docs
- 🔧 Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
- 💬 Open an issue on GitHub

Happy training! ✨
