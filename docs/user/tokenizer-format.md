# Tokenizer-Based Chat Formatting

## Overview

Merlina now supports automatic chat formatting using the tokenizer's built-in chat template! This feature allows you to train models using their native chat format without manually specifying templates.

## Why Use Tokenizer Format?

When you use `format_type: "tokenizer"`, Merlina will:
- Automatically read the `chat_template` from the model's `tokenizer_config.json`
- Apply the exact format the model was trained with
- Handle all special tokens and formatting correctly
- Ensure compatibility with the model's expected input format

This is especially useful when:
- Training instruction-tuned models (Llama 3, Qwen, Mistral, etc.)
- You want to maintain consistency with the base model's format
- You're not sure what chat format the model uses

## How to Use

Simply set `format_type` to `"tokenizer"` in your dataset configuration:

```json
{
  "base_model": "meta-llama/Llama-3.2-1B-Instruct",
  "output_name": "my-model",

  "dataset": {
    "source": {
      "source_type": "huggingface",
      "repo_id": "schneewolflabs/Athanor-DPO",
      "split": "train"
    },
    "format": {
      "format_type": "tokenizer"
    }
  }
}
```

That's it! The tokenizer will automatically format your data.

## How It Works

The TokenizerFormatter:

1. **Loads the model's tokenizer** during dataset preparation
2. **Reads the chat_template** from `tokenizer_config.json`
3. **Applies the template** using the tokenizer's `apply_chat_template()` method
4. **Formats prompts** with the correct system/user/assistant structure
5. **Handles special tokens** like `<|eot_id|>`, `<|im_end|>`, etc.

### Example Output

For a Llama 3 model, your data will be formatted as:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
```

The exact format depends on the model you're using!

## Fallback Behavior

If the tokenizer doesn't have a `chat_template` defined:
- Merlina will log a warning
- Fall back to simple concatenation format
- Your training will still work, just without special formatting

## Comparison with Other Formats

| Format Type | Use Case | When to Use |
|-------------|----------|-------------|
| `tokenizer` | **Recommended for most cases** | When training instruction models that have a chat template |
| `chatml` | OpenAI-style format | For models that use `<\|im_start\|>` and `<\|im_end\|>` |
| `llama3` | Llama 3 format | Manual override if tokenizer format doesn't work |
| `mistral` | Mistral format | For Mistral models without chat template |
| `custom` | Your own format | Special requirements or custom models |

## Dataset Preview Limitation

**Note**: When using the `/dataset/preview-formatted` endpoint, the tokenizer format cannot be previewed without loading the model. The preview will fall back to `chatml` format for display purposes only. Your actual training will still use the correct tokenizer format.

## Complete Training Example

```json
{
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "output_name": "my-qwen-model",

  "dataset": {
    "source": {
      "source_type": "local_file",
      "file_path": "/path/to/dataset.json"
    },
    "format": {
      "format_type": "tokenizer"
    },
    "test_size": 0.01,
    "max_samples": 1000
  },

  "lora_r": 64,
  "lora_alpha": 32,
  "learning_rate": 5e-6,
  "num_epochs": 2
}
```

## Testing the Feature

You can test the tokenizer formatter with the provided test script:

```bash
python3 test_tokenizer_formatter.py
```

This will:
- Load different tokenizers (Llama 3, GPT-2)
- Test chat template formatting
- Verify fallback behavior
- Validate error handling

## API Changes

### `DatasetFormat` Model

The `format_type` field now accepts `"tokenizer"` as a valid option:

```python
class DatasetFormat(BaseModel):
    format_type: str = Field(
        "chatml",
        description="Format type: chatml, llama3, mistral, tokenizer, custom"
    )
```

### `get_formatter()` Function

The formatter factory now accepts a `tokenizer` parameter:

```python
formatter = get_formatter(
    format_type='tokenizer',
    tokenizer=tokenizer  # Required for tokenizer format
)
```

## Technical Details

### TokenizerFormatter Class

Located in `dataset_handlers/formatters.py`:

```python
class TokenizerFormatter(DatasetFormatter):
    """
    Format dataset using the tokenizer's chat template from tokenizer_config.json.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        # Check if tokenizer has chat template
        # ...

    def format(self, row: dict) -> dict:
        # Apply chat template to format prompt/chosen/rejected
        # ...
```

### Integration Points

The tokenizer is passed to the formatter at these locations:

1. **Training pipeline** (`merlina.py:295`):
   ```python
   formatter = get_formatter(
       format_type=config.dataset.format.format_type,
       custom_templates=config.dataset.format.custom_templates,
       tokenizer=tokenizer if config.dataset.format.format_type == 'tokenizer' else None
   )
   ```

2. **Preview endpoints** (`merlina.py:521`, `merlina.py:586`):
   - Falls back to `chatml` for preview (model not loaded yet)

## Benefits

1. **Automatic Compatibility**: Always use the correct format for your model
2. **Less Configuration**: No need to manually specify templates
3. **Future-Proof**: Works with new models as they're released
4. **Error-Free**: Eliminates template syntax mistakes
5. **Consistent**: Matches the exact format the base model expects

## Troubleshooting

### "tokenizer required for 'tokenizer' format type"
- This error occurs if you try to use tokenizer format in preview mode
- During training, this shouldn't happen (tokenizer is loaded first)
- For preview, the system will fall back to chatml

### Tokenizer has no chat_template
- Some older models don't include chat templates
- The formatter will fall back to simple concatenation
- Consider using `chatml`, `llama3`, or `custom` format instead

### Unexpected formatting
- Check the tokenizer's chat template: `print(tokenizer.chat_template)`
- Verify your dataset has the required columns: `system`, `prompt`, `chosen`, `rejected`
- Try the test script to see actual output: `python3 test_tokenizer_formatter.py`

## Summary

The tokenizer format type makes it easy to train models with their native chat format. Just set `format_type: "tokenizer"` and Merlina handles the rest!

For most instruction-tuned models, this is now the **recommended approach**.
