# Formatted Preview Feature

## Overview

The formatted preview feature allows users to see exactly how their dataset will be transformed before training begins. This helps ensure the chat format is correct for their target model.

## How It Works

### Two Preview Modes

**1. Preview Raw Data** 🔍
- Shows first 3 samples in original JSON format
- Helps verify dataset structure
- Checks column names and values

**2. Preview Formatted** ✨
- Shows first sample after format transformation
- Displays side-by-side comparison:
  - Prompt (with system message and user input)
  - Chosen response (green highlight)
  - Rejected response (red highlight)
- Shows which format type is applied

## Visual Example

### Raw Data
```json
{
  "system": "You are a helpful assistant.",
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I don't know."
}
```

### Formatted Preview (ChatML)

**📝 Prompt:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant

```

**✓ Chosen Response:** (green background)
```
The capital of France is Paris.<|im_end|>
```

**✗ Rejected Response:** (red background)
```
I don't know.<|im_end|>
```

**📝 Format Type:** ChatML

### Formatted Preview (Llama 3)

**📝 Prompt:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

**✓ Chosen Response:**
```
The capital of France is Paris.<|eot_id|>
```

**✗ Rejected Response:**
```
I don't know.<|eot_id|>
```

**📝 Format Type:** Llama 3

## User Benefits

1. **Confidence** - See exactly what the model will receive
2. **Debug** - Catch formatting issues before training
3. **Compare** - Try different formats and see which looks best
4. **Learn** - Understand how chat templates work
5. **Verify** - Ensure special tokens are correct for your model

## UI Features

### Layout
- Two side-by-side buttons for easy switching
- Clean, color-coded display
- Scrollable sections for long text
- Monospace font for clarity
- Format type indicator

### Interactivity
- Click "Preview Raw Data" → See JSON
- Click "Preview Formatted" → See chat format
- Change format type → Preview updates when clicked again
- Toast notifications for success/errors

### Color Coding
- **Prompt section**: Light gray background
- **Chosen response**: Light green background (success color)
- **Rejected response**: Light red background (danger color)
- **Format indicator**: Purple background (brand color)

## Implementation Details

### Backend API
- Endpoint: `POST /dataset/preview-formatted`
- Accepts: `DatasetConfig` with source and format
- Returns: First sample after formatting
- Uses: `DatasetPipeline.preview_formatted(num_samples=1)`

### Frontend
- Two buttons: `#preview-dataset-button` and `#preview-formatted-button`
- Display toggle: Shows one preview at a time
- Format mapping: Friendly names (ChatML, Llama 3, etc.)
- Error handling: Clear messages if preview fails

### Format Support
All format types supported:
- ✅ ChatML
- ✅ Llama 3
- ✅ Mistral Instruct
- ✅ Custom templates

## Example Use Case

**Scenario:** User wants to fine-tune Llama 3.2 but isn't sure if their dataset is formatted correctly.

**Workflow:**
1. Upload dataset JSON file
2. Select "Llama 3" format type
3. Click "Preview Formatted"
4. See the formatted conversation with proper Llama 3 tokens
5. Verify it matches Llama 3's expected format
6. Proceed with training confidently!

**Without this feature:** User would need to:
- Start training and hope for the best
- Check logs for formatting issues
- Possibly waste hours on incorrect training
- Debug by examining model outputs

## Technical Architecture

```python
# Frontend sends:
{
  "source": {"source_type": "upload", "dataset_id": "..."},
  "format": {"format_type": "llama3"}
}

# Backend processes:
loader = create_loader(source)
formatter = get_formatter(format_type)
pipeline = DatasetPipeline(loader, formatter)
samples = pipeline.preview_formatted(num_samples=1)

# Backend returns:
{
  "status": "success",
  "samples": [
    {
      "prompt": "<|begin_of_text|>...",
      "chosen": "Response...<|eot_id|>",
      "rejected": "Bad response...<|eot_id|>"
    }
  ],
  "num_samples": 1
}

# Frontend displays in UI with color coding
```

## Testing

Tested with:
- ✅ Local JSON files
- ✅ All 4 format types
- ✅ Datasets with and without system prompts
- ✅ Different sample sizes
- ✅ Error conditions

See `test_formatted_preview.py` for automated tests.

## Future Enhancements

Possible improvements:
1. Show multiple samples (paginated view)
2. Compare formats side-by-side
3. Highlight differences between formats
4. Export formatted samples
5. Validate against known model tokenizers
6. Show token count for each section

## Conclusion

The formatted preview feature significantly improves the user experience by:
- ✅ Reducing training errors
- ✅ Building user confidence
- ✅ Teaching about chat formats
- ✅ Enabling quick iteration
- ✅ Preventing wasted compute time

Users can now see exactly what they're training on before spending GPU time!
