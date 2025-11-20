# ORPO Migration Guide

## Overview

This guide explains how to migrate Merlina from TRL's experimental `ORPOTrainer` to our standalone `MerlinaORPOTrainer`.

**Why migrate?**
- TRL has marked ORPO as experimental and may remove it
- Full control over the implementation
- No dependency on TRL's internal changes
- Ability to customize and optimize for Merlina's needs

**What we've extracted:**
- ~600 lines of standalone code vs 1,300+ lines in TRL
- Core ORPO algorithm (proven and tested)
- Direct inheritance from `transformers.Trainer` (not TRL's BaseTrainer)
- All essential features needed for training

---

## What's Included

### ✅ Implemented Features

| Feature | Status | Notes |
|---------|--------|-------|
| Core ORPO loss computation | ✅ | Exact implementation from TRL |
| NLL + Odds Ratio loss | ✅ | Full algorithm |
| Decoder-only models | ✅ | Llama, Qwen, Mistral, etc. |
| 4-bit quantization | ✅ | Works with BitsAndBytes |
| LoRA/PEFT support | ✅ | Via peft_config parameter |
| Gradient accumulation | ✅ | Native Transformers support |
| Mixed precision (bf16/fp16) | ✅ | Native Transformers support |
| Metrics logging | ✅ | Loss, rewards, margins |
| W&B integration | ✅ | Via TrainingArguments |

### ⚠️ Not Implemented (Yet)

| Feature | Priority | Workaround |
|---------|----------|------------|
| Encoder-decoder models | Low | Use TRL if needed |
| W&B generation logging | Low | Manual implementation |
| Auxiliary loss (MoE) | Low | Not commonly used |

---

## Migration Steps

### Step 1: Update Imports

**Before (using TRL):**
```python
from trl import ORPOConfig, ORPOTrainer
```

**After (using standalone):**
```python
from src.orpo_standalone import ORPOConfig, MerlinaORPOTrainer as ORPOTrainer
```

That's it! The API is designed to be drop-in compatible.

---

### Step 2: Update training_runner.py

**File:** `src/training_runner.py`

**Changes needed:**

```python
# At the top of the file (line 21)
# OLD:
from trl import ORPOConfig, ORPOTrainer

# NEW:
from src.orpo_standalone import ORPOConfig, MerlinaORPOTrainer as ORPOTrainer
```

**That's the only change needed!** The rest of your code stays exactly the same because:
- `ORPOConfig` has the same parameters
- `MerlinaORPOTrainer` accepts the same arguments
- The trainer is aliased as `ORPOTrainer` for compatibility

---

### Step 3: Update merlina.py (if used)

**File:** `merlina.py`

**Changes needed:**

```python
# At the top of the file (line 29)
# OLD:
from trl import ORPOConfig, ORPOTrainer

# NEW:
from src.orpo_standalone import ORPOConfig, MerlinaORPOTrainer as ORPOTrainer
```

**Note:** The old training function in `merlina.py` is deprecated, so this change is only needed if you're maintaining backwards compatibility.

---

### Step 4: Update requirements.txt (Optional)

You can now make TRL optional instead of required:

```txt
# Before:
trl>=0.7.0

# After (make it optional):
# trl>=0.7.0  # No longer required for ORPO training

# Or keep it for other features:
trl>=0.7.0  # Optional: only needed for other trainers (DPO, PPO, etc.)
```

---

## Testing the Migration

### Test 1: Quick Validation

Run a short training job with the new trainer:

```python
# Create a minimal test script: test_orpo_migration.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.orpo_standalone import ORPOConfig, create_orpo_trainer

# Load small model for testing
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer.pad_token = tokenizer.eos_token

# Load small dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:10]")

# Quick test config
config = ORPOConfig(
    output_dir="./test_orpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    max_steps=5,  # Just 5 steps
    logging_steps=1,
    beta=0.1,
    max_length=512,
)

# Create trainer
trainer = create_orpo_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    orpo_config=config
)

# Run quick test
print("Starting test training...")
trainer.train()
print("✅ Test completed successfully!")
```

Run it:
```bash
python test_orpo_migration.py
```

**Expected output:**
```
Starting test training...
[Training logs with loss, nll_loss, or_loss, rewards]
✅ Test completed successfully!
```

---

### Test 2: Full Merlina Training

Run a full training job through Merlina's web UI:

1. Start Merlina: `python merlina.py`
2. Configure a small training job:
   - Model: Any small model for testing
   - Dataset: Small dataset (max_samples: 100)
   - Epochs: 1
   - Enable W&B: Yes (to test metrics logging)

3. Monitor the logs for:
   - ✅ No import errors
   - ✅ Training starts successfully
   - ✅ Metrics logged: `nll_loss`, `or_loss`, `chosen_rewards`, `rejected_rewards`
   - ✅ W&B integration works
   - ✅ Model saves successfully

---

## Comparison: TRL vs Standalone

| Aspect | TRL ORPOTrainer | MerlinaORPOTrainer |
|--------|-----------------|-------------------|
| **Lines of code** | ~1,300 | ~600 |
| **Dependencies** | TRL (heavy) | Transformers only |
| **Base class** | TRL's BaseTrainer | transformers.Trainer |
| **Maintenance** | HuggingFace team | Your team |
| **Customization** | Limited | Full control |
| **Core algorithm** | ✅ Same | ✅ Same |
| **Performance** | Same | Same |
| **Encoder-decoder** | ✅ Full support | ⚠️ Basic support |
| **MoE aux loss** | ✅ Supported | ❌ Not implemented |
| **W&B generation** | ✅ Supported | ❌ Not implemented |

---

## Feature Parity Checklist

Use this to verify your migration:

- [ ] Training starts without errors
- [ ] Loss values are reasonable (not NaN/Inf)
- [ ] NLL loss is logged
- [ ] OR loss is logged
- [ ] Chosen rewards > Rejected rewards (preference learned)
- [ ] Reward margin increases over training
- [ ] Model saves successfully
- [ ] Evaluation runs (if enabled)
- [ ] W&B logging works (if enabled)
- [ ] HuggingFace Hub push works (if enabled)
- [ ] GPU memory usage is similar to before
- [ ] Training speed is similar to before

---

## Troubleshooting

### Issue: Import Error

**Error:**
```
ImportError: cannot import name 'MerlinaORPOTrainer' from 'src.orpo_standalone'
```

**Solution:**
Make sure `src/orpo_standalone.py` exists and check Python path:
```bash
ls -la src/orpo_standalone.py
python -c "from src.orpo_standalone import MerlinaORPOTrainer; print('OK')"
```

---

### Issue: Dataset Format Error

**Error:**
```
KeyError: 'chosen_input_ids'
```

**Solution:**
The dataset must have these fields (pre-tokenized):
- `chosen_input_ids`, `chosen_attention_mask`, `chosen_labels`
- `rejected_input_ids`, `rejected_attention_mask`, `rejected_labels`

Merlina's dataset pipeline should already handle this via the formatters.

**Check your formatter is applied:**
```python
# In training_runner.py, verify this section exists:
formatter = get_formatter(
    format_type=config.dataset.format.format_type,
    custom_templates=config.dataset.format.custom_templates,
    tokenizer=tokenizer if config.dataset.format.format_type == 'tokenizer' else None
)
```

---

### Issue: Loss is NaN

**Possible causes:**
1. Learning rate too high
2. Gradient clipping disabled
3. Beta value too large

**Solutions:**
```python
# Check these in your ORPOConfig:
config = ORPOConfig(
    learning_rate=5e-6,  # Try lower if NaN
    max_grad_norm=0.3,   # Enable gradient clipping
    beta=0.1,            # Try 0.01-0.5 range
)
```

---

### Issue: Rewards Not Increasing

**Expected behavior:**
- `chosen_rewards` should be higher than `rejected_rewards`
- `reward_margin` should increase over training

**If not happening:**
1. Check beta value (try 0.1 - 0.5)
2. Verify dataset has meaningful chosen vs rejected differences
3. Check learning rate (try 1e-6 to 1e-5)
4. Ensure labels are correctly formatted

---

## Advanced: Customization Examples

### Example 1: Custom Beta Scheduling

Adjust beta during training for better convergence:

```python
class CustomORPOTrainer(MerlinaORPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Increase beta over time
        initial_beta = 0.05
        final_beta = 0.3
        progress = self.state.global_step / self.state.max_steps
        self.beta = initial_beta + (final_beta - initial_beta) * progress

        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
```

---

### Example 2: Add Custom Metrics

Log additional metrics:

```python
class MetricsORPOTrainer(MerlinaORPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, metrics = super().compute_loss(model, inputs, return_outputs=True)

        # Add custom metric
        metrics["reward_accuracy"] = (
            (metrics["chosen_rewards"] > metrics["rejected_rewards"]).float().mean()
        )

        if return_outputs:
            return (loss, metrics)
        return loss
```

---

## Rollback Plan

If you need to rollback to TRL:

1. **Revert imports:**
   ```python
   # src/training_runner.py and merlina.py
   from trl import ORPOConfig, ORPOTrainer  # Back to TRL
   ```

2. **Verify TRL is installed:**
   ```bash
   pip install trl>=0.7.0
   ```

3. **Restart Merlina:**
   ```bash
   python merlina.py
   ```

Your training configs and datasets don't need any changes.

---

## Next Steps

After successful migration:

1. **Remove TRL dependency** (if not using other TRL features):
   ```bash
   pip uninstall trl
   ```

2. **Test thoroughly** with your production workloads

3. **Consider enhancements:**
   - Custom beta scheduling
   - Additional metrics
   - Performance optimizations
   - Support for your specific model architectures

4. **Monitor for issues:**
   - Keep an eye on training metrics
   - Compare results to TRL baseline
   - Report any anomalies

---

## FAQ

**Q: Is the math exactly the same as TRL?**
A: Yes, we extracted the exact implementation of the ORPO loss computation.

**Q: Will I lose any features?**
A: Only experimental features like MoE auxiliary loss and W&B generation logging. Core training is identical.

**Q: What about performance?**
A: Performance should be identical since the core computation is the same and both use the same base infrastructure.

**Q: Can I still use TRL for other trainers?**
A: Yes! You can keep TRL installed for DPO, PPO, or other trainers. Just use `MerlinaORPOTrainer` for ORPO.

**Q: What if I find a bug?**
A: Since you own the code, you can fix it immediately! Or open an issue in your repo.

**Q: Can I contribute improvements back?**
A: Absolutely! Any optimizations or enhancements you make can be shared with the community.

---

## Support

If you encounter issues during migration:

1. Check the troubleshooting section above
2. Review the test scripts
3. Compare your config to the working examples
4. Check logs for specific error messages

For questions or issues, refer to the codebase documentation or create an issue in your repository.

---

**Happy training with your standalone ORPO implementation! 🧙‍♀️✨**
