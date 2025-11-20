# ORPO Standalone - Quick Reference

## 🚀 Quick Start (3 Steps)

### Step 1: Update Import
```python
# src/training_runner.py (line 21)
from src.orpo_standalone import ORPOConfig, MerlinaORPOTrainer as ORPOTrainer
```

### Step 2: Use as Normal
```python
# Everything else stays the same!
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)
trainer.train()
```

### Step 3: Test
```bash
python merlina.py
# Run a small training job via UI
```

---

## 📋 Common Configurations

### Minimal Config
```python
config = ORPOConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
)
```

### Production Config
```python
config = ORPOConfig(
    output_dir="./output",

    # ORPO-specific
    beta=0.1,                    # Preference strength (0.01 - 1.0)
    max_length=2048,             # Total sequence length
    max_prompt_length=1024,      # Prompt length
    disable_dropout=True,        # Stability

    # Training
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    warmup_ratio=0.05,

    # Optimization
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    weight_decay=0.01,

    # Precision
    bf16=True,  # Use bf16 if GPU supports it

    # Logging
    logging_steps=10,
    report_to=["wandb"],
)
```

### High-Memory Config (Large Model)
```python
config = ORPOConfig(
    output_dir="./output",
    beta=0.1,
    max_length=4096,             # Longer sequences
    max_prompt_length=2048,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # Larger effective batch
    gradient_checkpointing=True,     # Save VRAM

    learning_rate=1e-6,          # Lower LR for stability
)
```

---

## 🎯 Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `beta` | 0.1 | 0.01 - 1.0 | Preference strength (λ) |
| `max_length` | 2048 | 512 - 8192 | Total sequence length |
| `max_prompt_length` | 1024 | 256 - 4096 | Maximum prompt length |
| `disable_dropout` | True | bool | Disable dropout for stability |
| `learning_rate` | 5e-6 | 1e-7 - 1e-4 | Learning rate |

---

## 📊 Expected Metrics

### During Training

```
Step 10:  loss=2.45  nll_loss=2.30  or_loss=0.15  reward_margin=0.52
Step 20:  loss=2.38  nll_loss=2.25  or_loss=0.13  reward_margin=0.68
Step 30:  loss=2.31  nll_loss=2.19  or_loss=0.12  reward_margin=0.83
```

**Good signs:**
- ✅ `loss` decreasing
- ✅ `nll_loss` decreasing
- ✅ `reward_margin` increasing (chosen > rejected)
- ✅ `chosen_rewards` > `rejected_rewards`

**Bad signs:**
- ❌ `loss` is NaN → Reduce learning rate
- ❌ `reward_margin` not increasing → Increase beta
- ❌ `or_loss` exploding → Reduce beta

---

## 🔧 Troubleshooting

### Loss is NaN
```python
config = ORPOConfig(
    learning_rate=1e-6,      # Reduce LR
    max_grad_norm=0.3,       # Enable clipping
    beta=0.05,               # Reduce beta
)
```

### Preference Not Learning (margin not increasing)
```python
config = ORPOConfig(
    beta=0.3,                # Increase beta
    learning_rate=1e-5,      # Increase LR slightly
)
```

### Out of Memory
```python
config = ORPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    max_length=1024,         # Reduce if needed
)
```

---

## 📈 Hyperparameter Tuning

### Beta (Preference Strength)

| Beta | Effect | Use When |
|------|--------|----------|
| 0.01 - 0.05 | Weak preference | Dataset has subtle differences |
| 0.1 (default) | Moderate | Balanced training |
| 0.3 - 0.5 | Strong preference | Clear chosen vs rejected |
| 0.5 - 1.0 | Very strong | Override other signals |

### Learning Rate

| LR | Effect | Use When |
|----|--------|----------|
| 1e-7 - 1e-6 | Very conservative | Large models, unstable training |
| 5e-6 (default) | Standard | Most cases |
| 1e-5 - 5e-5 | Aggressive | Small models, quick experiments |

---

## 🧪 Quick Test Script

```python
# test_orpo.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.orpo_standalone import ORPOConfig, create_orpo_trainer
from datasets import load_dataset

# Small model for testing
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer.pad_token = tokenizer.eos_token

# Small dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:50]")

# Quick config
config = ORPOConfig(
    output_dir="./test",
    max_steps=10,
    logging_steps=1,
    beta=0.1,
    max_length=512,
)

# Train
trainer = create_orpo_trainer(model, tokenizer, dataset, orpo_config=config)
trainer.train()

print("✅ Test passed!")
```

Run: `python test_orpo.py`

---

## 📚 More Info

- **Full Guide:** `docs/ORPO_MIGRATION_GUIDE.md`
- **Summary:** `docs/ORPO_STANDALONE_SUMMARY.md`
- **Source:** `src/orpo_standalone.py`
- **Tests:** `tests/test_orpo_standalone.py`
- **Paper:** https://arxiv.org/abs/2403.07691

---

## 🆘 Getting Help

1. Check troubleshooting section above
2. Review migration guide
3. Inspect logged metrics
4. Run test script to isolate issues
5. Check source code docstrings

**Most common issues:**
- Wrong dataset format (need chosen/rejected fields)
- Learning rate too high (use 1e-6 to 1e-5)
- Beta too high/low (start with 0.1)
- Out of memory (reduce batch size, enable checkpointing)
