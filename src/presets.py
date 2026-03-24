"""
Recommended training presets per method, backed by paper results and community best practices.

Each preset contains hyperparameters tuned for LoRA fine-tuning of 7B-9B models.
These are starting points — users should still sweep beta and learning_rate for
their specific model and dataset.

Sources:
- SFT: Unsloth LoRA guide, Databricks blog
- ORPO: arXiv:2403.07691 (ORPO paper), HF ORPO tutorial
- DPO: arXiv:2305.18290 (DPO paper), HF Alignment Handbook, HF pref-tuning blog
- SimPO: arXiv:2405.14734 (SimPO paper), SimPO GitHub
- CPO: arXiv:2401.08417 (CPO paper), TRL CPOConfig
- IPO: arXiv:2310.12036 (IPO paper), HF pref-tuning blog
- KTO: arXiv:2402.01306 (KTO paper), TRL KTO docs
"""

TRAINING_PRESETS = {
    "sft": {
        "label": "SFT - Supervised Fine-Tuning",
        "settings": {
            "learning_rate": 2e-4,
            "num_epochs": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "lora_dropout": 0.05,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "notes": (
            "LoRA learning rates are ~10x higher than full fine-tuning. "
            "2-3 epochs is safe; more risks overfitting on small datasets. "
            "SFT is the least sensitive method to hyperparameters."
        ),
    },
    "orpo": {
        "label": "ORPO - Odds Ratio Preference Optimization",
        "settings": {
            "learning_rate": 8e-6,
            "beta": 0.1,
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "lora_dropout": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "notes": (
            "ORPO combines SFT + preference in one pass, so it needs a higher LR "
            "than other preference methods. Beta 0.1-0.25 works for most models "
            "(smaller models may benefit from higher beta). No reference model needed. "
            "Source: ORPO paper, Table 4."
        ),
    },
    "dpo": {
        "label": "DPO - Direct Preference Optimization",
        "settings": {
            "learning_rate": 5e-6,
            "beta": 0.1,
            "label_smoothing": 0.0,
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "lora_dropout": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "notes": (
            "DPO is very sensitive to beta — sweep it. The HF blog found optimal "
            "beta ranges from 0.01 to 0.6 depending on the model. Train for 1 epoch; "
            "reward accuracy hitting 1.0 early means overfitting. Use label_smoothing "
            "0.1 if your preference data is noisy. "
            "Source: HF Alignment Handbook, HF pref-tuning blog."
        ),
    },
    "simpo": {
        "label": "SimPO - Simple Preference Optimization",
        "settings": {
            "learning_rate": 1e-6,
            "beta": 2.0,
            "gamma": 1.0,
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "lora_dropout": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "notes": (
            "SimPO beta is MUCH higher than DPO (2.0-10.0 vs 0.1) because it uses "
            "average log-probability as reward without a reference baseline. "
            "Learning rate is the most critical parameter — above 1e-5 causes "
            "incoherent output. Tune gamma/beta ratio in [0.1, 0.8]. "
            "Source: SimPO paper, Table 2 and GitHub."
        ),
    },
    "cpo": {
        "label": "CPO - Contrastive Preference Optimization",
        "settings": {
            "learning_rate": 5e-6,
            "beta": 0.1,
            "label_smoothing": 0.0,
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "lora_dropout": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "notes": (
            "CPO is reference-free DPO with an SFT regularizer. Shares most "
            "hyperparameter sensitivities with DPO. Use label_smoothing 0.1 "
            "for noisy preference data. "
            "Source: CPO paper (ALMA-R), TRL CPOConfig."
        ),
    },
    "ipo": {
        "label": "IPO - Identity Preference Optimization",
        "settings": {
            "learning_rate": 5e-6,
            "beta": 0.01,
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "lora_dropout": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "notes": (
            "IPO's beta has an inverted meaning vs DPO: target margin = 1/(2*beta). "
            "Beta 0.01 (margin=50) was optimal across models in HF benchmarks, making "
            "IPO less model-dependent than DPO. Its squared loss is naturally robust "
            "to noisy labels — no label_smoothing needed. "
            "Source: IPO paper, HF pref-tuning blog."
        ),
    },
    "kto": {
        "label": "KTO - Kahneman-Tversky Optimization",
        "settings": {
            "learning_rate": 5e-6,
            "beta": 0.1,
            "num_epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.0,
            "lora_dropout": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "notes": (
            "KTO estimates KL divergence from the batch, so per-step batch size >= 4 "
            "is critical (not just effective batch size). For each beta, there's a "
            "maximum tolerable LR — at beta=0.1, stay at or below 1e-5. "
            "If desirable/undesirable examples are imbalanced, adjust lambda_d/lambda_u. "
            "Source: KTO paper, TRL KTO docs."
        ),
    },
}


def get_preset(training_mode: str) -> dict | None:
    """Return the preset for a given training mode, or None if not found."""
    return TRAINING_PRESETS.get(training_mode)


def get_all_presets() -> dict:
    """Return all presets."""
    return TRAINING_PRESETS
