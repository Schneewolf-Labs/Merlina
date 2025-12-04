"""
Standalone ORPO (Odds Ratio Preference Optimization) Implementation for Merlina

This is a self-contained implementation of ORPO that doesn't depend on TRL.
Extracted from: https://github.com/huggingface/trl/blob/main/trl/trainer/orpo_trainer.py

Paper: "ORPO: Monolithic Preference Optimization without Reference Model"
       arXiv:2403.07691 (2024)

Key differences from TRL version:
- Inherits directly from transformers.Trainer (not TRL's BaseTrainer)
- Minimal dependencies - only uses PyTorch and Transformers
- Simplified configuration
- No experimental features (W&B generation, aux loss, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
from transformers import Trainer, TrainingArguments, PreTrainedModel
from transformers.trainer_utils import EvalPrediction
import logging

logger = logging.getLogger(__name__)


@dataclass
class ORPOConfig(TrainingArguments):
    """
    Configuration for ORPO training.

    Extends TrainingArguments with ORPO-specific parameters.
    """

    # ORPO-specific parameters
    max_length: int = 2048
    """Maximum sequence length (prompt + completion)"""

    max_prompt_length: int = 1024
    """Maximum prompt length"""

    max_completion_length: Optional[int] = None
    """Maximum completion length (calculated as max_length - max_prompt_length if None)"""

    beta: float = 0.1
    """
    Weight for the odds ratio loss (λ in the paper).
    Controls the strength of preference optimization.
    Typical range: 0.01 to 1.0
    """

    disable_dropout: bool = True
    """Whether to disable dropout during training"""

    label_pad_token_id: int = -100
    """Token ID used for padding labels (ignored in loss)"""

    padding_value: Optional[int] = None
    """Value used for padding input_ids (uses tokenizer default if None)"""

    def __post_init__(self):
        super().__post_init__()

        # Calculate max_completion_length if not provided
        if self.max_completion_length is None:
            self.max_completion_length = self.max_length - self.max_prompt_length

        # Set bf16 default if not specified
        if self.bf16 is None and not self.fp16:
            self.bf16 = True

        # Validation
        if self.max_prompt_length >= self.max_length:
            raise ValueError(
                f"max_prompt_length ({self.max_prompt_length}) must be less than "
                f"max_length ({self.max_length})"
            )


# ============================================================================
# Utility Functions
# ============================================================================

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int = 0, dim: int = -1) -> torch.Tensor:
    """
    Pad a tensor to a given length along a dimension.

    Args:
        tensor: Input tensor to pad
        length: Target length
        pad_value: Value to use for padding
        dim: Dimension to pad along

    Returns:
        Padded tensor
    """
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute log softmax for selected indices efficiently.

    Instead of computing full log_softmax and then gathering,
    this directly computes log P(label) for each position.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]

    Returns:
        Log probabilities: [batch_size, seq_len]
    """
    # Compute log_softmax efficiently
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    # gather expects index shape to match output shape except in the gather dimension
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    return per_token_logps


# ============================================================================
# ORPO Trainer
# ============================================================================

class MerlinaORPOTrainer(Trainer):
    """
    Standalone ORPO Trainer for Merlina.

    Implements ORPO (Odds Ratio Preference Optimization) without requiring TRL.
    Inherits from transformers.Trainer for maximum compatibility.

    Expected dataset format (each row must have):
        - prompt: str or list (user input)
        - chosen: str or list (preferred response)
        - rejected: str or list (non-preferred response)
        - OR pre-tokenized versions:
          - chosen_input_ids, chosen_attention_mask, chosen_labels
          - rejected_input_ids, rejected_attention_mask, rejected_labels
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        args: Optional[ORPOConfig] = None,
        processing_class = None,  # Tokenizer
        **kwargs
    ):
        """
        Initialize ORPO Trainer.

        Args:
            model: The model to train
            args: Training arguments (ORPOConfig)
            processing_class: Tokenizer for the model
            **kwargs: Additional arguments passed to Trainer
        """
        if args is None:
            raise ValueError("args (ORPOConfig) must be provided")

        if not isinstance(args, ORPOConfig):
            raise ValueError("args must be an instance of ORPOConfig")

        # Extract peft_config if provided (Trainer doesn't accept it)
        peft_config = kwargs.pop('peft_config', None)

        # Store ORPO-specific config
        self.beta = args.beta
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length

        # Store tokenizer
        self.tokenizer = processing_class
        if self.tokenizer and self.padding_value is None:
            self.padding_value = self.tokenizer.pad_token_id or 0

        # Apply PEFT if config provided
        if peft_config is not None and model is not None:
            from peft import get_peft_model
            logger.info(f"Applying PEFT config: {peft_config}")
            model = get_peft_model(model, peft_config)

        # Disable dropout if requested
        if args.disable_dropout and model is not None:
            self._disable_dropout_in_model(model)

        # We only support decoder-only models for now
        self.is_encoder_decoder = False
        if model is not None and hasattr(model.config, 'is_encoder_decoder'):
            self.is_encoder_decoder = model.config.is_encoder_decoder
            if self.is_encoder_decoder:
                logger.warning(
                    "Encoder-decoder models are not fully tested in this implementation. "
                    "Use at your own risk or fall back to TRL's ORPOTrainer."
                )

        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            **kwargs
        )

        logger.info(f"Initialized MerlinaORPOTrainer with beta={self.beta}")

    @staticmethod
    def _disable_dropout_in_model(model: nn.Module) -> None:
        """Disable dropout in the model for training stability."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

    # ========================================================================
    # Core ORPO Logic
    # ========================================================================

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """
        Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            average_log_prob: If True, return average log prob per sequence
            label_pad_token_id: Token ID to ignore in loss computation
            is_encoder_decoder: Whether model is encoder-decoder

        Returns:
            Log probabilities per sequence: [batch_size]
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits shape {logits.shape[:-1]} and labels shape {labels.shape} must match "
                "(batch and sequence length dim)"
            )

        # For decoder-only models, shift labels and logits
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        # Create mask for non-padding tokens
        loss_mask = labels != label_pad_token_id

        # Replace padding tokens with 0 for gather operation
        labels = torch.where(labels == label_pad_token_id, 0, labels)

        # Compute log probabilities for each token
        per_token_logps = selective_log_softmax(logits, labels)

        # Sum over sequence length
        if average_log_prob:
            # Average log prob (normalized by sequence length)
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            # Total log prob (sum over sequence)
            return (per_token_logps * loss_mask).sum(-1)

    def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute ORPO's odds ratio (OR) loss for a batch of policy log probabilities.

        ORPO Loss = NLL_loss + β * E[log(σ(log_odds))]

        where:
            log_odds = log(P_chosen / (1 - P_chosen)) - log(P_rejected / (1 - P_rejected))
                     = log(P_chosen) - log(1 - P_chosen) - log(P_rejected) + log(1 - P_rejected)

        Args:
            policy_chosen_logps: Log probabilities for chosen responses [batch_size]
            policy_rejected_logps: Log probabilities for rejected responses [batch_size]

        Returns:
            Tuple of:
                - losses: OR loss per example [batch_size]
                - chosen_rewards: Reward for chosen responses
                - rejected_rewards: Reward for rejected responses
                - log_odds_ratio: Mean log odds ratio (for logging)
                - log_odds: Mean log odds (for logging)
        """
        # Compute log odds for chosen and rejected
        # log(p / (1-p)) = log(p) - log(1-p)
        # Use log1p for numerical stability: log(1-p) = log1p(-p) when p = exp(log_p)
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) -
            torch.log1p(-torch.exp(policy_rejected_logps))
        )

        # Apply sigmoid to log_odds (in log space for stability)
        # log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x)) = -softplus(-x)
        ratio = F.logsigmoid(log_odds)

        # Final loss (negative because we want to maximize ratio)
        losses = self.beta * ratio

        # Compute rewards for logging
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return (
            losses,
            chosen_rewards,
            rejected_rewards,
            torch.mean(ratio).detach(),
            torch.mean(log_odds).detach()
        )

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, torch.LongTensor],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """
        Concatenate chosen and rejected inputs into a single batch for efficient processing.

        This allows a single forward pass instead of two separate passes.

        Args:
            batch: Dictionary with chosen_* and rejected_* keys
            is_encoder_decoder: Whether model is encoder-decoder
            label_pad_token_id: Token ID for label padding
            padding_value: Value for padding input_ids
            device: Target device

        Returns:
            Dictionary with concatenated_* keys
        """
        concatenated_batch = {}

        # Determine max length
        if is_encoder_decoder:
            max_length = max(
                batch["chosen_labels"].shape[1],
                batch["rejected_labels"].shape[1]
            )
        else:
            max_length = max(
                batch["chosen_input_ids"].shape[1],
                batch["rejected_input_ids"].shape[1]
            )

        # Concatenate chosen tensors first
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                # Determine padding value based on key
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    pad_value = 0

                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )

        # Concatenate rejected tensors
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                # Determine padding value based on key
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    pad_value = 0

                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        # For encoder-decoder, repeat prompt inputs
        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = (
                batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.LongTensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Run forward pass on concatenated chosen and rejected inputs.

        Args:
            model: The model to run
            batch: Batch with chosen_* and rejected_* keys

        Returns:
            Tuple of:
                - chosen_logps: Log probabilities for chosen responses
                - rejected_logps: Log probabilities for rejected responses
                - chosen_nll_loss: NLL loss for chosen responses
        """
        # Concatenate inputs
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device if hasattr(self, 'accelerator') else None,
        )

        len_chosen = batch["chosen_labels"].shape[0]

        # Forward pass
        model_kwargs = {}
        if self.is_encoder_decoder:
            # For encoder-decoder, shift labels for decoder input
            model_kwargs["decoder_input_ids"] = self._shift_right(
                concatenated_batch["concatenated_labels"]
            )

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        # Compute NLL loss for chosen responses
        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift for decoder-only models
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        # Prepare labels
        if self.is_encoder_decoder:
            labels = concatenated_batch["concatenated_labels"].clone()
        else:
            labels = concatenated_batch["concatenated_input_ids"].clone()
            attention_mask = concatenated_batch["concatenated_attention_mask"]
            labels = torch.where(attention_mask == 1, labels, self.label_pad_token_id)

        # Compute NLL for chosen
        chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        # Compute log probabilities for all responses
        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,  # Use average for better stability
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        # Split into chosen and rejected
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        return chosen_logps, rejected_logps, chosen_nll_loss

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute ORPO loss.

        Loss = NLL_loss + β * OR_loss

        Args:
            model: The model being trained
            inputs: Batch of inputs
            return_outputs: Whether to return additional outputs
            num_items_in_batch: Number of items in batch (unused)

        Returns:
            Loss tensor, or tuple of (loss, metrics) if return_outputs=True
        """
        # Forward pass
        chosen_logps, rejected_logps, chosen_nll_loss = self.concatenated_forward(
            model, inputs
        )

        # Compute ORPO loss
        or_losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds = self.odds_ratio_loss(
            chosen_logps, rejected_logps
        )

        # Combine NLL and OR losses
        # Note: or_losses is negative (we want to maximize), so we subtract it
        # Total loss = NLL - β * log(sigmoid(log_odds))
        or_loss = -or_losses.mean()
        total_loss = chosen_nll_loss + or_loss

        # Prepare metrics for logging
        metrics = {
            "loss": total_loss.detach(),
            "nll_loss": chosen_nll_loss.detach(),
            "or_loss": or_loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "log_odds_ratio": log_odds_ratio,
            "log_odds": log_odds,
            "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
        }

        # Store metrics for logging (if supported)
        if hasattr(self, 'store_metrics'):
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (total_loss, metrics)
        return total_loss

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Shift input ids one token to the right (for encoder-decoder models).
        """
        decoder_start_token_id = self.model.config.decoder_start_token_id
        pad_token_id = self.model.config.pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # Replace possible -100 values with pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


# ============================================================================
# Helper function to create trainer (for easy migration)
# ============================================================================

def create_orpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    orpo_config: Optional[ORPOConfig] = None,
    **trainer_kwargs
) -> MerlinaORPOTrainer:
    """
    Helper function to create an ORPO trainer with sensible defaults.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        orpo_config: ORPO configuration (if None, uses defaults)
        **trainer_kwargs: Additional arguments passed to trainer

    Returns:
        Configured MerlinaORPOTrainer
    """
    if orpo_config is None:
        orpo_config = ORPOConfig(
            output_dir="./orpo_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=5e-6,
            logging_steps=10,
        )

    trainer = MerlinaORPOTrainer(
        model=model,
        args=orpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **trainer_kwargs
    )

    return trainer
