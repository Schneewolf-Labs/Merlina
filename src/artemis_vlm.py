"""Schneewolf Labs — Project Artemis: ArtemisVLM.

A LLaVA-style vision-language wrapper that grafts Qwen3-VL's vision stack
onto an *unmodified* Schneewolf Labs A-series (Mistral) decoder.

Data flow (Path B):
    image
      -> Qwen3VLVisionModel (SigLIP-2 ViT + internal patch merger)
      -> .pooler_output            # merged: (sum_image_tokens, vision out_hidden)
      -> 2-layer MLP projector     # vision out_hidden -> text hidden
      -> spliced into text embeds at <|image_pad|> (image_token_id) positions
      -> A2 decoder (vanilla 1-D RoPE, byte-for-byte unchanged) -> logits

Deliberately NOT Qwen3-VL's decoder: no Interleaved-MRoPE, no DeepStack.
The vision tower's `deepstack_features` are intentionally ignored.

Scope: this module is Project Artemis piece #1 (model class + config).
Not yet here (tracked separately): image processor adaptation, multimodal
collator / Merlina data path, staged-freeze training integration, and
`prepare_inputs_for_generation` for multi-step `generate()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


class ArtemisVLMConfig(PretrainedConfig):
    """Composite config: a Qwen3-VL vision tower + an A-series (Mistral) text decoder."""

    model_type = "artemis_vlm"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id: int = 22,   # repurposed <|image_pad|> in the A-series tokenizer
        video_token_id: int = 23,   # repurposed <|video_pad|>
        projector_hidden_act: str = "gelu",
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Qwen3VLVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Qwen3VLVisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = AutoConfig.for_model("mistral")
        elif isinstance(text_config, dict):
            text_config = AutoConfig.for_model(**text_config)
        self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.projector_hidden_act = projector_hidden_act
        super().__init__(**kwargs)


class ArtemisVLMProjector(nn.Module):
    """Fresh 2-layer MLP bridging vision out_hidden -> text hidden. Trained from scratch
    (Stage-1 alignment) — Qwen3-VL's own merger output dim (3584) != A2 hidden (5120),
    so there is no warm-start for this module by construction."""

    def __init__(self, in_dim: int, out_dim: int, act: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = ACT2FN[act]
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ArtemisVLMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = ArtemisVLMConfig
    base_model_prefix = "artemis"
    _no_split_modules = ["Qwen3VLVisionBlock", "MistralDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: ArtemisVLMConfig, vision_model=None, language_model=None):
        super().__init__(config)
        # Pre-built submodules may be injected (assembly path) to avoid
        # double-instantiating a 12B decoder; otherwise build from config
        # (the from_pretrained path).
        self.visual = vision_model if vision_model is not None else Qwen3VLVisionModel(config.vision_config)
        self.language_model = (
            language_model if language_model is not None
            else AutoModelForCausalLM.from_config(config.text_config)
        )
        self.multi_modal_projector = ArtemisVLMProjector(
            config.vision_config.out_hidden_size,
            config.text_config.hidden_size,
            config.projector_hidden_act,
        )
        self.vocab_size = config.text_config.vocab_size

    # --- embedding passthrough (delegate to the A-series decoder) ---
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # --- vision path (mirrors transformers Qwen3VLModel.get_image_features,
    #     but uses only the MERGED pooler_output; DeepStack ignored) ---
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        image_embeds = vision_output.pooler_output            # merged: (sum_tokens, out_hidden)
        return self.multi_modal_projector(image_embeds)        # (sum_tokens, text_hidden)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            mask = input_ids == self.config.image_token_id
            n_tokens = int(mask.sum())
            if n_tokens != image_features.shape[0]:
                raise ValueError(
                    f"Image placeholder tokens ({n_tokens}) != image features "
                    f"({image_features.shape[0]}). The processor's <|image_pad|> "
                    f"expansion must equal grid.prod()//spatial_merge_size**2."
                )
            mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    # --- staged-freeze control (Project Artemis training recipe) ---
    def set_training_stage(self, stage: str, unfreeze_vision_top_n: int = 0):
        """Stage-1: train only the projector (freeze ViT + A2) — connector alignment.
        Stage-2: + A2 full fine-tune (projector stays trainable); ViT frozen unless
        `unfreeze_vision_top_n` top blocks are opened. Returns (#trainable, #total)."""
        if stage not in ("stage1", "stage2"):
            raise ValueError("stage must be 'stage1' or 'stage2'")
        for p in self.visual.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = (stage == "stage2")
        for p in self.multi_modal_projector.parameters():
            p.requires_grad = True
        if unfreeze_vision_top_n > 0:
            blocks = getattr(self.visual, "blocks", None)
            if blocks is not None:
                for blk in list(blocks)[-unfreeze_vision_top_n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tr, tot

    # --- generation: inject the image only on the first step ---
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        pixel_values=None, image_grid_thw=None, **kwargs
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, **kwargs
        )
        if past_key_values is None:           # step 0 only — image is in the prompt
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw
        else:
            model_inputs["pixel_values"] = None
            model_inputs["image_grid_thw"] = None
        return model_inputs

    @classmethod
    def from_a2_and_vision(
        cls,
        a2_path: str,
        vision_model: Qwen3VLVisionModel | None = None,
        vision_config: Qwen3VLVisionConfig | None = None,
        image_token_id: int = 22,
        video_token_id: int = 23,
        torch_dtype=torch.bfloat16,
    ) -> "ArtemisVLMForConditionalGeneration":
        """Assemble ArtemisVLM from a trained A-series checkpoint + a Qwen3-VL vision
        tower (pretrained module passed in, or random from `vision_config`).

        Loads the 12B decoder exactly once (no random double-instantiation)."""
        language_model = AutoModelForCausalLM.from_pretrained(a2_path, dtype=torch_dtype)
        if vision_model is None:
            vision_model = Qwen3VLVisionModel(vision_config or Qwen3VLVisionConfig())
        config = ArtemisVLMConfig(
            vision_config=vision_model.config.to_dict(),
            text_config=language_model.config.to_dict(),
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        model = cls(config, vision_model=vision_model, language_model=language_model)
        model.multi_modal_projector.to(language_model.device, torch_dtype)
        model.visual.to(language_model.device, torch_dtype)
        return model


def artemis_loss_fn(model, batch, training: bool = True):
    """grimoire-compatible loss_fn for Artemis training.

    The collator already emits `labels` (prompt + <|image_pad|> masked), so the
    transformers CausalLM computes the LM loss internally — no separate
    tokenization/loss path needed. Returns (loss, metrics) like grimoire losses.
    """
    out = model(**batch)
    return out.loss, {"nll_loss": out.loss.detach().item()}


class ArtemisVLMProcessor:
    """Image+text processor for ArtemisVLM.

    Wraps Qwen2-VL's image processor (which Qwen3-VL itself reuses) and the
    A-series tokenizer. Critically, patch/temporal/merge sizes are sourced from
    the model's `vision_config` so the processor's `<|image_pad|>` expansion can
    never drift from the model's merged feature count (the contract the model's
    forward enforces).

    Expansion mirrors transformers' `Qwen3VLProcessor.__call__`: each single
    `<|image_pad|>` (the chat template emits one per image, framed by
    `<|vision_start|>`/`<|vision_end|>`) is replaced by
    `grid_thw[i].prod() // merge_size**2` copies.
    """

    def __init__(self, tokenizer, vision_config, image_token="<|image_pad|>",
                 video_token="<|video_pad|>", min_pixels=32 * 32,
                 max_pixels=512 * 512):
        from transformers import Qwen2VLImageProcessor

        self.tokenizer = tokenizer
        # patch/temporal/merge from vision_config (can't drift from the model);
        # min/max_pixels cap dynamic-resolution token blow-up on large images
        # (e.g. 2560x1920 uncapped -> ~4800 image tokens).
        self.image_processor = Qwen2VLImageProcessor(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            merge_size=vision_config.spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.merge_length = vision_config.spatial_merge_size ** 2
        self.image_token = image_token
        self.video_token = video_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        from transformers.feature_extraction_utils import BatchFeature

        image_inputs = {}
        grid = None
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
            grid = image_inputs["image_grid_thw"]

        if text is None:
            return BatchFeature(data=image_inputs)
        if isinstance(text, str):
            text = [text]

        if grid is not None:
            idx = 0
            out_text = []
            for t in text:
                while self.image_token in t:
                    n = int(grid[idx].prod()) // self.merge_length
                    t = t.replace(self.image_token, "<|placeholder|>" * n, 1)
                    idx += 1
                out_text.append(t.replace("<|placeholder|>", self.image_token))
            text = out_text

        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        return BatchFeature(data={**text_inputs, **image_inputs})


class ArtemisDataCollator:
    """Multimodal collator for ArtemisVLM training.

    Each feature is a dict::

        {"images": [PIL.Image, ...], "messages": [chat turns ...]}

    where `messages` is a chat list (image placeholders + text); the final
    turn (assistant) is the training target. Produces a batch consumable by
    `ArtemisVLMForConditionalGeneration.forward`:

      input_ids / attention_mask / labels  -- right-padded across the batch
                                               (prompt + <|image_pad|> = -100)
      pixel_values                          -- flat concat over all images
      image_grid_thw                        -- (num_images_in_batch, 3)

    Label masking mirrors the validated text path (prefix trick): tokenize the
    prompt (everything but the last assistant turn, with `add_generation_prompt`)
    to get its length under the *same* image expansion, mask that prefix, and
    additionally mask every `<|image_pad|>` position (vision input, not a target).
    """

    def __init__(self, processor: "ArtemisVLMProcessor", label_pad: int = -100):
        self.proc = processor
        self.tok = processor.tokenizer
        self.label_pad = label_pad
        self.img_id = processor.image_token_id
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def _ids(self, messages, images, add_gen):
        text = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen
        )
        b = self.proc(text=text, images=images, return_tensors="pt")
        return b

    def __call__(self, features):
        import torch

        seqs, labels, pvs, grids = [], [], [], []
        for f in features:
            msgs, imgs = f["messages"], f.get("images")
            full = self._ids(msgs, imgs, add_gen=False)
            prompt = self._ids(msgs[:-1], imgs, add_gen=True)  # everything but target
            ids = full["input_ids"][0]
            plen = prompt["input_ids"].shape[1]
            lab = ids.clone()
            lab[:plen] = self.label_pad                          # mask prompt
            lab[ids == self.img_id] = self.label_pad             # mask image placeholders
            seqs.append(ids)
            labels.append(lab)
            if "pixel_values" in full:
                pvs.append(full["pixel_values"])
                grids.append(full["image_grid_thw"])

        maxlen = max(s.size(0) for s in seqs)
        input_ids, attn, lbl = [], [], []
        for s, l in zip(seqs, labels):
            pad = maxlen - s.size(0)
            input_ids.append(torch.cat([s, torch.full((pad,), self.pad_id, dtype=s.dtype)]))
            attn.append(torch.cat([torch.ones(s.size(0), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
            lbl.append(torch.cat([l, torch.full((pad,), self.label_pad, dtype=l.dtype)]))
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(lbl),
        }
        if pvs:
            batch["pixel_values"] = torch.cat(pvs, dim=0)        # flat over all images
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        return batch
