"""Per-architecture fixes that the generic training path can't infer.

DeepSeek-V4 is the first family that needs any:

- **No chat template.** The checkpoint ships a Python reference encoder (`encoding_dsv4.py`)
  instead of a Jinja template, and vLLM's `--tokenizer-mode deepseek_v4` renders with that
  encoder. `attach_chat_encoder` gives the tokenizer an `apply_chat_template` backed by the
  same encoder (vendored in `dataset_handlers/encoding_dsv4.py`), so the tokenizer formatter,
  the messages converter and the tool-call flattener all produce exactly what the server sees.
  The tokenizer's `chat_template` stays `None`, so nothing bogus is saved with the adapter.

- **LoRA targets.** Merlina's default targets (`q_proj`, `gate_proj`, ...) don't fit: the
  attention is `q_a_proj`/`q_b_proj`/`kv_proj`/`o_a_proj`/`o_b_proj`, the routed experts are
  fused 3-D parameters PEFT can't wrap, and bare `gate_proj`/`up_proj` also match the KV
  compressors and the indexer (so do bare `kv_proj`/`q_b_proj`). `o_a_proj` is a grouped (block-diagonal) linear whose forward
  reshapes its weight, so a LoRA wrapper around it computes the wrong thing.
  `resolve_target_modules` swaps in the attention + shared-expert set.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEEPSEEK_V4 = "deepseek_v4"

# Attention projections plus the always-on shared expert. Matched by PEFT as name suffixes.
DEEPSEEK_V4_LORA_TARGETS = [
    "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj", "self_attn.o_b_proj",
    "shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj",
]

# Mirrors TrainingConfig.target_modules' default.
GENERIC_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

# Bare names that would wrap the wrong DeepSeek-V4 modules, and what they should mean.
_DSV4_RENAMES = {
    "q_a_proj": "self_attn.q_a_proj",
    "q_b_proj": "self_attn.q_b_proj",
    "kv_proj": "self_attn.kv_proj",
    "o_b_proj": "self_attn.o_b_proj",
    "gate_proj": "shared_experts.gate_proj",
    "up_proj": "shared_experts.up_proj",
    "down_proj": "shared_experts.down_proj",
}
_DSV4_UNSUPPORTED = {"o_a_proj", "q_proj", "k_proj", "v_proj", "o_proj"}

THINKING_START = "<think>"
THINKING_END = "</think>"


def model_type_of(model_path: str) -> Optional[str]:
    """`model_type` from a checkpoint's config.json: a local dir, or a hub id already in the
    HF cache (nothing is downloaded). None when unknown."""
    path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if path is None:
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(model_path, "config.json")
            path = cached if isinstance(cached, str) else None
        except Exception:
            path = None
    if path is None:
        return None
    try:
        with open(path) as f:
            return json.load(f).get("model_type")
    except (OSError, ValueError, TypeError):
        return None


def resolve_target_modules(targets: Optional[list], model_type: Optional[str]) -> Optional[list]:
    """LoRA target list for this architecture. Non-DeepSeek-V4 targets pass through untouched."""
    if model_type != DEEPSEEK_V4:
        return targets
    if not targets or set(targets) == GENERIC_LORA_TARGETS:
        logger.info(f"deepseek_v4: using LoRA targets {DEEPSEEK_V4_LORA_TARGETS}")
        return list(DEEPSEEK_V4_LORA_TARGETS)
    resolved = []
    for t in targets:
        if t in _DSV4_UNSUPPORTED:
            logger.warning(f"deepseek_v4: dropping LoRA target {t!r} (absent, or a grouped linear LoRA can't wrap)")
            continue
        t = _DSV4_RENAMES.get(t, t)
        if t not in resolved:
            resolved.append(t)
    if not resolved:
        raise ValueError(f"No usable deepseek_v4 LoRA targets in {targets}; try {DEEPSEEK_V4_LORA_TARGETS}")
    return resolved


# ── chat encoding ─────────────────────────────────────────────────────────────

def _split_think(content: str) -> tuple[Optional[str], str]:
    """`<think>r</think>answer` -> ("r", "answer"); anything else -> (None, content)."""
    s = content.lstrip()
    if s.startswith(THINKING_START) and THINKING_END in s:
        reasoning, answer = s[len(THINKING_START):].split(THINKING_END, 1)
        return reasoning.strip("\n"), answer.lstrip("\n")
    return None, content


def _to_encoder_messages(conversation: list, tools: Optional[list]) -> list:
    msgs = []
    for m in conversation:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("role") == "assistant":
            if isinstance(m["content"], str) and not m.get("reasoning_content"):
                reasoning, answer = _split_think(m["content"])
                if reasoning is not None:
                    m["reasoning_content"], m["content"] = reasoning, answer
            calls = []
            for call in m.get("tool_calls") or []:
                fn = dict(call.get("function") or call)
                if not isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.dumps(fn.get("arguments") or {}, ensure_ascii=False)
                calls.append({**call, "function": fn})
            if calls:
                m["tool_calls"] = calls
        msgs.append(m)
    if tools:
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["tools"] = tools
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    return msgs


def encode_deepseek_v4(conversation: list, tools: Optional[list] = None, thinking: bool = False,
                       continue_final_message: bool = False) -> str:
    """Render a conversation the way vLLM's deepseek_v4 tokenizer mode does.

    The encoder always ends a trailing user turn with the assistant prefix (`<｜Assistant｜></think>`
    in chat mode, `<｜Assistant｜><think>` in thinking mode), so `add_generation_prompt` has no
    separate effect.
    """
    from dataset_handlers.encoding_dsv4 import encode_messages
    msgs = _to_encoder_messages(conversation, tools)
    if continue_final_message and msgs and msgs[-1].get("role") == "assistant":
        msgs[-1]["wo_eos"] = True
    return encode_messages(msgs, thinking_mode="thinking" if thinking else "chat")


def _apply_chat_template(self, conversation, tools=None, add_generation_prompt=False, tokenize=True,
                         continue_final_message=False, return_dict=False, enable_thinking=None,
                         thinking=None, chat_template_kwargs=None, **kwargs):
    if conversation and isinstance(conversation[0], (list, tuple)):
        return [_apply_chat_template(self, c, tools=tools, add_generation_prompt=add_generation_prompt,
                                     tokenize=tokenize, continue_final_message=continue_final_message,
                                     return_dict=return_dict, enable_thinking=enable_thinking,
                                     thinking=thinking, chat_template_kwargs=chat_template_kwargs)
                for c in conversation]
    extra = chat_template_kwargs or {}
    on = enable_thinking if enable_thinking is not None else thinking
    if on is None:
        on = extra.get("enable_thinking", extra.get("thinking", False))
    text = encode_deepseek_v4(list(conversation), tools=tools, thinking=bool(on),
                              continue_final_message=continue_final_message)
    if not tokenize:
        return text
    ids = self.encode(text, add_special_tokens=False)   # BOS is already in the text
    if return_dict:
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}
    return ids


def _rebuild(base: type, state: dict) -> Any:
    tokenizer = base.__new__(base)
    tokenizer.__dict__.update(state)
    tokenizer.__dict__.pop("merlina_chat_encoder", None)
    return attach_chat_encoder(tokenizer, model_type=DEEPSEEK_V4)


def _reduce(self):
    # The swapped-in class is built at runtime; pickle as the base class and re-attach.
    return _rebuild, (type(self).__mro__[1], dict(self.__dict__))


_SUBCLASSES: dict = {}


def has_chat_encoder(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "merlina_chat_encoder", None))


def attach_chat_encoder(tokenizer: Any, model_path: Optional[str] = None,
                        model_type: Optional[str] = None) -> Any:
    """Give a DeepSeek-V4 tokenizer without a chat template the reference encoder. Returns it.

    Swaps the instance onto a subclass rather than wrapping it, so `isinstance` checks in the
    trainer still see a PreTrainedTokenizer.
    """
    model_type = model_type or (model_type_of(model_path) if model_path else None)
    if model_type != DEEPSEEK_V4 or getattr(tokenizer, "chat_template", None) or has_chat_encoder(tokenizer):
        return tokenizer
    base = type(tokenizer)
    if base not in _SUBCLASSES:
        _SUBCLASSES[base] = type(f"DeepseekV4{base.__name__}", (base,),
                                 {"apply_chat_template": _apply_chat_template, "__reduce__": _reduce})
    tokenizer.__class__ = _SUBCLASSES[base]
    tokenizer.merlina_chat_encoder = DEEPSEEK_V4
    logger.info("deepseek_v4: chat rendering uses the DeepSeek-V4 reference encoder")
    return tokenizer
