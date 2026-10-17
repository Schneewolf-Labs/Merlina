#!/usr/bin/env python3
"""CPU-only smoke test for DeepSeek-V4 training support (needs transformers>=5.20.1, peft).

The unit suite mocks transformers, so this runs the parts it can't:

1. A tiny randomly initialised DeepseekV4ForCausalLM with Merlina's default LoRA targets
   resolved for deepseek_v4: checks exactly which modules get adapters (no compressor,
   indexer or grouped o_a_proj) and that a forward/backward pass only produces LoRA grads.
2. With --tokenizer <checkpoint dir>: the real tokenizer with the reference encoder attached
   renders a chat + tool conversation, and the encoded ids decode back to the same text
   (special tokens included), i.e. the strings Merlina trains on tokenize the way vLLM's do.

    python scripts/smoke_deepseek_v4.py [--tokenizer /path/to/DeepSeek-V4-Flash]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from src.model_families import attach_chat_encoder, resolve_target_modules  # noqa: E402

GENERIC = ["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"]
EXPECTED = {"self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj", "self_attn.o_b_proj",
            "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj"}


def tiny_model():
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    config = DeepseekV4Config(
        vocab_size=512, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, head_dim=32,
        qk_rope_head_dim=8, q_lora_rank=32, o_groups=2, o_lora_rank=16, n_routed_experts=8,
        num_experts_per_tok=2, moe_intermediate_size=32, index_n_heads=2, index_head_dim=16,
        index_topk=8, sliding_window=8, max_position_embeddings=256, num_nextn_predict_layers=0,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        layer_types=["compressed_sparse_attention", "heavily_compressed_attention",
                     "compressed_sparse_attention"],
        mlp_layer_types=["hash_moe", "moe", "moe"],
    )
    torch.manual_seed(0)
    return DeepseekV4ForCausalLM(config)


def check_lora():
    model = tiny_model()
    targets = resolve_target_modules(GENERIC, model.config.model_type)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                                  task_type="CAUSAL_LM"))
    wrapped = {n.split(".layers.")[1].split(".", 1)[1].rsplit(".lora_A", 1)[0]
               for n, _ in peft_model.named_parameters() if ".lora_A." in n}
    assert wrapped == EXPECTED, f"unexpected LoRA modules: {sorted(wrapped ^ EXPECTED)}"
    ids = torch.randint(0, 512, (2, 40))
    out = peft_model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = [n for n, p in peft_model.named_parameters() if p.grad is not None]
    assert grads and all("lora_" in n for n in grads), "gradients outside the adapter"
    print(f"lora ok: {len(wrapped)} module kinds, loss {out.loss.item():.3f}, {len(grads)} grads")


def check_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = attach_chat_encoder(AutoTokenizer.from_pretrained(path), path)
    assert getattr(tok, "merlina_chat_encoder", None), "encoder not attached (not a deepseek_v4 dir?)"
    tools = [{"type": "function", "function": {"name": "execute_command", "description": "Run a command.",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    conv = [{"role": "system", "content": "You are Kira."},
            {"role": "user", "content": "清理一下 ~/dev/logs"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {
                "name": "execute_command", "arguments": '{"command": "ls ~/dev/logs"}'}}]},
            {"role": "tool", "tool_call_id": "a", "content": "README.txt\napp.log"},
            {"role": "assistant", "content": "A README and a log. Delete both?"}]
    text = tok.apply_chat_template(conv, tools=tools, tokenize=False)
    ids = tok.apply_chat_template(conv, tools=tools)
    assert tok.decode(ids) == text, "ids don't decode back to the rendered text"
    specials = {t: tok.convert_tokens_to_ids(t) for t in ("<｜begin▁of▁sentence｜>", "<｜User｜>",
                                                          "<｜Assistant｜>", "</think>", "<｜end▁of▁sentence｜>")}
    assert all(i is not None and i != tok.unk_token_id for i in specials.values()), specials
    for t, i in specials.items():
        assert i in ids, f"{t} not encoded as a single special token"
    print(f"tokenizer ok: {len(ids)} ids, special tokens {specials}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", help="DeepSeek-V4 checkpoint dir (only tokenizer files are read)")
    args = ap.parse_args()
    check_lora()
    if args.tokenizer:
        check_tokenizer(args.tokenizer)
