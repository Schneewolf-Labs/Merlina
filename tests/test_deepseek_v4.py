"""DeepSeek-V4: chat rendering through the reference encoder, and LoRA target selection.

DeepSeek-V4 ships no Jinja chat template, only `encoding_dsv4.py`, which is what vLLM's
deepseek_v4 tokenizer mode renders with. Training data must come out byte-identical to that,
or the adapter learns a format the server never shows it. `transformers` is mocked in this
suite, so the tokenizer here is a stub class; scripts/smoke_deepseek_v4.py runs the same path
against the real tokenizer and a tiny real model.
"""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset_handlers.encoding_dsv4 import encode_messages
from dataset_handlers.formatters import get_formatter
from dataset_handlers.messages_converter import convert_messages_to_standard
from dataset_handlers.tool_calls import verify_roundtrip
from src.model_families import (
    DEEPSEEK_V4_LORA_TARGETS,
    attach_chat_encoder,
    encode_deepseek_v4,
    has_chat_encoder,
    model_type_of,
    resolve_target_modules,
)

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v4"
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
ASSISTANT = "<｜Assistant｜>"

TOOLS = [{"type": "function", "function": {
    "name": "execute_command", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


class StubTokenizer:
    """Stands in for PreTrainedTokenizerFast: no chat template, byte-ish encode."""
    chat_template = None
    eos_token = EOS

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 256 for c in text]


@pytest.fixture
def tok():
    return attach_chat_encoder(StubTokenizer(), model_type="deepseek_v4")


# ── encoder ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", [1, 2])
def test_vendored_encoder_matches_upstream_goldens(case):
    data = json.loads((FIXTURES / f"test_input_{case}.json").read_text())
    if isinstance(data, dict):
        messages = data["messages"]
        messages[0]["tools"] = data["tools"]
    else:
        messages = data
    gold = (FIXTURES / f"test_output_{case}.txt").read_text()
    assert encode_messages(messages, thinking_mode="thinking") == gold


def test_openai_tools_kwarg_lands_on_the_system_turn():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    with_kwarg = encode_deepseek_v4(msgs, tools=TOOLS)
    native = encode_messages([{**msgs[0], "tools": TOOLS}, msgs[1]], thinking_mode="chat")
    assert with_kwarg == native and "## Tools" in with_kwarg


def test_think_block_in_content_becomes_reasoning():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>\nr\n</think>\n\nanswer"}]
    assert encode_deepseek_v4(msgs, thinking=True) == f"{BOS}<｜User｜>q{ASSISTANT}<think>r</think>answer{EOS}"


def test_dict_arguments_are_serialized():
    call = {"id": "c1", "type": "function",
            "function": {"name": "execute_command", "arguments": {"command": "ls"}}}
    text = encode_deepseek_v4([{"role": "user", "content": "q"},
                               {"role": "assistant", "content": None, "tool_calls": [call]}], tools=TOOLS)
    assert '<｜DSML｜parameter name="command" string="true">ls</｜DSML｜parameter>' in text


# ── tokenizer hook ────────────────────────────────────────────────────────────

def test_attach_only_for_deepseek_v4_without_template():
    other = attach_chat_encoder(StubTokenizer(), model_type="llama")
    assert not has_chat_encoder(other) and type(other) is StubTokenizer
    templated = StubTokenizer()
    templated.chat_template = "{{ messages }}"
    assert not has_chat_encoder(attach_chat_encoder(templated, model_type="deepseek_v4"))


def test_apply_chat_template(tok):
    assert isinstance(tok, StubTokenizer) and tok.chat_template is None
    msgs = [{"role": "user", "content": "hi"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text == f"{BOS}<｜User｜>hi{ASSISTANT}</think>"
    assert tok.apply_chat_template(msgs, tokenize=False, enable_thinking=True).endswith(f"{ASSISTANT}<think>")
    assert tok.apply_chat_template(msgs, tokenize=False,
                                   chat_template_kwargs={"thinking": True}).endswith("<think>")
    assert tok.apply_chat_template(msgs) == tok.encode(text)
    assert tok.apply_chat_template(msgs, return_dict=True)["input_ids"] == tok.encode(text)


def test_pickles_with_the_encoder(tok):
    back = pickle.loads(pickle.dumps(tok))
    assert has_chat_encoder(back)
    assert back.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False).endswith("</think>")


def test_model_type_of(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    assert model_type_of(str(tmp_path)) == "deepseek_v4"
    assert model_type_of(str(tmp_path / "missing")) is None


# ── formatter / tool flattening ───────────────────────────────────────────────

def test_tokenizer_formatter_chat_row(tok):
    out = get_formatter("tokenizer", tokenizer=tok).format(
        {"system": "S", "prompt": "Q", "chosen": "yes", "rejected": "no"})
    assert out == {"prompt": f"{BOS}S<｜User｜>Q{ASSISTANT}</think>", "chosen": f"yes{EOS}", "rejected": f"no{EOS}"}


def test_tokenizer_formatter_reasoning_row(tok):
    out = get_formatter("tokenizer", tokenizer=tok).format(
        {"prompt": "Q", "chosen": "yes", "rejected": "no", "reasoning": "because"})
    assert out["prompt"].endswith(f"{ASSISTANT}<think>")
    assert out["chosen"] == f"because</think>yes{EOS}"


def test_tool_call_turn_round_trips(tok):
    """The flattener renders tool schemas and DSML calls through the encoder, byte for byte.

    (It joins all user turns, then all assistant turns, so multi-step trajectories don't
    round-trip for any template; pre-render those and train with format 'raw'.)
    """
    row = {"tools": TOOLS, "messages": [
        {"role": "system", "content": "You are Kira."},
        {"role": "user", "content": "clean up ~/dev/logs"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "execute_command", "arguments": '{"command": "ls -la ~/dev/logs"}'}}]},
    ]}
    flat = convert_messages_to_standard(row, tokenizer=tok)
    assert verify_roundtrip(tok, row, flat)
    assert "## Tools" in flat["system"]
    assert flat["chosen"].startswith("\n\n<｜DSML｜tool_calls>")   # EOS comes from the formatter


# ── LoRA targets ──────────────────────────────────────────────────────────────

def test_default_targets_swap_to_the_deepseek_v4_set():
    generic = ["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"]
    assert resolve_target_modules(generic, "deepseek_v4") == DEEPSEEK_V4_LORA_TARGETS
    assert resolve_target_modules(None, "deepseek_v4") == DEEPSEEK_V4_LORA_TARGETS
    assert resolve_target_modules(generic, "llama") == generic


def test_explicit_targets_are_scoped_and_unsafe_ones_dropped():
    assert resolve_target_modules(["kv_proj", "o_a_proj", "gate_proj"], "deepseek_v4") == [
        "self_attn.kv_proj", "shared_experts.gate_proj"]
    with pytest.raises(ValueError):
        resolve_target_modules(["o_a_proj"], "deepseek_v4")


# ── pre-flight ────────────────────────────────────────────────────────────────

def _cfg(path, fmt):
    return SimpleNamespace(base_model=str(path), use_lora=True, use_4bit=False,
                           dataset=SimpleNamespace(format=SimpleNamespace(format_type=fmt)),
                           remote=None)


def test_preflight_requires_tokenizer_format(tmp_path, monkeypatch):
    from src.preflight_checks import PreflightValidator
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    v = PreflightValidator()
    monkeypatch.setattr(v, "_is_remote", lambda config: True)
    v._check_model_family(_cfg(tmp_path, "chatml"))
    assert any("format 'tokenizer'" in e for e in v.errors)
    v = PreflightValidator()
    monkeypatch.setattr(v, "_is_remote", lambda config: True)
    v._check_model_family(_cfg(tmp_path, "tokenizer"))
    v._check_model_family(_cfg(tmp_path, "raw"))
    assert not v.errors
