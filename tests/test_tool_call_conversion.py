"""Tool-calling datasets must survive flattening in the target model's own dialect.

Three defects sat behind a single crash on a tool-calling dataset:

1. `msg.get('content', '')` returns None when the key exists and is null, which every assistant
   tool-call turn is. The join then died with "sequence item 0: expected str instance, NoneType
   found" and no dataset in this shape could be loaded at all.
2. With that fixed, `tool_calls` were dropped, so the assistant target was empty and the row was
   rejected as having no assistant messages -- the tool call *was* the training signal.
3. The `tools` column was dropped, so the model would have been trained to call functions it was
   never shown.

The fix renders every piece through the target tokenizer's chat template. That matters more than
it sounds: the obvious encoding is `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, which
is what Hermes and Qwen2 use and what the source data looks like, but Qwen3.5 emits a nested
`<function=name><parameter=key>` form. Train the JSON form, prompt the XML form, and the model's
calls never parse -- with a completely healthy-looking loss curve.

The guarantee under test is exact: re-templating the flattened strings reproduces the native
rendering byte for byte.

`tests/conftest.py` replaces `transformers` with mocks, so a real tokenizer is unavailable here
and a MagicMock's `apply_chat_template` returns a mock rather than text. These use a stub whose
template mirrors Qwen3.5's actual quirks -- the empty `<think>` block, tools prepended to system,
tool results folded into a user turn, and a hard error when no user turn is present. Those quirks
are what the slot inference has to survive, so a simpler stub would pass while the real thing
failed.
"""

import json

import pytest

from dataset_handlers.messages_converter import convert_messages_to_standard
from dataset_handlers.tool_calls import (
    ToolFlattenError,
    row_has_tools,
    verify_roundtrip,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "take_action",
        "description": "Take one legal action.",
        "parameters": {
            "type": "object",
            "properties": {"action_id": {"type": "string"}},
            "required": ["action_id"],
        },
    },
}]


class StubTokenizer:
    """A chat template with Qwen3.5's shape, including the parts that break naive flattening."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, **kwargs):
        if not any(m.get("role") in ("user", "tool") for m in messages):
            raise ValueError("No user query found in messages.")

        out = []
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""

            if role == "system":
                if tools:
                    block = "\n".join(json.dumps(t) for t in tools)
                    content = f"# Tools\n\n<tools>\n{block}\n</tools>\n\n{content}"
                out.append(f"<|im_start|>system\n{content}<|im_end|>\n")

            elif role == "user":
                out.append(f"<|im_start|>user\n{content}<|im_end|>\n")

            elif role == "tool":
                # Folded into a user turn, exactly as Qwen3.5 does.
                out.append(
                    f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
                )

            elif role == "assistant":
                body = content
                for call in m.get("tool_calls") or []:
                    fn = call.get("function", call)
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        raise TypeError("Can only get item pairs from a mapping.")
                    params = "".join(
                        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()
                    )
                    body += (
                        f"<tool_call>\n<function={fn['name']}>\n{params}</function>\n</tool_call>"
                    )
                # The empty think block is prefix, so it must cancel in slot inference.
                out.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{body}<|im_end|>\n")

        return "".join(out)


@pytest.fixture
def tok():
    return StubTokenizer()


def tool_row(arguments):
    return {
        "messages": [
            {"role": "system", "content": "You play one seat."},
            {"role": "user", "content": "Your turn."},
            {"role": "assistant", "content": None,
             "tool_calls": [{"type": "function",
                             "function": {"name": "take_action", "arguments": arguments}}]},
        ],
        "tools": TOOLS,
    }


def test_detects_rows_carrying_tools():
    assert row_has_tools(tool_row({"action_id": "a7"}))
    assert not row_has_tools({"messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})


def test_detects_tool_result_turns_without_a_tools_column():
    assert row_has_tools({"messages": [{"role": "tool", "content": "ok"}]})


def test_plain_conversation_still_converts_without_a_tokenizer():
    """The common path must not start requiring a tokenizer."""
    out = convert_messages_to_standard({"messages": [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]})
    assert out == {"system": "Be helpful", "prompt": "Hi", "chosen": "Hello!"}


def test_null_content_does_not_crash_the_join():
    """A null content on a non-tool turn is a value, not a missing key."""
    out = convert_messages_to_standard({"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": None},
    ]})
    assert out["chosen"] == "Hello"


def test_refuses_to_guess_a_dialect_without_a_tokenizer():
    """Silently picking an encoding here trains calls the model will never emit."""
    with pytest.raises(ToolFlattenError, match="tokenizer"):
        convert_messages_to_standard(tool_row({"action_id": "a7"}), tokenizer=None)


def test_tool_call_is_rendered_in_the_templates_dialect(tok):
    out = convert_messages_to_standard(tool_row({"action_id": "a7"}), tok)
    assert "<function=take_action>" in out["chosen"]
    assert "a7" in out["chosen"]
    # The JSON encoding would be wrong for this template.
    assert '{"name"' not in out["chosen"]


def test_the_empty_think_block_is_not_absorbed_into_the_target(tok):
    """It is template scaffolding, not content; training on it teaches a literal prefix."""
    out = convert_messages_to_standard(tool_row({"action_id": "a7"}), tok)
    assert "<think>" not in out["chosen"]


def test_tools_reach_the_system_message(tok):
    out = convert_messages_to_standard(tool_row({"action_id": "a7"}), tok)
    assert "take_action" in out["system"], "tool schemas dropped; model trained blind to them"
    assert "You play one seat." in out["system"], "original system text lost"


def test_roundtrip_is_byte_exact(tok):
    row = tool_row({"action_id": "a7"})
    assert verify_roundtrip(tok, row, convert_messages_to_standard(row, tok))


def test_string_encoded_arguments_are_accepted(tok):
    """Datasets commonly store arguments as a JSON string; templates want a mapping."""
    row = tool_row('{"action_id": "a7"}')
    out = convert_messages_to_standard(row, tok)
    assert "a7" in out["chosen"]
    assert verify_roundtrip(tok, row, out)


def test_tool_results_are_preserved_in_the_prompt(tok):
    row = {
        "messages": [
            {"role": "user", "content": "Your turn."},
            {"role": "assistant", "content": None,
             "tool_calls": [{"type": "function",
                             "function": {"name": "take_action",
                                          "arguments": {"action_id": "a7"}}}]},
            {"role": "tool", "content": "you moved"},
            {"role": "assistant", "content": "Done."},
        ],
        "tools": TOOLS,
    }
    out = convert_messages_to_standard(row, tok)
    assert "you moved" in out["prompt"], "tool result dropped from the conversation"
    assert "Done." in out["chosen"]


def test_assistant_text_alongside_a_call_is_kept(tok):
    row = tool_row({"action_id": "a7"})
    row["messages"][2]["content"] = "Taking the stars track."
    out = convert_messages_to_standard(row, tok)
    assert "Taking the stars track." in out["chosen"]
    assert "<function=take_action>" in out["chosen"]


def test_a_template_that_swallows_content_is_reported_not_guessed(tok):
    class Swallowing(StubTokenizer):
        def apply_chat_template(self, messages, tools=None, **kw):
            return "<|im_start|>assistant\nfixed<|im_end|>\n"

    with pytest.raises(ToolFlattenError):
        convert_messages_to_standard(tool_row({"action_id": "a7"}), Swallowing())


def test_pipeline_uses_its_own_tokenizer_not_the_formatters():
    """The tokenizer must reach conversion even when the formatter does not want one.

    `create_dataset_pipeline` only hands a tokenizer to the formatter for `format_type='tokenizer'`,
    and the default is `chatml`. Reading it off the formatter therefore found nothing on the
    default path, and a tool-calling dataset failed with "set model_name..." even though
    model_name had been set. Caught against a live server after the first fix shipped.
    """
    from dataset_handlers.base import DatasetPipeline

    class Loader:
        def load(self):
            raise AssertionError("not needed")

        def get_source_info(self):
            return {}

    pipeline = DatasetPipeline(
        loader=Loader(),
        formatter=object(),  # no .tokenizer attribute, like ChatMLFormatter
        tokenizer=StubTokenizer(),
    )
    assert pipeline.tokenizer is not None

    row = tool_row({"action_id": "a7"})
    out = convert_messages_to_standard(row, pipeline.tokenizer)
    assert "<function=take_action>" in out["chosen"]
