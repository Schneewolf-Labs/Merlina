"""
Test script for the chat template embedding feature.

When a model doesn't have a chat_template and the user selects a specific format
(e.g. chatml, llama3, mistral, qwen3), the corresponding Jinja2 template should
be written into the tokenizer so it's saved with the model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from dataset_handlers.formatters import (
    get_chat_template_for_format,
    CHAT_TEMPLATES,
    ChatMLFormatter,
    Llama3Formatter,
    MistralFormatter,
    Qwen3Formatter,
)


def test_chat_templates_exist():
    """All known format types should have a chat template."""
    expected_formats = ["chatml", "llama3", "mistral", "qwen3"]
    for fmt in expected_formats:
        template = get_chat_template_for_format(fmt)
        assert template is not None, f"Missing chat template for format: {fmt}"
        assert len(template) > 0, f"Empty chat template for format: {fmt}"


def test_unknown_format_returns_none():
    """Unknown or non-standard formats should return None."""
    assert get_chat_template_for_format("custom") is None
    assert get_chat_template_for_format("tokenizer") is None
    assert get_chat_template_for_format("nonexistent") is None


def test_case_insensitive():
    """Format lookup should be case-insensitive."""
    assert get_chat_template_for_format("ChatML") is not None
    assert get_chat_template_for_format("LLAMA3") is not None
    assert get_chat_template_for_format("Mistral") is not None


try:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

requires_jinja2 = pytest.mark.skipif(not HAS_JINJA2, reason="jinja2 required for template rendering tests")


def _render_template(template_str, **kwargs):
    """Helper to render a Jinja2 chat template string."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_str)
    return template.render(**kwargs)


@requires_jinja2
def test_chatml_template_renders():
    """ChatML template should produce correct output with Jinja2."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    # With generation prompt
    result = _render_template(
        CHAT_TEMPLATES["chatml"],
        messages=messages, add_generation_prompt=True,
    )
    assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
    assert "<|im_start|>user\nHello<|im_end|>" in result
    assert result.endswith("<|im_start|>assistant\n")

    # Without generation prompt (full conversation)
    messages_full = messages + [{"role": "assistant", "content": "Hi there!"}]
    result_full = _render_template(
        CHAT_TEMPLATES["chatml"],
        messages=messages_full, add_generation_prompt=False,
    )
    assert "<|im_start|>assistant\nHi there!<|im_end|>" in result_full


@requires_jinja2
def test_llama3_template_renders():
    """Llama3 template should produce correct output."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    # bos_token is a variable the tokenizer provides at render time
    result = _render_template(
        CHAT_TEMPLATES["llama3"],
        messages=messages, add_generation_prompt=True,
        bos_token="<|begin_of_text|>",
    )
    assert "<|begin_of_text|>" in result
    assert "<|start_header_id|>user<|end_header_id|>" in result
    assert "Hello<|eot_id|>" in result
    assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


@requires_jinja2
def test_mistral_template_renders():
    """Mistral template should produce correct output."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    result = _render_template(
        CHAT_TEMPLATES["mistral"],
        messages=messages, bos_token="<s>", eos_token="</s>",
    )
    assert "[INST] Hello [/INST]" in result

    # Test with system message
    messages_sys = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result_sys = _render_template(
        CHAT_TEMPLATES["mistral"],
        messages=messages_sys, bos_token="<s>", eos_token="</s>",
    )
    assert "Be helpful." in result_sys
    assert "[INST] Hello [/INST]" in result_sys


@requires_jinja2
def test_template_matches_formatter_output():
    """The Jinja2 template output should match the formatter's format() output."""
    formatter = ChatMLFormatter()

    row = {
        "system": "You are helpful.",
        "prompt": "What is 2+2?",
        "chosen": "4",
        "rejected": "5",
    }

    # Format using the formatter
    formatted = formatter.format(row)

    # Render using the template (prompt part)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    rendered_prompt = _render_template(
        CHAT_TEMPLATES["chatml"],
        messages=messages, add_generation_prompt=True,
    )

    assert formatted["prompt"] == rendered_prompt, (
        f"Mismatch!\nFormatter: {repr(formatted['prompt'])}\n"
        f"Template:  {repr(rendered_prompt)}"
    )

    # Check full conversation for chosen
    messages_chosen = messages + [{"role": "assistant", "content": "4"}]
    rendered_full = _render_template(
        CHAT_TEMPLATES["chatml"],
        messages=messages_chosen, add_generation_prompt=False,
    )
    rendered_chosen = rendered_full[len(rendered_prompt):]
    assert formatted["chosen"] == rendered_chosen, (
        f"Mismatch!\nFormatter: {repr(formatted['chosen'])}\n"
        f"Template:  {repr(rendered_chosen)}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Chat Template Embedding Tests")
    print("=" * 60)
    print()

    test_chat_templates_exist()
    test_unknown_format_returns_none()
    test_case_insensitive()
    test_chatml_template_renders()
    test_llama3_template_renders()
    test_mistral_template_renders()
    test_template_matches_formatter_output()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
