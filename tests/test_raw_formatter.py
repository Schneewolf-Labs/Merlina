"""
Tests for the RawFormatter (format_type: "raw").

Raw mode passes prompt/chosen/rejected through verbatim with no chat
template applied — for pre-formatted datasets or plain-text training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset_handlers.formatters import (
    RawFormatter,
    get_formatter,
    get_chat_template_for_format,
)


def test_raw_passes_columns_through_verbatim():
    """Prompt, chosen, and rejected must come back byte-for-byte unchanged."""
    formatter = RawFormatter()
    row = {
        "prompt": "<|im_start|>user\nAlready formatted!<|im_end|>\n",
        "chosen": "Line one\n\nLine two with {braces} and <tags>",
        "rejected": "  leading and trailing spaces  ",
    }
    result = formatter.format(row)
    assert result["prompt"] == row["prompt"]
    assert result["chosen"] == row["chosen"]
    assert result["rejected"] == row["rejected"]


def test_raw_ignores_system_and_reasoning():
    """System and reasoning columns must not leak into the output."""
    formatter = RawFormatter()
    row = {
        "system": "You are a helpful assistant",
        "prompt": "Hello",
        "chosen": "Hi!",
        "rejected": "Go away",
        "reasoning": "The user greeted me, so I should greet back.",
    }
    result = formatter.format(row)
    assert result["prompt"] == "Hello"
    assert result["chosen"] == "Hi!"
    assert result["rejected"] == "Go away"
    assert "helpful assistant" not in result["prompt"]
    assert "<think>" not in result["chosen"]


def test_raw_handles_missing_and_none_fields():
    """Missing or None columns become empty strings, not crashes."""
    formatter = RawFormatter()
    result = formatter.format({"prompt": "only a prompt", "chosen": None})
    assert result["prompt"] == "only a prompt"
    assert result["chosen"] == ""
    assert result["rejected"] == ""


def test_factory_returns_raw_formatter():
    """get_formatter should accept 'raw' (and the 'none' alias), case-insensitively."""
    assert isinstance(get_formatter("raw"), RawFormatter)
    assert isinstance(get_formatter("none"), RawFormatter)
    assert isinstance(get_formatter("Raw"), RawFormatter)


def test_raw_format_info():
    """Format info should identify the type as 'raw'."""
    info = RawFormatter().get_format_info()
    assert info["format_type"] == "raw"
    assert "no chat template" in info["description"].lower()


def test_raw_has_no_chat_template_to_embed():
    """No Jinja2 chat template should be written into the tokenizer for raw mode."""
    assert get_chat_template_for_format("raw") is None
    assert get_chat_template_for_format("none") is None


if __name__ == "__main__":
    print("=" * 60)
    print("Raw Formatter Tests")
    print("=" * 60)
    print()

    test_raw_passes_columns_through_verbatim()
    test_raw_ignores_system_and_reasoning()
    test_raw_handles_missing_and_none_fields()
    test_factory_returns_raw_formatter()
    test_raw_format_info()
    test_raw_has_no_chat_template_to_embed()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
