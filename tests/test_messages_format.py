"""
Tests for messages format conversion.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from dataset_handlers.messages_converter import (
    has_messages_format,
    convert_messages_to_standard,
    convert_messages_dataset
)


def test_has_messages_format():
    """Test detection of messages format"""
    print("Testing messages format detection...")

    # Test with messages format
    messages_dataset = Dataset.from_list([
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
    ])
    assert has_messages_format(messages_dataset), "Should detect messages format"

    # Test with standard format
    standard_dataset = Dataset.from_list([
        {"prompt": "Hello", "chosen": "Hi there!", "rejected": "Go away"}
    ])
    assert not has_messages_format(standard_dataset), "Should not detect messages format"

    print("✓ Messages format detection works")


def test_convert_single_turn():
    """Test conversion of single-turn conversation"""
    print("Testing single-turn conversation conversion...")

    row = {
        "messages": [
            {"role": "user", "content": "Who developed you?"},
            {"role": "assistant", "content": "I'm a language model finetuned by Schneewolf Labs, a software research and publishing company based in Pennsylvania."}
        ]
    }

    result = convert_messages_to_standard(row)

    assert result["prompt"] == "Who developed you?"
    assert result["chosen"] == "I'm a language model finetuned by Schneewolf Labs, a software research and publishing company based in Pennsylvania."
    assert result["system"] == ""

    print("✓ Single-turn conversion works")


def test_convert_multi_turn():
    """Test conversion of multi-turn conversation"""
    print("Testing multi-turn conversation conversion...")

    row = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"}
        ]
    }

    result = convert_messages_to_standard(row)

    assert result["prompt"] == "Hello\n\nHow are you?"
    assert result["chosen"] == "Hi there!\n\nI'm doing well, thanks!"

    print("✓ Multi-turn conversion works")


def test_convert_with_system():
    """Test conversion with system message"""
    print("Testing conversion with system message...")

    row = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]
    }

    result = convert_messages_to_standard(row)

    assert result["system"] == "You are a helpful assistant"
    assert result["prompt"] == "Hello"
    assert result["chosen"] == "Hi!"

    print("✓ System message conversion works")


def test_convert_dataset():
    """Test conversion of entire dataset"""
    print("Testing full dataset conversion...")

    dataset = Dataset.from_list([
        {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Turn 2"},
                {"role": "assistant", "content": "Response 2"}
            ]
        }
    ])

    converted = convert_messages_dataset(dataset)

    # Check first sample (no system)
    assert converted[0]["prompt"] == "Question 1"
    assert converted[0]["chosen"] == "Answer 1"
    assert converted[0]["system"] == ""

    # Check second sample (with system)
    assert converted[1]["system"] == "Be helpful"
    assert converted[1]["prompt"] == "Question 2"
    assert converted[1]["chosen"] == "Answer 2"

    # Check third sample (multi-turn)
    assert converted[2]["prompt"] == "Turn 1\n\nTurn 2"
    assert converted[2]["chosen"] == "Response 1\n\nResponse 2"
    assert converted[2]["system"] == ""

    print("✓ Full dataset conversion works")


def test_integration_with_pipeline():
    """Test integration with DatasetPipeline"""
    print("Testing integration with DatasetPipeline...")

    from dataset_handlers.base import DatasetPipeline
    from dataset_handlers.loaders import LocalFileLoader
    from dataset_handlers.formatters import ChatMLFormatter
    import json
    import tempfile

    # Create test dataset with messages format
    test_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "It's 4"}
            ]
        }
    ]

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        # Create pipeline
        loader = LocalFileLoader(temp_path)
        formatter = ChatMLFormatter()
        pipeline = DatasetPipeline(
            loader=loader,
            formatter=formatter,
            training_mode='sft'
        )

        # Test preview (should auto-convert)
        preview = pipeline.preview(num_samples=2)
        assert len(preview) == 2
        assert "prompt" in preview[0]
        assert "chosen" in preview[0]
        assert preview[0]["prompt"] == "Hello"

        # Test formatted preview
        formatted = pipeline.preview_formatted(num_samples=2)
        assert len(formatted) == 2
        assert "<|im_start|>user" in formatted[0]["prompt"]

        print("✓ Pipeline integration works")

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    print("Running messages format tests...\n")

    test_has_messages_format()
    test_convert_single_turn()
    test_convert_multi_turn()
    test_convert_with_system()
    test_convert_dataset()
    test_integration_with_pipeline()

    print("\n✓ All tests passed!")
