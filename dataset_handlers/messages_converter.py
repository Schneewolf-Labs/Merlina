"""
Utilities for converting messages format to standard Merlina format.

Supports datasets with this format:
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well!"}
    ]
}

Converts to:
{
    "prompt": "Hello\n\nHow are you?",
    "chosen": "Hi there!\n\nI'm doing well!",
    "system": "You are a helpful assistant"
}
"""

import logging
from typing import Any
from datasets import Dataset

logger = logging.getLogger(__name__)


def has_messages_format(dataset: Dataset) -> bool:
    """
    Check if dataset uses messages format.

    Args:
        dataset: Dataset to check

    Returns:
        True if dataset has 'messages' column with list of message dicts
    """
    if 'messages' not in dataset.column_names:
        return False

    # Check first sample to verify structure
    try:
        first_sample = dataset[0]
        messages = first_sample.get('messages')

        if not isinstance(messages, list) or len(messages) == 0:
            return False

        # Check if messages have required role and content fields
        first_message = messages[0]
        return isinstance(first_message, dict) and 'role' in first_message and 'content' in first_message

    except (IndexError, TypeError, KeyError):
        return False


def convert_messages_to_standard(row: dict) -> dict:
    """
    Convert a row with messages format to standard Merlina format.

    Args:
        row: Dictionary with 'messages' key containing list of message dicts

    Returns:
        Dictionary with 'system', 'prompt', 'chosen' keys

    Logic:
    - Extracts system messages (if any) into 'system' field
    - Combines all user messages into 'prompt' field
    - Combines all assistant messages into 'chosen' field
    - Multi-turn conversations are serialized with double newlines between turns

    Example:
        Input: {"messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Great!"}
        ]}

        Output: {
            "system": "Be helpful",
            "prompt": "Hi\n\nHow are you?",
            "chosen": "Hello!\n\nGreat!"
        }
    """
    messages = row.get('messages', [])

    if not messages:
        raise ValueError("Row has empty messages list")

    # Separate messages by role
    system_messages = []
    user_messages = []
    assistant_messages = []

    for msg in messages:
        role = msg.get('role', '').lower()
        content = msg.get('content', '')

        if role == 'system':
            system_messages.append(content)
        elif role == 'user':
            user_messages.append(content)
        elif role == 'assistant':
            assistant_messages.append(content)
        else:
            logger.warning(f"Unknown role '{role}' in messages, skipping")

    # Combine system messages (if multiple)
    system = '\n\n'.join(system_messages) if system_messages else ''

    # Combine user messages with double newlines
    prompt = '\n\n'.join(user_messages) if user_messages else ''

    # Combine assistant messages with double newlines
    chosen = '\n\n'.join(assistant_messages) if assistant_messages else ''

    if not prompt:
        raise ValueError("No user messages found in messages list")

    if not chosen:
        raise ValueError("No assistant messages found in messages list")

    # Always include system field for consistency across dataset
    result = {
        'system': system,
        'prompt': prompt,
        'chosen': chosen,
    }

    return result


def convert_messages_dataset(dataset: Dataset) -> Dataset:
    """
    Convert entire dataset from messages format to standard format.

    Args:
        dataset: Dataset with 'messages' column

    Returns:
        Dataset with 'system', 'prompt', 'chosen' columns
    """
    if not has_messages_format(dataset):
        raise ValueError("Dataset does not have valid messages format")

    logger.info(f"Converting {len(dataset)} samples from messages format to standard format")

    # Convert all rows
    converted = dataset.map(
        convert_messages_to_standard,
        remove_columns=['messages'],
        desc="Converting messages format"
    )

    logger.info("Successfully converted messages format dataset")

    return converted
