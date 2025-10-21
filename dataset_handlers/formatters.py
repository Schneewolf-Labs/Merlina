"""
Dataset formatter implementations for different chat template formats
"""

import logging
from typing import Optional, Any
from .base import DatasetFormatter

logger = logging.getLogger(__name__)


class ChatMLFormatter(DatasetFormatter):
    """
    Format dataset using ChatML template.
    This is the original Merlina format.
    """

    def format(self, row: dict) -> dict:
        """
        Format row using ChatML template.

        Expected input columns: system (optional), prompt, chosen, rejected
        Output columns: prompt, chosen, rejected
        """
        system = str(row.get('system', ''))
        prompt = str(row.get('prompt', ''))
        chosen = str(row.get('chosen', ''))
        rejected = str(row.get('rejected', ''))

        # ChatML format
        system_prefix = f"<|im_start|>system\n{system}<|im_end|>\n" if system.strip() else ""

        return {
            "prompt": f"{system_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "chosen": f"{chosen}<|im_end|>\n",
            "rejected": f"{rejected}<|im_end|>\n"
        }

    def get_format_info(self) -> dict:
        """Get information about the format type"""
        return {
            "format_type": "chatml",
            "description": "ChatML format with <|im_start|> and <|im_end|> tags",
            "example_prompt": "<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n<|im_start|>user\\nHello<|im_end|>\\n<|im_start|>assistant\\n"
        }


class Llama3Formatter(DatasetFormatter):
    """
    Format dataset using Llama 3 template.
    Uses <|begin_of_text|>, <|start_header_id|>, <|end_header_id|>, <|eot_id|>
    """

    def format(self, row: dict) -> dict:
        """
        Format row using Llama 3 template.

        Expected input columns: system (optional), prompt, chosen, rejected
        Output columns: prompt, chosen, rejected
        """
        system = str(row.get('system', ''))
        prompt = str(row.get('prompt', ''))
        chosen = str(row.get('chosen', ''))
        rejected = str(row.get('rejected', ''))

        # Llama 3 format
        system_part = ""
        if system.strip():
            system_part = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"

        formatted_prompt = (
            f"<|begin_of_text|>{system_part}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        return {
            "prompt": formatted_prompt,
            "chosen": f"{chosen}<|eot_id|>",
            "rejected": f"{rejected}<|eot_id|>"
        }

    def get_format_info(self) -> dict:
        """Get information about the format type"""
        return {
            "format_type": "llama3",
            "description": "Llama 3 format with special header tags",
            "example_prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        }


class MistralFormatter(DatasetFormatter):
    """
    Format dataset using Mistral Instruct template.
    Uses [INST] and [/INST] tags
    """

    def format(self, row: dict) -> dict:
        """
        Format row using Mistral template.

        Expected input columns: system (optional), prompt, chosen, rejected
        Output columns: prompt, chosen, rejected
        """
        system = str(row.get('system', ''))
        prompt = str(row.get('prompt', ''))
        chosen = str(row.get('chosen', ''))
        rejected = str(row.get('rejected', ''))

        # Combine system and prompt if system exists
        full_prompt = f"{system}\n\n{prompt}" if system.strip() else prompt

        # Mistral format
        formatted_prompt = f"[INST] {full_prompt} [/INST] "

        return {
            "prompt": formatted_prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    def get_format_info(self) -> dict:
        """Get information about the format type"""
        return {
            "format_type": "mistral",
            "description": "Mistral Instruct format with [INST] tags",
            "example_prompt": "[INST] Hello, how are you? [/INST] "
        }


class CustomFormatter(DatasetFormatter):
    """
    Format dataset using custom user-defined templates.
    Supports variable substitution with {system}, {prompt}, {chosen}, {rejected}
    """

    def __init__(
        self,
        prompt_template: str,
        chosen_template: str = "{chosen}",
        rejected_template: str = "{rejected}"
    ):
        """
        Initialize custom formatter.

        Args:
            prompt_template: Template for prompt with variables like {system}, {prompt}
            chosen_template: Template for chosen response with {chosen}
            rejected_template: Template for rejected response with {rejected}
        """
        self.prompt_template = prompt_template
        self.chosen_template = chosen_template
        self.rejected_template = rejected_template

    def format(self, row: dict) -> dict:
        """
        Format row using custom templates.

        Expected input columns: system (optional), prompt, chosen, rejected
        Output columns: prompt, chosen, rejected
        """
        system = str(row.get('system', ''))
        prompt = str(row.get('prompt', ''))
        chosen = str(row.get('chosen', ''))
        rejected = str(row.get('rejected', ''))

        # Replace variables in templates
        formatted_prompt = self.prompt_template.format(
            system=system,
            prompt=prompt
        )
        formatted_chosen = self.chosen_template.format(chosen=chosen)
        formatted_rejected = self.rejected_template.format(rejected=rejected)

        return {
            "prompt": formatted_prompt,
            "chosen": formatted_chosen,
            "rejected": formatted_rejected
        }

    def get_format_info(self) -> dict:
        """Get information about the format type"""
        return {
            "format_type": "custom",
            "description": "Custom user-defined template format",
            "prompt_template": self.prompt_template,
            "chosen_template": self.chosen_template,
            "rejected_template": self.rejected_template
        }


class TokenizerFormatter(DatasetFormatter):
    """
    Format dataset using the tokenizer's chat template from tokenizer_config.json.
    This automatically uses the correct format for the model being trained.
    """

    def __init__(self, tokenizer: Any):
        """
        Initialize tokenizer-based formatter.

        Args:
            tokenizer: HuggingFace tokenizer with chat_template support
        """
        self.tokenizer = tokenizer

        # Check if tokenizer has chat template
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            logger.warning(
                "Tokenizer does not have a chat_template. "
                "Falling back to simple concatenation format."
            )
            self.has_chat_template = False
        else:
            self.has_chat_template = True
            logger.info(f"Using tokenizer chat template for formatting")

    def format(self, row: dict) -> dict:
        """
        Format row using tokenizer's chat template.

        Expected input columns: system (optional), prompt, chosen, rejected
        Output columns: prompt, chosen, rejected
        """
        system = str(row.get('system', ''))
        prompt = str(row.get('prompt', ''))
        chosen = str(row.get('chosen', ''))
        rejected = str(row.get('rejected', ''))

        if not self.has_chat_template:
            # Fallback: simple concatenation
            system_prefix = f"{system}\n\n" if system.strip() else ""
            return {
                "prompt": f"{system_prefix}{prompt}\n",
                "chosen": chosen,
                "rejected": rejected
            }

        # Build message list for prompt (without assistant response)
        messages_prompt = []
        if system.strip():
            messages_prompt.append({"role": "system", "content": system})
        messages_prompt.append({"role": "user", "content": prompt})

        # Format prompt using chat template (without assistant message)
        # We use add_generation_prompt=True to add the assistant prefix
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using fallback.")
            system_prefix = f"{system}\n\n" if system.strip() else ""
            formatted_prompt = f"{system_prefix}{prompt}\n"

        # For chosen and rejected, we just need the response content
        # The tokenizer template will handle the closing tags
        # But we need to extract what comes after the generation prompt

        # Get the full conversation with chosen response to extract the suffix format
        messages_chosen = messages_prompt + [{"role": "assistant", "content": chosen}]
        try:
            full_chosen = self.tokenizer.apply_chat_template(
                messages_chosen,
                tokenize=False,
                add_generation_prompt=False
            )
            # Extract just the response part (everything after the prompt)
            formatted_chosen = full_chosen[len(formatted_prompt):]
        except Exception as e:
            logger.warning(f"Failed to format chosen response: {e}")
            formatted_chosen = chosen

        # Same for rejected
        messages_rejected = messages_prompt + [{"role": "assistant", "content": rejected}]
        try:
            full_rejected = self.tokenizer.apply_chat_template(
                messages_rejected,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_rejected = full_rejected[len(formatted_prompt):]
        except Exception as e:
            logger.warning(f"Failed to format rejected response: {e}")
            formatted_rejected = rejected

        return {
            "prompt": formatted_prompt,
            "chosen": formatted_chosen,
            "rejected": formatted_rejected
        }

    def get_format_info(self) -> dict:
        """Get information about the format type"""
        return {
            "format_type": "tokenizer",
            "description": "Uses the tokenizer's chat_template from tokenizer_config.json",
            "has_chat_template": self.has_chat_template,
            "tokenizer_name": getattr(self.tokenizer, 'name_or_path', 'unknown')
        }


def get_formatter(
    format_type: str,
    custom_templates: Optional[dict] = None,
    tokenizer: Optional[Any] = None
) -> DatasetFormatter:
    """
    Factory function to get the appropriate formatter.

    Args:
        format_type: One of 'chatml', 'llama3', 'mistral', 'tokenizer', 'custom'
        custom_templates: Required if format_type is 'custom'
                         Dict with keys: prompt_template, chosen_template, rejected_template
        tokenizer: Required if format_type is 'tokenizer'
                  HuggingFace tokenizer with chat_template support

    Returns:
        DatasetFormatter instance

    Raises:
        ValueError: If format_type is invalid or required parameters are missing
    """
    format_type = format_type.lower()

    if format_type == 'chatml':
        return ChatMLFormatter()

    elif format_type == 'llama3':
        return Llama3Formatter()

    elif format_type == 'mistral':
        return MistralFormatter()

    elif format_type == 'tokenizer':
        if tokenizer is None:
            raise ValueError("tokenizer required for 'tokenizer' format type")

        return TokenizerFormatter(tokenizer=tokenizer)

    elif format_type == 'custom':
        if not custom_templates:
            raise ValueError("custom_templates required for 'custom' format type")

        return CustomFormatter(
            prompt_template=custom_templates.get('prompt_template', '{prompt}'),
            chosen_template=custom_templates.get('chosen_template', '{chosen}'),
            rejected_template=custom_templates.get('rejected_template', '{rejected}')
        )

    else:
        raise ValueError(
            f"Invalid format_type: {format_type}. "
            f"Must be one of: chatml, llama3, mistral, tokenizer, custom"
        )
