"""
Transformation Engine for Data Editor

Provides data transformation capabilities including column mapping,
preference pair generation, and data augmentation.
"""

import json
import random
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

from . import EditorRow, TransformationConfig

logger = logging.getLogger(__name__)


class TransformationEngine:
    """
    Engine for transforming dataset rows
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pair_generators = {
            'truncate_30': self._generate_truncated(0.3),
            'truncate_50': self._generate_truncated(0.5),
            'truncate_70': self._generate_truncated(0.7),
            'degrade_formatting': self._generate_degraded_formatting,
            'add_errors': self._generate_with_errors,
            'shuffle_sentences': self._generate_shuffled_sentences,
            'remove_details': self._generate_without_details,
        }

    def transform_row(self, raw_row: Dict[str, Any], config: TransformationConfig) -> EditorRow:
        """
        Transform a raw data row into EditorRow format

        Args:
            raw_row: Raw data dictionary
            config: Transformation configuration with column mapping

        Returns: EditorRow with transformed data
        """
        # Apply column mapping
        mapped_data = self._apply_column_mapping(raw_row, config.column_mapping)

        # Create EditorRow
        row = EditorRow(
            idx=0,  # Will be set by caller
            prompt=mapped_data.get('prompt'),
            chosen=mapped_data.get('chosen'),
            rejected=mapped_data.get('rejected'),
            system=mapped_data.get('system'),
            reasoning=mapped_data.get('reasoning'),
            metadata={'original': raw_row}
        )

        # Generate rejected if needed
        if config.generate_rejected and row.chosen and not row.rejected:
            strategy = config.rejected_strategy or 'truncate_50'
            row.rejected = self.generate_rejected_response(row.chosen, strategy)

        # Add system message if configured
        if config.add_system_message and not row.system:
            row.system = config.add_system_message

        # Extract reasoning if configured
        if config.extract_reasoning and not row.reasoning:
            row.reasoning = self._extract_reasoning(row.chosen or '')

        # Apply custom transforms
        if config.custom_transforms:
            row = self._apply_custom_transforms(row, config.custom_transforms)

        return row

    def _apply_column_mapping(self, raw_row: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Apply column mapping to transform raw row

        Mapping can be:
        - Direct: {"prompt": "instruction"}
        - Nested: {"prompt": "messages[0].content"}
        - Template: {"prompt": "{instruction}\n{input}"}
        - Constant: {"system": "You are a helpful assistant"}
        """
        result = {}

        for target_field, source_spec in mapping.items():
            if source_spec is None:
                result[target_field] = None
                continue

            if isinstance(source_spec, dict):
                # Complex mapping with transformation
                source = source_spec.get('source')
                transform = source_spec.get('transform')

                value = self._extract_value(raw_row, source)

                if transform:
                    value = self._apply_transform(value, transform)

                result[target_field] = value

            elif isinstance(source_spec, str):
                # Simple mapping or template
                if '{' in source_spec and '}' in source_spec:
                    # Template string
                    result[target_field] = self._apply_template(raw_row, source_spec)
                else:
                    # Direct or nested path
                    result[target_field] = self._extract_value(raw_row, source_spec)
            else:
                # Constant value
                result[target_field] = str(source_spec)

        return result

    def _extract_value(self, data: Dict[str, Any], path: str) -> Optional[str]:
        """
        Extract value from nested dictionary using path notation

        Examples:
            "instruction" -> data["instruction"]
            "messages[0].content" -> data["messages"][0]["content"]
            "data.response" -> data["data"]["response"]
        """
        if not path:
            return None

        # Handle nested paths
        parts = self._parse_path(path)
        value = data

        for part in parts:
            if isinstance(part, int):
                # Array index
                if isinstance(value, list) and 0 <= part < len(value):
                    value = value[part]
                else:
                    return None
            else:
                # Dict key
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None

            if value is None:
                return None

        # Convert to string
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, dict) or isinstance(value, list):
            return json.dumps(value)
        else:
            return str(value)

    def _parse_path(self, path: str) -> List[Any]:
        """Parse path string into list of keys and indices"""
        import re
        parts = []
        tokens = re.split(r'\.', path)

        for token in tokens:
            match = re.match(r'(\w+)\[(\d+)\]', token)
            if match:
                parts.append(match.group(1))
                parts.append(int(match.group(2)))
            else:
                parts.append(token)

        return parts

    def _apply_template(self, data: Dict[str, Any], template: str) -> str:
        """Apply template string with placeholders"""
        # Simple template replacement
        result = template

        # Find all placeholders {field}
        placeholders = re.findall(r'\{(\w+)\}', template)

        for placeholder in placeholders:
            value = self._extract_value(data, placeholder)
            if value:
                result = result.replace(f'{{{placeholder}}}', value)
            else:
                result = result.replace(f'{{{placeholder}}}', '')

        # Clean up extra whitespace
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)

        return result.strip()

    def _apply_transform(self, value: Optional[str], transform: str) -> Optional[str]:
        """Apply transformation function to value"""
        if value is None:
            return None

        transform = transform.lower().strip()

        if transform == 'lowercase':
            return value.lower()
        elif transform == 'uppercase':
            return value.upper()
        elif transform == 'title':
            return value.title()
        elif transform == 'strip':
            return value.strip()
        elif transform == 'trim':
            return ' '.join(value.split())
        else:
            # Unknown transform, return as-is
            return value

    def generate_rejected_response(self, chosen: str, strategy: str = 'truncate_50') -> str:
        """
        Generate a rejected response from a chosen response

        Strategies:
        - truncate_30/50/70: Truncate at 30/50/70% of length
        - degrade_formatting: Remove formatting, make less readable
        - add_errors: Introduce spelling/grammar errors
        - shuffle_sentences: Shuffle sentence order
        - remove_details: Remove specific details and examples
        """
        if strategy in self.pair_generators:
            return self.pair_generators[strategy](chosen)
        else:
            # Default to truncate_50
            return self._generate_truncated(0.5)(chosen)

    def _generate_truncated(self, ratio: float) -> Callable[[str], str]:
        """Return a truncation function for given ratio"""
        def truncate(text: str) -> str:
            if not text:
                return text

            # Truncate at sentence boundary near the ratio
            target_pos = int(len(text) * ratio)

            # Find nearest sentence boundary
            sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text)]

            if sentence_ends:
                # Find closest sentence end to target
                closest = min(sentence_ends, key=lambda x: abs(x - target_pos))
                return text[:closest].strip()
            else:
                # No sentence boundaries, just truncate
                return text[:target_pos].strip() + '...'

        return truncate

    def _generate_degraded_formatting(self, text: str) -> str:
        """Remove formatting and make text less readable"""
        if not text:
            return text

        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.+?)`', r'\1', text)  # Code
        text = re.sub(r'#+\s+', '', text)  # Headers

        # Remove bullet points and numbering
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        # Collapse paragraphs
        text = re.sub(r'\n\n+', ' ', text)

        # Remove extra spacing
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _generate_with_errors(self, text: str) -> str:
        """Introduce spelling and grammar errors"""
        if not text:
            return text

        words = text.split()
        num_errors = max(1, len(words) // 20)  # ~5% error rate

        # Common typos
        typo_map = {
            'the': 'teh',
            'and': 'adn',
            'you': 'yuo',
            'are': 'aer',
            'for': 'fro',
            'this': 'thsi',
            'that': 'taht',
            'with': 'wiht',
            'have': 'ahve',
            'from': 'form',
        }

        # Randomly introduce errors
        for _ in range(num_errors):
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower()

            if word in typo_map:
                # Use predefined typo
                words[idx] = typo_map[word]
            elif len(word) > 3:
                # Swap two adjacent characters
                pos = random.randint(0, len(word) - 2)
                word_list = list(word)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                words[idx] = ''.join(word_list)

        return ' '.join(words)

    def _generate_shuffled_sentences(self, text: str) -> str:
        """Shuffle sentence order"""
        if not text:
            return text

        # Split into sentences
        sentences = re.split(r'([.!?]\s+)', text)

        # Recombine sentence with its punctuation
        sentence_pairs = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_pairs.append(sentences[i] + sentences[i + 1])
            else:
                sentence_pairs.append(sentences[i])

        # Shuffle
        random.shuffle(sentence_pairs)

        return ''.join(sentence_pairs)

    def _generate_without_details(self, text: str) -> str:
        """Remove specific details and examples"""
        if not text:
            return text

        # Remove content in parentheses (often examples/details)
        text = re.sub(r'\([^)]+\)', '', text)

        # Remove "for example" sections
        text = re.sub(r'[Ff]or example,?.+?[.!?]', '', text)
        text = re.sub(r'[Ff]or instance,?.+?[.!?]', '', text)

        # Remove numbered lists (often examples)
        text = re.sub(r'^\s*\d+\..+$', '', text, flags=re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    def _extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning/thinking from text if present"""
        # Look for common reasoning markers
        patterns = [
            r'(?:Let me think|Let\'s think|Let\'s analyze|Here\'s my reasoning)(.+?)(?:\n\n|$)',
            r'(?:Step \d+:|First,|Second,|Third,)(.+?)(?:\n\n|$)',
            r'(?:Analysis:|Reasoning:|Thought process:)(.+?)(?:\n\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        return None

    def _apply_custom_transforms(self, row: EditorRow, transforms: Dict[str, str]) -> EditorRow:
        """Apply custom transformation functions"""
        # This is a placeholder for more advanced custom transformations
        # In a full implementation, this could support Python code execution
        # or more complex transformation logic
        return row

    def batch_transform(self, raw_rows: List[Dict[str, Any]], config: TransformationConfig) -> List[EditorRow]:
        """
        Transform multiple rows at once

        Args:
            raw_rows: List of raw data dictionaries
            config: Transformation configuration

        Returns: List of EditorRow objects
        """
        transformed_rows = []

        for idx, raw_row in enumerate(raw_rows):
            try:
                row = self.transform_row(raw_row, config)
                row.idx = idx
                transformed_rows.append(row)
            except Exception as e:
                self.logger.error(f"Error transforming row {idx}: {e}")
                # Create error row
                error_row = EditorRow(
                    idx=idx,
                    prompt=None,
                    chosen=None,
                    rejected=None,
                    metadata={'error': str(e), 'original': raw_row}
                )
                error_row.validation_errors.append(f"Transformation failed: {e}")
                transformed_rows.append(error_row)

        return transformed_rows


# Convenience functions
def apply_column_mapping(raw_rows: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[EditorRow]:
    """
    Convenience function to apply column mapping

    Args:
        raw_rows: List of raw dictionaries
        mapping: Column mapping dict

    Returns: List of EditorRow objects
    """
    engine = TransformationEngine()
    config = TransformationConfig(column_mapping=mapping)

    return engine.batch_transform(raw_rows, config)


def generate_preference_pairs(rows: List[EditorRow], strategy: str = 'truncate_50') -> List[EditorRow]:
    """
    Generate rejected responses for rows that only have chosen

    Args:
        rows: List of EditorRow objects
        strategy: Generation strategy

    Returns: List of EditorRow objects with rejected responses
    """
    engine = TransformationEngine()

    for row in rows:
        if row.chosen and not row.rejected:
            row.rejected = engine.generate_rejected_response(row.chosen, strategy)

    return rows
