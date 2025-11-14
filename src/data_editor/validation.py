"""
Validation Engine for Data Editor

Provides comprehensive validation for ORPO dataset schema and quality checks.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass, field
from collections import Counter

from . import EditorRow, EditorSession, ValidationResult

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Engine for validating dataset rows and providing quality metrics
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_session(self, session: EditorSession) -> ValidationResult:
        """
        Validate all rows in a session

        Returns: ValidationResult with errors, warnings, and statistics
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            statistics={},
            row_issues={}
        )

        if not session.rows:
            result.is_valid = False
            result.errors.append("Dataset is empty - no rows to validate")
            return result

        # Validate each row
        valid_rows = 0
        for row in session.rows:
            row_result = self.validate_row(row)

            if row_result.errors:
                result.row_issues[row.idx] = {
                    "errors": row_result.errors,
                    "warnings": row_result.warnings
                }
                result.is_valid = False
            elif row_result.warnings:
                result.row_issues[row.idx] = {
                    "errors": [],
                    "warnings": row_result.warnings
                }
                valid_rows += 1
            else:
                valid_rows += 1

        # Aggregate statistics
        result.statistics = self._compute_statistics(session.rows)

        # Global validation checks
        global_errors, global_warnings = self._global_checks(session.rows, result.statistics)
        result.errors.extend(global_errors)
        result.warnings.extend(global_warnings)

        # Summary
        result.statistics['total_rows'] = len(session.rows)
        result.statistics['valid_rows'] = valid_rows
        result.statistics['invalid_rows'] = len(session.rows) - valid_rows
        result.statistics['error_rate'] = (len(session.rows) - valid_rows) / len(session.rows) if session.rows else 0

        self.logger.info(f"Validation complete: {valid_rows}/{len(session.rows)} valid rows")

        return result

    def validate_row(self, row: EditorRow) -> ValidationResult:
        """
        Validate a single row

        Checks:
        - Required fields present and non-empty
        - Field types
        - Content quality (length, encoding, etc.)
        - Chosen vs rejected similarity
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Check required fields
        if not row.prompt or not row.prompt.strip():
            result.errors.append("Prompt is required and cannot be empty")
            result.is_valid = False

        if not row.chosen or not row.chosen.strip():
            result.errors.append("Chosen response is required and cannot be empty")
            result.is_valid = False

        if not row.rejected or not row.rejected.strip():
            result.errors.append("Rejected response is required and cannot be empty")
            result.is_valid = False

        # If required fields missing, don't continue with other checks
        if not result.is_valid:
            return result

        # Content quality checks
        prompt_issues = self._check_content_quality(row.prompt, "prompt")
        chosen_issues = self._check_content_quality(row.chosen, "chosen")
        rejected_issues = self._check_content_quality(row.rejected, "rejected")

        result.warnings.extend(prompt_issues)
        result.warnings.extend(chosen_issues)
        result.warnings.extend(rejected_issues)

        # Check optional fields if present
        if row.system:
            system_issues = self._check_content_quality(row.system, "system")
            result.warnings.extend(system_issues)

        if row.reasoning:
            reasoning_issues = self._check_content_quality(row.reasoning, "reasoning")
            result.warnings.extend(reasoning_issues)

        # Check similarity between chosen and rejected
        similarity = self._compute_similarity(row.chosen, row.rejected)
        if similarity > 0.9:
            result.warnings.append(f"Chosen and rejected responses are very similar ({similarity:.1%})")
        elif similarity > 0.95:
            result.errors.append("Chosen and rejected responses are nearly identical - poor training signal")
            result.is_valid = False

        # Token length checks
        prompt_tokens = self._estimate_tokens(row.prompt)
        chosen_tokens = self._estimate_tokens(row.chosen)
        rejected_tokens = self._estimate_tokens(row.rejected)

        if prompt_tokens > 4096:
            result.warnings.append(f"Prompt is very long ({prompt_tokens} tokens) - may cause training issues")

        if chosen_tokens > 4096:
            result.warnings.append(f"Chosen response is very long ({chosen_tokens} tokens)")

        if rejected_tokens > 4096:
            result.warnings.append(f"Rejected response is very long ({rejected_tokens} tokens)")

        if prompt_tokens + max(chosen_tokens, rejected_tokens) > 8192:
            result.errors.append("Combined prompt + response exceeds 8192 tokens - will be truncated in training")
            result.is_valid = False

        return result

    def _check_content_quality(self, content: str, field_name: str) -> List[str]:
        """Check content quality for a field"""
        issues = []

        if not content:
            return issues

        # Check length
        if len(content) < 10:
            issues.append(f"{field_name} is very short (< 10 characters)")

        # Check for placeholder text
        placeholder_patterns = [
            r'\[.*?\]',  # [placeholder]
            r'TODO',
            r'FIXME',
            r'XXX',
            r'<.*?>',  # <placeholder>
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"{field_name} may contain placeholder text")
                break

        # Check for encoding issues
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            issues.append(f"{field_name} has encoding issues")

        # Check for excessive whitespace
        if len(content.strip()) < len(content) * 0.8:
            issues.append(f"{field_name} has excessive whitespace")

        # Check for repeated characters
        if re.search(r'(.)\1{10,}', content):
            issues.append(f"{field_name} has repeated characters (possible generation error)")

        return issues

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple character-level similarity between two texts

        Returns: similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Jaccard similarity on character n-grams (trigrams)
        def get_trigrams(text):
            return set(text[i:i+3] for i in range(len(text) - 2))

        trigrams1 = get_trigrams(text1)
        trigrams2 = get_trigrams(text2)

        if not trigrams1 or not trigrams2:
            return 0.0

        intersection = len(trigrams1 & trigrams2)
        union = len(trigrams1 | trigrams2)

        return intersection / union if union > 0 else 0.0

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (actual tokenization would require model tokenizer)

        Uses simple heuristic: ~4 characters per token for English text
        """
        if not text:
            return 0
        return len(text) // 4

    def _compute_statistics(self, rows: List[EditorRow]) -> Dict[str, Any]:
        """Compute dataset statistics"""
        if not rows:
            return {}

        stats = {
            'num_rows': len(rows),
            'has_system': sum(1 for r in rows if r.system),
            'has_reasoning': sum(1 for r in rows if r.reasoning),
        }

        # Token length statistics
        prompt_lengths = [self._estimate_tokens(r.prompt) for r in rows if r.prompt]
        chosen_lengths = [self._estimate_tokens(r.chosen) for r in rows if r.chosen]
        rejected_lengths = [self._estimate_tokens(r.rejected) for r in rows if r.rejected]

        if prompt_lengths:
            stats['prompt_tokens'] = {
                'min': min(prompt_lengths),
                'max': max(prompt_lengths),
                'avg': sum(prompt_lengths) / len(prompt_lengths),
                'median': sorted(prompt_lengths)[len(prompt_lengths) // 2]
            }

        if chosen_lengths:
            stats['chosen_tokens'] = {
                'min': min(chosen_lengths),
                'max': max(chosen_lengths),
                'avg': sum(chosen_lengths) / len(chosen_lengths),
                'median': sorted(chosen_lengths)[len(chosen_lengths) // 2]
            }

        if rejected_lengths:
            stats['rejected_tokens'] = {
                'min': min(rejected_lengths),
                'max': max(rejected_lengths),
                'avg': sum(rejected_lengths) / len(rejected_lengths),
                'median': sorted(rejected_lengths)[len(rejected_lengths) // 2]
            }

        # Character length statistics
        prompt_chars = [len(r.prompt) for r in rows if r.prompt]
        chosen_chars = [len(r.chosen) for r in rows if r.chosen]
        rejected_chars = [len(r.rejected) for r in rows if r.rejected]

        stats['avg_prompt_length'] = sum(prompt_chars) / len(prompt_chars) if prompt_chars else 0
        stats['avg_chosen_length'] = sum(chosen_chars) / len(chosen_chars) if chosen_chars else 0
        stats['avg_rejected_length'] = sum(rejected_chars) / len(rejected_chars) if rejected_chars else 0

        return stats

    def _global_checks(self, rows: List[EditorRow], statistics: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Perform global dataset checks

        Returns: (errors, warnings)
        """
        errors = []
        warnings = []

        # Check dataset size
        num_rows = len(rows)

        if num_rows < 10:
            warnings.append(f"Dataset is very small ({num_rows} rows) - consider adding more examples")
        elif num_rows < 50:
            warnings.append(f"Dataset is small ({num_rows} rows) - may not be enough for effective training")

        # Check for duplicate prompts
        prompts = [r.prompt for r in rows if r.prompt]
        prompt_counts = Counter(prompts)
        duplicates = {prompt: count for prompt, count in prompt_counts.items() if count > 1}

        if duplicates:
            num_duplicates = len(duplicates)
            total_duplicate_rows = sum(duplicates.values()) - num_duplicates
            warnings.append(f"Found {num_duplicates} duplicate prompts ({total_duplicate_rows} total duplicate rows)")

        # Check for empty rows
        empty_rows = sum(1 for r in rows if not (r.prompt and r.chosen and r.rejected))
        if empty_rows > 0:
            errors.append(f"Found {empty_rows} rows with missing required fields")

        # Check token length distribution
        if 'prompt_tokens' in statistics:
            max_prompt = statistics['prompt_tokens']['max']
            if max_prompt > 8192:
                errors.append(f"Some prompts exceed 8192 tokens (max: {max_prompt}) - will cause errors")
            elif max_prompt > 4096:
                warnings.append(f"Some prompts are very long (max: {max_prompt} tokens)")

        # Check response length balance
        if 'chosen_tokens' in statistics and 'rejected_tokens' in statistics:
            avg_chosen = statistics['chosen_tokens']['avg']
            avg_rejected = statistics['rejected_tokens']['avg']

            ratio = avg_chosen / avg_rejected if avg_rejected > 0 else 0

            if ratio > 3 or ratio < 0.33:
                warnings.append(f"Chosen and rejected responses have very different lengths (ratio: {ratio:.2f})")

        return errors, warnings

    def quick_validate_row(self, prompt: Optional[str], chosen: Optional[str],
                          rejected: Optional[str]) -> Tuple[bool, List[str]]:
        """
        Quick validation for a single row (used in UI for real-time feedback)

        Returns: (is_valid, error_messages)
        """
        errors = []

        if not prompt or not prompt.strip():
            errors.append("Prompt is required")

        if not chosen or not chosen.strip():
            errors.append("Chosen response is required")

        if not rejected or not rejected.strip():
            errors.append("Rejected response is required")

        if chosen and rejected and chosen == rejected:
            errors.append("Chosen and rejected responses must be different")

        return len(errors) == 0, errors


# Convenience function
def validate_dataset(rows: List[Dict[str, Any]]) -> ValidationResult:
    """
    Validate a list of raw dictionaries

    Returns: ValidationResult
    """
    engine = ValidationEngine()

    # Convert to EditorRow objects
    editor_rows = []
    for i, row_dict in enumerate(rows):
        editor_row = EditorRow(
            idx=i,
            prompt=row_dict.get('prompt'),
            chosen=row_dict.get('chosen'),
            rejected=row_dict.get('rejected'),
            system=row_dict.get('system'),
            reasoning=row_dict.get('reasoning')
        )
        editor_rows.append(editor_row)

    # Create temporary session
    from datetime import datetime
    session = EditorSession(
        session_id="temp",
        name="temp",
        rows=editor_rows,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    return engine.validate_session(session)
