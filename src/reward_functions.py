"""
GRPO Reward Functions for Merlina.

Provides configurable reward functions for Group Relative Policy Optimization.
Each reward function takes (prompts, completions) and returns a list of float scores.

Built-in reward types:
- length: Longer completions score higher (normalized within batch)
- format: Rewards structured output markers (lists, code blocks, etc.)
- combined: Weighted combination of length + format
- answer_match: Compares model's extracted answer against ground truth (R1-style)
- regex: Rewards completions matching a user-defined regex pattern
- answer_and_format: Composite of answer_match + regex with configurable weights
"""

import re
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def build_length_reward() -> Callable:
    """Simple length-based reward: longer completions score higher (normalized)."""
    def reward_fn(prompts, completions):
        lengths = [len(c) for c in completions]
        max_len = max(lengths) if lengths else 1
        return [l / max_len for l in lengths]
    return reward_fn


def build_format_reward() -> Callable:
    """Format-based reward: checks for structured output markers."""
    def reward_fn(prompts, completions):
        rewards = []
        for c in completions:
            score = 0.0
            if any(marker in c for marker in ['\n', '- ', '* ', '1.', '```']):
                score += 0.5
            if len(c.strip()) > 10:
                score += 0.5
            rewards.append(score)
        return rewards
    return reward_fn


def build_combined_reward() -> Callable:
    """Combined reward: length (normalized) + non-empty bonus."""
    def reward_fn(prompts, completions):
        rewards = []
        lengths = [len(c) for c in completions]
        max_len = max(lengths) if lengths else 1
        for c in completions:
            score = len(c) / max_len * 0.7
            if len(c.strip()) > 10:
                score += 0.3
            rewards.append(score)
        return rewards
    return reward_fn


def build_answer_match_reward(
    prompt_to_answer: dict[str, str],
    answer_extraction_pattern: str = r"\\boxed\{(.*?)\}",
    strip_whitespace: bool = True,
    case_sensitive: bool = False,
) -> Callable:
    """Answer-match reward (R1-style): extract answer from completion and compare to ground truth.

    Args:
        prompt_to_answer: Mapping from prompt text to expected answer string.
        answer_extraction_pattern: Regex with a capture group to extract the model's answer
            from the completion. Common patterns:
            - r"\\\\boxed\\{(.*?)\\}" for LaTeX boxed answers
            - r"#### (.*)" for markdown heading answers
            - r"(?:Answer|ANSWER):\\s*(.*)" for "Answer: X" format
            - r"<answer>(.*?)</answer>" for XML-tagged answers
        strip_whitespace: Strip whitespace from extracted and expected answers before comparison.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        Reward function: 1.0 if extracted answer matches ground truth, 0.0 otherwise.
    """
    try:
        pattern = re.compile(answer_extraction_pattern, re.DOTALL)
    except re.error as e:
        logger.error(f"Invalid answer extraction pattern '{answer_extraction_pattern}': {e}")
        raise ValueError(f"Invalid regex pattern for answer extraction: {e}")

    def reward_fn(prompts, completions):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            expected = prompt_to_answer.get(prompt)
            if expected is None:
                # No ground truth for this prompt — give neutral score
                rewards.append(0.0)
                continue

            # Extract answer from completion
            match = pattern.search(completion)
            if match:
                extracted = match.group(1)
            else:
                # No match found — wrong format
                rewards.append(0.0)
                continue

            # Compare
            if strip_whitespace:
                extracted = extracted.strip()
                expected = expected.strip()
            if not case_sensitive:
                extracted = extracted.lower()
                expected = expected.lower()

            rewards.append(1.0 if extracted == expected else 0.0)
        return rewards

    return reward_fn


def build_regex_reward(
    format_pattern: str,
) -> Callable:
    """Regex reward: scores 1.0 if completion matches the pattern, 0.0 otherwise.

    Args:
        format_pattern: Regex pattern the completion should match.
            Examples:
            - r"<think>.*?</think>" for thinking tags
            - r"\\{.*\\}" for JSON-like output
            - r"^\\d+\\." for numbered lists

    Returns:
        Reward function scoring 1.0 for matches, 0.0 otherwise.
    """
    try:
        pattern = re.compile(format_pattern, re.DOTALL)
    except re.error as e:
        logger.error(f"Invalid format pattern '{format_pattern}': {e}")
        raise ValueError(f"Invalid regex pattern for format reward: {e}")

    def reward_fn(prompts, completions):
        return [1.0 if pattern.search(c) else 0.0 for c in completions]

    return reward_fn


def build_composite_reward(
    reward_fns: list[tuple[Callable, float]],
) -> Callable:
    """Composite reward: weighted combination of multiple reward functions.

    Args:
        reward_fns: List of (reward_fn, weight) tuples.
            Weights are used as-is (not normalized) so they can sum to >1.0 or <1.0.

    Returns:
        Reward function returning weighted sum of component rewards.
    """
    def reward_fn(prompts, completions):
        combined = [0.0] * len(completions)
        for fn, weight in reward_fns:
            scores = fn(prompts, completions)
            for i, s in enumerate(scores):
                combined[i] += s * weight
        return combined

    return reward_fn


def build_reward_function(
    reward_type: str,
    prompt_to_answer: Optional[dict[str, str]] = None,
    answer_extraction_pattern: str = r"\\boxed\{(.*?)\}",
    answer_case_sensitive: bool = False,
    format_pattern: Optional[str] = None,
    accuracy_weight: float = 0.8,
    format_weight: float = 0.2,
) -> Callable:
    """Factory function to build a reward function from config parameters.

    Args:
        reward_type: One of 'length', 'format', 'combined', 'answer_match',
            'regex', 'answer_and_format'.
        prompt_to_answer: Required for 'answer_match' and 'answer_and_format'.
            Maps prompt text to expected answer.
        answer_extraction_pattern: Regex to extract answer from completion.
        answer_case_sensitive: Whether answer comparison is case-sensitive.
        format_pattern: Required for 'regex' and 'answer_and_format'.
        accuracy_weight: Weight for answer_match in 'answer_and_format' composite.
        format_weight: Weight for regex/format in 'answer_and_format' composite.

    Returns:
        A reward function with signature (prompts: list[str], completions: list[str]) -> list[float].
    """
    if reward_type == "length":
        return build_length_reward()

    elif reward_type == "format":
        return build_format_reward()

    elif reward_type == "combined":
        return build_combined_reward()

    elif reward_type == "answer_match":
        if not prompt_to_answer:
            raise ValueError(
                "answer_match reward requires a dataset with an 'answer' column. "
                "Map your ground-truth column to 'answer' in the column mapping."
            )
        return build_answer_match_reward(
            prompt_to_answer=prompt_to_answer,
            answer_extraction_pattern=answer_extraction_pattern,
            case_sensitive=answer_case_sensitive,
        )

    elif reward_type == "regex":
        if not format_pattern:
            raise ValueError(
                "regex reward requires a format_pattern. "
                "Provide a regex pattern that completions should match."
            )
        return build_regex_reward(format_pattern=format_pattern)

    elif reward_type == "answer_and_format":
        components = []

        if prompt_to_answer:
            answer_fn = build_answer_match_reward(
                prompt_to_answer=prompt_to_answer,
                answer_extraction_pattern=answer_extraction_pattern,
                case_sensitive=answer_case_sensitive,
            )
            components.append((answer_fn, accuracy_weight))
        else:
            raise ValueError(
                "answer_and_format reward requires a dataset with an 'answer' column."
            )

        if format_pattern:
            regex_fn = build_regex_reward(format_pattern=format_pattern)
            components.append((regex_fn, format_weight))
        else:
            # Fall back to built-in format reward
            components.append((build_format_reward(), format_weight))

        return build_composite_reward(components)

    else:
        raise ValueError(
            f"Unknown reward type: '{reward_type}'. "
            f"Supported: length, format, combined, answer_match, regex, answer_and_format"
        )
