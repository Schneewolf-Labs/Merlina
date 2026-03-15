"""
Tests for GRPO reward functions.

Tests the reward function builders in src/reward_functions.py including
length, format, combined, answer_match, regex, and answer_and_format types.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly from the module file to avoid src/__init__.py pulling in
# fastapi/torch dependencies that aren't available in the test environment
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "reward_functions",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'reward_functions.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_length_reward = _mod.build_length_reward
build_format_reward = _mod.build_format_reward
build_combined_reward = _mod.build_combined_reward
build_answer_match_reward = _mod.build_answer_match_reward
build_regex_reward = _mod.build_regex_reward
build_composite_reward = _mod.build_composite_reward
build_reward_function = _mod.build_reward_function


class TestLengthReward:
    def test_longer_scores_higher(self):
        fn = build_length_reward()
        prompts = ["q1", "q2", "q3"]
        completions = ["short", "a medium length answer", "a very long and detailed answer that goes on"]
        rewards = fn(prompts, completions)
        assert rewards[0] < rewards[1] < rewards[2]

    def test_equal_lengths(self):
        fn = build_length_reward()
        rewards = fn(["q"], ["abc", "def"])
        assert rewards[0] == rewards[1] == 1.0

    def test_empty_completions(self):
        fn = build_length_reward()
        rewards = fn(["q"], ["", "hello"])
        assert rewards[0] == 0.0
        assert rewards[1] == 1.0


class TestFormatReward:
    def test_structured_output(self):
        fn = build_format_reward()
        rewards = fn(["q"], ["1. First item\n2. Second item"])
        assert rewards[0] == 1.0  # has newline + numbered list + len > 10

    def test_plain_short(self):
        fn = build_format_reward()
        rewards = fn(["q"], ["yes"])
        assert rewards[0] == 0.0  # no structure, len <= 10

    def test_code_block(self):
        fn = build_format_reward()
        rewards = fn(["q"], ["Here is the code:\n```python\nprint('hi')\n```"])
        assert rewards[0] == 1.0


class TestCombinedReward:
    def test_returns_scores(self):
        fn = build_combined_reward()
        rewards = fn(["q1", "q2"], ["short", "a much longer answer with content"])
        assert len(rewards) == 2
        assert all(0 <= r <= 1 for r in rewards)


class TestAnswerMatchReward:
    def test_correct_boxed_answer(self):
        lookup = {"What is 2+2?": "4"}
        fn = build_answer_match_reward(lookup, answer_extraction_pattern=r"\\boxed\{(.*?)\}")
        rewards = fn(
            ["What is 2+2?"],
            ["Let me calculate... \\boxed{4}"]
        )
        assert rewards[0] == 1.0

    def test_wrong_boxed_answer(self):
        lookup = {"What is 2+2?": "4"}
        fn = build_answer_match_reward(lookup, answer_extraction_pattern=r"\\boxed\{(.*?)\}")
        rewards = fn(
            ["What is 2+2?"],
            ["Let me calculate... \\boxed{5}"]
        )
        assert rewards[0] == 0.0

    def test_no_boxed_answer(self):
        lookup = {"What is 2+2?": "4"}
        fn = build_answer_match_reward(lookup, answer_extraction_pattern=r"\\boxed\{(.*?)\}")
        rewards = fn(
            ["What is 2+2?"],
            ["The answer is 4"]
        )
        assert rewards[0] == 0.0  # no extraction match

    def test_case_insensitive(self):
        lookup = {"question": "Paris"}
        fn = build_answer_match_reward(
            lookup,
            answer_extraction_pattern=r"Answer: (.*)",
            case_sensitive=False,
        )
        rewards = fn(["question"], ["Answer: paris"])
        assert rewards[0] == 1.0

    def test_case_sensitive(self):
        lookup = {"question": "Paris"}
        fn = build_answer_match_reward(
            lookup,
            answer_extraction_pattern=r"Answer: (.*)",
            case_sensitive=True,
        )
        rewards = fn(["question"], ["Answer: paris"])
        assert rewards[0] == 0.0

    def test_unknown_prompt(self):
        lookup = {"known": "yes"}
        fn = build_answer_match_reward(lookup, answer_extraction_pattern=r"(.*)")
        rewards = fn(["unknown"], ["anything"])
        assert rewards[0] == 0.0

    def test_xml_answer_pattern(self):
        lookup = {"q": "42"}
        fn = build_answer_match_reward(lookup, answer_extraction_pattern=r"<answer>(.*?)</answer>")
        rewards = fn(["q"], ["Let me think... <answer>42</answer>"])
        assert rewards[0] == 1.0

    def test_markdown_heading_pattern(self):
        lookup = {"q": "42"}
        fn = build_answer_match_reward(lookup, answer_extraction_pattern=r"#### (.*)")
        rewards = fn(["q"], ["Calculation steps...\n#### 42"])
        assert rewards[0] == 1.0


class TestRegexReward:
    def test_matching_pattern(self):
        fn = build_regex_reward(r"<think>.*?</think>")
        rewards = fn(["q"], ["<think>Let me reason</think> The answer is 4."])
        assert rewards[0] == 1.0

    def test_non_matching(self):
        fn = build_regex_reward(r"<think>.*?</think>")
        rewards = fn(["q"], ["The answer is 4."])
        assert rewards[0] == 0.0

    def test_json_pattern(self):
        fn = build_regex_reward(r"\{.*\}")
        rewards = fn(["q"], ['Here: {"key": "value"}'])
        assert rewards[0] == 1.0

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            build_regex_reward(r"[invalid")


class TestCompositeReward:
    def test_weighted_sum(self):
        fn1 = lambda p, c: [1.0] * len(c)
        fn2 = lambda p, c: [0.5] * len(c)
        composite = build_composite_reward([(fn1, 0.8), (fn2, 0.2)])
        rewards = composite(["q"], ["answer"])
        assert abs(rewards[0] - 0.9) < 1e-6  # 0.8*1.0 + 0.2*0.5

    def test_multiple_completions(self):
        fn1 = lambda p, c: [float(i) for i in range(len(c))]
        fn2 = lambda p, c: [1.0] * len(c)
        composite = build_composite_reward([(fn1, 0.5), (fn2, 0.5)])
        rewards = composite(["q", "q", "q"], ["a", "b", "c"])
        assert len(rewards) == 3
        assert abs(rewards[0] - 0.5) < 1e-6   # 0.5*0 + 0.5*1
        assert abs(rewards[1] - 1.0) < 1e-6   # 0.5*1 + 0.5*1
        assert abs(rewards[2] - 1.5) < 1e-6   # 0.5*2 + 0.5*1


class TestBuildRewardFunction:
    def test_length(self):
        fn = build_reward_function("length")
        rewards = fn(["q"], ["hello world"])
        assert len(rewards) == 1

    def test_format(self):
        fn = build_reward_function("format")
        rewards = fn(["q"], ["hello"])
        assert len(rewards) == 1

    def test_combined(self):
        fn = build_reward_function("combined")
        rewards = fn(["q"], ["hello"])
        assert len(rewards) == 1

    def test_answer_match(self):
        fn = build_reward_function(
            "answer_match",
            prompt_to_answer={"q": "42"},
            answer_extraction_pattern=r"#### (.*)",
        )
        rewards = fn(["q"], ["#### 42"])
        assert rewards[0] == 1.0

    def test_answer_match_no_lookup_raises(self):
        with pytest.raises(ValueError, match="answer_match reward requires"):
            build_reward_function("answer_match")

    def test_regex(self):
        fn = build_reward_function(
            "regex",
            format_pattern=r"\{.*\}",
        )
        rewards = fn(["q"], ['{"a": 1}'])
        assert rewards[0] == 1.0

    def test_regex_no_pattern_raises(self):
        with pytest.raises(ValueError, match="regex reward requires"):
            build_reward_function("regex")

    def test_answer_and_format(self):
        fn = build_reward_function(
            "answer_and_format",
            prompt_to_answer={"q": "42"},
            answer_extraction_pattern=r"<answer>(.*?)</answer>",
            format_pattern=r"<think>.*?</think>",
            accuracy_weight=0.8,
            format_weight=0.2,
        )
        # Both correct
        rewards = fn(["q"], ["<think>reasoning</think> <answer>42</answer>"])
        assert abs(rewards[0] - 1.0) < 1e-6  # 0.8*1 + 0.2*1

        # Answer correct, no format
        rewards = fn(["q"], ["<answer>42</answer>"])
        assert abs(rewards[0] - 0.8) < 1e-6  # 0.8*1 + 0.2*0

        # Wrong answer, has format
        rewards = fn(["q"], ["<think>reasoning</think> <answer>99</answer>"])
        assert abs(rewards[0] - 0.2) < 1e-6  # 0.8*0 + 0.2*1

    def test_answer_and_format_no_lookup_raises(self):
        with pytest.raises(ValueError, match="answer_and_format reward requires"):
            build_reward_function("answer_and_format")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reward type"):
            build_reward_function("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
