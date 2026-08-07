#!/usr/bin/env python3
"""
Tests for the architecture-aware VRAM estimator (src/vram_estimator.py).

Covers model profiling (local config.json, safetensors index, name-heuristic
fallback, VLM text_config nesting), parameter-count derivation accuracy,
the component breakdown, and the monotonicity properties the pre-flight
suggestions rely on (4-bit < bf16, checkpointing < none, paired preference
> SFT, liger < full logits, 8-bit optimizer < 32-bit, LoRA < full FT).

Runs standalone (python tests/test_vram_estimator.py) or under pytest.
No GPU and no network required — torch is faked where needed and Hub
lookups are never exercised.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Standalone-run support: importing src triggers torch via preflight_checks.
if 'torch' not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        _mock_torch = mock.MagicMock()
        _mock_torch.__spec__ = mock.MagicMock()
        _mock_torch.cuda.is_available.return_value = False
        sys.modules['torch'] = _mock_torch
if 'psutil' not in sys.modules:
    try:
        import psutil  # noqa: F401
    except ImportError:
        _mock_psutil = mock.MagicMock()
        _mock_psutil.__spec__ = mock.MagicMock()
        sys.modules['psutil'] = _mock_psutil

from src.vram_estimator import (  # noqa: E402
    ModelProfile,
    _derive_param_count,
    _profile_from_config_dict,
    _profile_from_local_path,
    clear_profile_cache,
    estimate_vram,
    get_model_profile,
    get_model_size_from_name,
    suggest_reductions,
)

LLAMA3_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}


def _llama3_8b_profile() -> ModelProfile:
    return _profile_from_config_dict("test/llama3-8b", LLAMA3_8B_CONFIG, "local_config")


# ---------------------------------------------------------------------------
# Name heuristic
# ---------------------------------------------------------------------------

def test_size_from_name_basic():
    assert get_model_size_from_name("meta-llama/Meta-Llama-3-8B-Instruct") == 8.0
    assert get_model_size_from_name("mistralai/Mistral-7B-v0.1") == 7.0
    assert get_model_size_from_name("Qwen/Qwen2.5-0.5B") == 0.5
    assert get_model_size_from_name("my-model-13b-chat") == 13.0
    print("✓ size from name: basic patterns")


def test_size_from_name_moe_and_misses():
    # MoE total params, not per-expert
    assert get_model_size_from_name("mistralai/Mixtral-8x7B-v0.1") == 56.0
    # No size in name
    assert get_model_size_from_name("mistralai/Mistral-Nemo-Instruct-2407") is None
    assert get_model_size_from_name("my-cool-model") is None
    print("✓ size from name: MoE + misses")


# ---------------------------------------------------------------------------
# Profile building
# ---------------------------------------------------------------------------

def test_derive_param_count_llama3_8b():
    profile = _llama3_8b_profile()
    # Real Llama-3-8B has 8.03B params; the derivation should land within 2%.
    assert profile.num_params is not None
    billions = profile.num_params / 1e9
    assert 7.9 < billions < 8.2, f"derived {billions:.2f}B, expected ~8.03B"
    print(f"✓ derived Llama-3-8B params: {billions:.2f}B")


def test_profile_from_vlm_text_config():
    vlm_config = {
        "architectures": ["SomeVLMForConditionalGeneration"],
        "vision_config": {"hidden_size": 1152},
        "text_config": dict(LLAMA3_8B_CONFIG),
    }
    profile = _profile_from_config_dict("test/vlm", vlm_config, "hub_config")
    assert profile.hidden_size == 4096
    assert profile.vocab_size == 128256
    print("✓ VLM text_config nesting")


def test_profile_from_local_path_with_safetensors_index():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with open(tmp_path / "config.json", "w") as f:
            json.dump(LLAMA3_8B_CONFIG, f)
        # 8.03B params in bf16 = 16.06e9 bytes
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": 16_060_000_000}, "weight_map": {}}, f)

        profile = _profile_from_local_path(str(tmp_path))
        assert profile is not None
        assert profile.source == "local_config"
        assert profile.params_exact
        assert abs(profile.num_params - 8.03e9) < 0.01e9
        print("✓ local path profile with exact safetensors params")


def test_get_model_profile_name_fallback():
    clear_profile_cache()
    # Hub unreachable → name heuristic with preset dims
    with mock.patch("src.vram_estimator._profile_from_hub", return_value=None):
        profile = get_model_profile("nonexistent-org/some-model-8b", use_cache=False)
    assert profile is not None
    assert profile.source == "name_heuristic"
    assert profile.num_params == 8.0e9
    assert profile.has_dims
    print("✓ name-heuristic fallback profile")


def test_get_model_profile_unknown_returns_none():
    clear_profile_cache()
    with mock.patch("src.vram_estimator._profile_from_hub", return_value=None):
        profile = get_model_profile("nonexistent-org/mystery-model", use_cache=False)
    assert profile is None
    print("✓ unknown model returns None")


# ---------------------------------------------------------------------------
# Estimation: absolute sanity + component breakdown
# ---------------------------------------------------------------------------

def test_qlora_orpo_estimate_in_plausible_range():
    est = estimate_vram(
        _llama3_8b_profile(),
        batch_size=1, max_length=2048, training_mode="orpo",
        use_4bit=True, use_lora=True, gradient_checkpointing=True,
    )
    # 8B QLoRA ORPO with checkpointing runs comfortably on a 24GB card.
    assert 8 < est.total_gb < 24, f"got {est.total_gb:.1f}GB"
    assert est.confidence == "medium"  # config-derived, not safetensors-exact
    print(f"✓ QLoRA ORPO ckpt estimate: {est.total_gb:.1f}GB")


def test_breakdown_components_present():
    est = estimate_vram(_llama3_8b_profile(), training_mode="orpo")
    for key in ("model_weights", "trainable_params", "gradients",
                "optimizer_states", "activations", "logits_and_loss",
                "cuda_context", "fragmentation_buffer"):
        assert key in est.breakdown, f"missing {key}"
        assert est.breakdown[key] >= 0
    assert abs(sum(est.breakdown.values()) - est.total_gb) < 0.01
    print("✓ breakdown components present and consistent")


def test_paired_mode_doubles_forward_batch():
    profile = _llama3_8b_profile()
    sft = estimate_vram(profile, training_mode="sft")
    orpo = estimate_vram(profile, training_mode="orpo")
    assert orpo.breakdown["activations"] > sft.breakdown["activations"] * 1.9
    assert orpo.breakdown["logits_and_loss"] > sft.breakdown["logits_and_loss"] * 1.9
    print("✓ paired preference mode doubles activations/logits")


def test_reference_model_only_for_dpo_kto_without_lora():
    profile = _llama3_8b_profile()
    dpo_full = estimate_vram(profile, training_mode="dpo", use_lora=False, use_4bit=False)
    dpo_lora = estimate_vram(profile, training_mode="dpo", use_lora=True, use_4bit=False)
    orpo_full = estimate_vram(profile, training_mode="orpo", use_lora=False, use_4bit=False)
    assert dpo_full.breakdown.get("reference_model", 0) > 10
    assert dpo_lora.breakdown.get("reference_model", 0) == 0
    assert orpo_full.breakdown.get("reference_model", 0) == 0
    print("✓ reference model counted only for DPO/KTO without LoRA")


def test_monotonicity_properties():
    profile = _llama3_8b_profile()
    base = dict(batch_size=1, max_length=2048, training_mode="sft")

    bf16 = estimate_vram(profile, use_4bit=False, **base)
    q4 = estimate_vram(profile, use_4bit=True, **base)
    assert q4.breakdown["model_weights"] < bf16.breakdown["model_weights"] * 0.6

    no_ckpt = estimate_vram(profile, gradient_checkpointing=False, **base)
    ckpt = estimate_vram(profile, gradient_checkpointing=True, **base)
    assert ckpt.breakdown["activations"] < no_ckpt.breakdown["activations"] * 0.5

    liger = estimate_vram(profile, use_liger=True, **base)
    no_liger = estimate_vram(profile, use_liger=False, **base)
    assert liger.breakdown["logits_and_loss"] < no_liger.breakdown["logits_and_loss"] * 0.3

    opt8 = estimate_vram(profile, use_lora=False, optimizer_type="paged_adamw_8bit", **base)
    opt32 = estimate_vram(profile, use_lora=False, optimizer_type="adamw_torch", **base)
    assert opt8.breakdown["optimizer_states"] < opt32.breakdown["optimizer_states"]

    lora = estimate_vram(profile, use_lora=True, **base)
    full = estimate_vram(profile, use_lora=False, use_4bit=False, **base)
    assert lora.total_gb < full.total_gb

    eager = estimate_vram(profile, attn_implementation="eager", **base)
    flash = estimate_vram(profile, attn_implementation="flash_attention_2", **base)
    assert eager.breakdown["activations"] > flash.breakdown["activations"]

    longer = estimate_vram(profile, batch_size=1, max_length=8192, training_mode="sft")
    shorter = estimate_vram(profile, batch_size=1, max_length=1024, training_mode="sft")
    assert longer.total_gb > shorter.total_gb
    print("✓ monotonicity: 4bit/ckpt/liger/8bit-opt/lora/flash/seq-length")


def test_modules_to_save_increases_trainable():
    profile = _llama3_8b_profile()
    without = estimate_vram(profile, modules_to_save=[])
    with_embed = estimate_vram(profile, modules_to_save=["embed_tokens", "lm_head"])
    # Two full 128k×4096 matrices dwarf the LoRA adapters.
    assert with_embed.breakdown["optimizer_states"] > without.breakdown["optimizer_states"] * 3
    print("✓ modules_to_save counted as trainable")


def test_vocab_size_drives_logits_memory():
    small_vocab = dict(LLAMA3_8B_CONFIG, vocab_size=32000)
    p_small = _profile_from_config_dict("test/small-vocab", small_vocab, "local_config")
    p_big = _llama3_8b_profile()
    small = estimate_vram(p_small, training_mode="orpo")
    big = estimate_vram(p_big, training_mode="orpo")
    ratio = big.breakdown["logits_and_loss"] / small.breakdown["logits_and_loss"]
    assert 3.0 < ratio < 4.5, f"expected ~4x (128k/32k vocab), got {ratio:.1f}x"
    print("✓ logits memory scales with vocab size")


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def test_suggestions_target_dominant_component():
    from types import SimpleNamespace
    profile = _llama3_8b_profile()

    # Activations dominate without checkpointing → suggest it first.
    est = estimate_vram(profile, batch_size=4, max_length=4096,
                        training_mode="orpo", gradient_checkpointing=False)
    config = SimpleNamespace(gradient_checkpointing=False, use_liger=False,
                             use_4bit=True, batch_size=4, max_length=4096,
                             optimizer_type="paged_adamw_8bit")
    suggestions = suggest_reductions(est, config)
    assert any("gradient_checkpointing" in s for s in suggestions)
    assert "gradient_checkpointing" in suggestions[0]

    # With checkpointing on and a big vocab, logits dominate → suggest liger.
    est2 = estimate_vram(profile, batch_size=1, max_length=4096,
                         training_mode="orpo", gradient_checkpointing=True)
    config2 = SimpleNamespace(gradient_checkpointing=True, use_liger=False,
                              use_4bit=True, batch_size=1, max_length=4096,
                              optimizer_type="paged_adamw_8bit")
    suggestions2 = suggest_reductions(est2, config2)
    if est2.dominant_component() == "logits_and_loss":
        assert any("use_liger" in s for s in suggestions2)
    print("✓ suggestions target the dominant component")


def test_profile_cache():
    clear_profile_cache()
    with mock.patch("src.vram_estimator._profile_from_hub", return_value=None):
        p1 = get_model_profile("cached-org/model-7b")
        p2 = get_model_profile("cached-org/model-7b")
    assert p1 is p2
    clear_profile_cache()
    print("✓ profile cache")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} VRAM estimator tests...\n")
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"✗ {test.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
