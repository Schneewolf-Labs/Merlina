"""
Tests for MoE-aware remote sizing (src/remote/sizing.py).

Run standalone: python -m pytest tests/test_remote_sizing.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.remote.sizing import (
    FALLBACK_GPU_CATALOG,
    estimate_disk_gb,
    estimate_train_vram_gb,
    parse_model_specs,
    pick_instance,
)
from src.remote.spec import GpuOffer


# Realistic config.json shapes
LLAMA_7B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "intermediate_size": 11008,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "vocab_size": 32000,
    "tie_word_embeddings": False,
}

# Kimi-K2-style DeepSeek-V3 MoE: 1T total / ~32B active
KIMI_K2_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 128,
    "vocab_size": 163840,
    "tie_word_embeddings": False,
}


class TestParseModelSpecs:
    def test_dense_7b_param_estimate(self):
        specs = parse_model_specs("meta-llama/Llama-2-7b", LLAMA_7B_CONFIG)
        assert not specs.is_moe
        assert 6.0 <= specs.total_params_b <= 8.0
        # Dense model: active == total
        assert specs.active_params_b == specs.total_params_b

    def test_kimi_k2_total_params_near_1t(self):
        specs = parse_model_specs("moonshotai/Kimi-K2", KIMI_K2_CONFIG)
        assert specs.is_moe
        assert 900 <= specs.total_params_b <= 1150

    def test_kimi_k2_active_params_far_below_total(self):
        specs = parse_model_specs("moonshotai/Kimi-K2", KIMI_K2_CONFIG)
        assert 20 <= specs.active_params_b <= 60
        assert specs.active_params_b < specs.total_params_b / 10

    def test_missing_fields_degrade_to_none(self):
        specs = parse_model_specs("mystery/model", {"architectures": ["X"]})
        assert specs.total_params_b is None
        assert not specs.is_moe

    def test_nested_text_config(self):
        specs = parse_model_specs("some/vlm", {"text_config": LLAMA_7B_CONFIG})
        assert specs.hidden_size == 4096

    def test_weight_bytes_passthrough(self):
        specs = parse_model_specs("m/x", LLAMA_7B_CONFIG, weight_bytes_total=14_000_000_000)
        assert specs.weight_bytes_total == 14_000_000_000


class TestVramEstimation:
    def test_7b_qlora_fits_consumer_gpu(self):
        specs = parse_model_specs("m/7b", LLAMA_7B_CONFIG)
        vram = estimate_train_vram_gb(specs, use_4bit=True)
        assert 4 <= vram <= 16

    def test_bf16_needs_more_than_4bit(self):
        specs = parse_model_specs("m/7b", LLAMA_7B_CONFIG)
        assert estimate_train_vram_gb(specs, use_4bit=False) > \
               estimate_train_vram_gb(specs, use_4bit=True)

    def test_kimi_k2_qlora_needs_multi_gpu_class_vram(self):
        specs = parse_model_specs("moonshotai/Kimi-K2", KIMI_K2_CONFIG)
        vram = estimate_train_vram_gb(specs, use_4bit=True)
        # ~570 GB of 4-bit weights alone; estimate must reflect that scale
        assert vram > 550
        # ...but not be the absurd >5000 GB the name-based estimator gives
        assert vram < 1200

    def test_preference_mode_costs_more_than_sft(self):
        specs = parse_model_specs("m/7b", LLAMA_7B_CONFIG)
        sft = estimate_train_vram_gb(specs, preference_mode=False)
        pref = estimate_train_vram_gb(specs, preference_mode=True)
        assert pref > sft

    def test_longer_sequences_cost_more(self):
        specs = parse_model_specs("m/7b", LLAMA_7B_CONFIG)
        assert estimate_train_vram_gb(specs, max_length=8192) > \
               estimate_train_vram_gb(specs, max_length=1024)


class TestDiskEstimation:
    def test_train_disk_covers_weights(self):
        specs = parse_model_specs("m/7b", LLAMA_7B_CONFIG)
        assert estimate_disk_gb(specs) > 14  # bf16 download + overhead

    def test_merge_stage_needs_two_copies(self):
        specs = parse_model_specs("m/7b", LLAMA_7B_CONFIG)
        assert estimate_disk_gb(specs, merge_stage=True) > estimate_disk_gb(specs)


class TestPickInstance:
    def test_small_job_gets_cheapest_single_gpu(self):
        decision = pick_instance(8, 60, FALLBACK_GPU_CATALOG)
        assert decision.gpu_count == 1
        assert decision.parallelism == "single_gpu"
        # cheapest catalog entry that fits 8 GB is the 4090
        assert "4090" in decision.gpu_type_id

    def test_huge_job_goes_model_parallel(self):
        decision = pick_instance(689, 700, FALLBACK_GPU_CATALOG)
        assert decision.gpu_count > 1
        assert decision.parallelism == "model_parallel"
        assert decision.multi_gpu_strategy == "single"  # device_map=auto

    def test_picked_instance_actually_fits(self):
        decision = pick_instance(689, 700, FALLBACK_GPU_CATALOG)
        offer = next(o for o in FALLBACK_GPU_CATALOG
                     if o.gpu_type_id == decision.gpu_type_id)
        usable = (offer.vram_gb * 0.94 - 2.0) * decision.gpu_count
        assert usable >= 689

    def test_cheapest_feasible_wins(self):
        offers = [
            GpuOffer("expensive", "Big", 100, price_per_hr_secure=10.0),
            GpuOffer("cheap", "Small", 100, price_per_hr_secure=1.0),
        ]
        decision = pick_instance(50, 100, offers)
        assert decision.gpu_type_id == "cheap"

    def test_unavailable_offers_skipped(self):
        offers = [
            GpuOffer("gone", "Gone", 100, price_per_hr_secure=1.0, available=False),
            GpuOffer("here", "Here", 100, price_per_hr_secure=5.0),
        ]
        assert pick_instance(50, 100, offers).gpu_type_id == "here"

    def test_budget_filters_picks(self):
        offers = [GpuOffer("only", "Only", 100, price_per_hr_secure=5.0)]
        with pytest.raises(ValueError, match="under"):
            pick_instance(50, 100, offers, max_cost_per_hr=1.0)

    def test_nothing_fits_raises(self):
        offers = [GpuOffer("tiny", "Tiny", 8, price_per_hr_secure=0.2, max_gpu_count=1)]
        with pytest.raises(ValueError, match="fits"):
            pick_instance(500, 100, offers)

    def test_explicit_override_respected(self):
        decision = pick_instance(
            8, 60, FALLBACK_GPU_CATALOG, gpu_type_id="NVIDIA H200", gpu_count=2
        )
        assert decision.gpu_type_id == "NVIDIA H200"
        assert decision.gpu_count == 2

    def test_explicit_override_too_small_raises(self):
        with pytest.raises(ValueError, match="usable VRAM"):
            pick_instance(689, 700, FALLBACK_GPU_CATALOG,
                          gpu_type_id="NVIDIA GeForce RTX 4090", gpu_count=1)

    def test_community_cloud_pricing_used(self):
        offers = [GpuOffer("g", "G", 100, price_per_hr_secure=2.0,
                           price_per_hr_community=1.0)]
        decision = pick_instance(50, 100, offers, cloud_type="community")
        assert decision.est_cost_per_hr == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
