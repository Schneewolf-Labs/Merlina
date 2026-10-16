"""
Tests for remote inference sizing and planning (src/remote/sizing.py serve
helpers, src/remote/inference.py). No network: model specs are injected.

Run standalone: python -m pytest tests/test_remote_inference_plan.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_manager import JobManager
from src.remote.inference import (
    RemoteInferenceRequest,
    build_serve_plan,
    serve_spec,
)
from src.remote.lifecycle import RemoteLifecycle
from src.remote.plan import RemotePlanError
from src.remote.sizing import (
    DEFAULT_KV_BYTES_PER_TOKEN,
    GIB,
    estimate_serve_vram_gb,
    kv_bytes_per_token,
    parse_model_specs,
    pick_instance,
    serve_vram_breakdown,
    serve_weights_gb,
    tensor_parallel_counts,
)
from src.remote.spec import GpuOffer
from tests.test_remote_sizing import KIMI_K2_CONFIG, LLAMA_7B_CONFIG

SETTINGS = SimpleNamespace(remote_serve_image="ghcr.io/example/merlina-serve:test")

# A DeepSeek-V3-style MLA MoE shipped as an FP8 checkpoint (the shape of
# DeepSeek-V4-Flash-class models): the safetensors index reports ~167 GB,
# far below what the parameter count would need in bf16.
FP8_MLA_MOE_CONFIG = {
    **KIMI_K2_CONFIG,
    "n_routed_experts": 256,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "num_attention_heads": 128,
    "max_position_embeddings": 163840,
}
FP8_TOTAL_SIZE = 167_000_000_000

QWEN_LIKE_CONFIG = {
    **LLAMA_7B_CONFIG,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 32768,
}


def specs(config=LLAMA_7B_CONFIG, total_size=None, model_id="org/model"):
    return parse_model_specs(model_id, config, total_size)


def request(**overrides):
    body = {"model": {"hf_repo": "org/model"}}
    body.update(overrides)
    return RemoteInferenceRequest(**body)


def plan_for(req=None, *, model_specs=None, offers=None, **kwargs):
    return build_serve_plan(req or request(), model_specs=model_specs or specs(),
                            offers=offers, settings=SETTINGS, **kwargs)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

class TestServeSizing:
    def test_fp8_checkpoint_sized_from_total_size(self):
        s = specs(FP8_MLA_MOE_CONFIG, FP8_TOTAL_SIZE)
        assert serve_weights_gb(s) == pytest.approx(FP8_TOTAL_SIZE / GIB)
        # The bf16 assumption would be several times larger for this MoE.
        assert s.total_params_b * 2e9 / GIB > 3 * serve_weights_gb(s)

    def test_missing_index_falls_back_to_bf16_params(self):
        s = specs(LLAMA_7B_CONFIG)
        assert serve_weights_gb(s) == pytest.approx(s.total_params_b * 2e9 / GIB)

    def test_gqa_kv_bytes_per_token(self):
        # 32 layers * (K+V) * 32 kv heads * 128 head_dim * 2 bytes
        assert kv_bytes_per_token(specs()) == 32 * 2 * 32 * 128 * 2
        # GQA: only the 4 KV heads are cached
        assert kv_bytes_per_token(specs(QWEN_LIKE_CONFIG)) == 32 * 2 * 4 * 128 * 2

    def test_mla_kv_uses_compressed_latent(self):
        s = specs(FP8_MLA_MOE_CONFIG, FP8_TOTAL_SIZE)
        assert kv_bytes_per_token(s) == 61 * (512 + 64) * 2

    def test_unknown_attention_uses_conservative_default(self):
        s = specs({"architectures": ["Mystery"], "vocab_size": 1000})
        assert kv_bytes_per_token(s) == DEFAULT_KV_BYTES_PER_TOKEN

    def test_kv_allowance_scales_with_context_and_concurrency(self):
        s = specs()
        one = serve_vram_breakdown(s, 4096, 1)
        many = serve_vram_breakdown(s, 4096, 8)
        assert one["kv_tokens"] == 4096
        assert one["kv_cache_gb"] == pytest.approx(4096 * 32 * 2 * 32 * 128 * 2 / GIB, abs=0.1)
        assert many["kv_cache_gb"] == pytest.approx(8 * one["kv_cache_gb"], abs=0.2)
        assert estimate_serve_vram_gb(s, 4096, 8) > estimate_serve_vram_gb(s, 4096, 1)

    def test_lora_allowance(self):
        s = specs()
        assert estimate_serve_vram_gb(s, 2048, 1, lora=True) > estimate_serve_vram_gb(s, 2048, 1)

    def test_tensor_parallel_counts_divide_heads(self):
        assert tensor_parallel_counts(specs()) == [1, 2, 4, 8]                   # 32 heads
        assert tensor_parallel_counts(specs(QWEN_LIKE_CONFIG)) == [1, 2, 4]      # 28 heads
        assert tensor_parallel_counts(None) == [1, 2, 4, 8]

    def test_pick_instance_respects_allowed_counts(self):
        offers = [GpuOffer("G24", "G24", 24, price_per_hr_secure=1.0)]
        # ~60 GB needs 3 of these, but TP can't use 3 -> 4
        assert pick_instance(60, 50, offers).gpu_count == 3
        assert pick_instance(60, 50, offers, gpu_counts=[1, 2, 4, 8]).gpu_count == 4


# ---------------------------------------------------------------------------
# Plans from HF repos
# ---------------------------------------------------------------------------

class TestServePlan:
    def test_fp8_model_fits_where_bf16_assumption_would_not(self):
        exact = plan_for(model_specs=specs(FP8_MLA_MOE_CONFIG, FP8_TOTAL_SIZE))
        assert exact.vram_breakdown["weights_exact"] is True
        assert exact.sizing.gpu_count * 192 >= exact.vram_breakdown["total_gb"]
        guessed = plan_for(model_specs=specs(FP8_MLA_MOE_CONFIG))
        assert guessed.vram_breakdown["total_gb"] > 3 * exact.vram_breakdown["total_gb"]
        assert guessed.sizing.gpu_count > exact.sizing.gpu_count
        assert any("safetensors index" in w for w in guessed.warnings)

    def test_tensor_parallel_defaults_to_gpu_count(self):
        plan = plan_for(request(gpu_type_id="NVIDIA GeForce RTX 4090", gpu_count=4),
                        model_specs=specs())
        assert plan.sizing.gpu_count == 4
        assert plan.tensor_parallel_size == 4
        assert plan.vllm_command[plan.vllm_command.index("--tensor-parallel-size") + 1] == "4"

    def test_auto_pick_uses_tp_compatible_counts(self):
        offers = [GpuOffer("G24", "G24", 24, price_per_hr_secure=1.0)]
        plan = plan_for(request(max_num_seqs=1, max_model_len=2048),
                        model_specs=specs(QWEN_LIKE_CONFIG, 50 * GIB), offers=offers)
        assert plan.sizing.gpu_count in (1, 2, 4)
        assert 28 % plan.tensor_parallel_size == 0

    def test_max_model_len_defaults_to_capped_model_max(self):
        assert plan_for(model_specs=specs(QWEN_LIKE_CONFIG)).max_model_len == 8192
        assert plan_for(request(max_model_len=4096),
                        model_specs=specs(QWEN_LIKE_CONFIG)).max_model_len == 4096
        over = plan_for(request(max_model_len=65536), model_specs=specs(QWEN_LIKE_CONFIG))
        assert any("max_position_embeddings" in w for w in over.warnings)

    def test_runtime_cap_is_the_deadline_without_a_cost_cap(self):
        plan = plan_for(request(max_runtime_hours=3))
        assert plan.effective_max_hours == 3
        assert plan.deadline_reason == "max_runtime_hours"
        assert plan.est_max_total_cost_usd == pytest.approx(3 * plan.sizing.est_cost_per_hr, abs=0.01)

    def test_total_cost_cap_shortens_the_deadline(self):
        offers = [GpuOffer("G80", "G80", 80, price_per_hr_secure=2.0)]
        plan = plan_for(request(max_runtime_hours=4, max_total_cost_usd=3.0), offers=offers)
        assert plan.effective_max_hours == pytest.approx(1.5)
        assert plan.deadline_reason == "max_total_cost_usd"
        assert plan.est_max_total_cost_usd == pytest.approx(3.0)

    def test_runtime_cap_wins_when_earlier_than_cost_cap(self):
        offers = [GpuOffer("G80", "G80", 80, price_per_hr_secure=2.0)]
        plan = plan_for(request(max_runtime_hours=1, max_total_cost_usd=100), offers=offers)
        assert plan.deadline_reason == "max_runtime_hours"
        assert plan.est_max_total_cost_usd == pytest.approx(2.0)

    def test_cost_cap_needs_a_price(self):
        with pytest.raises(RemotePlanError, match="price"):
            plan_for(request(gpu_type_id="UNLISTED-GPU", max_total_cost_usd=5))

    def test_tiny_budget_warns(self):
        offers = [GpuOffer("G80", "G80", 80, price_per_hr_secure=4.0)]
        plan = plan_for(request(max_total_cost_usd=0.5), offers=offers)
        assert any("minutes" in w for w in plan.warnings)

    def test_max_cost_per_hr_filters(self):
        offers = [GpuOffer("CHEAP", "c", 24, price_per_hr_secure=0.5),
                  GpuOffer("BIG", "b", 80, price_per_hr_secure=3.0)]
        with pytest.raises(RemotePlanError):
            plan_for(request(max_cost_per_hr=1.0), model_specs=specs(LLAMA_7B_CONFIG, 60 * GIB),
                     offers=offers)

    def test_hf_command_has_revision_and_no_secret(self):
        plan = plan_for(request(model={"hf_repo": "org/model", "revision": "abc123"},
                                hf_token="hf_secret", vllm_args=["--enable-prefix-caching"]))
        cmd = plan.vllm_command
        assert cmd[:3] == ["vllm", "serve", "org/model"]
        assert cmd[cmd.index("--revision") + 1] == "abc123"
        assert cmd[cmd.index("--served-model-name") + 1] == "model"
        assert cmd[-1] == "--enable-prefix-caching"
        assert "--api-key" not in cmd
        assert "hf_secret" not in str(plan.to_dict())

    def test_worker_image_default_and_override(self):
        assert plan_for().worker_image == SETTINGS.remote_serve_image
        assert plan_for(request(worker_image="my/img:1")).worker_image == "my/img:1"

    def test_serve_spec_round_trip(self):
        plan = plan_for(request(served_model_name="mine"))
        spec = serve_spec(plan)
        assert spec["source_kind"] == "hf"
        assert spec["served_model_name"] == "mine"
        assert spec["tensor_parallel_size"] == plan.tensor_parallel_size


class TestRequestValidation:
    def test_exactly_one_source(self):
        with pytest.raises(ValidationError):
            RemoteInferenceRequest(model={})
        with pytest.raises(ValidationError):
            RemoteInferenceRequest(model={"hf_repo": "a/b", "job_id": "job_1"})

    def test_local_paths_rejected(self):
        with pytest.raises(ValidationError):
            RemoteInferenceRequest(model={"hf_repo": "/models/mine"})

    def test_runtime_capped_at_24h(self):
        with pytest.raises(ValidationError):
            request(max_runtime_hours=25)
        assert request().max_runtime_hours == 2.0

    @pytest.mark.parametrize("flag", ["--api-key", "--port=9000", "--host", "--served-model-name"])
    def test_reserved_vllm_flags_rejected(self, flag):
        with pytest.raises(ValidationError, match="Merlina manages"):
            request(vllm_args=[flag, "x"])


# ---------------------------------------------------------------------------
# Plans from training jobs (train -> serve)
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    manager = JobManager(db_path=str(tmp_path / "jobs.db"))
    return manager, RemoteLifecycle(manager.db_path)


def finished_remote_job(db, job_id="job_1", *, artifacts=("adapter",), status="completed",
                        remote=True, **config):
    manager, lifecycle = db
    cfg = {"base_model": "org/base", "output_name": "you/my-tune", "use_lora": True,
           "lora_r": 64, "hf_token": "hf_job", "remote": {"enabled": True} if remote else None}
    cfg.update(config)
    manager.create_job(job_id, cfg)
    manager.update_job(job_id, status=status,
                       metrics={"artifact_store": {"kind": "local", "base_dir": "/store"}})
    owner = lifecycle.claim(job_id, 1)
    lifecycle.intent(job_id, owner, "train", "runpod")
    lifecycle.stage_state(job_id, "train", "completed", artifacts=list(artifacts))
    return job_id


def job_request(job_id="job_1", artifact="adapter", **overrides):
    return request(model={"job_id": job_id, "artifact": artifact}, **overrides)


class TestJobSource:
    def test_adapter_serves_base_plus_lora(self, db):
        finished_remote_job(db)
        plan = plan_for(job_request(), job_manager=db[0], lifecycle=db[1])
        assert plan.lora is True
        assert plan.model_id == "org/base"
        assert plan.served_model_name == "my-tune"
        assert plan.base_served_model_name == "org/base"
        assert plan.max_lora_rank == 64
        assert plan.artifact_store == {"kind": "local", "base_dir": "/store"}
        cmd = plan.vllm_command
        assert "--enable-lora" in cmd
        assert cmd[cmd.index("--lora-modules") + 1] == "my-tune=<artifact:adapter>"
        assert cmd[cmd.index("--served-model-name") + 1] == "org/base"
        assert plan.vram_breakdown["lora_gb"] > 0

    def test_lora_rank_rounds_up_to_vllm_value(self, db):
        finished_remote_job(db, lora_r=48)
        assert plan_for(job_request(), job_manager=db[0], lifecycle=db[1]).max_lora_rank == 64

    def test_full_finetune_adapter_artifact_serves_as_model(self, db):
        finished_remote_job(db, use_lora=False)
        plan = plan_for(job_request(), job_manager=db[0], lifecycle=db[1])
        assert plan.lora is False
        assert plan.vllm_command[2] == "<artifact:adapter>"

    def test_merged_artifact(self, db):
        finished_remote_job(db, artifacts=("adapter", "merged"))
        plan = plan_for(job_request(artifact="merged"), job_manager=db[0], lifecycle=db[1])
        assert plan.lora is False and plan.artifact == "merged"

    def test_missing_merged_explains_merge_strategy(self, db):
        finished_remote_job(db)
        with pytest.raises(RemotePlanError, match="merge_strategy='remote'"):
            plan_for(job_request(artifact="merged"), job_manager=db[0], lifecycle=db[1])

    def test_hub_published_merge_points_to_hf_repo(self, db):
        finished_remote_job(db, artifacts=("adapter", "hub:you/my-tune"))
        with pytest.raises(RemotePlanError, match="hf_repo='you/my-tune'"):
            plan_for(job_request(artifact="merged"), job_manager=db[0], lifecycle=db[1])

    def test_local_job_refused(self, db):
        finished_remote_job(db, remote=False)
        with pytest.raises(RemotePlanError, match="trained locally"):
            plan_for(job_request(), job_manager=db[0], lifecycle=db[1])

    def test_unfinished_job_refused(self, db):
        finished_remote_job(db, status="training")
        with pytest.raises(RemotePlanError, match="only completed"):
            plan_for(job_request(), job_manager=db[0], lifecycle=db[1])

    def test_unknown_job(self, db):
        with pytest.raises(RemotePlanError, match="not found"):
            plan_for(job_request("nope"), job_manager=db[0], lifecycle=db[1])

    def test_modules_to_save_warns(self, db):
        finished_remote_job(db, modules_to_save=["embed_tokens", "lm_head"])
        plan = plan_for(job_request(), job_manager=db[0], lifecycle=db[1])
        assert any("merged" in w for w in plan.warnings)

    def test_job_token_never_in_plan(self, db):
        finished_remote_job(db)
        plan = plan_for(job_request(), job_manager=db[0], lifecycle=db[1])
        assert "hf_job" not in str(plan.to_dict())
