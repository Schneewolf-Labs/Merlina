"""
Tests for remote run planning (src/remote/plan.py).

Run standalone: python -m pytest tests/test_remote_plan.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.remote.plan import (
    RemotePlanError,
    build_remote_plan,
    check_remote_compatible,
)
from src.remote.sizing import parse_model_specs
from tests.test_remote_sizing import KIMI_K2_CONFIG, LLAMA_7B_CONFIG


def make_remote(**overrides):
    base = dict(
        enabled=True, provider="runpod", gpu_type_id=None, gpu_count=None,
        cloud_type="secure", container_disk_gb=None, volume_gb=None,
        max_cost_per_hr=None, max_runtime_hours=48.0, merge_strategy="auto",
        artifact_repo=None, worker_image=None, keep_instance_on_failure=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_config(*, source_type="huggingface", training_mode="sft", remote=None,
                base_model="some/model"):
    source = SimpleNamespace(source_type=source_type)
    dataset = SimpleNamespace(source=source, additional_sources=[])
    return SimpleNamespace(
        base_model=base_model,
        output_name="test-model",
        hf_token="hf_test",
        use_4bit=True,
        lora_r=64,
        batch_size=1,
        max_length=2048,
        training_mode=training_mode,
        dataset=dataset,
        remote=remote or make_remote(),
    )


class TestRemoteCompatibility:
    def test_hf_dataset_is_fine(self):
        assert check_remote_compatible(make_config()) == []

    def test_local_file_dataset_rejected(self):
        problems = check_remote_compatible(make_config(source_type="local_file"))
        assert problems and "local_file" in problems[0]

    def test_uploaded_dataset_rejected(self):
        assert check_remote_compatible(make_config(source_type="upload"))

    def test_vlm_mode_rejected(self):
        assert check_remote_compatible(make_config(training_mode="vlm_stage1"))


class TestBuildRemotePlan:
    def _plan(self, config_json, *, config=None, model_id="m/x"):
        config = config or make_config()
        specs = parse_model_specs(model_id, config_json)
        return build_remote_plan(config, model_specs=specs, job_id="job_1")

    def test_small_model_plans_train_plus_local_merge(self):
        plan = self._plan(LLAMA_7B_CONFIG)
        train = plan.stage("train")
        merge = plan.stage("merge")
        assert train is not None and train.target == "remote"
        assert train.artifacts_out == ["adapter"]
        assert merge is not None and merge.target == "local"
        assert merge.artifacts_in == ["adapter"]

    def test_kimi_k2_auto_plan_is_adapter_only(self):
        """1T-class bases skip the merge: the run produces an adapter."""
        plan = self._plan(KIMI_K2_CONFIG, model_id="moonshotai/Kimi-K2")
        assert plan.stage("merge") is None
        assert any("adapter-only" in w for w in plan.warnings)

    def test_kimi_k2_train_stage_is_model_parallel(self):
        plan = self._plan(KIMI_K2_CONFIG, model_id="moonshotai/Kimi-K2")
        sizing = plan.stage("train").sizing
        assert sizing.gpu_count > 1
        assert sizing.multi_gpu_strategy == "single"

    def test_explicit_remote_merge_respected_for_big_model(self):
        config = make_config(remote=make_remote(merge_strategy="remote"))
        plan = self._plan(KIMI_K2_CONFIG, config=config)
        assert plan.stage("merge").target == "remote"

    def test_skip_strategy_never_merges_small_model(self):
        config = make_config(remote=make_remote(merge_strategy="skip"))
        plan = self._plan(LLAMA_7B_CONFIG, config=config)
        assert plan.stage("merge") is None

    def test_incompatible_dataset_fails_planning(self):
        config = make_config(source_type="upload")
        specs = parse_model_specs("m/x", LLAMA_7B_CONFIG)
        with pytest.raises(RemotePlanError, match="huggingface"):
            build_remote_plan(config, model_specs=specs)

    def test_plan_to_dict_is_json_safe(self):
        import json
        plan = self._plan(KIMI_K2_CONFIG)
        json.dumps(plan.to_dict())  # must not raise

    def test_static_catalog_warning_when_no_offers(self):
        plan = self._plan(LLAMA_7B_CONFIG)
        assert any("static GPU catalog" in w for w in plan.warnings)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
