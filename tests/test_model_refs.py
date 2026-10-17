"""store:<name> base-model references: resolution into the local model cache,
offline fallback, pre-flight, sizing and model-card metadata. Fake S3 only."""
import json
from types import SimpleNamespace

import pytest

from src import model_refs
from src.model_refs import (
    ModelRefError,
    cached_ref_path,
    is_store_ref,
    resolve_model_ref,
    restore_adapter_base_ref,
    store_name,
)
from src.remote.s3_artifacts import S3ArtifactStore
from tests.test_s3_artifacts import FakeS3

CONFIG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096,
          "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
          "intermediate_size": 14336, "vocab_size": 128256, "max_position_embeddings": 8192,
          "torch_dtype": "bfloat16"}


class InfoS3(FakeS3):
    def __init__(self):
        super().__init__()
        self.gets = 0
        self.down = False

    def info(self, key):
        return {"size": len(self.objects[key])}

    def get_file(self, key, local):
        self.gets += 1
        super().get_file(key, local)

    def cat_file(self, key, start=None, end=None):
        if self.down:
            raise ConnectionError("store unreachable")
        return super().cat_file(key, start, end)


@pytest.fixture
def store(tmp_path):
    s = S3ArtifactStore("merlina", "models", fs=InfoS3())
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(CONFIG))
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16_000_000_000}, "weight_map": {}}))
    (src / "model-00001-of-00001.safetensors").write_bytes(b"w" * 64)
    s.push_dir(src, "C0", snapshot=False)
    s.push_dir(src, "hf/meta-llama/Llama-3.1-8B", snapshot=False)
    return s


@pytest.fixture(autouse=True)
def clear_memo():
    model_refs._MEMO.clear()
    yield
    model_refs._MEMO.clear()


def test_non_store_refs_pass_through():
    assert resolve_model_ref("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"
    assert resolve_model_ref("./models/x") == "./models/x"
    assert not is_store_ref("Qwen/Qwen3-8B") and is_store_ref("store:C0")


def test_names_are_validated():
    assert store_name("store:hf/org/repo") == "hf/org/repo"
    for bad in ("store:", "store:../x", "store:a//b", "store:/abs", "store:a/b/c/d/e"):
        with pytest.raises(ModelRefError):
            store_name(bad)


def test_pull_once_then_cache_hit(store, tmp_path):
    cache = tmp_path / "cache"
    path = resolve_model_ref("store:C0", store=store, cache_root=cache)
    assert json.loads((tmp_path / "cache/merlina/C0").joinpath(
        path.rsplit("/", 1)[-1], "config.json").read_text()) == CONFIG
    pulls = store.fs.gets
    assert resolve_model_ref("store:C0", store=store, cache_root=cache) == path
    assert store.fs.gets == pulls, "a cached generation is not pulled again"


def test_new_generation_replaces_old(store, tmp_path):
    cache = tmp_path / "cache"
    old = resolve_model_ref("store:C0", store=store, cache_root=cache)
    src = tmp_path / "src"
    (src / "config.json").write_text(json.dumps({**CONFIG, "hidden_size": 8192}))
    store.push_dir(src, "C0", snapshot=False)
    new = resolve_model_ref("store:C0", store=store, cache_root=cache)
    assert new != old
    assert json.loads(open(f"{new}/config.json").read())["hidden_size"] == 8192
    assert not (cache / "merlina/C0" / old.rsplit("/", 1)[-1]).exists(), "superseded copy removed"


def test_offline_falls_back_to_cache(store, tmp_path):
    cache = tmp_path / "cache"
    path = resolve_model_ref("store:C0", store=store, cache_root=cache)
    store.fs.down = True
    assert resolve_model_ref("store:C0", store=store, cache_root=cache) == path
    assert cached_ref_path("store:C0", cache) == path
    with pytest.raises(ModelRefError, match="nothing is cached"):
        resolve_model_ref("store:hf/meta-llama/Llama-3.1-8B", store=store, cache_root=cache)


def test_missing_model_names_the_fix(store, tmp_path):
    with pytest.raises(ModelRefError, match="merlina models push"):
        resolve_model_ref("store:nope", store=store, cache_root=tmp_path)
    with pytest.raises(ModelRefError, match="merlina models mirror org/x"):
        resolve_model_ref("store:hf/org/x", store=store, cache_root=tmp_path)


def test_interrupted_pull_is_not_used(store, tmp_path):
    cache = tmp_path / "cache"
    gen = store._manifest("C0")["generation"]
    half = cache / "merlina/C0" / gen
    half.mkdir(parents=True)
    (half / "config.json").write_text("{}")          # no completion marker
    path = resolve_model_ref("store:C0", store=store, cache_root=cache)
    assert json.loads(open(f"{path}/config.json").read()) == CONFIG


def test_default_path_memoizes(store, tmp_path, monkeypatch):
    monkeypatch.setattr(model_refs, "_cache_root", lambda settings=None: tmp_path / "cache")
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    first = resolve_model_ref("store:C0")
    store.fs.down = True       # a memo hit doesn't touch the store
    assert resolve_model_ref("store:C0") == first


def test_restore_adapter_base_ref(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "/data/model-cache/merlina/C0/abc", "r": 16}))
    restore_adapter_base_ref(tmp_path, "store:C0")
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg == {"base_model_name_or_path": "store:C0", "r": 16}
    restore_adapter_base_ref(tmp_path / "missing", "store:C0")   # no adapter: no-op


# ── pre-flight ────────────────────────────────────────────────────────────────

def test_preflight_store_model(store, monkeypatch, tmp_path):
    from src.preflight_checks import PreflightValidator, is_local_model_path
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    assert not is_local_model_path("store:hf/org/repo")
    v = PreflightValidator()
    assert v._check_model_access(SimpleNamespace(base_model="store:C0", hf_token=None))["exists"]
    assert not v.errors
    v._check_model_access(SimpleNamespace(base_model="store:hf/org/missing", hf_token=None))
    assert "merlina models mirror org/missing" in v.errors[0]


def test_preflight_store_unreachable_uses_cache(store, monkeypatch, tmp_path):
    from src.preflight_checks import PreflightValidator
    monkeypatch.setattr(model_refs, "_cache_root", lambda settings=None: tmp_path / "cache")
    resolve_model_ref("store:C0", store=store, cache_root=tmp_path / "cache")

    def down(settings=None):
        raise ConnectionError("no route")
    monkeypatch.setattr("src.remote.artifacts.model_store", down)
    v = PreflightValidator()
    assert v._check_model_access(SimpleNamespace(base_model="store:C0", hf_token=None))["cached"]
    assert not v.errors and "cached copy" in v.warnings[0]


def test_preflight_missing_hf_dataset_with_mirror_is_a_warning(monkeypatch):
    from huggingface_hub.utils import RepositoryNotFoundError
    from src.preflight_checks import PreflightValidator

    class Api:
        def dataset_info(self, *a, **k):
            raise RepositoryNotFoundError("gone", response=SimpleNamespace(headers={}, status_code=404, request=None))

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    source = SimpleNamespace(repo_id="org/gone", config_name=None, split="train", streaming=False)
    v = PreflightValidator()
    monkeypatch.setattr(v, "_hf_mirror_available", lambda src: True)
    v._check_hf_dataset_resolves("org/gone", None, "Dataset", source=source)
    assert not v.errors and "mirrored copy" in v.warnings[0]
    v = PreflightValidator()
    monkeypatch.setattr(v, "_hf_mirror_available", lambda src: False)
    v._check_hf_dataset_resolves("org/gone", None, "Dataset", source=source)
    assert v.errors and "does not resolve" in v.errors[0]


# ── sizing / metadata ─────────────────────────────────────────────────────────

def test_sizing_and_vram_profile_read_the_store(store, monkeypatch, tmp_path):
    from src.remote.sizing import fetch_model_specs
    from src.vram_estimator import get_model_profile
    monkeypatch.setattr("src.remote.artifacts.model_store", lambda settings=None: store)
    monkeypatch.setattr(model_refs, "_cache_root", lambda settings=None: tmp_path / "empty")
    specs = fetch_model_specs("store:C0")
    assert specs.weight_bytes_total == 16_000_000_000 and specs.num_layers == 32
    profile = get_model_profile("store:C0", use_cache=False)
    assert profile.source == "store_config" and profile.hidden_size == 4096
    assert profile.num_params == 8e9          # 16 GB of bf16


def test_model_card_credits_hub_mirrors_only():
    from src.model_card import generate_model_readme
    src = SimpleNamespace(source_type="upload", repo_id=None)
    base = dict(output_name="x", use_lora=True, dataset=SimpleNamespace(source=src, additional_sources=[]))
    mirrored = generate_model_readme(SimpleNamespace(base_model="store:hf/meta-llama/Llama-3.1-8B", **base), "sft")
    assert "base_model:\n- meta-llama/Llama-3.1-8B" in mirrored
    ours = generate_model_readme(SimpleNamespace(base_model="store:C0", **base), "sft")
    assert "base_model:" not in ours and "store:C0" in ours
