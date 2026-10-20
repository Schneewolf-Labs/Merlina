#!/usr/bin/env python3
"""
Model merging with Kessel: the /merge API and src/kessel_merge.

The API tests stub the Kessel side (the app is imported with mocked torch, like the other API
tests). The real merge runs in a subprocess, where torch and Kessel are real, on tiny
safetensors checkpoints written there; it is skipped when Kessel isn't installed.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Mock GPU/ML dependencies before importing merlina (mirrors the other API tests)
# ---------------------------------------------------------------------------


class FakeTensor:
    pass


class FakeModule:
    pass


if 'torch' not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.__spec__ = MagicMock()
    mock_torch.Tensor = FakeTensor
    mock_torch.nn = MagicMock()
    mock_torch.nn.Module = FakeModule
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0
    mock_torch.cuda.empty_cache = Mock()
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.float16 = "float16"
    sys.modules['torch'] = mock_torch
    sys.modules['torch.cuda'] = mock_torch.cuda
    sys.modules['torch.nn'] = mock_torch.nn

if 'transformers' not in sys.modules:
    class FakeTokenizerBase:
        pass

    mock_transformers = MagicMock()
    mock_transformers.PreTrainedTokenizerBase = FakeTokenizerBase
    sys.modules['transformers'] = mock_transformers

for _module in [
    'trl', 'peft', 'accelerate', 'accelerate.utils', 'bitsandbytes', 'wandb',
    'psutil', 'pynvml', 'grimoire', 'grimoire.losses', 'grimoire.data',
    'huggingface_hub', 'datasets', 'datasets.arrow_dataset', 'datasets.load',
    'pyarrow', 'artemis_vlm',
]:
    if _module not in sys.modules:
        try:
            __import__(_module)
        except ImportError:
            sys.modules[_module] = MagicMock()

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import merlina  # noqa: E402
from src import kessel_merge  # noqa: E402

RECIPE = {
    "models": [{"model": "org/ft1"}, {"model": "org/ft2"}],
    "merge_method": "model_stock",
    "base_model": "org/base",
}


@pytest.fixture
def client():
    with TestClient(merlina.app) as test_client:
        yield test_client


@pytest.fixture
def kessel_stub(monkeypatch):
    """Kessel 'installed', recipes accepted as-is, queue submissions captured instead of run."""
    submitted = []
    monkeypatch.setattr(kessel_merge, "kessel_available", lambda: True)
    monkeypatch.setattr(kessel_merge, "normalize",
                        lambda recipe=None, yaml_text=None: (recipe or {"merge_method": "linear"}, "yaml: text\n"))
    monkeypatch.setattr(merlina.job_queue, "submit",
                        lambda job_id, config, callback, *a, **k: submitted.append((job_id, config, callback)) or 0)
    return submitted


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(merlina, "_resolve_models_dir", lambda: d)
    return d


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_endpoints_explain_when_kessel_is_missing(client, monkeypatch):
    monkeypatch.setattr(kessel_merge, "kessel_available", lambda: False)
    r = client.get("/merge/methods")
    assert r.status_code == 501
    assert "kessel" in r.json()["detail"].lower()


def test_preview_maps_recipe_errors_to_400(client, monkeypatch):
    monkeypatch.setattr(kessel_merge, "kessel_available", lambda: True)

    def bad(recipe=None, yaml_text=None):
        raise kessel_merge.MergeRecipeError("model_stock needs a base_model")

    monkeypatch.setattr(kessel_merge, "normalize", bad)
    r = client.post("/merge/preview", json={"recipe": {"merge_method": "model_stock"}})
    assert r.status_code == 400
    assert r.json()["detail"] == "model_stock needs a base_model"


def test_submit_queues_a_merge_job(client, kessel_stub, models_dir):
    r = client.post("/merge", json={"recipe": RECIPE, "output_name": "Dragon-9B"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"].startswith("merge-") and body["output_name"] == "Dragon-9B"
    (job_id, config, _), = kessel_stub
    assert job_id == body["job_id"]
    assert config["type"] == "merge" and config["output_name"] == "Dragon-9B"
    assert config["merge_recipe"] == RECIPE
    assert config["base_model"] == "org/base"
    job = merlina.job_manager.get_job(job_id)
    assert job.config["merge_recipe"] == RECIPE


@pytest.mark.parametrize("name", ["../escape", "", "-lead", "a b", "x" * 129])
def test_submit_rejects_unsafe_output_names(client, kessel_stub, models_dir, name):
    r = client.post("/merge", json={"recipe": RECIPE, "output_name": name})
    assert r.status_code in (400, 422)
    assert not kessel_stub


def test_submit_refuses_to_overwrite_an_existing_model(client, kessel_stub, models_dir):
    (models_dir / "Taken").mkdir()
    r = client.post("/merge", json={"recipe": RECIPE, "output_name": "Taken"})
    assert r.status_code == 409
    assert not kessel_stub


def test_describe_falls_back_to_the_first_model():
    assert kessel_merge.describe(RECIPE) == "org/base"
    assert kessel_merge.describe({"models": [{"model": "a"}], "merge_method": "linear"}) == "a"
    assert kessel_merge.describe({"slices": [{"sources": [{"model": "s", "layer_range": [0, 2]}]}]}) == "s"


# ---------------------------------------------------------------------------
# Real Kessel, in a subprocess (this process has mocked torch)
# ---------------------------------------------------------------------------

_REAL = textwrap.dedent(r"""
    import json, sys
    from pathlib import Path
    import torch
    from safetensors.torch import save_file
    sys.path.insert(0, sys.argv[1])
    from src import kessel_merge as km

    root = Path(sys.argv[2])
    g = torch.Generator().manual_seed(0)
    base = {f"model.layers.{i}.mlp.up_proj.weight": torch.randn(8, 4, generator=g) for i in range(3)}
    base["lm_head.weight"] = torch.randn(16, 4, generator=g)
    for name, noise in (("base", 0), ("ft1", 0.1), ("ft2", 0.2)):
        d = root / name
        d.mkdir()
        save_file({k: v + noise * torch.randn(v.shape, generator=g) for k, v in base.items()}, str(d / "model.safetensors"))
        (d / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 3}))

    out = {}
    out["methods"] = {m["name"]: m for m in km.list_methods()}

    recipe, text = km.normalize(yaml_text=f"merge_method: model_stock\nbase_model: {root/'base'}\n"
                                          f"models:\n  - model: {root/'ft1'}\n  - model: {root/'ft2'}\n")
    out["yaml"] = text
    for bad in ({"merge_method": "model_stock", "models": [{"model": "a"}, {"model": "b"}]},
                {"merge_method": "linear", "models": [{"model": "a"}]},
                {"merge_method": "slerp", "base_model": "a", "models": [{"model": "a"}, {"model": "b"}]},
                {"merge_method": "nope", "models": [{"model": "a"}]},
                {"merge_method": "linear", "models": []}):
        try:
            km.normalize(bad)
            out.setdefault("errors", []).append(None)
        except km.MergeRecipeError as e:
            out.setdefault("errors", []).append(str(e))

    class JM:
        def __init__(self, stop=False):
            self.updates, self.stop = [], stop
        def get_job(self, job_id):
            return type("J", (), {"stop_requested": self.stop})()
        def update_job(self, job_id, **f):
            self.updates.append(f)

    models = root / "app" / "models"
    jm, sent = JM(), []
    km.run_merge("merge-x", recipe, models / "Merged", jm, publish=lambda jid, m: sent.append((jid, m["type"], m.get("status"))))
    out["sent"] = sent
    out["files"] = sorted(p.name for p in (models / "Merged").iterdir())
    out["statuses"] = [u.get("status") for u in jm.updates]
    out["last"] = jm.updates[-1]
    out["steps"] = [(u.get("current_step"), u.get("total_steps")) for u in jm.updates if "current_step" in u]
    out["leftover"] = sorted(p.name for p in (root / "app").iterdir())

    jm, sent = JM(stop=True), []
    km.run_merge("merge-y", recipe, models / "Stopped", jm, publish=lambda jid, m: sent.append(m.get("status")))
    out["sent_stop"] = sent
    out["stopped_exists"] = (models / "Stopped").exists()
    out["stopped_status"] = jm.updates[-1]
    out["leftover_after_stop"] = sorted(p.name for p in (root / "app").iterdir())
    try:
        km.run_merge("merge-z", recipe, models / "Merged", JM())
    except FileExistsError:
        out["exists_refused"] = True
    print(json.dumps(out))
""")


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if importlib.util.find_spec("kessel") is None:
        pytest.skip("Kessel not installed")
    root = tmp_path_factory.mktemp("kessel")
    proc = subprocess.run([sys.executable, "-c", _REAL, str(ROOT), str(root)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_methods_list_parameters_and_required_flags(real):
    m = real["methods"]
    assert m["model_stock"]["needs_base"] is True
    assert m["linear"]["model_params"]["weight"] == {"default": None, "required": True}
    assert m["slerp"]["global_params"]["t"]["required"] is True
    assert m["ties"]["model_params"]["density"] == {"default": 1.0, "required": False}


def test_normalize_reads_yaml_and_rejects_recipes_kessel_cannot_run(real):
    assert "merge_method: model_stock" in real["yaml"]
    no_base, no_weight, no_t, unknown, empty = real["errors"]
    assert empty == "Add at least one model"
    assert "needs a base_model" in no_base
    assert "weight" in no_weight
    assert "t" in no_t
    assert "nope" in unknown


def test_run_merge_writes_the_model_and_reports_progress(real):
    assert "model.safetensors" in real["files"] and "kessel_config.yaml" in real["files"]
    assert real["statuses"][0] == "loading_model"
    assert real["last"]["status"] == "completed" and real["last"]["progress"] == 1.0
    assert real["steps"][0] == [0, 4] and real["steps"][-1] == [4, 4]
    assert real["leftover"] == ["models"]  # the hidden partial dir was moved into place
    # the job's WebSocket gets the same updates, ending in a completion message
    assert real["sent"][0] == ["merge-x", "status_update", "loading_model"]
    assert ["merge-x", "status_update", "merging"] in real["sent"]
    assert real["sent"][-1] == ["merge-x", "completed", None]


def test_stop_request_removes_the_partial_output(real):
    assert real["stopped_status"] == {"status": "stopped"}
    assert real["sent_stop"][-1] == "stopped"
    assert real["stopped_exists"] is False
    assert real["leftover_after_stop"] == ["models"]
    assert real["exists_refused"] is True
