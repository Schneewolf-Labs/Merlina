#!/usr/bin/env python3
"""
API-level tests for HuggingFace namespace selection.

Covers the ``/hf/namespaces`` discovery endpoint and the ``hf_namespace``
field that carries the chosen org through TrainingConfig and the two
upload request models.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


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

from merlina import TrainingConfig, ModelUploadRequest, UploadJobRequest, app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_training_config_defaults_to_no_namespace():
    config = TrainingConfig(output_name="my-model")
    assert config.hf_namespace is None


def test_training_config_keeps_namespace():
    config = TrainingConfig(output_name="my-model", hf_namespace="Schneewolf-Labs")
    assert config.hf_namespace == "Schneewolf-Labs"


def test_training_config_normalizes_namespace():
    config = TrainingConfig(output_name="my-model", hf_namespace="  Schneewolf-Labs/ ")
    assert config.hf_namespace == "Schneewolf-Labs"


def test_training_config_blank_namespace_becomes_none():
    config = TrainingConfig(output_name="my-model", hf_namespace="   ")
    assert config.hf_namespace is None


def test_training_config_rejects_invalid_namespace():
    with pytest.raises(Exception) as exc_info:
        TrainingConfig(output_name="my-model", hf_namespace="not a namespace!")
    assert "Invalid HuggingFace namespace" in str(exc_info.value)


def test_upload_requests_accept_namespace():
    job_request = UploadJobRequest(hf_namespace="Schneewolf-Labs/")
    assert job_request.hf_namespace == "Schneewolf-Labs"

    model_request = ModelUploadRequest(hf_namespace="Schneewolf-Labs")
    assert model_request.hf_namespace == "Schneewolf-Labs"


def test_upload_request_distinguishes_omitted_from_explicit_null():
    """Re-upload keeps the job's org unless the caller clears it on purpose."""
    omitted = UploadJobRequest()
    assert "hf_namespace" not in omitted.model_fields_set

    cleared = UploadJobRequest(hf_namespace=None)
    assert "hf_namespace" in cleared.model_fields_set
    assert cleared.hf_namespace is None

    blank = UploadJobRequest(hf_namespace="  ")
    assert "hf_namespace" in blank.model_fields_set
    assert blank.hf_namespace is None


def test_namespace_survives_config_round_trip():
    """The upload runs off the stored job config, so it must round-trip."""
    from pydantic import TypeAdapter

    original = TrainingConfig(
        output_name="Qwen3.6-27B-Chud-LoRA",
        push_to_hub=True,
        hf_namespace="Schneewolf-Labs",
    )
    restored = TypeAdapter(TrainingConfig).validate_python(original.model_dump())
    assert restored.hf_namespace == "Schneewolf-Labs"

    from src.hf_namespaces import resolve_hub_repo_id
    assert resolve_hub_repo_id(
        restored.output_name, restored.hf_namespace
    ) == "Schneewolf-Labs/Qwen3.6-27B-Chud-LoRA"


# ---------------------------------------------------------------------------
# /hf/namespaces endpoint
# ---------------------------------------------------------------------------

def test_namespaces_endpoint_returns_user_and_orgs(client):
    payload = {
        "user": "nbeerbower",
        "namespaces": [
            {"name": "nbeerbower", "type": "user", "full_name": "", "role": "owner", "can_write": True},
            {"name": "Schneewolf-Labs", "type": "org", "full_name": "Schneewolf Labs", "role": "admin", "can_write": True},
        ],
    }
    with patch("src.hf_namespaces.list_namespaces", return_value=payload) as mock_list:
        response = client.post("/hf/namespaces", json={"hf_token": "hf_test"})

    assert response.status_code == 200
    assert response.json() == payload
    mock_list.assert_called_once_with("hf_test")


def test_namespaces_endpoint_reports_token_problems(client):
    from src.hf_namespaces import HFNamespaceError

    with patch("src.hf_namespaces.list_namespaces", side_effect=HFNamespaceError("bad token")):
        response = client.post("/hf/namespaces", json={"hf_token": "nope"})

    assert response.status_code == 400
    assert "bad token" in response.json()["detail"]


def test_namespaces_endpoint_reports_hub_outage(client):
    with patch("src.hf_namespaces.list_namespaces", side_effect=RuntimeError("boom")):
        response = client.post("/hf/namespaces", json={"hf_token": "hf_test"})

    assert response.status_code == 502
    assert "boom" in response.json()["detail"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
