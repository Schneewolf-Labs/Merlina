"""
Tests for the RunPod provider (src/remote/providers/runpod.py) against a
fake HTTP session — no network, no real API key.

Run standalone: python -m pytest tests/test_remote_provider_runpod.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.remote.providers.base import ProviderError, get_provider
from src.remote.providers.runpod import RunPodProvider
from src.remote.spec import InstanceSpec


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = self.content.decode()

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeSession:
    """Records requests; replies from a scripted route table."""

    def __init__(self):
        self.headers = {}
        self.routes = {}            # (method, path) -> FakeResponse
        self.calls = []             # (method, path, json_payload)

    def request(self, method, url, timeout=None, **kwargs):
        path = url.replace("https://rest.runpod.io/v1", "")
        self.calls.append((method, path, kwargs.get("json")))
        return self.routes.get((method, path), FakeResponse(404, None))


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def provider(session):
    return RunPodProvider("rp_test_key", session=session)


def basic_spec(**overrides):
    spec = InstanceSpec(
        name="merlina-job_1-train",
        image="ghcr.io/schneewolf-labs/merlina:latest",
        gpu_type_id="NVIDIA H200",
        gpu_count=4,
        container_disk_gb=700,
        env={"MERLINA_WORKER_TOKEN": "tok"},
        docker_start_cmd=["python", "-m", "src.remote.worker_entry"],
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


class TestProvision:
    def test_payload_shape(self, provider, session):
        session.routes[("POST", "/pods")] = FakeResponse(200, {"id": "pod_abc", "desiredStatus": "CREATED"})
        instance = provider.provision(basic_spec())
        assert instance.instance_id == "pod_abc"
        assert instance.status == "pending"

        method, path, payload = session.calls[0]
        assert payload["gpuTypeIds"] == ["NVIDIA H200"]
        assert payload["gpuCount"] == 4
        assert payload["containerDiskInGb"] == 700
        assert payload["cloudType"] == "SECURE"
        assert payload["ports"] == ["8000/http"]
        assert payload["env"]["MERLINA_WORKER_TOKEN"] == "tok"
        assert payload["dockerStartCmd"] == ["python", "-m", "src.remote.worker_entry"]

    def test_no_id_raises(self, provider, session):
        session.routes[("POST", "/pods")] = FakeResponse(200, {})
        with pytest.raises(ProviderError, match="no id"):
            provider.provision(basic_spec())

    def test_api_error_raises(self, provider, session):
        session.routes[("POST", "/pods")] = FakeResponse(500, {"error": "boom"})
        with pytest.raises(ProviderError, match="500"):
            provider.provision(basic_spec())

    def test_auth_header_set(self, session):
        RunPodProvider("rp_secret", session=session)
        assert session.headers["Authorization"] == "Bearer rp_secret"


class TestStatusAndTerminate:
    def test_status_mapping(self, provider, session):
        session.routes[("GET", "/pods/pod_abc")] = FakeResponse(
            200, {"id": "pod_abc", "desiredStatus": "RUNNING", "costPerHr": 15.96, "gpuCount": 4})
        instance = provider.get_instance("pod_abc")
        assert instance.is_running
        assert instance.cost_per_hr == 15.96

    def test_missing_pod_is_terminated(self, provider, session):
        instance = provider.get_instance("gone")
        assert instance.status == "terminated"

    def test_terminate_is_idempotent(self, provider, session):
        # 404 on DELETE must not raise — pod already gone is success
        provider.terminate("already_gone")

    def test_unknown_status_degrades(self, provider, session):
        session.routes[("GET", "/pods/p")] = FakeResponse(200, {"id": "p", "desiredStatus": "WEIRD"})
        assert provider.get_instance("p").status == "unknown"


class TestOffers:
    def test_offer_parsing(self, provider, session):
        session.routes[("GET", "/gputypes")] = FakeResponse(200, [
            {"id": "NVIDIA H200", "displayName": "H200 SXM", "memoryInGb": 141,
             "securePrice": 3.99, "communityPrice": 3.45, "maxGpuCount": 8},
            {"displayName": "junk row without id"},
        ])
        offers = provider.list_gpu_offers()
        assert len(offers) == 1
        assert offers[0].vram_gb == 141
        assert offers[0].price_per_hr("community") == 3.45


class TestFactory:
    def test_runpod_requires_key(self):
        with pytest.raises(ProviderError, match="API key"):
            get_provider("runpod", api_key=None)

    def test_unknown_provider(self):
        with pytest.raises(ProviderError, match="Unknown"):
            get_provider("vastai", api_key="x")

    def test_proxy_url(self, provider):
        from src.remote.spec import RemoteInstance
        inst = RemoteInstance(instance_id="pod_abc", provider="runpod")
        assert provider.proxy_url(inst, 8000) == "https://pod_abc-8000.proxy.runpod.net"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
