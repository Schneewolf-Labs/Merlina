"""
Tests for the RunPod provider (src/remote/providers/runpod.py) against a
fake HTTP session — no network, no real API key. Response shapes follow
RunPod REST API v2 (https://api.runpod.io/v2).

Run standalone: python -m pytest tests/test_remote_provider_runpod.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.remote.providers.base import ProviderError, ProvisionRejected, get_provider
from src.remote.providers.runpod import API_BASE, RunPodProvider
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
    """Records requests; replies from a scripted route table.

    A route is a FakeResponse, or a callable taking the query params and
    returning one (for paginated or per-cloud endpoints).
    """

    def __init__(self):
        self.headers = {}
        self.routes = {}            # (method, path) -> FakeResponse | callable(params)
        self.calls = []             # (method, path, json_payload, params)

    def request(self, method, url, timeout=None, **kwargs):
        assert url.startswith(API_BASE), f"request outside the v2 API: {url}"
        path = url[len(API_BASE):]
        params = kwargs.get("params") or {}
        self.calls.append((method, path, kwargs.get("json"), params))
        route = self.routes.get((method, path), FakeResponse(404, None))
        return route(params) if callable(route) else route


def problem(status, detail):
    """An RFC 9457 problem body, as v2 returns for errors."""
    return {"title": "Error", "status": status, "detail": detail}


def pod(pod_id="pod_abc", status="PROVISIONING", **extra):
    return {"id": pod_id, "name": "merlina-job_1-train", "status": status, "cost": 15.96,
            "gpu": {"id": "NVIDIA H200", "count": 4, "vcpuCount": 64, "memory": 1000},
            "env": {"MERLINA_WORKER_TOKEN": "tok"}, **extra}


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
        session.routes[("POST", "/pods")] = FakeResponse(201, pod())
        instance = provider.provision(basic_spec())
        assert instance.instance_id == "pod_abc"
        assert instance.status == "pending"

        method, path, payload, _ = session.calls[0]
        assert (method, path) == ("POST", "/pods")
        assert payload == {
            "name": "merlina-job_1-train",
            "image": "ghcr.io/schneewolf-labs/merlina:latest",
            "cloud": "SECURE",
            "gpu": {"id": "NVIDIA H200", "count": 4},
            "disk": 700,
            "ports": ["8000/http"],
            "env": {"MERLINA_WORKER_TOKEN": "tok"},
            "cmd": ["python", "-m", "src.remote.worker_entry"],
        }

    def test_community_cloud_and_volume(self, provider, session):
        session.routes[("POST", "/pods")] = FakeResponse(201, pod())
        provider.provision(basic_spec(cloud_type="community", volume_gb=200))
        payload = session.calls[0][2]
        assert payload["cloud"] == "COMMUNITY"
        assert payload["mounts"] == {"persistent": {"size": 200, "path": "/workspace"}}

    def test_small_volume_raised_to_v2_minimum(self, provider, session):
        session.routes[("POST", "/pods")] = FakeResponse(201, pod())
        provider.provision(basic_spec(volume_gb=5))
        assert session.calls[0][2]["mounts"]["persistent"]["size"] == 10

    @pytest.mark.parametrize("overrides", [{"interruptible": True}, {"gpu_type_id": None}])
    def test_unsupported_specs_rejected_before_any_request(self, provider, session, overrides):
        with pytest.raises(ProvisionRejected):
            provider.provision(basic_spec(**overrides))
        assert session.calls == []

    def test_no_id_raises(self, provider, session):
        session.routes[("POST", "/pods")] = FakeResponse(201, {})
        with pytest.raises(ProviderError, match="no id"):
            provider.provision(basic_spec())

    def test_problem_detail_in_error_message(self, provider, session):
        session.routes[("POST", "/pods")] = FakeResponse(
            400, problem(400, "No capacity for NVIDIA H200 on SECURE cloud"))
        with pytest.raises(ProvisionRejected, match="No capacity for NVIDIA H200"):
            provider.provision(basic_spec())

    @pytest.mark.parametrize("response", [
        FakeResponse(400, problem(400, "no capacity")),
        FakeResponse(401, None),
        FakeResponse(422, problem(422, "unknown field")),
        FakeResponse(429, problem(429, "rate limited")),
    ])
    def test_definite_refusals_are_rejections(self, provider, session, response):
        session.routes[("POST", "/pods")] = response
        with pytest.raises(ProvisionRejected):
            provider.provision(basic_spec())

    def test_missing_create_endpoint_is_rejection(self, provider, session):
        with pytest.raises(ProvisionRejected):
            provider.provision(basic_spec())

    @pytest.mark.parametrize("response", [
        FakeResponse(201, {}),                        # accepted but unparseable
        FakeResponse(502, None),                      # gateway, no provider body
        FakeResponse(500, problem(500, "internal")),  # provider-authored, still ambiguous
        FakeResponse(503, {"error": "x", "id": "pod_abc"}),
    ])
    def test_ambiguous_failures_are_not_rejections(self, provider, session, response):
        session.routes[("POST", "/pods")] = response
        with pytest.raises(ProviderError) as info:
            provider.provision(basic_spec())
        assert not isinstance(info.value, ProvisionRejected)

    def test_network_error_is_not_rejection(self, provider, session):
        import requests
        def boom(*args, **kwargs):
            raise requests.ConnectionError("reset")
        session.request = boom
        with pytest.raises(ProviderError) as info:
            provider.provision(basic_spec())
        assert not isinstance(info.value, ProvisionRejected)

    def test_auth_header_set(self, session):
        RunPodProvider("rp_secret", session=session)
        assert session.headers["Authorization"] == "Bearer rp_secret"


class TestStatusAndTerminate:
    def test_status_mapping(self, provider, session):
        session.routes[("GET", "/pods/pod_abc")] = FakeResponse(200, pod(status="RUNNING"))
        instance = provider.get_instance("pod_abc")
        assert instance.is_running
        assert instance.cost_per_hr == 15.96
        assert instance.gpu_type_id == "NVIDIA H200"
        assert instance.gpu_count == 4

    @pytest.mark.parametrize("raw,expected", [
        ("PROVISIONING", "pending"), ("STARTING", "pending"), ("RUNNING", "running"),
        ("EXITED", "exited"), ("ERROR", "exited"), ("TERMINATED", "terminated"),
        ("WEIRD", "unknown"),
    ])
    def test_every_v2_status(self, provider, session, raw, expected):
        session.routes[("GET", "/pods/p")] = FakeResponse(200, pod("p", status=raw))
        assert provider.get_instance("p").status == expected

    def test_missing_pod_is_terminated(self, provider, session):
        instance = provider.get_instance("gone")
        assert instance.status == "terminated"

    def test_terminate_uses_delete(self, provider, session):
        session.routes[("DELETE", "/pods/pod_abc")] = FakeResponse(204, None)
        provider.terminate("pod_abc")
        assert session.calls[-1][:2] == ("DELETE", "/pods/pod_abc")

    def test_terminate_is_idempotent(self, provider, session):
        # 404 on DELETE must not raise — pod already gone is success
        provider.terminate("already_gone")

    def test_termination_errors_propagate(self, provider, session):
        session.routes[("DELETE", "/pods/pod")] = FakeResponse(503, problem(503, "unavailable"))
        with pytest.raises(ProviderError):
            provider.terminate("pod")


def catalog(secure, community):
    """Route for GET /catalog/gpus that answers per requested cloud."""
    def route(params):
        assert params["include"] == "AVAILABILITY" and params["product"] == "POD"
        return FakeResponse(200, {"gpus": secure if params["cloud"] == "SECURE" else community})
    return route


def gpu(gpu_id, availability, *, secure=4.59, community=3.59, memory=141, **extra):
    return {"id": gpu_id, "name": gpu_id.replace("NVIDIA ", ""), "memory": memory,
            "secure": True, "community": True,
            "price": {"secure": secure, "community": community, "serverless": None},
            "maxCount": {"secure": 8, "community": 4}, "availability": availability, **extra}


class TestOffers:
    def test_offer_parsing(self, provider, session):
        session.routes[("GET", "/catalog/gpus")] = catalog(
            secure=[gpu("NVIDIA H200", "HIGH"), gpu("NVIDIA H200 NVL", "NONE", memory=143),
                    {"name": "junk row without id"}],
            community=[gpu("NVIDIA H200", "LOW"), gpu("NVIDIA H200 NVL", "NONE", memory=143)],
        )
        offers = provider.list_gpu_offers()
        assert [o.gpu_type_id for o in offers] == ["NVIDIA H200", "NVIDIA H200 NVL"]
        h200, nvl = offers
        assert h200.display_name == "H200"
        assert h200.vram_gb == 141
        assert h200.price_per_hr("secure") == 4.59
        assert h200.price_per_hr("community") == 3.59
        assert h200.available is True
        assert h200.max_gpu_count == 8
        # No stock anywhere: not rentable, whatever the listed price says.
        assert nvl.available is False
        assert nvl.price_per_hr("secure") is None
        assert sorted(c[3]["cloud"] for c in session.calls) == ["COMMUNITY", "SECURE"]

    def test_stock_on_one_cloud_prices_only_that_cloud(self, provider, session):
        session.routes[("GET", "/catalog/gpus")] = catalog(
            secure=[gpu("NVIDIA A100", "NONE")],
            community=[gpu("NVIDIA A100", "MEDIUM", community=1.19)],
        )
        (offer,) = provider.list_gpu_offers()
        assert offer.available is True
        assert offer.price_per_hr("community") == 1.19
        assert offer.price_per_hr("secure") is None
        assert offer.max_gpu_count == 4

    def test_gpu_not_sold_on_cloud_has_no_price_there(self, provider, session):
        session.routes[("GET", "/catalog/gpus")] = catalog(
            secure=[gpu("NVIDIA RTX 4090", "HIGH", secure=0.69, community=0.34) | {"secure": False}],
            community=[gpu("NVIDIA RTX 4090", "HIGH", community=0.34)],
        )
        (offer,) = provider.list_gpu_offers()
        assert offer.price_per_hr("secure") is None
        assert offer.price_per_hr("community") == 0.34

    def test_catalog_errors_raise(self, provider, session):
        session.routes[("GET", "/catalog/gpus")] = FakeResponse(403, problem(403, "missing catalog:read"))
        with pytest.raises(ProviderError, match="catalog:read"):
            provider.list_gpu_offers()

    def test_malformed_catalog_raises(self, provider, session):
        session.routes[("GET", "/catalog/gpus")] = FakeResponse(200, {"gpuTypes": []})
        with pytest.raises(ProviderError, match="catalog"):
            provider.list_gpu_offers()


def paged(*pages):
    """Route for GET /pods serving `pages` by cursor ("c1", "c2", ...)."""
    def route(params):
        index = int(params.get("cursor", "c0")[1:])
        last = index == len(pages) - 1
        return FakeResponse(200, {"pods": pages[index], "pagination": {
            "hasNextPage": not last, "nextCursor": None if last else f"c{index + 1}"}})
    return route


class TestInventoryCleanup:
    def test_inventory_preserves_discovery_label(self, provider, session):
        session.routes[("GET", "/pods")] = paged([
            {"id": "pod", "name": "merlina-owned-train", "status": "RUNNING"}])
        instances = provider.list_instances()
        assert instances[0].raw["name"] == "merlina-owned-train"

    def test_inventory_follows_every_page(self, provider, session):
        session.routes[("GET", "/pods")] = paged(
            [pod("a", status="RUNNING")], [pod("b")], [pod("c", status="EXITED")])
        instances = provider.list_instances()
        assert [i.instance_id for i in instances] == ["a", "b", "c"]
        assert [c[3].get("cursor") for c in session.calls] == [None, "c1", "c2"]

    @pytest.mark.parametrize("body", [
        [{"id": "pod"}],                                   # v1 bare list
        {"unexpected": []},
        {"pods": [{"name": "no id"}]},
    ])
    def test_malformed_inventory_is_not_empty_success(self, provider, session, body):
        session.routes[("GET", "/pods")] = FakeResponse(200, body)
        with pytest.raises(ProviderError, match="inventory"):
            provider.list_instances()

    def test_next_page_without_cursor_is_incomplete(self, provider, session):
        session.routes[("GET", "/pods")] = FakeResponse(200, {
            "pods": [pod()], "pagination": {"hasNextPage": True, "nextCursor": None}})
        with pytest.raises(ProviderError, match="cursor"):
            provider.list_instances()


class TestFailedCreateTeardown:
    """A create that fails ambiguously, but did land, is found and torn down.

    Drives the real reconciler through the v2 provider: the 502 leaves the
    attempt ``provisioning_unknown``, and the pod is only discoverable by name
    on the second inventory page.
    """

    def test_ambiguous_create_found_on_later_page_and_terminated(self, provider, session, tmp_path):
        from src.job_manager import JobManager
        from src.remote.cleanup import reconcile_once
        from src.remote.lifecycle import RemoteLifecycle

        now = [1000.0]
        manager = JobManager(str(tmp_path / "jobs.db"))
        manager.create_job("job", {})
        lifecycle = RemoteLifecycle(manager.db_path, lease_seconds=10, clock=lambda: now[0])
        owner = lifecycle.claim("job", 1)
        label = lifecycle.intent("job", owner, "train", "runpod")

        session.routes[("POST", "/pods")] = FakeResponse(502, None)
        with pytest.raises(ProviderError) as info:
            provider.provision(basic_spec(name=label))
        assert not isinstance(info.value, ProvisionRejected)
        lifecycle.stage_state("job", "train", "provisioning_unknown", error="ProviderError")
        lifecycle.finish("job", owner, "failed")

        # The create landed after all; it shows up behind someone else's pod.
        orphan = pod("pod_orphan", status="RUNNING") | {"name": label}
        session.routes[("GET", "/pods")] = paged(
            [pod("unrelated", status="RUNNING") | {"name": "someone-else"}], [orphan])
        deleted = []
        def delete(params):
            deleted.append("pod_orphan")
            return FakeResponse(204, None)
        session.routes[("DELETE", "/pods/pod_orphan")] = delete
        # After the DELETE, GET 404s, which is what confirms termination.

        result = reconcile_once(lifecycle, {"runpod": provider}, job_manager=manager)

        assert deleted == ["pod_orphan"]
        assert ("DELETE", "/pods/unrelated") not in [c[:2] for c in session.calls]
        (resource,) = result["resources"]
        assert resource["instance_id"] == "pod_orphan"
        assert resource["cleanup_status"] == "terminated"
        assert lifecycle.attempts()[0]["state"] == "failed"
        assert result["alerts"] == []


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
