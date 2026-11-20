"""
RunPod compute provider, against the REST API v2 (https://api.runpod.io/v2).

REST v1 (rest.runpod.io/v1) and the GraphQL API are retired; v2 covers both
pod management and the GPU catalog. Migration notes:
https://docs.runpod.io/api-reference-v2/migrate-from-v1

The API key needs the ``pods:read``, ``pods:write`` and ``catalog:read``
scopes. Response parsing is deliberately defensive (``.get`` everywhere) so
minor API drift degrades to "unknown" rather than crashing a run
mid-orchestration; inventory listing is the exception, because a silently
empty inventory would let an ambiguous create look like it never happened.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from ..spec import GpuOffer, InstanceSpec, RemoteInstance
from .base import ComputeProvider, ProviderError, ProvisionRejected

logger = logging.getLogger(__name__)

API_BASE = "https://api.runpod.io/v2"
REQUEST_TIMEOUT = 30  # seconds
# Smallest persistent volume v2 accepts (mounts.persistent.size minimum).
MIN_PERSISTENT_VOLUME_GB = 10
# Safety stop for inventory pagination (1000 pods per page).
MAX_INVENTORY_PAGES = 100

_STATUS_MAP = {
    "PROVISIONING": "pending",
    "STARTING": "pending",
    "RUNNING": "running",
    "EXITED": "exited",
    "ERROR": "exited",
    "TERMINATED": "terminated",
}

# GpuType.availability values, with include=AVAILABILITY.
_IN_STOCK = {"LOW", "MEDIUM", "HIGH"}


class RunPodAPIError(ProviderError):
    """A RunPod REST call returned an HTTP error status."""

    def __init__(self, message: str, status_code: int, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class RunPodProvider(ComputeProvider):
    name = "runpod"

    def __init__(self, api_key: str, *, session: Optional[requests.Session] = None):
        self._session = session or requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    # ---- HTTP plumbing -------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{API_BASE}{path}"
        try:
            resp = self._session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as e:
            raise ProviderError(f"RunPod API request failed ({method} {path}): {e}") from e
        if resp.status_code == 404:
            return None
        if not resp.ok:
            try:
                body = resp.json()
            except ValueError:
                body = None
            # v2 errors are RFC 9457 problem details: {title, status, detail, errors?}.
            if isinstance(body, dict) and (body.get("detail") or body.get("title")):
                detail = str(body.get("detail") or body.get("title"))[:500]
            else:
                detail = resp.text[:500]
            raise RunPodAPIError(f"RunPod API error {resp.status_code} on {method} {path}: {detail}",
                                 resp.status_code, body)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderError(f"RunPod API returned non-JSON on {method} {path}") from e

    # ---- ComputeProvider -----------------------------------------------

    def provision(self, spec: InstanceSpec) -> RemoteInstance:
        # Refuse before any HTTP call, so these are definite: nothing was created.
        if spec.interruptible:
            raise ProvisionRejected("RunPod API v2 has no interruptible (spot) pods")
        if not spec.gpu_type_id:
            raise ProvisionRejected("CPU-only RunPod pods are not supported")

        payload: Dict[str, Any] = {
            "name": spec.name,
            "image": spec.image,
            "cloud": spec.cloud_type.upper(),
            "gpu": {"id": spec.gpu_type_id, "count": spec.gpu_count},
            "disk": spec.container_disk_gb,
            "ports": [f"{p}/http" for p in spec.http_ports],
            "env": dict(spec.env),
        }
        if spec.volume_gb:
            size = max(spec.volume_gb, MIN_PERSISTENT_VOLUME_GB)
            if size != spec.volume_gb:
                logger.info("RunPod persistent volumes start at %d GB; using %d GB instead of %d GB",
                            MIN_PERSISTENT_VOLUME_GB, size, spec.volume_gb)
            payload["mounts"] = {"persistent": {"size": size, "path": spec.volume_mount_path}}
        if spec.docker_start_cmd:
            # Replaces the image CMD (v1's dockerStartCmd); the ENTRYPOINT is kept.
            payload["cmd"] = list(spec.docker_start_cmd)

        try:
            data = self._request("POST", "/pods", json=payload)
        except RunPodAPIError as e:
            if _definitely_not_created(e):
                raise ProvisionRejected(str(e)) from e
            raise
        if data is None:
            # 404: the create endpoint itself was not found; nothing was made.
            raise ProvisionRejected("RunPod pod creation endpoint returned 404")
        if not data or not data.get("id"):
            raise ProviderError(f"RunPod pod creation returned no id: {data!r}")
        logger.info("Provisioned RunPod pod %s (%s× %s)", data["id"], spec.gpu_count, spec.gpu_type_id)
        return self._to_instance(data)

    def get_instance(self, instance_id: str) -> RemoteInstance:
        data = self._request("GET", f"/pods/{instance_id}")
        if data is None:
            return RemoteInstance(instance_id=instance_id, provider=self.name, status="terminated")
        return self._to_instance(data)

    def terminate(self, instance_id: str) -> None:
        # _request treats only 404 as already gone. Other failures must remain
        # visible so the independent reconciler retries them.
        self._request("DELETE", f"/pods/{instance_id}")
        logger.info("Requested termination of RunPod pod %s", instance_id)

    def list_instances(self) -> List[RemoteInstance]:
        instances: List[RemoteInstance] = []
        params: Dict[str, Any] = {"limit": 1000}
        for _ in range(MAX_INVENTORY_PAGES):
            data = self._request("GET", "/pods", params=dict(params))
            pods = data.get("pods") if isinstance(data, dict) else None
            if not isinstance(pods, list) or any(not isinstance(item, dict) or not item.get("id") for item in pods):
                raise ProviderError("RunPod inventory returned an unexpected response; reconciliation incomplete")
            instances.extend(self._to_instance(item) for item in pods)
            pagination = data.get("pagination") or {}
            if not pagination.get("hasNextPage"):
                return instances
            cursor = pagination.get("nextCursor")
            if not cursor:
                raise ProviderError("RunPod inventory has more pages but no cursor; reconciliation incomplete")
            params["cursor"] = cursor
        raise ProviderError("RunPod inventory exceeded the page limit; reconciliation incomplete")

    def list_gpu_offers(self) -> List[GpuOffer]:
        # Availability is reported per cloud, so ask once for each. A cloud with no
        # stock contributes no price, which is how sizing knows not to pick it there.
        by_cloud = {cloud: self._gpu_catalog(cloud) for cloud in ("SECURE", "COMMUNITY")}
        offers: Dict[str, GpuOffer] = {}
        order: List[str] = []
        for cloud, items in by_cloud.items():
            key = cloud.lower()
            for item in items:
                gpu_id = item.get("id")
                if not gpu_id:
                    continue
                offer = offers.get(gpu_id)
                if offer is None:
                    offer = offers[gpu_id] = GpuOffer(
                        gpu_type_id=gpu_id,
                        display_name=item.get("name") or gpu_id,
                        vram_gb=float(item.get("memory") or 0),
                        max_gpu_count=0,
                        available=False,
                    )
                    order.append(gpu_id)
                sold_here = item.get(key) is not False
                in_stock = str(item.get("availability") or "").upper() in _IN_STOCK
                if not (sold_here and in_stock):
                    continue
                price = (item.get("price") or {}).get(key)
                if not price:
                    continue
                setattr(offer, f"price_per_hr_{key}", float(price))
                offer.available = True
                offer.max_gpu_count = max(offer.max_gpu_count,
                                          int((item.get("maxCount") or {}).get(key) or 8))
        for offer in offers.values():
            offer.max_gpu_count = offer.max_gpu_count or 8
        return [offers[gpu_id] for gpu_id in order]

    def _gpu_catalog(self, cloud: str) -> List[Dict[str, Any]]:
        data = self._request("GET", "/catalog/gpus", params={
            "include": "AVAILABILITY", "product": "POD", "cloud": cloud, "count": 1,
        })
        gpus = data.get("gpus") if isinstance(data, dict) else None
        if not isinstance(gpus, list):
            raise ProviderError(f"RunPod GPU catalog returned an unexpected response for {cloud} cloud")
        return [item for item in gpus if isinstance(item, dict)]

    def proxy_url(self, instance: RemoteInstance, port: int) -> Optional[str]:
        return f"https://{instance.instance_id}-{port}.proxy.runpod.net"

    # ---- helpers ---------------------------------------------------------

    def _to_instance(self, data: Dict[str, Any]) -> RemoteInstance:
        raw_status = str(data.get("status") or "").upper()
        gpu = data.get("gpu") if isinstance(data.get("gpu"), dict) else {}
        return RemoteInstance(
            instance_id=str(data["id"]),
            provider=self.name,
            status=_STATUS_MAP.get(raw_status, "unknown"),
            cost_per_hr=data.get("cost"),
            gpu_type_id=gpu.get("id"),
            gpu_count=int(gpu.get("count") or 0),
            raw=data,
        )


def _definitely_not_created(error: RunPodAPIError) -> bool:
    """Only a request refusal proves no pod exists.

    A 4xx (validation, auth, no capacity for the requested GPU, rate limit) is
    a refusal. A 5xx stays ambiguous whatever its body says: the create may
    have landed, so the reconciler resolves it from inventory by name.
    """
    return 400 <= error.status_code < 500
