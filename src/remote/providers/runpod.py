"""
RunPod compute provider, against the REST API (https://rest.runpod.io/v1).

Field names follow the documented v1 REST schema; parsing is deliberately
defensive (``.get`` everywhere) so minor API drift degrades to "unknown"
rather than crashing a run mid-orchestration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from ..spec import GpuOffer, InstanceSpec, RemoteInstance
from .base import ComputeProvider, ProviderError

logger = logging.getLogger(__name__)

API_BASE = "https://rest.runpod.io/v1"
REQUEST_TIMEOUT = 30  # seconds

_STATUS_MAP = {
    "CREATED": "pending",
    "PENDING": "pending",
    "RUNNING": "running",
    "EXITED": "exited",
    "TERMINATED": "terminated",
    "DEAD": "exited",
}


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
            detail = resp.text[:500]
            raise ProviderError(f"RunPod API error {resp.status_code} on {method} {path}: {detail}")
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderError(f"RunPod API returned non-JSON on {method} {path}") from e

    # ---- ComputeProvider -----------------------------------------------

    def provision(self, spec: InstanceSpec) -> RemoteInstance:
        payload: Dict[str, Any] = {
            "name": spec.name,
            "imageName": spec.image,
            "cloudType": spec.cloud_type.upper(),
            "containerDiskInGb": spec.container_disk_gb,
            "ports": [f"{p}/http" for p in spec.http_ports],
            "env": dict(spec.env),
            "interruptible": spec.interruptible,
        }
        if spec.gpu_type_id:
            payload["gpuTypeIds"] = [spec.gpu_type_id]
            payload["gpuCount"] = spec.gpu_count
        else:
            payload["computeType"] = "CPU"
        if spec.volume_gb:
            payload["volumeInGb"] = spec.volume_gb
            payload["volumeMountPath"] = spec.volume_mount_path
        if spec.docker_start_cmd:
            payload["dockerStartCmd"] = spec.docker_start_cmd

        data = self._request("POST", "/pods", json=payload)
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
        try:
            self._request("DELETE", f"/pods/{instance_id}")
            logger.info("Terminated RunPod pod %s", instance_id)
        except ProviderError as e:
            # Idempotency: a pod that is already gone is success.
            logger.warning("Terminate of pod %s reported: %s", instance_id, e)

    def list_gpu_offers(self) -> List[GpuOffer]:
        data = self._request("GET", "/gputypes")
        if isinstance(data, dict):  # tolerate a wrapped list
            data = data.get("gpuTypes") or data.get("data") or []
        offers: List[GpuOffer] = []
        for item in data or []:
            gpu_id = item.get("id")
            if not gpu_id:
                continue
            offers.append(GpuOffer(
                gpu_type_id=gpu_id,
                display_name=item.get("displayName", gpu_id),
                vram_gb=float(item.get("memoryInGb") or 0),
                price_per_hr_secure=item.get("securePrice"),
                price_per_hr_community=item.get("communityPrice"),
                max_gpu_count=int(item.get("maxGpuCount") or 8),
                available=bool(item.get("secureCloud") or item.get("communityCloud") or True),
            ))
        return offers

    def proxy_url(self, instance: RemoteInstance, port: int) -> Optional[str]:
        return f"https://{instance.instance_id}-{port}.proxy.runpod.net"

    # ---- helpers ---------------------------------------------------------

    def _to_instance(self, data: Dict[str, Any]) -> RemoteInstance:
        raw_status = (data.get("desiredStatus") or data.get("status") or "").upper()
        return RemoteInstance(
            instance_id=str(data["id"]),
            provider=self.name,
            status=_STATUS_MAP.get(raw_status, "unknown"),
            cost_per_hr=data.get("costPerHr"),
            gpu_type_id=(data.get("gpuTypeIds") or [None])[0] if data.get("gpuTypeIds")
                        else data.get("gpuTypeId"),
            gpu_count=int(data.get("gpuCount") or 0),
            raw=data,
        )
