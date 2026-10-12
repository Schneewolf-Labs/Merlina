"""
Compute provider abstraction.

A provider knows how to rent machines: create one from an
:class:`~src.remote.spec.InstanceSpec`, report its status, expose an HTTP
proxy URL to reach the worker on it, and tear it down. Stages never talk
to provider APIs directly — they go through this interface, which is what
lets a plan split stages across providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..spec import GpuOffer, InstanceSpec, RemoteInstance


class ProviderError(RuntimeError):
    """A compute provider API call failed."""


class ProvisionRejected(ProviderError):
    """The provider definitively refused a create: nothing was provisioned.

    Raise only when the response proves no instance exists (e.g. a 4xx, or an
    explicit error body without an instance id such as "no capacity").
    Timeouts and unparseable responses stay plain ProviderError: ambiguous.
    """


class ComputeProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def provision(self, spec: InstanceSpec) -> RemoteInstance:
        """Create an instance. Returns immediately; poll with get_instance()."""

    @abstractmethod
    def get_instance(self, instance_id: str) -> RemoteInstance:
        """Fetch current status of an instance."""

    @abstractmethod
    def terminate(self, instance_id: str) -> None:
        """Destroy an instance. Must be idempotent — safe on already-gone ids."""

    def list_instances(self) -> List[RemoteInstance]:
        """Complete account inventory, including each instance's raw name.

        Unsupported inventory is an error, never an empty successful scan.
        """
        raise ProviderError(f"Provider {self.name} does not support inventory reconciliation")

    @abstractmethod
    def list_gpu_offers(self) -> List[GpuOffer]:
        """List rentable GPU types with pricing."""

    @abstractmethod
    def proxy_url(self, instance: RemoteInstance, port: int) -> Optional[str]:
        """Public HTTPS URL that reaches ``port`` on the instance, if any."""


def get_provider(name: str, *, api_key: Optional[str] = None) -> ComputeProvider:
    """Factory for providers by name."""
    if name == "runpod":
        from .runpod import RunPodProvider
        if not api_key:
            raise ProviderError(
                "RunPod provider requires an API key — set RUNPOD_API_KEY in .env "
                "(create one at https://www.runpod.io/console/user/settings)."
            )
        return RunPodProvider(api_key)
    raise ProviderError(f"Unknown compute provider: {name!r} (supported: runpod)")
