"""
Client side of the control-plane ↔ remote-worker protocol.

The worker (``src/remote/worker_entry.py``) exposes a tiny authenticated
HTTP API on the instance; the orchestrator polls it. Pull-only design:
the user's machine needs no public address, and any provider that can
expose one HTTP port works.

Endpoints:
  GET  /health                  → {"ok": true, ...}
  GET  /status?since_step=N     → WorkerStatus payload (see below)
  POST /stop                    → request graceful stop

All requests carry ``X-Merlina-Worker-Token``; the worker rejects
anything else, since provider proxy URLs are guessable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

WORKER_PORT = 8000
TOKEN_HEADER = "X-Merlina-Worker-Token"

# Worker top-level states
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_PUSHING = "pushing_artifacts"
STATE_DONE = "done"
STATE_FAILED = "failed"
TERMINAL_STATES = (STATE_DONE, STATE_FAILED)


@dataclass
class WorkerStatus:
    """Decoded /status response."""
    state: str = STATE_STARTING
    error: Optional[str] = None
    job: Dict[str, Any] = field(default_factory=dict)      # mirrored JobRecord fields
    new_metrics: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerStatus":
        return cls(
            state=data.get("state", STATE_STARTING),
            error=data.get("error"),
            job=data.get("job") or {},
            new_metrics=data.get("new_metrics") or [],
            artifacts=data.get("artifacts") or [],
        )


class WorkerClient(ABC):
    @abstractmethod
    def health(self) -> bool:
        """True once the worker's HTTP server is up and authenticated."""

    @abstractmethod
    def status(self, since_step: int = 0) -> WorkerStatus:
        """Poll worker state; ``since_step`` filters already-seen metrics."""

    @abstractmethod
    def request_stop(self) -> bool:
        """Ask the worker to stop training gracefully."""


class HttpWorkerClient(WorkerClient):
    def __init__(self, base_url: str, token: str, *, timeout: float = 15.0,
                 session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({TOKEN_HEADER: token})

    def health(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
            return resp.ok and bool(resp.json().get("ok"))
        except Exception:
            return False

    def status(self, since_step: int = 0) -> WorkerStatus:
        resp = self._session.get(
            f"{self.base_url}/status",
            params={"since_step": since_step},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return WorkerStatus.from_dict(resp.json())

    def request_stop(self) -> bool:
        try:
            resp = self._session.post(f"{self.base_url}/stop", timeout=self.timeout)
            return resp.ok
        except Exception as e:
            logger.warning("Failed to deliver stop request to worker: %s", e)
            return False
