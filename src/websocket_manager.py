"""
WebSocket fan-out for real-time training updates.

Clients subscribe to one job (``/ws/{job_id}``). Producers are almost never
on the event loop: training runs on a queue worker thread or in a separate
process whose progress a monitor thread tails. The one thread-safe entry
point for them is :meth:`WebSocketManager.publish`, which hops onto the event
loop bound at startup via :meth:`bind_loop`. Everything else here is plain
asyncio and must be awaited on that loop.

Connection bookkeeping is guarded by an ``asyncio.Lock``; sends happen
outside the lock so one slow client can't stall the others, and a client
whose send fails is dropped.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Tracks connections per job and broadcasts JSON messages to them."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self.job_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------ #
    # Thread-safe producer API
    # ------------------------------------------------------------------ #

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the server's event loop so worker threads can publish."""
        self._loop = loop

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def publish(self, job_id: str, message: Dict[str, Any], loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        Queue a message for a job's subscribers from any thread.

        Best-effort: with no event loop bound (tests, CLI tools) the message
        is dropped. Never raises.
        """
        loop = loop or self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast_to_job(job_id, message), loop)
        except Exception as e:
            logger.debug(f"Could not schedule WebSocket message for {job_id}: {e}")

    def schedule(self, coro, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Run one of this manager's coroutines on the event loop from any thread."""
        loop = loop or self._loop
        if loop is None or loop.is_closed():
            coro.close()
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:
            coro.close()
            logger.debug(f"Could not schedule WebSocket update: {e}")

    # ------------------------------------------------------------------ #
    # Connection management (event loop only)
    # ------------------------------------------------------------------ #

    async def connect(self, websocket: WebSocket, job_id: Optional[str] = None) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
            if job_id:
                self.job_connections.setdefault(job_id, set()).add(websocket)
        logger.info(f"WebSocket connected ({'job ' + job_id if job_id else 'global'})")

    async def disconnect(self, websocket: WebSocket, job_id: Optional[str] = None) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)
            if job_id:
                subscribers = self.job_connections.get(job_id)
                if subscribers is not None:
                    subscribers.discard(websocket)
                    if not subscribers:
                        del self.job_connections[job_id]
        logger.info(f"WebSocket disconnected ({'job ' + job_id if job_id else 'global'})")

    async def _send_all(self, connections: Set[WebSocket], message: Dict[str, Any], job_id: Optional[str]) -> None:
        if not connections:
            return
        text = json.dumps(message)
        failed = []
        for connection in connections:
            try:
                await connection.send_text(text)
            except Exception as e:
                logger.warning(f"Dropping WebSocket client after send failure: {e}")
                failed.append(connection)
        for connection in failed:
            await self.disconnect(connection, job_id)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send to every connected client."""
        async with self._lock:
            connections = set(self.active_connections)
        await self._send_all(connections, message, None)

    async def broadcast_to_job(self, job_id: str, message: Dict[str, Any]) -> None:
        """Send to the clients subscribed to ``job_id``."""
        message.setdefault("job_id", job_id)
        async with self._lock:
            connections = set(self.job_connections.get(job_id, ()))
        await self._send_all(connections, message, job_id)

    def get_connection_count(self, job_id: Optional[str] = None) -> int:
        # Unlocked read: a slightly stale count is fine for /stats.
        if job_id:
            return len(self.job_connections.get(job_id, ()))
        return len(self.active_connections)

    # ------------------------------------------------------------------ #
    # Message builders. Each has an awaitable form (for the event loop) and
    # the same payload is what worker threads pass to publish().
    # ------------------------------------------------------------------ #

    @staticmethod
    def status_message(job_id: str, status: str, progress: Optional[float] = None, **extra: Any) -> Dict[str, Any]:
        message = {"type": "status_update", "job_id": job_id, "status": status, "timestamp": time.time()}
        if progress is not None:
            message["progress"] = progress
        message.update({k: v for k, v in extra.items() if v is not None})
        return message

    async def send_status_update(
        self,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gpu_memory: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        await self.broadcast_to_job(job_id, self.status_message(
            job_id, status, progress,
            current_step=current_step, total_steps=total_steps, loss=loss,
            eval_loss=eval_loss, learning_rate=learning_rate, gpu_memory=gpu_memory,
            **kwargs,
        ))

    async def send_metric_update(self, job_id: str, step: int, metrics: Dict[str, float]) -> None:
        await self.broadcast_to_job(job_id, {
            "type": "metrics", "job_id": job_id, "step": step,
            "metrics": metrics, "timestamp": time.time(),
        })

    async def send_error(self, job_id: str, error: str) -> None:
        await self.broadcast_to_job(job_id, {
            "type": "error", "job_id": job_id, "error": error, "timestamp": time.time(),
        })

    async def send_completion(
        self,
        job_id: str,
        output_dir: str,
        final_metrics: Optional[Dict[str, Any]] = None,
        upload_error: Optional[str] = None,
    ) -> None:
        message: Dict[str, Any] = {
            "type": "completed", "job_id": job_id, "output_dir": output_dir, "timestamp": time.time(),
        }
        if final_metrics:
            message["final_metrics"] = final_metrics
        if upload_error:
            message["upload_error"] = upload_error
        await self.broadcast_to_job(job_id, message)

    async def send_gguf_progress(
        self,
        job_id: str,
        stage: str,
        message: str,
        *,
        quant_type: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        artifact_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "gguf_progress", "job_id": job_id, "stage": stage,
            "message": message, "timestamp": time.time(),
        }
        for key, value in (
            ("quant_type", quant_type), ("current", current), ("total", total),
            ("artifact_path", artifact_path), ("error", error),
        ):
            if value is not None:
                payload[key] = value
        await self.broadcast_to_job(job_id, payload)


# Global instance
websocket_manager = WebSocketManager()
