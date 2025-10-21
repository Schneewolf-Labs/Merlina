"""
WebSocket Manager for Real-time Training Updates
Broadcasts training progress, metrics, and status to connected clients
"""

import asyncio
import logging
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket
import json

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts updates to clients.
    """

    def __init__(self):
        # Set of active WebSocket connections
        self.active_connections: Set[WebSocket] = set()

        # Job-specific connections (job_id -> set of websockets)
        self.job_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: Optional[str] = None):
        """
        Accept and register a WebSocket connection.

        Args:
            websocket: WebSocket connection
            job_id: Optional job ID to subscribe to specific job updates
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        if job_id:
            if job_id not in self.job_connections:
                self.job_connections[job_id] = set()
            self.job_connections[job_id].add(websocket)
            logger.info(f"WebSocket connected for job {job_id}")
        else:
            logger.info("WebSocket connected (global)")

    def disconnect(self, websocket: WebSocket, job_id: Optional[str] = None):
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection
            job_id: Optional job ID if subscribed to specific job
        """
        self.active_connections.discard(websocket)

        if job_id and job_id in self.job_connections:
            self.job_connections[job_id].discard(websocket)

            # Clean up empty job connection sets
            if not self.job_connections[job_id]:
                del self.job_connections[job_id]

            logger.info(f"WebSocket disconnected from job {job_id}")
        else:
            logger.info("WebSocket disconnected (global)")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected clients.

        Args:
            message: Dictionary to send as JSON
        """
        if not self.active_connections:
            return

        # Convert to JSON
        json_message = json.dumps(message)

        # Send to all connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_job(self, job_id: str, message: Dict[str, Any]):
        """
        Broadcast message to clients subscribed to a specific job.

        Args:
            job_id: Job identifier
            message: Dictionary to send as JSON
        """
        if job_id not in self.job_connections:
            return

        connections = self.job_connections[job_id]
        if not connections:
            return

        # Add job_id to message if not present
        if "job_id" not in message:
            message["job_id"] = job_id

        # Convert to JSON
        json_message = json.dumps(message)

        # Send to job-specific connections
        disconnected = set()
        for connection in connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.warning(f"Failed to send to job {job_id} connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection, job_id)

    async def send_status_update(
        self,
        job_id: str,
        status: str,
        progress: float,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gpu_memory: Optional[float] = None,
        **kwargs
    ):
        """
        Send a formatted status update for a job.

        Args:
            job_id: Job identifier
            status: Current status
            progress: Progress percentage (0.0 to 1.0)
            current_step: Current training step
            total_steps: Total training steps
            loss: Training loss
            eval_loss: Evaluation loss
            learning_rate: Current learning rate
            gpu_memory: GPU memory used in GB
            **kwargs: Additional fields
        """
        message = {
            "type": "status_update",
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "timestamp": asyncio.get_event_loop().time()
        }

        if current_step is not None:
            message["current_step"] = current_step

        if total_steps is not None:
            message["total_steps"] = total_steps

        if loss is not None:
            message["loss"] = loss

        if eval_loss is not None:
            message["eval_loss"] = eval_loss

        if learning_rate is not None:
            message["learning_rate"] = learning_rate

        if gpu_memory is not None:
            message["gpu_memory"] = gpu_memory

        # Add any additional kwargs
        message.update(kwargs)

        await self.broadcast_to_job(job_id, message)

    async def send_metric_update(
        self,
        job_id: str,
        step: int,
        metrics: Dict[str, float]
    ):
        """
        Send training metrics update.

        Args:
            job_id: Job identifier
            step: Training step
            metrics: Dictionary of metric names to values
        """
        message = {
            "type": "metrics",
            "job_id": job_id,
            "step": step,
            "metrics": metrics,
            "timestamp": asyncio.get_event_loop().time()
        }

        await self.broadcast_to_job(job_id, message)

    async def send_error(self, job_id: str, error: str):
        """
        Send error notification.

        Args:
            job_id: Job identifier
            error: Error message
        """
        message = {
            "type": "error",
            "job_id": job_id,
            "error": error,
            "timestamp": asyncio.get_event_loop().time()
        }

        await self.broadcast_to_job(job_id, message)

    async def send_completion(
        self,
        job_id: str,
        output_dir: str,
        final_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Send job completion notification.

        Args:
            job_id: Job identifier
            output_dir: Path to saved model
            final_metrics: Final training metrics
        """
        message = {
            "type": "completed",
            "job_id": job_id,
            "output_dir": output_dir,
            "timestamp": asyncio.get_event_loop().time()
        }

        if final_metrics:
            message["final_metrics"] = final_metrics

        await self.broadcast_to_job(job_id, message)

    def get_connection_count(self, job_id: Optional[str] = None) -> int:
        """
        Get number of active connections.

        Args:
            job_id: Optional job ID to get job-specific count

        Returns:
            Number of connections
        """
        if job_id:
            return len(self.job_connections.get(job_id, set()))
        return len(self.active_connections)


# Global instance
websocket_manager = WebSocketManager()
