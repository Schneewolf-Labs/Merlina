"""
Reference-counted shared merge artifact.

Lives in its own module (zero ML dependencies) so the GGUF tests and the
upload-path tests can exercise the lifecycle without dragging in
torch / peft / grimoire / etc.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MergeArtifact:
    """
    Reference-counted handle to a merged-model directory shared between
    the HF-upload and GGUF-export background threads.

    The sync merge step creates one of these and seeds it with the
    number of background consumers. Each consumer calls :meth:`release`
    exactly once in its ``finally`` block; the directory is removed when
    the last consumer releases.

    A failed merge is represented by ``path=None`` and ``error`` set —
    consumers fall back to adapter-only behavior or skip gracefully.
    """

    def __init__(
        self,
        path: Optional[Path],
        num_consumers: int,
        *,
        error: Optional[str] = None,
    ):
        self.path = path
        self.error = error
        self._remaining = max(num_consumers, 0)
        self._lock = threading.Lock()

    def release(self) -> None:
        """Decrement the consumer count; clean up when zero remain."""
        cleanup_now = False
        with self._lock:
            if self._remaining <= 0:
                return
            self._remaining -= 1
            cleanup_now = self._remaining == 0 and self.path is not None

        if cleanup_now:
            try:
                shutil.rmtree(self.path, ignore_errors=True)
                logger.info("Cleaned up shared merged directory: %s", self.path)
            except Exception as exc:
                logger.warning("Could not clean up merged dir %s: %s", self.path, exc)
