"""
Per-model HuggingFace upload state.

Stored as a sidecar ``upload_state.json`` inside each model directory,
not in the jobs DB, so:
  - state survives moving / renaming / copying the model dir
  - state is inspectable with plain ``cat`` / ``jq`` for debugging
  - no migration needed — missing file means "never uploaded"

Shape (``format_version=1``):
    {
      "format_version": 1,
      "model_name": "my-model",           # local dir name
      "uploads": [
        {
          "timestamp": "2026-04-20T14:00:00+00:00",
          "repo_id": "user/my-model",
          "repo_url": "https://huggingface.co/user/my-model",
          "private": true,
          "commit_message": "...",
          "artifacts": ["adapter", "gguf:Q4_K_M", "gguf:Q8_0"],
          "status": "success" | "failed",
          "error": null
        },
        ...
      ]
    }

The ``uploads`` list is ordered oldest → newest. Readers only need the
last entry for "currently uploaded" checks; the full list is shown in
the UI as history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
SIDECAR_FILENAME = "upload_state.json"


@dataclass
class UploadEvent:
    """One upload attempt, success or failure."""

    timestamp: str
    repo_id: str
    repo_url: str
    private: bool
    commit_message: str
    artifacts: List[str] = field(default_factory=list)
    status: str = "success"  # "success" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UploadEvent":
        return cls(
            timestamp=data.get("timestamp", ""),
            repo_id=data.get("repo_id", ""),
            repo_url=data.get("repo_url", ""),
            private=bool(data.get("private", True)),
            commit_message=data.get("commit_message", ""),
            artifacts=list(data.get("artifacts", [])),
            status=data.get("status", "success"),
            error=data.get("error"),
        )


def _sidecar_path(model_dir: Path) -> Path:
    return Path(model_dir) / SIDECAR_FILENAME


def read_upload_state(model_dir: Path) -> dict:
    """
    Return the raw sidecar payload. Empty scaffold if the file is missing
    or unreadable — we never raise here so callers can blindly render the
    "never uploaded" state.
    """
    path = _sidecar_path(model_dir)
    if not path.is_file():
        return {"format_version": FORMAT_VERSION, "model_name": Path(model_dir).name, "uploads": []}

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s — treating as empty", path, exc)
        return {"format_version": FORMAT_VERSION, "model_name": Path(model_dir).name, "uploads": []}

    # Defensive: ensure the shape is as expected even if a human edited it.
    data.setdefault("format_version", FORMAT_VERSION)
    data.setdefault("model_name", Path(model_dir).name)
    data.setdefault("uploads", [])
    return data


def record_upload(
    model_dir: Path,
    *,
    repo_id: str,
    repo_url: str,
    private: bool,
    commit_message: str,
    artifacts: List[str],
    status: str = "success",
    error: Optional[str] = None,
) -> UploadEvent:
    """Append an upload event to the sidecar and return the event."""
    event = UploadEvent(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        repo_id=repo_id,
        repo_url=repo_url,
        private=private,
        commit_message=commit_message,
        artifacts=list(artifacts),
        status=status,
        error=error,
    )

    state = read_upload_state(model_dir)
    state["uploads"].append(event.to_dict())

    path = _sidecar_path(Path(model_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except OSError as exc:
        logger.error("Could not persist upload state to %s: %s", path, exc)

    return event


def last_successful_upload(model_dir: Path) -> Optional[UploadEvent]:
    """Return the most recent successful upload event, or ``None``."""
    state = read_upload_state(model_dir)
    for entry in reversed(state.get("uploads", [])):
        if entry.get("status") == "success":
            return UploadEvent.from_dict(entry)
    return None


def local_is_newer_than_last_upload(model_dir: Path) -> bool:
    """
    True when any file in ``model_dir`` has an mtime later than the
    timestamp of the most recent successful upload. Used to gray out
    "already uploaded ✓" hints when the user has re-trained.
    """
    last = last_successful_upload(model_dir)
    if last is None:
        return True  # never uploaded → "newer" is the relevant prompt

    try:
        upload_ts = datetime.fromisoformat(last.timestamp)
    except ValueError:
        return True

    upload_epoch = upload_ts.timestamp()
    root = Path(model_dir)
    for path in root.rglob("*"):
        # Ignore the sidecar itself — writing it is what we're comparing against.
        if path.name == SIDECAR_FILENAME:
            continue
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime > upload_epoch:
                return True
        except OSError:
            continue
    return False


def summary_for_api(model_dir: Path) -> dict:
    """
    Compact payload for ``GET /models/{name}/upload-state`` and list views.
    """
    state = read_upload_state(model_dir)
    last = last_successful_upload(model_dir)
    return {
        "has_uploads": bool(state.get("uploads")),
        "last_upload": last.to_dict() if last else None,
        "local_is_newer": local_is_newer_than_last_upload(model_dir),
        "history": state.get("uploads", []),
    }
