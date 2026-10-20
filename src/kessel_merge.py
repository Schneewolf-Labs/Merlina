"""Model merging (mergekit-style recipes) with Kessel, run as Merlina jobs.

A merge recipe is mergekit's YAML (``models`` or ``slices``, ``merge_method``, ``base_model``,
``parameters``, ``dtype``...), which Kessel reads as-is. The API takes it either as a JSON object
or as YAML text; ``normalize`` parses it with Kessel and returns both forms, so the page's YAML
preview and the job's stored recipe are what Kessel will actually run.

Kessel streams one output tensor at a time, so a merge needs little RAM beyond the tensors in
flight; it still runs on the job queue so it never overlaps a training run's model load.
The output is written next to ``models/`` and moved in only when complete, so a half-written
merge never shows up in the model lists (export, upload, playground).

Kessel is an optional dependency (``pip install git+https://github.com/Schneewolf-Labs/kessel``).
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# How often the progress callback writes to the job DB and checks for a stop request
_REPORT_EVERY_S = 1.0


class MergeRecipeError(ValueError):
    """The recipe can't be parsed or doesn't fit its merge method."""


class MergeStopped(Exception):
    """The job was asked to stop."""


def kessel_available() -> bool:
    try:
        import kessel.merge  # noqa: F401
    except ImportError:
        return False
    return True


def list_methods() -> list[dict]:
    """Every merge method with its parameters: name -> {default, required} for globals and per model."""
    from kessel import methods as kmethods

    def params(d: dict) -> dict:
        return {k: {"default": None if v is kmethods.REQUIRED else v, "required": v is kmethods.REQUIRED} for k, v in d.items()}

    return [
        {
            "name": m.name,
            "needs_base": m.needs_base,
            "global_params": params(m.global_params),
            "model_params": params(m.model_params),
        }
        for m in sorted(kmethods.METHODS.values(), key=lambda m: m.name)
    ]


def normalize(recipe: Optional[dict] = None, yaml_text: Optional[str] = None) -> tuple[dict, str]:
    """Parse a recipe (dict or YAML) with Kessel and check it against its method.

    Returns ``(recipe, yaml)`` in Kessel's canonical form. Raises MergeRecipeError.
    """
    import yaml
    from kessel import methods as kmethods
    from kessel.config import parse
    from kessel.merge import _config_dict

    if (recipe is None) == (yaml_text is None):
        raise MergeRecipeError("Send either a recipe object or YAML text")
    if yaml_text is not None:
        try:
            recipe = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise MergeRecipeError(f"Invalid YAML: {exc}") from exc
        if not isinstance(recipe, dict):
            raise MergeRecipeError("The YAML must be a mapping (merge_method, models, ...)")
    if not recipe.get("models") and not recipe.get("slices"):
        raise MergeRecipeError("Add at least one model")
    try:
        cfg = parse(recipe)
        method = kmethods.get(cfg.merge_method)
    except (ValueError, KeyError, TypeError) as exc:
        raise MergeRecipeError(str(exc)) from exc

    if method.needs_base and not cfg.base_model:
        raise MergeRecipeError(f"{method.name} needs a base_model")
    sources = cfg.models if cfg.models is not None else [s for sl in cfg.slices for s in sl.sources]
    if not sources:
        raise MergeRecipeError("Add at least one model")
    missing = [k for k, v in method.global_params.items()
               if v is kmethods.REQUIRED and k not in (cfg.parameters or {})]
    if missing:
        raise MergeRecipeError(f"{method.name} needs parameter(s): {', '.join(missing)}")
    base = str(cfg.base_model) if cfg.base_model else None
    for s in sources:
        if str(s.model) == base:
            continue
        gaps = [k for k, v in method.model_params.items()
                if v is kmethods.REQUIRED and k not in (s.parameters or {}) and k not in (cfg.parameters or {})]
        if gaps:
            raise MergeRecipeError(f"{s.model}: {method.name} needs {', '.join(gaps)} for each model")

    canonical = _config_dict(cfg)
    return canonical, yaml.safe_dump(canonical, sort_keys=False)


def describe(recipe: dict) -> str:
    """A short label for the job list: the base model, else the first model."""
    if recipe.get("base_model"):
        return str(recipe["base_model"])
    srcs = recipe.get("models") or [s for sl in recipe.get("slices", []) for s in sl.get("sources", [])]
    return str(srcs[0]["model"]) if srcs else ""


def run_merge(
    job_id: str,
    recipe: dict,
    out_dir: Path,
    job_manager,
    device: str = "cpu",
    publish: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Run a Kessel merge into ``out_dir`` (which must not exist), reporting progress on the job.

    Status goes to the job DB and, through ``publish(job_id, message)`` (e.g.
    ``websocket_manager.publish``), to the job's WebSocket subscribers in the same message shapes
    training uses. Writes to a hidden sibling first and renames it into place on success; on
    failure or stop the partial output is removed. Returns Kessel's report as a dict.
    """
    from kessel.merge import run

    def send(message: dict) -> None:
        if publish:
            publish(job_id, {"job_id": job_id, "timestamp": time.time(), **message})

    def status(**fields) -> None:
        job_manager.update_job(job_id, **fields)
        send({"type": "status_update", **{k: v for k, v in fields.items() if v is not None}})

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir.name} already exists in the models directory")
    partial = out_dir.parent.parent / f".merge-{job_id}"
    shutil.rmtree(partial, ignore_errors=True)

    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if done not in (0, total) and now - last[0] < _REPORT_EVERY_S:
            return
        last[0] = now
        job = job_manager.get_job(job_id)
        if job is not None and job.stop_requested:
            raise MergeStopped()
        status(status="merging", current_step=done, total_steps=total, progress=done / total if total else 0.0)

    status(status="loading_model", progress=0.0)
    try:
        report = run(recipe, partial, device=device, progress=progress)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(partial), str(out_dir))
    except MergeStopped:
        shutil.rmtree(partial, ignore_errors=True)
        status(status="stopped")
        logger.info("Merge %s stopped", job_id)
        return {}
    except BaseException as exc:
        shutil.rmtree(partial, ignore_errors=True)
        send({"type": "error", "error": str(exc)})
        raise  # the job queue marks the job failed with this error

    result = _report_dict(report)
    job_manager.update_job(job_id, status="completed", progress=1.0, output_dir=str(out_dir))
    send({"type": "completed", "output_dir": str(out_dir), "final_metrics": result})
    logger.info("Merge %s finished: %s (%s)", job_id, out_dir, result)
    return result


def _report_dict(report: Any) -> dict:
    try:
        from dataclasses import asdict
        return {k: v for k, v in asdict(report).items() if isinstance(v, (int, float, str))}
    except TypeError:
        return {}
