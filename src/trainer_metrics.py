"""Turn a trainer's log dict into the extra metrics Merlina persists.

grimoire's ``on_log`` / ``on_evaluate`` callbacks receive dicts such as::

    {"train/loss": 0.9, "train/learning_rate": 5e-6, "train/epoch": 0.4,
     "train/global_step": 10, "train/progress": 0.1,
     "train/nll_loss": 0.8, "train/or_loss": 0.1, "train/reward_margin": 0.02, ...}

The loss function decides the component keys (ORPO logs ``nll_loss`` and
``or_loss``, DPO logs rewards and margins, and so on), so they are kept
generically, under the names the trainer logged them with. Values that
already have a dedicated ``training_metrics`` column, and bookkeeping values
that duplicate the row's ``step``, are left out.
"""

import math
from numbers import Real
from typing import Any, Dict, Mapping, Optional

# Keys stored in their own training_metrics columns.
COLUMN_KEYS = frozenset({"train/loss", "train/learning_rate", "eval/loss"})
# Bookkeeping that duplicates the row's step.
REDUNDANT_KEYS = frozenset({"train/global_step", "train/progress"})


def _to_float(value: Any) -> Optional[float]:
    """Coerce a scalar (python number, numpy scalar, 0-d tensor) to float.

    Returns ``None`` for non-finite values, since the API's JSON encoder
    rejects NaN/inf. Raises ``TypeError``/``ValueError`` for non-numeric input.
    """
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, Real):
        raise TypeError(f"not numeric: {type(value).__name__}")
    value = float(value)
    return value if math.isfinite(value) else None


def extract_extra_metrics(metrics: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """Return every numeric metric the trainer logged that has no column of its own.

    Non-finite values are kept as ``None`` (the metric was logged, its value
    wasn't representable); non-numeric values are dropped.
    """
    extra: Dict[str, Optional[float]] = {}
    for key, value in (metrics or {}).items():
        if not isinstance(key, str) or key in COLUMN_KEYS or key in REDUNDANT_KEYS:
            continue
        try:
            extra[key] = _to_float(value)
        except (TypeError, ValueError):
            continue
    return extra


def finite_or_none(value: Any) -> Optional[float]:
    """Float value or ``None`` for missing/non-numeric/non-finite input."""
    if value is None:
        return None
    try:
        return _to_float(value)
    except (TypeError, ValueError):
        return None
