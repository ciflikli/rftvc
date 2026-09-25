"""Validation of the counting-process survival target."""

import numpy as np

SURV_DTYPE = np.dtype([("start", "f8"), ("stop", "f8"), ("event", "?")])


def make_survival_y(stop, event, start=None):
    """Build a structured survival target with fields ``start``, ``stop``, ``event``.

    ``start`` defaults to 0 (no delayed entry).
    """
    stop = np.asarray(stop, dtype=float)
    y = np.empty(stop.shape[0], dtype=SURV_DTYPE)
    y["start"] = 0.0 if start is None else np.asarray(start, dtype=float)
    y["stop"] = stop
    y["event"] = _as_bool(np.asarray(event))
    return y


def _as_bool(event):
    if event.dtype == bool:
        return event
    if np.issubdtype(event.dtype, np.integer) and np.isin(event, (0, 1)).all():
        return event.astype(bool)
    raise TypeError("event must be boolean (or integers in {0, 1})")


def check_survival_y(y):
    """Validate a structured survival target and return ``(start, stop, event)``.

    Raises ``TypeError`` for a wrong dtype and ``ValueError`` for invalid values.
    """
    y = np.asarray(y)
    names = y.dtype.names or ()
    missing = {"start", "stop", "event"} - set(names)
    if y.ndim != 1 or missing:
        raise TypeError(
            "y must be a 1-d structured array with fields 'start', 'stop', 'event' "
            f"(missing: {sorted(missing) or 'none'}); see rftvc.make_survival_y"
        )
    try:
        start = np.ascontiguousarray(y["start"], dtype=float)
        stop = np.ascontiguousarray(y["stop"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("'start' and 'stop' must be numeric") from exc
    event = np.ascontiguousarray(_as_bool(np.asarray(y["event"])))
    if not (np.isfinite(start).all() and np.isfinite(stop).all()):
        raise ValueError("'start' and 'stop' must be finite (no NaN or inf)")
    if (start >= stop).any():
        raise ValueError("every row must satisfy start < stop")
    if not event.any():
        raise ValueError("y contains no events")
    return start, stop, event
