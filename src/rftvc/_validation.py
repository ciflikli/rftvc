"""Validation of the counting-process survival target."""

from typing import NamedTuple

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


def _start_stop(y, required):
    y = np.asarray(y)
    names = y.dtype.names or ()
    missing = set(required) - set(names)
    if y.ndim != 1 or missing:
        raise TypeError(
            f"expected a 1-d structured array with fields {list(required)} "
            f"(missing: {sorted(missing) or 'none'}); see rftvc.make_survival_y"
        )
    try:
        start = np.ascontiguousarray(y["start"], dtype=float)
        stop = np.ascontiguousarray(y["stop"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("'start' and 'stop' must be numeric") from exc
    if not (np.isfinite(start).all() and np.isfinite(stop).all()):
        raise ValueError("'start' and 'stop' must be finite (no NaN or inf)")
    if (start >= stop).any():
        raise ValueError("every row must satisfy start < stop")
    return y, start, stop


def check_survival_y(y):
    """Validate a structured survival target and return ``(start, stop, event)``.

    Raises ``TypeError`` for a wrong dtype and ``ValueError`` for invalid values.
    """
    y, start, stop = _start_stop(y, ("start", "stop", "event"))
    event = np.ascontiguousarray(_as_bool(np.asarray(y["event"])))
    if not event.any():
        raise ValueError("y contains no events")
    return start, stop, event


def check_intervals(intervals):
    """Validate prediction intervals (fields ``start``, ``stop``; ``event`` ignored)."""
    _, start, stop = _start_stop(intervals, ("start", "stop"))
    return start, stop


class CountingProcess(NamedTuple):
    """Row grouping of validated counting-process data.

    ``group[r]`` is the id index (``0..n_groups``) of row ``r``; ``order`` sorts
    rows by (id index, start); ``offsets`` delimits each id in that order.
    Ids are numbered by first appearance.
    """

    group: np.ndarray
    n_groups: int
    order: np.ndarray
    offsets: np.ndarray


def check_counting_process(start, stop, event=None, ids=None, *, measured_at=None, gap_policy="error"):
    """Validate the per-id structure of counting-process rows.

    Per id, rows must be non-overlapping and contiguous (``stop_j == start_{j+1}``),
    and only the last row may carry an event. A gap raises ``ValueError`` unless
    ``gap_policy="split_id"``, which treats each contiguous segment as its own id
    (delayed re-entry; an explicit modelling assumption). If ``measured_at`` is
    given, covariates must be known at the row's ``start`` (``measured_at <= start``).
    """
    if gap_policy not in ("error", "split_id"):
        raise ValueError(f"gap_policy must be 'error' or 'split_id', got {gap_policy!r}")
    n = start.shape[0]
    if ids is None:
        ids = np.arange(n)
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.shape[0] != n:
        raise ValueError(f"ids must be 1-d with {n} entries, got shape {ids.shape}")
    if measured_at is not None:
        measured_at = np.asarray(measured_at, dtype=float)
        if measured_at.shape != (n,):
            raise ValueError(f"measured_at must have {n} entries")
        late = measured_at > start
        if late.any():
            raise ValueError(
                f"{late.sum()} rows use covariates measured after the row's start "
                "(look-ahead); covariates on (start, stop] must be known at start"
            )

    _, first, codes = np.unique(ids, return_index=True, return_inverse=True)
    codes = np.argsort(np.argsort(first))[codes]  # number ids by first appearance
    order = np.lexsort((start, codes))
    s_codes, s_start, s_stop = codes[order], start[order], stop[order]
    same = s_codes[1:] == s_codes[:-1]
    overlap = same & (s_start[1:] < s_stop[:-1])
    if overlap.any():
        raise ValueError(f"{overlap.sum()} overlapping rows within an id")
    if event is not None:
        s_event = event[order]
        not_last = np.r_[same, False]
        if (s_event & not_last).any():
            raise ValueError("an event occurs on a row that is not the id's last row")
    gap = same & (s_start[1:] > s_stop[:-1])
    if gap.any() and gap_policy == "error":
        raise ValueError(
            f"{gap.sum()} gaps between consecutive rows of an id; "
            "fill them, or pass gap_policy='split_id' to treat segments as delayed re-entry"
        )
    # New group at every id change and (under split_id) at every gap.
    new_group = np.r_[True, ~same | gap]
    s_group = np.cumsum(new_group) - 1
    group = np.empty(n, dtype=np.int64)
    group[order] = s_group
    offsets = np.r_[np.flatnonzero(new_group), n].astype(np.uint64)
    return CountingProcess(group.astype(np.uint32), int(s_group[-1]) + 1 if n else 0, order, offsets)
