"""Validation of the counting-process survival target."""

import numbers
import warnings
from typing import NamedTuple

import narwhals as nw
import numpy as np

SURV_DTYPE = np.dtype([("start", "f8"), ("stop", "f8"), ("event", "?")])
CR_DTYPE = np.dtype([("start", "f8"), ("stop", "f8"), ("event", "i8")])
MAX_CAUSES = 255


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


def make_competing_risks_y(stop, event, start=None):
    """Build a structured competing-risks target with fields ``start``, ``stop``, ``event``.

    ``event`` holds cause labels: 0 = censored, any positive integer = an event
    of that cause (labels need not be contiguous). A boolean event maps to {0, 1}.
    ``start`` defaults to 0 (no delayed entry).
    """
    stop = np.asarray(stop, dtype=float)
    y = np.empty(stop.shape[0], dtype=CR_DTYPE)
    y["start"] = 0.0 if start is None else np.asarray(start, dtype=float)
    y["stop"] = stop
    y["event"] = _as_labels(np.asarray(event))
    return y


def _frame(obj):
    """``obj`` as an eager narwhals DataFrame, or ``None`` if it is not a DataFrame."""
    df = nw.from_native(obj, eager_only=True, pass_through=True)
    return df if isinstance(df, nw.DataFrame) else None


def split_frame(X, ids=None):
    """Numeric ``X``, its column names and ``ids``, for array or DataFrame input.

    A DataFrame ``X`` (pandas, polars, pyarrow, ... via narwhals) gives its
    column names; nullable missing values become NaN. ``ids`` may name a column
    of ``X``: that column is returned as ``ids`` and removed from the features.
    Arrays pass through with ``names=None``.
    """
    df = _frame(X)
    if df is None:
        if isinstance(ids, str):
            raise ValueError(f"ids={ids!r} names a column, but X is not a DataFrame")
        return X, None, ids
    if isinstance(ids, str):
        if ids not in df.columns:
            raise ValueError(f"ids column {ids!r} not in X")
        ids, df = df[ids].to_numpy(), df.drop(ids)
    names = [str(c) for c in df.columns]
    return df.select(nw.all().cast(nw.Float64)).to_numpy(), np.asarray(names, dtype=object), ids


def _structured(y, required, make=make_survival_y):
    """A DataFrame ``y`` with the ``required`` columns (``start`` optional) as a structured array."""
    df = _frame(y)
    if df is None:
        return y
    missing = set(required) - set(df.columns) - {"start"}
    if missing:
        raise TypeError(f"y DataFrame is missing columns {sorted(missing)}")
    start = df["start"].to_numpy().astype(float) if "start" in df.columns else None
    stop = df["stop"].to_numpy().astype(float)
    if "event" not in required:
        out = np.empty(stop.shape[0], dtype=[("start", "f8"), ("stop", "f8")])
        out["start"], out["stop"] = 0.0 if start is None else start, stop
        return out
    return make(stop, df["event"].to_numpy(), start=start)


def _as_bool(event):
    if event.dtype == bool:
        return event
    if np.issubdtype(event.dtype, np.integer) and np.isin(event, (0, 1)).all():
        return event.astype(bool)
    raise TypeError(
        "event must be boolean (or integers in {0, 1}); for cause labels (competing risks) "
        "use rftvc.CompetingRisksForestTV with rftvc.make_competing_risks_y"
    )


def _as_labels(values, what="event"):
    """Non-negative integer labels as int64 (bool maps to {0, 1}; integral floats are accepted)."""
    values = np.asarray(values)
    if values.dtype == bool:
        return values.astype(np.int64)
    if np.issubdtype(values.dtype, np.integer):
        out = values.astype(np.int64)
    elif np.issubdtype(values.dtype, np.floating):
        if not (np.isfinite(values).all() and (values == np.round(values)).all()):
            raise ValueError(f"{what} labels must be integers")
        if (np.abs(values) >= 2.0**63).any():
            raise ValueError(f"{what} labels are out of the int64 range")
        out = values.astype(np.int64)
    else:
        raise TypeError(f"{what} must hold non-negative integer labels (0 = censored)")
    if (out < 0).any():
        raise ValueError(f"{what} labels must be non-negative (0 = censored)")
    return out


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


def check_survival_y(y, *, require_events=True):
    """Validate a structured survival target and return ``(start, stop, event)``.

    Raises ``TypeError`` for a wrong dtype and ``ValueError`` for invalid values
    (including no events, unless ``require_events=False``).
    """
    y = _structured(y, ("start", "stop", "event"))
    y, start, stop = _start_stop(y, ("start", "stop", "event"))
    event = np.ascontiguousarray(_as_bool(np.asarray(y["event"])))
    if require_events and not event.any():
        raise ValueError("y contains no events")
    return start, stop, event


def _check_causes(causes):
    """A cause vocabulary as sorted unique positive int64 labels."""
    causes = np.asarray(causes)
    if causes.ndim != 1 or causes.size == 0:
        raise ValueError("causes must be a non-empty 1-d array of labels")
    labels = _as_labels(causes, "causes")
    if (labels == 0).any():
        raise ValueError("causes must be positive labels (0 means censored)")
    if np.unique(labels).size != labels.size:
        raise ValueError("causes must not repeat a label")
    if labels.size > MAX_CAUSES:
        raise ValueError(f"at most {MAX_CAUSES} causes are supported, got {labels.size}")
    return np.sort(labels)


def check_competing_risks_y(y, causes=None, *, require_events=True):
    """Validate a competing-risks target; return ``(start, stop, codes, causes_)``.

    ``causes_`` is the sorted cause vocabulary: the observed non-zero labels,
    or ``causes`` when given. ``codes`` (``uint8``) maps each label to its
    1-based position in ``causes_`` (0 stays censored). With ``causes``, a label
    outside it raises ``ValueError``, and a cause with no events gives a
    ``UserWarning`` (its hazard is then zero). Raises ``ValueError`` for no
    events unless ``require_events=False``.
    """
    y = _structured(y, ("start", "stop", "event"), make=make_competing_risks_y)
    y, start, stop = _start_stop(y, ("start", "stop", "event"))
    labels = _as_labels(np.asarray(y["event"]))
    observed = np.unique(labels[labels != 0])
    if require_events and observed.size == 0:
        raise ValueError("y contains no events")
    if causes is None:
        causes_ = observed
        if causes_.size > MAX_CAUSES:
            raise ValueError(f"at most {MAX_CAUSES} causes are supported, got {causes_.size}")
    else:
        causes_ = _check_causes(causes)
        outside = np.setdiff1d(observed, causes_)
        if outside.size:
            raise ValueError(f"event labels {outside.tolist()} are not in causes={causes_.tolist()}")
        missing = np.setdiff1d(causes_, observed)
        if missing.size:
            warnings.warn(
                f"causes {missing.tolist()} have no events in y; their hazard and incidence are zero",
                UserWarning,
                stacklevel=3,
            )
    codes = np.zeros(labels.shape[0], dtype=np.uint8)
    nz = labels != 0
    codes[nz] = np.searchsorted(causes_, labels[nz]) + 1
    return start, stop, codes, causes_


def check_intervals(intervals):
    """Validate prediction intervals (fields or columns ``start``, ``stop``; ``event`` ignored)."""
    _, start, stop = _start_stop(_structured(intervals, ("start", "stop")), ("start", "stop"))
    return start, stop


class CountingProcess(NamedTuple):
    """Row grouping of validated counting-process data.

    ``group[r]`` is the chain index (``0..n_groups``) of row ``r``: its id, or
    under ``gap_policy="split_id"`` its contiguous segment. ``order`` sorts rows
    by (chain, start); ``offsets`` delimits each chain in that order.
    ``unit[r]`` is the original id index (``0..n_units``), the resampling unit:
    segments of one id stay one unit. Both are numbered by first appearance.
    """

    group: np.ndarray
    n_groups: int
    order: np.ndarray
    offsets: np.ndarray
    unit: np.ndarray
    n_units: int


def _check_ids(ids):
    """Id labels as an array, rejecting mixes of numbers and strings.

    ``np.asarray([1, "1"])`` coerces both to ``"1"`` and would merge two subjects.
    """
    if not isinstance(ids, np.ndarray) or ids.dtype == object:
        values = list(ids) if not isinstance(ids, np.ndarray) else ids.tolist()
        kinds = {"number" if isinstance(v, numbers.Number) else type(v).__name__ for v in values}
        if len(kinds) > 1:
            raise ValueError(f"ids mix label types ({sorted(kinds)}); use a single type")
        if kinds == {"number"}:
            return np.asarray(values)
        return np.asarray(values, dtype=object) if kinds - {"str"} else np.asarray(values, dtype=str)
    return ids


def check_counting_process(
    start, stop, event=None, ids=None, *, measured_at=None, gap_policy="error", layout="counting_process"
):
    """Validate the per-id structure of counting-process rows.

    ``layout="stacked"`` is for landmark stacks, where an id's rows are separate
    observations that may overlap in time: ids then only define resampling
    units, and the per-id structure checks below are skipped.

    Per id, rows must be non-overlapping and contiguous (``stop_j == start_{j+1}``),
    and only the last row may carry an event. A gap raises ``ValueError`` unless
    ``gap_policy="split_id"``: each contiguous segment is then its own chain
    (delayed re-entry after the gap, which is not at risk; an explicit modelling
    assumption), while the id stays one resampling unit (``unit``). If ``measured_at`` is
    given, covariates must be known at the row's ``start`` (``measured_at <= start``).
    """
    if layout not in ("counting_process", "stacked"):
        raise ValueError(f"layout must be 'counting_process' or 'stacked', got {layout!r}")
    if gap_policy not in ("error", "split_id"):
        raise ValueError(f"gap_policy must be 'error' or 'split_id', got {gap_policy!r}")
    n = start.shape[0]
    if n == 0:
        raise ValueError("no rows")
    ids = np.arange(n) if ids is None else _check_ids(ids)
    if ids.ndim != 1 or ids.shape[0] != n:
        raise ValueError(f"ids must be 1-d with {n} entries, got shape {ids.shape}")
    if measured_at is not None:
        measured_at = np.asarray(measured_at, dtype=float)
        if measured_at.shape != (n,):
            raise ValueError(f"measured_at must have {n} entries")
        if not np.isfinite(measured_at).all():
            raise ValueError("measured_at must be finite: an unknown measurement time cannot be checked")
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
    if layout == "stacked":
        new_group = np.r_[True, ~same]
        s_group = np.cumsum(new_group) - 1
        group = np.empty(n, dtype=np.int64)
        group[order] = s_group
        offsets = np.r_[np.flatnonzero(new_group), n].astype(np.uint64)
        group = group.astype(np.uint32)
        return CountingProcess(group, int(s_group[-1]) + 1, order, offsets, group, int(s_group[-1]) + 1)
    overlap = same & (s_start[1:] < s_stop[:-1])
    if overlap.any():
        raise ValueError(f"{overlap.sum()} overlapping rows within an id")
    if event is not None:
        s_event = np.asarray(event)[order] != 0  # any cause label, not only 1
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
    n_units = int(codes.max()) + 1
    return CountingProcess(
        group.astype(np.uint32), int(s_group[-1]) + 1, order, offsets, codes.astype(np.uint32), n_units
    )
