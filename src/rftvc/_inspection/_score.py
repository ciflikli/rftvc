"""Permutation-importance core on a ``predict`` callable, with PE scoring."""

import warnings
from typing import Callable, NamedTuple

import numpy as np

from ..metrics import UndefinedMetricError, piecewise_exponential_score
from . import _boot, _strata


class Evaluation(NamedTuple):
    """Scored rows and how to predict them.

    ``predict(X_rows, rows)`` returns the cumulative hazards at ``windows``
    (``(k, M+1)`` or ``(k, J, M+1)``) of ``X_rows``, the covariates of the
    evaluation rows ``rows`` (the index is used by OOB prediction only).
    """

    X: np.ndarray
    y: np.ndarray
    start: np.ndarray
    ids: np.ndarray
    windows: np.ndarray
    null: np.ndarray
    alpha: float
    causes: object
    cause: object
    predict: Callable


class Strata(NamedTuple):
    kind: object  # "time", None or "user"
    user: object  # integer codes per row (kind "user")
    n_strata: int
    conditional_on: list
    n_bins: int


def score(ev, y, H, ids=None, quiet=False):
    """PE score (sums) of predictions ``H`` on rows ``y``."""
    with warnings.catch_warnings():
        if quiet:  # zero-rate warnings of permuted predictions are not the user's concern
            warnings.simplefilter("ignore", UserWarning)
        return piecewise_exponential_score(
            y, H, ev.windows, null_cumhaz=ev.null, alpha=ev.alpha, causes=ev.causes, cause=ev.cause,
            ids=ids, reduce="sum",
        )


def labels(X, start, st, cols, rows=None):
    """Stratum label per row for the unit ``cols``: the base strata crossed with
    bins of the conditioning columns outside the unit."""
    if st.kind == "time":
        base = _strata.bin_codes(start, st.n_strata)
    elif st.kind == "user":
        base = st.user if rows is None else st.user[rows]
    else:
        base = np.zeros(X.shape[0], dtype=np.intp)
    extra = [_strata.bin_codes(X[:, c], st.n_bins) for c in st.conditional_on if c not in cols]
    return _strata.combine(base, *extra)


def _permuted(ev, H, X, cols, lab, rng, rows):
    """Predictions after one within-stratum permutation of the unit's columns.

    ``X`` / ``H`` are the (possibly resampled) rows, ``rows`` their evaluation
    row indices; only rows that receive another row's values are re-predicted.
    """
    src = _strata.donors(lab, rng)
    changed = np.flatnonzero(src != np.arange(src.size))
    Hp = H.copy()
    if changed.size:
        Xc = X[changed].copy()
        Xc[:, cols] = X[src[changed]][:, cols]
        Hp[changed] = ev.predict(Xc, rows[changed])
    return Hp


class UnitResult(NamedTuple):
    drops: np.ndarray  # (R,)
    window: np.ndarray  # (M,)
    cause: object  # (J,) or None
    by_id: np.ndarray  # (n_ids,)
    se: float
    n_unpermuted: int


def unit_importance(ev, H, intact, cols, st, stream, n_repeats, n_bootstrap):
    """Importance of one unit: repeats, then the id-cluster bootstrap SE."""
    n = ev.X.shape[0]
    rows = np.arange(n)
    N = intact.n_events
    lab = labels(ev.X, ev.start, st, cols)
    rng = np.random.Generator(np.random.PCG64(stream.spawn(1)[0]))
    drops, window, by_id = np.empty(n_repeats), 0.0, 0.0
    cause = None if intact.by_cause is None else 0.0
    for r in range(n_repeats):
        s = score(ev, ev.y, _permuted(ev, H, ev.X, cols, lab, rng, rows), ids=ev.ids, quiet=True)
        drops[r] = (intact.total - s.total) / N
        window = window + (intact.by_window - s.by_window) / N
        by_id = by_id + (intact.by_id - s.by_id) / N
        if cause is not None:
            cause = cause + (intact.by_cause - s.by_cause) / N
    se = np.nan
    if n_bootstrap >= 2:
        se = _bootstrap_se(ev, H, cols, st, stream, n_repeats, n_bootstrap)
    return UnitResult(
        drops, window / n_repeats, None if cause is None else cause / n_repeats, by_id / n_repeats, se,
        _strata.n_singletons(lab),
    )


def _bootstrap_se(ev, H, cols, st, stream, n_repeats, n_bootstrap):
    """SD over id-cluster bootstrap replicates of the repeat-averaged per-event drop."""
    values, attempts = [], 0
    while len(values) < n_bootstrap:
        if attempts == 2 * n_bootstrap:
            raise UndefinedMetricError("too many bootstrap replicates without a scored event")
        attempts += 1
        rng = np.random.Generator(np.random.PCG64(stream.spawn(1)[0]))
        ri, _ = _boot._resample_ids(ev.ids, rng)
        y_b, H_b, X_b = ev.y[ri], H[ri], ev.X[ri]
        try:
            base = score(ev, y_b, H_b, quiet=True)
        except UndefinedMetricError:
            continue
        lab = labels(X_b, ev.start[ri], st, cols, rows=ri)
        d = 0.0
        for _ in range(n_repeats):
            d += base.total - score(ev, y_b, _permuted(ev, H_b, X_b, cols, lab, rng, ri), quiet=True).total
        values.append(d / (n_repeats * base.n_events))
    return float(np.std(values, ddof=1))
