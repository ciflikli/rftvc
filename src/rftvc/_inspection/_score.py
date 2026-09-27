"""Permutation-importance core on a ``predict`` callable, with PE scoring."""

import warnings
from typing import Callable, NamedTuple

import numpy as np

from ..metrics import UndefinedMetricError, piecewise_exponential_score
from ..model_selection import _score_landmark
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


# --- landmark Brier / IBS importance (design §3, §6): a censoring model per landmark ------


class LandmarkLossResult(NamedTuple):
    importances: np.ndarray  # (p, n_repeats)
    importances_mean: np.ndarray  # (p,)
    importances_std: np.ndarray  # (p,)
    importances_se: np.ndarray  # (p,)
    baseline_score: float


def _loss_times(horizon, scoring, n_times):
    if scoring == "brier":
        return np.array([horizon])
    return np.linspace(0, horizon, n_times + 1)[1:]


def _pooled(vals, ns):
    """The n-weighted mean of per-landmark scores."""
    return float(np.sum(np.asarray(ns) * np.asarray(vals)) / np.sum(ns))


def _permuted_X(X, cols, lab, rng):
    """``(Xp, changed)``: ``X`` after one within-stratum permutation of ``cols``, and the
    rows that received another row's values."""
    src = _strata.donors(lab, rng)
    changed = np.flatnonzero(src != np.arange(src.size))
    Xp = X.copy()
    if changed.size:
        Xp[np.ix_(changed, cols)] = X[np.ix_(src[changed], cols)]
    return Xp, changed


def _landmark_scores(X, y, groups, curve_of, name, horizon, times, censoring_estimator, g_min, cause):
    """Per-landmark ``_score_landmark`` output, one per risk set in ``groups``."""
    return [_score_landmark(curve_of(X[m]), y[m], horizon, (name,), times, censoring_estimator, g_min, cause) for m in groups]


def _rescore_changed(Xp, changed, groups, vals, ns, y, curve_of, name, horizon, times, censoring_estimator, g_min, cause):
    """The pooled score after only the landmarks touched by ``changed`` rows are rescored."""
    mask = np.zeros(Xp.shape[0], dtype=bool)
    mask[changed] = True
    new_vals = vals.copy()
    for i, m in enumerate(groups):
        if mask[m].any():
            out = _score_landmark(curve_of(Xp[m]), y[m], horizon, (name,), times, censoring_estimator, g_min, cause)
            new_vals[i] = out[name]
    return _pooled(new_vals, ns)


def landmark_loss_importance(model, Xe, ye, row_ids, s, units, scoring, cause, n_repeats, n_bootstrap, entropy,
                              n_times, censoring_estimator, g_min, competing, st):
    """Brier/IBS permutation importance of a fitted landmark model: ``pooled(permuted) -
    pooled(baseline)`` (a loss, so positive = permuting made it worse), pooled over
    landmarks by risk-set size, with a per-landmark censoring fit (``_score_landmark``,
    as ``model_selection.landmark_cross_validate``), never a pooled stacked-data IPCW."""
    if competing and cause is None:
        raise ValueError("cause is required for Brier/IBS importance of a competing-risks landmark model")
    horizon = model.horizon
    times = _loss_times(horizon, scoring, n_times)
    name = "brier" if scoring == "brier" else "integrated_brier"
    landmarks = np.unique(s)
    groups = [np.flatnonzero(s == lm) for lm in landmarks]

    def curve_of(Xr):
        if competing:
            return model.forest_.predict_cumulative_incidence(Xr, times, cause=cause)
        return model.forest_.predict_survival_function(Xr, times)

    base_scores = _landmark_scores(Xe, ye, groups, curve_of, name, horizon, times, censoring_estimator, g_min, cause)
    ns = np.array([o["n"] for o in base_scores], dtype=float)
    vals = np.array([o[name] for o in base_scores], dtype=float)
    baseline = _pooled(vals, ns)
    start0 = np.zeros(Xe.shape[0])  # landmark strata never read Evaluation.start

    imp, std, se = [], [], []
    for cols in units:
        lab = labels(Xe, start0, st, cols)
        stream = np.random.SeedSequence(entropy, spawn_key=(int(cols.min()),))
        rng = np.random.Generator(np.random.PCG64(stream.spawn(1)[0]))
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp, changed = _permuted_X(Xe, cols, lab, rng)
            permuted = _rescore_changed(Xp, changed, groups, vals, ns, ye, curve_of, name, horizon, times,
                                        censoring_estimator, g_min, cause)
            drops[r] = permuted - baseline
        imp.append(drops)
        std.append(float(drops.std()))
        se.append(
            _bootstrap_se_loss(
                Xe, ye, s, cols, st, stream, n_repeats, n_bootstrap, curve_of, name, horizon, times,
                censoring_estimator, g_min, cause, row_ids,
            )
            if n_bootstrap >= 2
            else np.nan
        )
    imp = np.array(imp)
    return LandmarkLossResult(imp, imp.mean(axis=1), np.array(std), np.array(se), baseline)


def _bootstrap_se_loss(Xe, ye, s, cols, st, stream, n_repeats, n_bootstrap, curve_of, name, horizon, times,
                        censoring_estimator, g_min, cause, row_ids):
    """SD over id-cluster bootstrap replicates of the repeat-averaged pooled-loss drop; each
    replicate re-fits every landmark's censoring model on its own resampled outcomes."""
    values, attempts = [], 0
    while len(values) < n_bootstrap:
        if attempts == 2 * n_bootstrap:
            raise UndefinedMetricError("too many bootstrap replicates without a scored event")
        attempts += 1
        rng = np.random.Generator(np.random.PCG64(stream.spawn(1)[0]))
        ri, _ = _boot._resample_ids(row_ids, rng)
        Xb, yb, sb = Xe[ri], ye[ri], s[ri]
        landmarks_b = np.unique(sb)
        groups_b = [np.flatnonzero(sb == lm) for lm in landmarks_b]
        try:
            base_b = _landmark_scores(Xb, yb, groups_b, curve_of, name, horizon, times, censoring_estimator, g_min, cause)
        except UndefinedMetricError:
            continue
        ns_b = np.array([o["n"] for o in base_b], dtype=float)
        vals_b = np.array([o[name] for o in base_b], dtype=float)
        baseline_b = _pooled(vals_b, ns_b)
        lab_b = labels(Xb, np.zeros(Xb.shape[0]), st, cols, rows=ri)
        d = 0.0
        for _ in range(n_repeats):
            Xp, changed = _permuted_X(Xb, cols, lab_b, rng)
            permuted_b = _rescore_changed(Xp, changed, groups_b, vals_b, ns_b, yb, curve_of, name, horizon, times,
                                          censoring_estimator, g_min, cause)
            d += permuted_b - baseline_b
        values.append(d / n_repeats)
    return float(np.std(values, ddof=1))
