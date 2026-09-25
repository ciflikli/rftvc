"""Evaluation metrics for survival predictions.

Landmark metrics score one landmark at a time on the reset clock: ``y`` holds
the outcomes of the landmark risk set (fields ``start`` = 0, ``stop``,
``event``), as built by ``make_landmark_data``, and ``w`` is the horizon on that
clock. They are ordinary right-censored metrics and apply to any data with
``start == 0``.

Classification at ``w``:

- **case**: ``event`` and ``stop <= w`` (event within the horizon);
- **control**: ``stop >= w`` and not a case (event-free through ``w``; a
  subject administratively censored exactly at ``w`` is a control);
- otherwise **censored before w**: status unknown, weight 0 under IPCW.

Censoring weights use the left limits ``G(stop-)`` and ``G(w-)`` of the censoring
survival ``G(t) = P(C > t)``: a subject's status at ``t`` is observed iff
``C >= t``. When no test subject is censored before ``w`` (complete follow-up)
all weights are 1 and no censoring model is used (the *exact path*).
"""

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from ._validation import check_survival_y

__all__ = [
    "KaplanMeierCensoring",
    "UndefinedMetricError",
    "brier_landmark",
    "calibration_table",
    "cindex_dynamic",
    "concordance_index_cp",
    "integrated_brier",
]


class UndefinedMetricError(ValueError):
    """The metric is undefined on these outcomes (e.g. no case or no comparable pair)."""


class KaplanMeierCensoring(BaseEstimator):
    """Reverse Kaplan–Meier estimate of the censoring survival ``G(t) = P(C > t)``.

    At tied times events precede censoring (as in scikit-survival): the
    censoring hazard at ``t`` is ``c_t / (Y_t - d_t)``. ``fit`` takes
    right-censored outcomes (``start`` must be 0); ``predict(times, left=True)``
    returns the left limit ``G(t-)``. Any fitted object with the same
    ``predict`` signature can be passed as ``censoring_estimator``.
    """

    def fit(self, y):
        _, stop, event = _check_right_censored(y, "y")
        times = np.unique(stop)
        at_risk = stop.size - np.searchsorted(np.sort(stop), times, side="left")
        d = np.bincount(np.searchsorted(times, stop[event]), minlength=times.size)
        c = np.bincount(np.searchsorted(times, stop[~event]), minlength=times.size)
        denom = at_risk - d
        hazard = np.divide(c, denom, out=np.zeros(times.size), where=denom > 0)
        self.times_ = times
        self.survival_ = np.cumprod(1.0 - hazard)
        return self

    def predict(self, times, *, left=False):
        times = np.asarray(times, dtype=float)
        idx = np.searchsorted(self.times_, times, side="left" if left else "right")
        return np.r_[1.0, self.survival_][idx]


def _check_right_censored(y, name):
    start, stop, event = check_survival_y(y, require_events=False)
    if np.any(start != 0):
        raise ValueError(f"{name} must be right-censored outcomes on the reset clock (start == 0)")
    return start, stop, event


def _outcome_classes(stop, event, w):
    case = event & (stop <= w)
    control = (stop >= w) & ~case
    return case, control


def _censoring(y_censor, censoring_estimator):
    if (y_censor is None) == (censoring_estimator is None):
        raise ValueError(
            "pass exactly one of y_censor or a fitted censoring_estimator"
        )
    if censoring_estimator is not None:
        return censoring_estimator
    return KaplanMeierCensoring().fit(y_censor)


def _ipcw(stop, event, w, y_censor, censoring_estimator, g_min):
    """Per-subject weights at horizon ``w`` and the diagnostic info."""
    if not 0 < g_min <= 1:
        raise ValueError("g_min must lie in (0, 1]")
    case, control = _outcome_classes(stop, event, w)
    exact = bool(np.all(case | control))
    info = {"exact": exact, "n": int(stop.size), "n_cases": int(case.sum()), "n_controls": int(control.sum())}
    if exact:
        if y_censor is not None and censoring_estimator is not None:
            raise ValueError("pass at most one of y_censor or censoring_estimator")
        info.update(n_censored=0, n_clipped=0)
        return case, control, np.ones(stop.size), info
    cens = _censoring(y_censor, censoring_estimator)
    g = np.ones(stop.size)
    g[case] = cens.predict(stop[case], left=True)
    g[control] = cens.predict(np.array([w]), left=True)[0]
    weighted = case | control
    clipped = weighted & (g < g_min)
    g = np.maximum(g, g_min)
    weights = np.where(weighted, 1.0 / g, 0.0)
    info.update(n_censored=int((~weighted).sum()), n_clipped=int(clipped.sum()))
    return case, control, weights, info


def _check_pred(values, n, name):
    values = np.asarray(values, dtype=float)
    if values.shape[0] != n:
        raise ValueError(f"{name} has {values.shape[0]} rows but y has {n}")
    if np.isnan(values).any():
        raise ValueError(f"{name} must not contain NaN")
    return values


def brier_landmark(
    y_test, risk, w, *, y_censor=None, censoring_estimator=None, g_min=0.05, return_info=False
):
    """IPCW Brier score of ``risk = P(T <= w)`` at horizon ``w``.

    ``mean_i weight_i * (1{case_i} - risk_i)^2`` with weights ``1/G(stop_i-)``
    for cases, ``1/G(w-)`` for controls and 0 for subjects censored before
    ``w`` (Graf et al. 1999). ``G`` is clipped below at ``g_min``.

    Parameters
    ----------
    y_test : structured array (``start`` = 0, ``stop``, ``event``)
        Test outcomes, e.g. one landmark's risk set on the reset clock.
    risk : array-like of shape (n,)
    w : float
        Horizon on the same clock.
    y_censor : structured array, optional
        Outcomes a ``KaplanMeierCensoring`` is fitted on (in CV: the test risk set at the landmark).
    censoring_estimator : fitted object, optional
        ``predict(times, left=...) -> G``. Pass exactly one of the two when
        IPCW is needed; neither is needed under complete follow-up.
    g_min : float, default=0.05
        Positivity floor for ``G``.
    return_info : bool, default=False
        Also return a dict: ``exact`` (no IPCW needed), counts of cases,
        controls, censored-before-``w`` and clipped weights.
    """
    _, stop, event = _check_right_censored(y_test, "y_test")
    risk = _check_pred(risk, stop.size, "risk").ravel()
    if not w > 0:
        raise ValueError("w must be positive")
    case, _, weights, info = _ipcw(stop, event, w, y_censor, censoring_estimator, g_min)
    score = float(np.mean(weights * np.square(case - risk)))
    return (score, info) if return_info else score


def integrated_brier(
    y_test, surv, times, *, y_censor=None, censoring_estimator=None, g_min=0.05, return_info=False
):
    """Integrated IPCW Brier score of survival curves over ``times``.

    ``surv[:, j]`` is ``P(T > times[j])``. The Brier score at each time (as in
    ``brier_landmark``) is integrated by the trapezoid rule and divided by
    ``times[-1] - times[0]`` (the scikit-survival convention). ``info`` holds the
    per-time scores and the largest clipped count.
    """
    _, stop, event = _check_right_censored(y_test, "y_test")
    times = np.asarray(times, dtype=float).ravel()
    if times.size < 2 or np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("times must be at least two strictly increasing positive values")
    surv = _check_pred(surv, stop.size, "surv")
    if surv.shape != (stop.size, times.size):
        raise ValueError(f"surv must have shape (n, {times.size})")
    complete = [np.all(np.logical_or(*_outcome_classes(stop, event, t))) for t in times]
    cens = None if all(complete) else _censoring(y_censor, censoring_estimator)
    scores, clipped = np.empty(times.size), 0
    for j, t in enumerate(times):
        case, _, weights, info = _ipcw(stop, event, t, None, cens, g_min)
        scores[j] = np.mean(weights * np.square(case - (1.0 - surv[:, j])))
        clipped = max(clipped, info["n_clipped"])
    exact = all(complete)
    score = float(np.trapezoid(scores, times) / (times[-1] - times[0]))
    info = {"exact": exact, "n_clipped": clipped, "times": times, "brier": scores}
    return (score, info) if return_info else score


class _Fenwick:
    """Counts over ranks ``0..n-1``: point updates, prefix sums."""

    def __init__(self, n):
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def add(self, i, v):
        i += 1
        tree, n = self.tree, self.tree.size
        while i < n:
            tree[i] += v
            i += i & -i

    def prefix(self, i):
        """Sum over ranks ``< i``."""
        s, tree = 0, self.tree
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s


def _concordance(start, stop, event, risk, weights, groups=None, event_mask=None):
    """Weighted counting-process concordance: ``(numerator, denominator, n_pairs)``.

    For each event row ``i`` (where ``event_mask``), comparators are rows at
    risk at ``T_i = stop_i`` (``start < T_i <= stop``) except rows with an event
    at ``T_i`` and rows of the same group. The pair is concordant when
    ``risk_i`` is larger; risk ties count ½. Each pair has weight ``weights[i]``.
    """
    n = stop.size
    ev = event if event_mask is None else event & event_mask
    ranks = np.unique(risk, return_inverse=True)[1].ravel()
    bit = _Fenwick(ranks.max() + 1 if n else 1)
    by_start, by_stop = np.argsort(start, kind="stable"), np.argsort(stop, kind="stable")
    ev_rows = np.flatnonzero(ev)
    ev_rows = ev_rows[np.argsort(stop[ev_rows], kind="stable")]
    tied_event = event & np.isin(stop, stop[ev_rows])
    if groups is not None:
        g_order = np.argsort(groups, kind="stable")
        g_sorted = groups[g_order]
    num = den = 0.0
    n_pairs = 0
    p = q = size = 0
    k = 0
    while k < ev_rows.size:
        t = stop[ev_rows[k]]
        end = k
        while end < ev_rows.size and stop[ev_rows[end]] == t:
            end += 1
        while p < n and start[by_start[p]] < t:
            bit.add(ranks[by_start[p]], 1)
            p, size = p + 1, size + 1
        while q < n and stop[by_stop[q]] < t:
            bit.add(ranks[by_stop[q]], -1)
            q, size = q + 1, size - 1
        # All events at t (not only those in event_mask) are excluded as comparators.
        tied = np.flatnonzero(tied_event & (stop == t) & (start < t))
        tied_ranks = np.sort(ranks[tied])
        for i in ev_rows[k:end]:
            r = ranks[i]
            less = bit.prefix(r) - np.searchsorted(tied_ranks, r, side="left")
            equal = bit.prefix(r + 1) - bit.prefix(r) - (
                np.searchsorted(tied_ranks, r, side="right") - np.searchsorted(tied_ranks, r, side="left")
            )
            total = size - tied.size
            if groups is not None:
                lo, hi = np.searchsorted(g_sorted, groups[i], side="left"), np.searchsorted(
                    g_sorted, groups[i], side="right"
                )
                same = g_order[lo:hi]
                same = same[(start[same] < t) & (t <= stop[same]) & ~(tied_event[same] & (stop[same] == t))]
                less -= int((ranks[same] < r).sum())
                equal -= int((ranks[same] == r).sum())
                total -= same.size
            num += weights[i] * (less + 0.5 * equal)
            den += weights[i] * total
            n_pairs += total
        k = end
    return num, den, n_pairs


def concordance_index_cp(y, risk, ids=None):
    """Concordance for counting-process data with time-varying risk scores.

    ``risk[r]`` is row ``r``'s risk, in force on its ``(start, stop]``. At each
    event time ``T_i`` the event row is compared with every row at risk at
    ``T_i`` (``start < T_i <= stop``), excluding rows with an event at ``T_i``
    and, when ``ids`` is given, rows of the same id. Concordant when the event
    row's risk is larger; ties count ½. With ``start == 0`` and one row per
    subject this is Harrell's C; with counting-process rows it matches R
    ``survival::concordance(Surv(start, stop, event) ~ risk, reverse=TRUE)``.
    """
    start, stop, event = check_survival_y(y, require_events=False)
    risk = _check_pred(risk, stop.size, "risk").ravel()
    groups = None if ids is None else np.unique(np.asarray(ids), return_inverse=True)[1].ravel()
    num, den, _ = _concordance(start, stop, event, risk, np.ones(stop.size), groups)
    if den == 0:
        raise UndefinedMetricError("no comparable pairs")
    return num / den


def cindex_dynamic(
    y_test, risk, w, *, kind="cumulative", y_censor=None, censoring_estimator=None, g_min=0.05,
    return_info=False,
):
    """Time-dependent discrimination of ``risk`` at horizon ``w``.

    - ``kind="cumulative"``: cumulative/dynamic AUC (Uno et al. 2007): cases
      (events by ``w``, weighted ``1/G(stop-)``) against controls (event-free
      through ``w``); ``P(risk_case > risk_control)``, ties ½.
    - ``kind="incident"``: the incident/dynamic concordance integrated over
      ``(0, w]`` (Heagerty & Zheng 2005), estimated by Uno's truncated C (Uno
      et al. 2011): each event by ``w`` against the subjects still at risk at
      its time, weighted ``1/G(stop-)^2``. A pairwise estimand, not an AUC at
      a single time.

    Weights are 1 under complete follow-up to ``w`` (exact path).
    """
    _, stop, event = _check_right_censored(y_test, "y_test")
    risk = _check_pred(risk, stop.size, "risk").ravel()
    if kind not in ("cumulative", "incident"):
        raise ValueError(f"kind must be 'cumulative' or 'incident', got {kind!r}")
    case, control, weights, info = _ipcw(stop, event, w, y_censor, censoring_estimator, g_min)
    if kind == "cumulative":
        if not case.any() or not control.any():
            raise UndefinedMetricError("need at least one case and one control at w")
        ctrl = np.sort(risk[control])
        rc = risk[case]
        less = np.searchsorted(ctrl, rc, side="left")
        equal = np.searchsorted(ctrl, rc, side="right") - less
        wc = weights[case]
        score = float(np.sum(wc * (less + 0.5 * equal)) / (wc.sum() * ctrl.size))
    else:
        num, den, _ = _concordance(np.zeros(stop.size), stop, event, risk, np.square(weights), event_mask=case)
        if den == 0:
            raise UndefinedMetricError("no comparable pairs before w")
        score = num / den
    return (score, info) if return_info else score


def _km_at(stop, event, w):
    """Kaplan–Meier ``P(T > w)``, right-continuous (events at ``w`` included)."""
    times = np.unique(stop[event & (stop <= w)])
    s = 1.0
    for t in times:
        s *= 1.0 - np.sum(event & (stop == t)) / np.sum(stop >= t)
    return s


def calibration_table(y_test, risk, w, n_bins=10):
    """Observed vs predicted risk at ``w`` in quantile bins of ``risk``.

    Observed risk is ``1 - KM(w)`` within each bin, so censoring before ``w``
    is handled by the Kaplan–Meier estimator. This is descriptive: it assumes
    censoring independent of the outcome *within each bin*, which is stronger
    than independence given the full history.
    Returns a polars DataFrame: ``bin, n, n_events, mean_risk, observed_risk``.
    """
    _, stop, event = _check_right_censored(y_test, "y_test")
    risk = _check_pred(risk, stop.size, "risk").ravel()
    edges = np.unique(np.quantile(risk, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, risk, side="right") - 1, 0, max(edges.size - 2, 0))
    rows = []
    for b in np.unique(bins):
        m = bins == b
        rows.append(
            {
                "bin": int(b),
                "n": int(m.sum()),
                "n_events": int((event[m] & (stop[m] <= w)).sum()),
                "mean_risk": float(risk[m].mean()),
                "observed_risk": 1.0 - _km_at(stop[m], event[m], w),
            }
        )
    return pl.DataFrame(rows)
