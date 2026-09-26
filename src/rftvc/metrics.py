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

import numbers
import warnings
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from ._validation import check_survival_y, competing_risks_labels

__all__ = [
    "KaplanMeierCensoring",
    "PEScore",
    "UndefinedMetricError",
    "brier_landmark",
    "calibration_table",
    "cindex_dynamic",
    "concordance_index_cp",
    "concordance_index_cr",
    "event_windows",
    "integrated_brier",
    "piecewise_exponential_score",
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
        """Fit on right-censored outcomes; cause labels are accepted (any label != 0 is an event)."""
        stop, labels = _landmark_outcomes(y, "y", any_labels=True)
        event = labels != 0
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


def _check_cause(cause):
    if not isinstance(cause, numbers.Integral) or isinstance(cause, (bool, np.bool_)) or cause <= 0:
        raise ValueError(f"cause must be a positive integer label, got {cause!r}")


def _landmark_outcomes(y, name, cause=None, any_labels=False):
    """``(stop, event)`` of right-censored outcomes: bool events for ``cause=None``,
    int64 cause labels with a ``cause`` (or with ``any_labels``, no cause check)."""
    if cause is None and not any_labels:
        _, stop, event = _check_right_censored(y, name)
        return stop, event
    if not any_labels:
        _check_cause(cause)
    start, stop, labels = competing_risks_labels(y)
    if np.any(start != 0):
        raise ValueError(f"{name} must be right-censored outcomes on the reset clock (start == 0)")
    return stop, labels


def _outcome_classes(stop, event, w):
    case = event & (stop <= w)
    control = (stop >= w) & ~case
    return case, control


def _classes(stop, event, w, cause=None):
    """``(case, competing, control)`` at horizon ``w``.

    Without ``cause``: ``event`` is boolean and ``competing`` is empty. With
    ``cause=k``: ``event`` holds labels; a case is a cause-``k`` event by ``w``,
    a competing event by ``w`` is an observed non-case, and a control is
    event-free through ``w`` (``stop >= w``, no event by ``w``).
    """
    if cause is None:
        case, control = _outcome_classes(stop, event, w)
        return case, np.zeros(stop.size, dtype=bool), control
    by_w = (event != 0) & (stop <= w)
    case = by_w & (event == cause)
    competing = by_w & ~case
    control = (stop >= w) & ~by_w
    return case, competing, control


def _censoring(y_censor, censoring_estimator):
    if (y_censor is None) == (censoring_estimator is None):
        raise ValueError(
            "pass exactly one of y_censor or a fitted censoring_estimator"
        )
    if censoring_estimator is not None:
        return censoring_estimator
    return KaplanMeierCensoring().fit(y_censor)


def _ipcw(stop, event, w, y_censor, censoring_estimator, g_min, cause=None):
    """Per-subject weights at horizon ``w`` and the diagnostic info: ``(case, control, weights, info)``.

    Cases and competing events (``cause`` given) are weighted ``1/G(stop-)``,
    controls ``1/G(w-)``, subjects censored before ``w`` 0.
    """
    case, _, control, weights, info = _ipcw_classes(stop, event, w, y_censor, censoring_estimator, g_min, cause)
    return case, control, weights, info


def _ipcw_classes(stop, event, w, y_censor, censoring_estimator, g_min, cause=None):
    """As ``_ipcw``, also returning the competing mask: ``(case, competing, control, weights, info)``."""
    if not 0 < g_min <= 1:
        raise ValueError("g_min must lie in (0, 1]")
    case, competing, control = _classes(stop, event, w, cause)
    observed_event = case | competing
    exact = bool(np.all(observed_event | control))
    info = {"exact": exact, "n": int(stop.size), "n_cases": int(case.sum()), "n_controls": int(control.sum())}
    if cause is not None:
        info["n_competing"] = int(competing.sum())
    if exact:
        if y_censor is not None and censoring_estimator is not None:
            raise ValueError("pass at most one of y_censor or censoring_estimator")
        info.update(n_censored=0, n_clipped=0)
        return case, competing, control, np.ones(stop.size), info
    cens = _censoring(y_censor, censoring_estimator)
    g = np.ones(stop.size)
    g[observed_event] = cens.predict(stop[observed_event], left=True)
    g[control] = cens.predict(np.array([w]), left=True)[0]
    weighted = observed_event | control
    clipped = weighted & (g < g_min)
    g = np.maximum(g, g_min)
    weights = np.where(weighted, 1.0 / g, 0.0)
    info.update(n_censored=int((~weighted).sum()), n_clipped=int(clipped.sum()))
    return case, competing, control, weights, info


def _check_pred(values, n, name):
    values = np.asarray(values, dtype=float)
    if values.shape[0] != n:
        raise ValueError(f"{name} has {values.shape[0]} rows but y has {n}")
    if np.isnan(values).any():
        raise ValueError(f"{name} must not contain NaN")
    return values


def brier_landmark(
    y_test, risk, w, *, cause=None, y_censor=None, censoring_estimator=None, g_min=0.05, return_info=False
):
    """IPCW Brier score of ``risk = P(T <= w)`` at horizon ``w``.

    ``mean_i weight_i * (1{case_i} - risk_i)^2`` with weights ``1/G(stop_i-)``
    for cases, ``1/G(w-)`` for controls and 0 for subjects censored before
    ``w`` (Graf et al. 1999). ``G`` is clipped below at ``g_min``.

    With ``cause=k`` (competing risks), ``y_test`` holds cause labels and
    ``risk = F_k(w) = P(T <= w, cause k)``. A case is a cause-``k`` event by
    ``w``; a competing event by ``w`` is an observed outcome 0, weighted
    ``1/G(stop-)`` (it can never become a case); event-free subjects are controls.

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
    stop, event = _landmark_outcomes(y_test, "y_test", cause)
    risk = _check_pred(risk, stop.size, "risk").ravel()
    if not w > 0:
        raise ValueError("w must be positive")
    case, _, weights, info = _ipcw(stop, event, w, y_censor, censoring_estimator, g_min, cause)
    score = float(np.mean(weights * np.square(case - risk)))
    return (score, info) if return_info else score


def integrated_brier(
    y_test, surv, times, *, cause=None, y_censor=None, censoring_estimator=None, g_min=0.05, return_info=False
):
    """Integrated IPCW Brier score of survival curves over ``times``.

    ``surv[:, j]`` is ``P(T > times[j])``. The Brier score at each time (as in
    ``brier_landmark``) is integrated by the trapezoid rule and divided by
    ``times[-1] - times[0]`` (the scikit-survival convention). ``info`` holds the
    per-time scores and the largest clipped count.

    With ``cause=k``, the second argument holds the **cumulative incidence**
    ``F_k(times[j])`` (a risk, not a survival probability) and ``y_test`` holds
    cause labels; each time is scored as in ``brier_landmark(..., cause=k)``.
    """
    stop, event = _landmark_outcomes(y_test, "y_test", cause)
    times = np.asarray(times, dtype=float).ravel()
    if times.size < 2 or np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("times must be at least two strictly increasing positive values")
    surv = _check_pred(surv, stop.size, "surv")
    if surv.shape != (stop.size, times.size):
        raise ValueError(f"surv must have shape (n, {times.size})")
    complete = [np.all(np.logical_or.reduce(_classes(stop, event, t, cause))) for t in times]
    cens = None if all(complete) else _censoring(y_censor, censoring_estimator)
    scores, clipped = np.empty(times.size), 0
    for j, t in enumerate(times):
        case, _, weights, info = _ipcw(stop, event, t, None, cens, g_min, cause)
        risk = surv[:, j] if cause is not None else 1.0 - surv[:, j]
        scores[j] = np.mean(weights * np.square(case - risk))
        clipped = max(clipped, info["n_clipped"])
    exact = all(complete)
    score = float(np.trapezoid(scores, times) / (times[-1] - times[0]))
    info = {"exact": exact, "n_clipped": clipped, "times": times, "brier": scores}
    return (score, info) if return_info else score


class _Fenwick:
    """Counts over ranks ``0..n-1``: point updates, prefix sums."""

    def __init__(self, n, dtype=np.int64):
        self.tree = np.zeros(n + 1, dtype=dtype)

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


def _groups(ids, n):
    """Group codes of ``ids`` (one per row), or ``None``."""
    if ids is None:
        return None
    ids = np.asarray(ids)
    if ids.shape != (n,):
        raise ValueError(f"ids must be 1-d with {n} entries, got shape {ids.shape}")
    return np.unique(ids, return_inverse=True)[1].ravel()


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
    groups = _groups(ids, stop.size)
    num, den, _ = _concordance(start, stop, event, risk, np.ones(stop.size), groups)
    if den == 0:
        raise UndefinedMetricError("no comparable pairs")
    return num / den


def _competing_pairs(stop, labels, cause, risk, groups, case_mask=None, weights=None):
    """Type-B pairs of Wolbers' C: ``(numerator, denominator)``.

    Each case (a ``cause`` event at ``T_i``, restricted to ``case_mask``)
    against every row with a competing event (any other cause) at
    ``T_j <= T_i`` of another group, using that row's risk; concordant when the
    case's risk is larger, ties ½. With ``weights``, a pair counts
    ``weights[i] * weights[j]`` (IPCW); otherwise 1.
    """
    is_case = labels == cause if case_mask is None else case_mask
    cases = np.flatnonzero(is_case)
    cases = cases[np.argsort(stop[cases], kind="stable")]
    comp = np.flatnonzero((labels != 0) & (labels != cause))
    comp = comp[np.argsort(stop[comp], kind="stable")]
    if cases.size == 0 or comp.size == 0:
        return 0.0, 0.0
    wt = np.ones(stop.size) if weights is None else weights
    ranks = np.unique(risk, return_inverse=True)[1].ravel()
    bit = _Fenwick(int(ranks.max()) + 1, dtype=np.float64)
    num = den = 0.0
    q = 0
    added = 0.0
    for i in cases:
        t = stop[i]
        while q < comp.size and stop[comp[q]] <= t:
            bit.add(ranks[comp[q]], wt[comp[q]])
            added += wt[comp[q]]
            q += 1
        r = ranks[i]
        less = bit.prefix(r)
        equal = bit.prefix(r + 1) - less
        total = added
        if groups is not None:
            same = comp[:q][groups[comp[:q]] == groups[i]]
            less -= wt[same][ranks[same] < r].sum()
            equal -= wt[same][ranks[same] == r].sum()
            total -= wt[same].sum()
        num += wt[i] * (less + 0.5 * equal)
        den += wt[i] * total
    return num, den


def concordance_index_cr(y, risk, cause, ids=None):
    """Cause-specific concordance (Wolbers et al. 2014) on counting-process rows.

    ``y`` holds cause labels (0 = censored; see ``make_competing_risks_y``) and
    ``risk[r]`` is row ``r``'s risk of ``cause``, in force on its
    ``(start, stop]``. Each case, a row with a ``cause`` event at ``T_i``, is
    compared with

    - (A) every row at risk at ``T_i`` (``start < T_i <= stop``), excluding
      rows with an event of any cause at ``T_i``, as in ``concordance_index_cp``;
    - (B) every row with a **competing** event at ``T_j <= T_i``, with that
      row's risk: such a subject can no longer have a ``cause`` event. Ties
      ``T_j = T_i`` are included (they are not type-A comparators).

    Rows of the case's id are excluded when ``ids`` is given. Concordant when
    the case's risk is larger; ties count ½. With a single cause there are no
    type-B pairs and this equals ``concordance_index_cp``.
    """
    _check_cause(cause)
    start, stop, labels = competing_risks_labels(y)
    risk = _check_pred(risk, stop.size, "risk").ravel()
    groups = _groups(ids, stop.size)
    num_a, den_a, _ = _concordance(
        start, stop, labels != 0, risk, np.ones(stop.size), groups, event_mask=labels == cause
    )
    num_b, den_b = _competing_pairs(stop, labels, cause, risk, groups)
    if den_a + den_b == 0:
        raise UndefinedMetricError("no comparable pairs")
    return (num_a + num_b) / (den_a + den_b)


def cindex_dynamic(
    y_test, risk, w, *, kind="cumulative", cause=None, y_censor=None, censoring_estimator=None, g_min=0.05,
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

    With ``cause=k`` (competing risks; ``y_test`` holds cause labels, ``risk``
    is ``F_k(w)``):

    - ``kind="cumulative"``: the cumulative/dynamic AUC of Blanche et al.
      (2013), definition 2. Cases (cause-``k`` events by ``w``, weight
      ``1/G(T_i-)``) against every observed non-case: subjects event-free
      through ``w`` (``stop >= w``, weight ``1/G(w-)``) and subjects with a
      competing event by ``w`` (weight ``1/G(T_j-)``). Each pair counts
      ``w_i w_j``. A subject censored exactly at ``w`` is an event-free control,
      as in ``brier_landmark`` (``comprisk`` / ``timeROC`` use ``stop > w`` and
      ``G(w)``; the two agree when no time equals ``w``).
    - ``kind="incident"``: the IPCW Wolbers C truncated at ``w``. Each
      cause-``k`` event by ``w`` is compared with (A) the subjects still at risk
      at its time, weight ``1/G(T_i-)^2`` as above, and (B) the subjects with a
      competing event at ``T_j <= T_i``, weight ``1/(G(T_i-) G(T_j-))``.

    With one cause both equal their survival versions above.
    """
    stop, event = _landmark_outcomes(y_test, "y_test", cause)
    risk = _check_pred(risk, stop.size, "risk").ravel()
    if kind not in ("cumulative", "incident"):
        raise ValueError(f"kind must be 'cumulative' or 'incident', got {kind!r}")
    case, competing, control, weights, info = _ipcw_classes(
        stop, event, w, y_censor, censoring_estimator, g_min, cause
    )
    if kind == "cumulative" and competing.any():
        score = _cumulative_auc_weighted(risk, case, control | competing, weights)
    elif kind == "cumulative":
        if not case.any() or not control.any():
            raise UndefinedMetricError("need at least one case and one control at w")
        ctrl = np.sort(risk[control])
        rc = risk[case]
        less = np.searchsorted(ctrl, rc, side="left")
        equal = np.searchsorted(ctrl, rc, side="right") - less
        wc = weights[case]
        score = float(np.sum(wc * (less + 0.5 * equal)) / (wc.sum() * ctrl.size))
    else:
        num, den, _ = _concordance(np.zeros(stop.size), stop, event != 0, risk, np.square(weights), event_mask=case)
        if cause is not None:
            num_b, den_b = _competing_pairs(stop, event, cause, risk, None, case_mask=case, weights=weights)
            num, den = num + num_b, den + den_b
        if den == 0:
            raise UndefinedMetricError("no comparable pairs before w")
        score = num / den
    return (score, info) if return_info else score


def _cumulative_auc_weighted(risk, case, ctrl_mask, weights):
    """``sum_ij w_i w_j (1{r_i > r_j} + 1/2 1{r_i = r_j}) / (sum_i w_i sum_j w_j)`` over cases i, controls j."""
    if not case.any() or not ctrl_mask.any():
        raise UndefinedMetricError("need at least one case and one control at w")
    order = np.argsort(risk[ctrl_mask], kind="stable")
    r_ctrl, w_ctrl = risk[ctrl_mask][order], weights[ctrl_mask][order]
    cum = np.r_[0.0, np.cumsum(w_ctrl)]
    rc, wc = risk[case], weights[case]
    lo = np.searchsorted(r_ctrl, rc, side="left")
    hi = np.searchsorted(r_ctrl, rc, side="right")
    num = np.sum(wc * (cum[lo] + 0.5 * (cum[hi] - cum[lo])))
    return float(num / (wc.sum() * cum[-1]))


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


# --- piecewise-exponential score (docs/plans/tvc-deviance.md) -----------------


class PEScore(NamedTuple):
    """Result of ``piecewise_exponential_score`` (higher is better).

    With ``reduce="per_event"`` every score field is divided by ``n_events``.
    ``by_cause`` / ``by_cause_window`` are ``None`` for survival data and when
    one ``cause`` is selected; ``by_id`` / ``id_labels`` are ``None`` without ``ids``.
    """

    total: float
    by_window: np.ndarray
    by_cause: object
    by_cause_window: object
    by_id: object
    id_labels: object
    zero_rate_share: float
    null_total: float
    n_events: int
    n_truncated_events: int


def event_windows(estimator, n_windows=8):
    """Scoring windows from a fitted forest's training events.

    Interior edges are inverse-CDF quantiles of the training event times
    counted with multiplicity: the time of the ``ceil(k·D/n_windows)``-th
    event, ``k = 1..n_windows-1``, with ``D`` training events. The edges are
    ``[0, interior..., tau]`` with ``tau`` the last event time, deduplicated,
    so heavy ties give fewer windows. Windows are right-closed ``(w_{m-1}, w_m]``.
    """
    if not isinstance(n_windows, numbers.Integral) or isinstance(n_windows, (bool, np.bool_)) or n_windows < 1:
        raise ValueError(f"n_windows must be a positive integer, got {n_windows!r}")
    counts = getattr(estimator, "_event_counts_", None)
    if counts is None:
        raise AttributeError("refit: this forest predates baseline_cumhaz_ (or is not fitted)")
    times = np.asarray(estimator.event_times_, dtype=float)
    total = counts.sum()
    if total == 0 or times[-1] <= 0:
        raise ValueError("the forest has no training events at positive times")
    ranks = np.ceil(np.arange(1, n_windows) * total / n_windows)
    interior = times[np.searchsorted(np.cumsum(counts), ranks, side="left")]
    return np.unique(np.r_[0.0, interior[interior > 0], times[-1]])


def _baseline_at(estimator, windows):
    """The estimator's ``baseline_cumhaz_`` (a right-continuous step function on
    ``event_times_``) at ``windows``: ``(M+1,)`` or ``(J, M+1)``."""
    base = getattr(estimator, "baseline_cumhaz_", None)
    if base is None:
        raise AttributeError("refit: this forest predates baseline_cumhaz_ (or is not fitted)")
    idx = np.searchsorted(estimator.event_times_, np.asarray(windows, dtype=float), side="right") - 1
    padded = np.concatenate([np.zeros(base.shape[:-1] + (1,)), base], axis=-1)
    return padded[..., idx + 1]


def _check_windows(windows):
    w = np.asarray(windows, dtype=float)
    if w.ndim != 1 or w.size < 2 or not np.isfinite(w).all() or w[0] != 0 or np.any(np.diff(w) <= 0):
        raise ValueError("windows must be finite, strictly increasing edges starting at 0, with at least one window")
    return w


def _pe_codes(y, cumhaz, null_cumhaz, causes, cause, m1):
    """``(start, stop, codes, H (n, J, M+1), H0 (J, M+1), competing)``; ``codes`` 0 = censored, j+1 = axis j."""
    if cumhaz.ndim == 2:
        if causes is not None or cause is not None:
            raise ValueError("causes and cause apply to competing-risks cumhaz of shape (n, J, M+1)")
        start, stop, event = check_survival_y(y, require_events=False)
        codes = event.astype(np.int64)
        H, H0 = cumhaz[:, None, :], np.asarray(null_cumhaz, dtype=float)[None, :]
        competing = False
    elif cumhaz.ndim == 3:
        if causes is None:
            raise ValueError("competing-risks cumhaz needs causes (the estimator's causes_) to map labels to axes")
        causes = np.asarray(causes).ravel()
        start, stop, labels = competing_risks_labels(y)
        codes = np.zeros(labels.size, dtype=np.int64)
        known = labels == 0
        for j, c in enumerate(causes.tolist()):
            hit = labels == c
            codes[hit], known = j + 1, known | hit
        if not known.all():
            raise ValueError(f"y has cause labels {sorted(set(labels[~known].tolist()))} not in causes")
        if cumhaz.shape[1] != causes.size:
            raise ValueError(f"cumhaz has {cumhaz.shape[1]} causes but causes has {causes.size}")
        H, H0 = cumhaz, np.asarray(null_cumhaz, dtype=float)
        competing = True
        if cause is not None:
            pos = np.flatnonzero(causes == cause) if isinstance(cause, numbers.Integral) else []
            if len(pos) != 1:
                raise ValueError(f"cause={cause!r} is not in causes {causes.tolist()}")
            k = int(pos[0])
            H, H0 = H[:, k : k + 1], H0[k : k + 1]
            codes = np.where(codes == k + 1, 1, 0)
            competing = False
    else:
        raise ValueError("cumhaz must have shape (n, M+1) or (n, J, M+1)")
    if H.shape[0] != stop.size or H.shape[2] != m1:
        raise ValueError(f"cumhaz must have {stop.size} rows and {m1} edge columns, got shape {cumhaz.shape}")
    if H0.shape != (H.shape[1], m1):
        raise ValueError(f"null_cumhaz must have shape {(H.shape[1], m1) if competing or H.shape[1] > 1 else (m1,)}")
    if not np.isfinite(H).all():
        raise ValueError("cumhaz must be finite (drop OOB rows that have no out-of-bag tree)")
    if not np.isfinite(H0).all():
        raise ValueError("null_cumhaz must be finite")
    return start, stop, codes, H, H0, competing


def piecewise_exponential_score(
    y, cumhaz, windows, *, null_cumhaz, alpha=0.01, causes=None, cause=None, ids=None, reduce="per_event"
):
    """Piecewise-exponential log score of cumulative-hazard predictions on counting-process rows.

    For windows ``W_m = (w_{m-1}, w_m]`` and rows ``(start, stop]`` with
    covariates fixed on the row, the score is::

        S = sum_{r, m} [ N_rm log(rate_rm) - rate_rm * e_rm ]
        rate_rm = ((1 - alpha) (cumhaz[r, m] - cumhaz[r, m-1])
                   + alpha (null[m] - null[m-1])) / (w_m - w_{m-1})

    with ``e_rm`` the row's exposure in ``W_m`` and ``N_rm`` its event there.
    It is the log-likelihood of the piecewise-constant hazard ``rate``: proper
    for piecewise-constant predictions, IPCW-free, and additive over windows,
    causes and ids (``docs/plans/tvc-deviance.md``). Higher is better; the
    Poisson deviance is ``-2 S`` plus a data-only term.

    Parameters
    ----------
    y : survival or competing-risks target with ``start``, ``stop``, ``event``.
    cumhaz : ndarray of shape (n, M+1) or (n, J, M+1)
        Each row's predicted cumulative hazard at ``windows`` (fixed profile,
        e.g. ``predict_cumulative_hazard(X, times=windows)``).
    windows : ndarray of shape (M+1,)
        Strictly increasing edges starting at 0 (see ``event_windows``). Rows are
        administratively truncated at ``tau = windows[-1]``; exposure outside
        ``(0, tau]`` is ignored, and events outside it are counted in
        ``n_truncated_events`` and not scored.
    null_cumhaz : ndarray of shape (M+1,) or (J, M+1)
        The training-set null (e.g. the estimator's ``baseline_cumhaz_``) at ``windows``.
        It must come from training data, not from ``y``.
    alpha : float in [0, 1), default=0.01
        Weight of the null in the scored rate. It keeps event cells with a zero
        predicted rate finite; with ``alpha=0`` such a cell scores ``-inf``.
    causes : array-like, required for competing risks
        The labels of the cause axis of ``cumhaz`` (the estimator's ``causes_``).
    cause : int, default=None
        Score one cause only; ``None`` sums all causes.
    ids : array-like of shape (n,), default=None
        Row subjects, for ``by_id``.
    reduce : {"per_event", "sum"}, default="per_event"
        ``"per_event"`` divides the scores by the number of scored events.

    Returns
    -------
    PEScore
        ``zero_rate_share`` is the fraction of scored events whose cell has a
        zero predicted (unmixed) rate; above 1% the windows are too fine for
        the model, and a ``UserWarning`` is raised. ``null_total`` is the null's
        own score on the same cells.
    """
    if reduce not in ("per_event", "sum"):
        raise ValueError(f"reduce must be 'per_event' or 'sum', got {reduce!r}")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real) or not 0 <= alpha < 1:
        raise ValueError(f"alpha must be in [0, 1), got {alpha!r}")
    w = _check_windows(windows)
    M = w.size - 1
    start, stop, codes, H, H0, competing = _pe_codes(
        y, np.asarray(cumhaz, dtype=float), null_cumhaz, causes, cause, M + 1
    )
    if ids is not None:
        ids = np.asarray(ids)
        if ids.shape != (stop.size,):
            raise ValueError(f"ids must have {stop.size} entries")
        id_labels, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")  # labels in order of first appearance
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        inv, id_labels = rank[inv.ravel()], id_labels[order]
        by_id = np.zeros(id_labels.size)
    tau = w[-1]
    event = codes > 0
    scored = event & (stop > 0) & (stop <= tau)
    n_events = int(scored.sum())
    n_truncated = int((event & ~scored).sum())
    if n_events == 0:
        raise UndefinedMetricError("no events in (0, windows[-1]] to score")
    J = H.shape[1]
    cw = np.zeros((J, M))
    null_sum = 0.0
    zero_cells = 0
    with np.errstate(divide="ignore"):
        for m in range(M):
            lo, hi = w[m], w[m + 1]
            width = hi - lo
            e = np.clip(np.minimum(stop, hi) - np.maximum(start, lo), 0.0, None)
            in_window = scored & (stop > lo)
            in_window &= stop <= hi
            for j in range(J):
                d = H[:, j, m + 1] - H[:, j, m]
                d0 = H0[j, m + 1] - H0[j, m]
                rate = ((1.0 - alpha) * d + alpha * d0) / width
                hit = in_window & (codes == j + 1)
                contrib = -rate * e
                contrib[hit] += np.log(rate[hit])
                zero_cells += int((d[hit] <= 0).sum())
                cw[j, m] = contrib.sum()
                if ids is not None:
                    by_id += np.bincount(inv, weights=contrib, minlength=by_id.size)
                rate0 = d0 / width
                n_hit = int(hit.sum())
                null_sum += (n_hit * np.log(rate0) if n_hit else 0.0) - rate0 * e.sum()
    div = n_events if reduce == "per_event" else 1.0
    share = zero_cells / n_events
    if share > 0.01:
        warnings.warn(
            f"{share:.1%} of events fall in windows where the predicted rate is 0; "
            "the windows are too fine for this model (use fewer windows)",
            UserWarning,
            stacklevel=2,
        )
    return PEScore(
        total=float(cw.sum() / div),
        by_window=cw.sum(axis=0) / div,
        by_cause=cw.sum(axis=1) / div if competing else None,
        by_cause_window=cw / div if competing else None,
        by_id=None if ids is None else by_id / div,
        id_labels=None if ids is None else id_labels,
        zero_rate_share=float(share),
        null_total=float(null_sum / div),
        n_events=n_events,
        n_truncated_events=n_truncated,
    )
