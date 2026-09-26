"""Model inspection for the survival forests: permutation importance."""

import numbers
import warnings

import numpy as np
from joblib import effective_n_jobs
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._competing import CompetingRisksForestTV
from ._estimator import _BaseForestTV
from ._inspection import _score, _strata, _units
from ._validation import check_survival_y, competing_risks_labels, make_competing_risks_y, make_survival_y
from .metrics import _baseline_at, _check_windows, event_windows

__all__ = ["permutation_importance"]


def _family(estimator):
    from .landmark import _LandmarkBase

    if isinstance(estimator, _LandmarkBase):
        raise NotImplementedError("permutation_importance for landmark models is not implemented yet")
    if not isinstance(estimator, _BaseForestTV):
        raise TypeError(
            "permutation_importance supports SurvivalForestTV and CompetingRisksForestTV, "
            f"got {type(estimator).__name__}"
        )
    check_is_fitted(estimator, "forest_")
    return isinstance(estimator, CompetingRisksForestTV)


def _entropy(random_state):
    """One non-negative integer seeding every unit's stream."""
    if random_state is None:
        return int(np.random.SeedSequence().entropy)
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(np.iinfo(np.int64).max))
    if isinstance(random_state, numbers.Integral) and not isinstance(random_state, (bool, np.bool_)):
        if random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {random_state}")
        return int(random_state)
    return int(check_random_state(random_state).randint(np.iinfo(np.int64).max, dtype=np.int64))


def _windows(estimator, windows):
    if isinstance(windows, numbers.Integral) and not isinstance(windows, (bool, np.bool_)):
        return event_windows(estimator, windows)
    w = _check_windows(windows)
    tau = estimator.event_times_[-1]
    if w[-1] > tau:
        raise ValueError(f"windows[-1] = {w[-1]} is beyond the last training event time {tau}")
    return w


def _target(y, competing):
    """``(y as a structured array, start)``, checked for the estimator family."""
    if competing:
        start, stop, labels = competing_risks_labels(y)
        return make_competing_risks_y(stop, labels, start=start), start
    start, stop, event = check_survival_y(y, require_events=False)
    return make_survival_y(stop, event, start=start), start


def _subset_csr(offsets, units, rows):
    """The CSR sets ``(offsets, units)`` of ``rows``."""
    offsets = np.asarray(offsets, dtype=np.int64)
    lo, hi = offsets[rows], offsets[rows + 1]
    lengths = hi - lo
    pos = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    sub = np.asarray(units)[np.repeat(lo, lengths) + pos]
    return np.r_[0, np.cumsum(lengths)].astype(np.uint64), np.ascontiguousarray(sub, dtype=np.uint32)


def _heldout(estimator, X, y, ids, competing, w, n_jobs):
    """Evaluation rows and predict callable for held-out data."""
    fit_ids = getattr(estimator, "ids_column_", None)
    if ids is None and fit_ids is not None and hasattr(X, "columns") and fit_ids in list(X.columns):
        ids = fit_ids
    Xe, _, ids = estimator._check_predict(X, None, ids)
    y, start = _target(y, competing)
    if y.shape[0] != Xe.shape[0]:
        raise ValueError(f"X has {Xe.shape[0]} rows but y has {y.shape[0]}")
    row_ids = np.arange(Xe.shape[0]) if ids is None else np.asarray(ids)
    if row_ids.shape != (Xe.shape[0],):
        raise ValueError(f"ids must have {Xe.shape[0]} entries")
    forest = estimator.forest_
    if competing:
        def predict(Xr, rows):
            return forest.predict_cause_cumhaz(np.ascontiguousarray(Xr), w, n_jobs)
    else:
        def predict(Xr, rows):
            return forest.predict_cumhaz(np.ascontiguousarray(Xr), w, estimator.aggregate, n_jobs)
    return Xe, y, start, row_ids, predict, None, 0


def _oob(estimator, X, y, ids, measured_at, block_time, competing, w, n_jobs):
    """Evaluation rows (the fit-time design's rows with an OOB tree) and the OOB predict callable."""
    d = estimator._rebuild_design(X, y, ids, measured_at=measured_at, block_time=block_time)
    offsets, units = d.oob_set
    forest = estimator.forest_
    if competing:
        H, n_trees = forest.oob_cause_cumhaz(d.X, offsets, units, w, n_jobs)
    else:
        H, n_trees = forest.oob_cumhaz(d.X, offsets, units, w, estimator.aggregate, n_jobs)
    ok = np.flatnonzero(n_trees > 0)
    if ok.size == 0:
        raise ValueError("no row has an out-of-bag tree")
    off, uni = _subset_csr(offsets, units, ok)

    def predict(Xr, rows):
        o, u = _subset_csr(off, uni, rows)
        Xr = np.ascontiguousarray(Xr)
        if competing:
            return forest.oob_cause_cumhaz(Xr, o, u, w, n_jobs)[0]
        return forest.oob_cumhaz(Xr, o, u, w, estimator.aggregate, n_jobs)[0]

    y_design = estimator._oob_target(d.stop, d.event, d.start)[ok]
    orig = np.arange(d.n_rows) if d.kept is None else d.kept
    row_ids = (orig if d.ids_values is None else np.asarray(d.ids_values)[orig])[ok]
    excluded = d.n_rows - ok.size
    return np.ascontiguousarray(d.X[ok]), y_design, d.start[ok], row_ids, predict, (H[ok], orig[ok], d.n_rows), excluded


def permutation_importance(
    estimator,
    X,
    y=None,
    *,
    ids=None,
    features=None,
    groups=None,
    strata="time",
    n_strata=10,
    conditional_on=None,
    n_bins=4,
    scoring="pe",
    windows=8,
    alpha=0.01,
    cause=None,
    oob=False,
    measured_at=None,
    block_time=None,
    n_repeats=5,
    n_bootstrap=100,
    random_state=None,
    n_jobs=None,
):
    """Permutation importance on counting-process rows, scored by the piecewise-exponential log score.

    The importance of a unit (a feature, or a group of columns permuted
    jointly) is the drop in the PE score (``metrics.piecewise_exponential_score``,
    per scored event) when the unit's values are permuted among the evaluation
    rows, averaged over ``n_repeats`` permutations. Predictions are each row's
    fixed-profile cumulative hazard at the window edges.

    By default rows are permuted **within time strata** (bins of the rows'
    ``start``), so a permuted value comes from a row at a similar time and
    stays on the observed (time, value) support. ``strata=None`` is the naive
    shuffle across all rows: with time-varying covariates it can extrapolate
    off the (time, value) support (a late value placed on an early row), which
    biases importance; it is available for comparison only and warns.

    Parameters
    ----------
    estimator : fitted SurvivalForestTV or CompetingRisksForestTV
    X : array-like or DataFrame of shape (n_rows, n_features)
        Held-out counting-process rows (or, with ``oob=True``, the training data).
    y : survival or competing-risks target of the rows (``start``, ``stop``, ``event``).
    ids : array-like of shape (n_rows,) or str, default=None
        Subject of each row (or a column of a DataFrame ``X``). ``None`` makes
        every row its own subject; the bootstrap then resamples rows.
    features : list of names or indices, default=None
        One unit per feature; default all features.
    groups : dict name -> list of columns, default=None
        Units permuted jointly (e.g. a covariate with its lags); exclusive with ``features``.
    strata : "time", None or array-like of shape (n_rows,), default="time"
        ``"time"``: ``n_strata`` quantile bins of ``start``. An array gives
        user strata (e.g. calendar period). ``None``: no strata (naive shuffle; see above).
    n_strata : int, default=10
    conditional_on : list of names or indices, default=None
        Columns whose ``n_bins`` quantile bins are crossed with the strata
        (conditional permutation); a unit's own columns are left out of its
        conditioning set. Rows alone in their stratum are not permuted and are
        counted in ``n_unpermuted``.
    n_bins : int, default=4
        Bins per conditioning column; a column with at most ``n_bins`` distinct
        values is binned by value.
    scoring : "pe", default="pe"
        The piecewise-exponential log score.
    windows : int or array-like, default=8
        Number of scoring windows (``metrics.event_windows``) or their edges.
    alpha : float, default=0.01
        Mixture weight of the training null in the scored rate.
    cause : int, default=None
        Competing risks: score one cause only; ``None`` sums causes and reports
        ``importances_cause``.
    oob : bool, default=False
        Score the training rows out of bag. ``X``, ``y``, ``ids`` (and
        ``measured_at`` / ``block_time`` if the fit used them) must be the
        training data; they are checked against the fit. Rows are the fit-time
        design rows (after ``ntime`` coarsening), those without an out-of-bag
        tree are excluded. No standard error is computed.
    measured_at, block_time : array-like, default=None
        As passed to ``fit``; only with ``oob=True``.
    n_repeats : int, default=5
    n_bootstrap : int, default=100
        Id-cluster bootstrap replicates for ``importances_se`` (held-out only).
        Each replicate resamples subjects with replacement, redraws the
        ``n_repeats`` within-stratum permutations and rescores, so the SE is
        that of the reported repeat average over evaluation subjects,
        conditional on the fitted forest (refit variability is not included).
        It costs ``n_units × n_bootstrap × n_repeats`` predictions; 0 skips it.
    random_state : int, RandomState, Generator or None
        Each unit draws from its own stream keyed by its first column, so a
        unit's result does not depend on the other units requested.
    n_jobs : int, default=None
        Threads for prediction.

    Returns
    -------
    Bunch with, per unit (in score units per scored event):

    - ``importances (p, n_repeats)``, ``importances_mean``, ``importances_std``, ``importances_se``;
    - ``importances_window (p, M)`` (sums to ``importances_mean``) and ``window_edges``;
    - ``importances_cause (p, J)`` (competing risks, ``cause=None``; else None);
    - ``importances_id (p, n_ids)`` with ``id_labels``;
    - ``share_of_gain``: ``importances_mean / (baseline_score - null_score)``,
      an unbounded ratio (NaN when the model does not beat the training null);
    - ``baseline_score``, ``null_score``, ``zero_rate_share``, ``n_events``,
      ``n_truncated_events``, ``n_rows_excluded``, ``n_unpermuted``,
      ``feature_names``, ``units`` (column indices per unit).
    """
    competing = _family(estimator)
    if y is None:
        raise ValueError("y is required for counting-process estimators")
    if scoring != "pe":
        raise ValueError(f"scoring must be 'pe' for counting-process estimators (brier/ibs: landmark models), got {scoring!r}")
    n_repeats = _strata.check_count(n_repeats, "n_repeats")
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, numbers.Integral) or n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be an integer >= 0, got {n_bootstrap!r}")
    n_strata = _strata.check_count(n_strata, "n_strata")
    n_bins = _strata.check_count(n_bins, "n_bins")
    if not oob and (measured_at is not None or block_time is not None):
        raise ValueError("measured_at and block_time are only used with oob=True")
    w = np.ascontiguousarray(_windows(estimator, windows))
    null = _baseline_at(estimator, w)
    threads = effective_n_jobs(n_jobs)
    if oob:
        Xe, ye, start, row_ids, predict, pre, excluded = _oob(
            estimator, X, y, ids, measured_at, block_time, competing, w, threads
        )
        H, input_rows, n_input = pre
    else:
        Xe, ye, start, row_ids, predict, _, excluded = _heldout(estimator, X, y, ids, competing, w, threads)
        H, input_rows, n_input = predict(Xe, np.arange(Xe.shape[0])), None, Xe.shape[0]
    names = getattr(estimator, "feature_names_in_", None)
    ids_column = getattr(estimator, "ids_column_", None)
    units, unit_names = _units.resolve_units(features, groups, names, Xe.shape[1], ids_column)
    cond = []
    if conditional_on is not None:
        refs = [conditional_on] if isinstance(conditional_on, (str, numbers.Integral)) else list(conditional_on)
        cond = [_units._column(c, names, Xe.shape[1], ids_column) for c in refs]
    if isinstance(strata, str) and strata == "time":
        st = _score.Strata("time", None, n_strata, cond, n_bins)
    elif strata is None:
        warnings.warn(
            "strata=None: naive permutation can extrapolate off the (time, value) support; prefer strata='time'",
            UserWarning,
            stacklevel=2,
        )
        st = _score.Strata(None, None, n_strata, cond, n_bins)
    elif isinstance(strata, str):
        raise ValueError(f"strata must be 'time', None or an array of labels, got {strata!r}")
    else:
        codes = _strata.user_labels(strata, n_input)
        if input_rows is not None:
            codes = codes[input_rows]
        st = _score.Strata("user", codes, n_strata, cond, n_bins)
    ev = _score.Evaluation(Xe, ye, start, row_ids, w, null, alpha, estimator.causes_ if competing else None, cause, predict)
    intact = _score.score(ev, ye, H, ids=row_ids)
    entropy = _entropy(random_state)
    boot = 0 if oob else n_bootstrap
    res = [
        _score.unit_importance(
            ev, H, intact, cols, st, np.random.SeedSequence(entropy, spawn_key=(int(cols.min()),)), n_repeats, boot
        )
        for cols in units
    ]
    n_unpermuted = np.array([r.n_unpermuted for r in res])
    if (n_unpermuted > 0.1 * Xe.shape[0]).any():
        warnings.warn(
            f"more than 10% of rows are alone in their stratum and not permuted (max {n_unpermuted.max()} of "
            f"{Xe.shape[0]}); importance is biased toward 0 (use fewer strata or bins)",
            UserWarning,
            stacklevel=2,
        )
    N = intact.n_events
    imp = np.array([r.drops for r in res])
    mean = imp.mean(axis=1)
    baseline, null_score = intact.total / N, intact.null_total / N
    gain = baseline - null_score
    if gain > 1e-12 * max(1.0, abs(null_score)):  # a model equal to the null ties up to rounding
        share = mean / gain
    else:
        warnings.warn(
            "the model does not beat the training null on these data; share_of_gain is NaN", UserWarning, stacklevel=2
        )
        share = np.full(mean.shape, np.nan)
    return Bunch(
        importances=imp,
        importances_mean=mean,
        importances_std=imp.std(axis=1),
        importances_se=np.array([r.se for r in res]),
        importances_window=np.array([r.window for r in res]),
        window_edges=w,
        importances_cause=None if intact.by_cause is None else np.array([r.cause for r in res]),
        importances_id=np.array([r.by_id for r in res]),
        id_labels=intact.id_labels,
        share_of_gain=share,
        baseline_score=baseline,
        null_score=null_score,
        zero_rate_share=intact.zero_rate_share,
        n_events=N,
        n_truncated_events=intact.n_truncated_events,
        n_rows_excluded=int(excluded),
        n_unpermuted=n_unpermuted,
        feature_names=np.array(unit_names, dtype=object),
        units=units,
    )
