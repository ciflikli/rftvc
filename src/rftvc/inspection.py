"""Model inspection for the survival forests: permutation and drop-column (LOCO) importance."""

import numbers
import warnings

import numpy as np
from joblib import effective_n_jobs
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ._competing import CompetingRisksForestTV
from ._estimator import _BaseForestTV
from ._inspection import _effects, _loco, _score, _strata, _units
from ._validation import (
    check_counting_process,
    check_intervals,
    check_random_state_or_generator,
    check_survival_y,
    competing_risks_labels,
    make_competing_risks_y,
    make_survival_y,
    split_frame,
)
from .metrics import _baseline_at, _check_windows, event_windows

__all__ = ["drop_column_importance", "hazard_effect", "path_effect", "permutation_importance"]


def _family(estimator, fn="permutation_importance", fitted=True):
    """``(is_landmark, competing)``, checked for ``fn``'s supported estimator types."""
    from .landmark import LandmarkCompetingRisksForest, _LandmarkBase

    is_landmark = isinstance(estimator, _LandmarkBase)
    if not is_landmark and not isinstance(estimator, _BaseForestTV):
        raise TypeError(
            f"{fn} supports SurvivalForestTV, CompetingRisksForestTV, LandmarkSurvivalForest and "
            f"LandmarkCompetingRisksForest, got {type(estimator).__name__}"
        )
    if fitted:
        check_is_fitted(estimator, "forest_")
    competing = isinstance(estimator, (CompetingRisksForestTV, LandmarkCompetingRisksForest))
    return is_landmark, competing


def _reject_brier_kwargs(censoring_estimator, g_min, n_times, context):
    if censoring_estimator is not None or g_min != 0.05 or n_times != 10:
        raise ValueError(f"censoring_estimator, g_min and n_times are only used with scoring in {{'brier', 'ibs'}} ({context})")


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
    return int(check_random_state_or_generator(random_state).randint(np.iinfo(np.int64).max, dtype=np.int64))


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
    censoring_estimator=None,
    g_min=0.05,
    n_times=10,
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

    For a fitted ``LandmarkSurvivalForest`` / ``LandmarkCompetingRisksForest``,
    ``X`` is the raw long frame (``y``, ``ids``, ``oob``, ``measured_at`` and
    ``block_time`` do not apply and must be left at their defaults: outcomes
    and ids come from the model's own ``event``/``stop``/``id`` columns).
    Units default to *raw* history columns (every derived feature of a column
    moves together); rows are permuted within each landmark's own risk set
    (``strata`` is fixed to that partition: ``"time"``, the shared default, or
    the explicit alias ``"landmark"``; anything else raises). ``scoring`` also
    accepts ``"brier"`` / ``"ibs"`` (IPCW, fitted per landmark as
    ``model_selection.landmark_cross_validate`` does, then pooled by risk-set
    size); as these are losses, importance there is ``pooled(permuted) -
    pooled(baseline)`` (positive = permuting made the loss worse), the mirror
    of the PE score's ``intact - permuted``, and ``share_of_gain`` is ``None``.

    Parameters
    ----------
    estimator : fitted SurvivalForestTV, CompetingRisksForestTV, LandmarkSurvivalForest or LandmarkCompetingRisksForest
    X : array-like or DataFrame of shape (n_rows, n_features)
        Held-out counting-process rows (or, with ``oob=True``, the training
        data); for a landmark estimator, the raw long frame.
    y : survival or competing-risks target of the rows (``start``, ``stop``, ``event``).
        Must be ``None`` for a landmark estimator.
    ids : array-like of shape (n_rows,) or str, default=None
        Subject of each row (or a column of a DataFrame ``X``). ``None`` makes
        every row its own subject; the bootstrap then resamples rows. Must be
        ``None`` for a landmark estimator (ids come from its own id column).
    features : list of names or indices, default=None
        One unit per feature; default all features. For a landmark estimator,
        a raw column name expands to every derived feature built on it
        (the default unit); a derived feature name/index is a singleton.
    groups : dict name -> list of columns, default=None
        Units permuted jointly (e.g. a covariate with its lags); exclusive with ``features``.
    strata : "time", None or array-like of shape (n_rows,), default="time"
        ``"time"``: ``n_strata`` quantile bins of ``start``. An array gives
        user strata (e.g. calendar period). ``None``: no strata (naive shuffle; see above).
        For a landmark estimator this is fixed to the landmark partition;
        pass ``"time"`` (the default) or the explicit alias ``"landmark"``.
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
    random_state : int, RandomState instance, Generator, or None, default=None
        Each unit draws from its own stream keyed by its first column, so a
        unit's result does not depend on the other units requested.
    n_jobs : int, default=None
        Threads for prediction.
    censoring_estimator, g_min, n_times : default=None, 0.05, 10
        Landmark ``scoring in {"brier", "ibs"}`` only (same names/defaults as
        ``model_selection.landmark_cross_validate`` / ``metrics.brier_landmark`` /
        ``metrics.integrated_brier``); non-default with ``scoring="pe"`` or a
        counting-process estimator raises.

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

    For a landmark estimator with ``scoring in {"brier", "ibs"}`` the Bunch
    instead has ``importances``, ``importances_mean``, ``importances_std``,
    ``importances_se``, ``share_of_gain=None``, ``baseline_score`` (the pooled
    baseline loss), ``feature_names``, ``units``; the PE-only decompositions
    (``importances_window``, ``importances_cause``, ``importances_id``,
    ``zero_rate_share``, ``n_truncated_events``, ``n_unpermuted``) are ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from rftvc import SurvivalForestTV, make_survival_y, inspection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> t = rng.exponential(np.exp(-0.5 * X[:, 0]))
    >>> y = make_survival_y(np.minimum(t, 2.0), t <= 2.0)
    >>> forest = SurvivalForestTV(n_estimators=100, random_state=0, oob_score=True).fit(X, y)
    >>> result = inspection.permutation_importance(
    ...     forest, X, y, oob=True, n_repeats=3, n_bootstrap=0, random_state=0
    ... )
    >>> result.importances.shape
    (3, 3)
    """
    is_landmark, competing = _family(estimator)
    if is_landmark:
        if oob:
            raise ValueError("oob=True is only for counting-process estimators (SurvivalForestTV/CompetingRisksForestTV)")
        if y is not None:
            raise ValueError("y must be None for a landmark estimator: outcomes come from X's own event/stop columns")
        if ids is not None:
            raise ValueError("ids must be None for a landmark estimator: ids come from X's own id column (model.id)")
        if measured_at is not None or block_time is not None:
            raise ValueError("measured_at and block_time are only used with oob=True (counting-process estimators)")
        if scoring not in ("pe", "brier", "ibs"):
            raise ValueError(f"scoring must be 'pe', 'brier' or 'ibs', got {scoring!r}")
        return _pe_landmark(
            estimator, X, features, groups, strata, n_strata, conditional_on, n_bins, scoring, windows, alpha,
            cause, n_repeats, n_bootstrap, random_state, n_jobs, censoring_estimator, g_min, n_times, competing,
        )
    if y is None:
        raise ValueError("y is required for counting-process estimators")
    if scoring != "pe":
        raise ValueError(f"scoring must be 'pe' for counting-process estimators (brier/ibs: landmark models), got {scoring!r}")
    _reject_brier_kwargs(censoring_estimator, g_min, n_times, "counting-process estimator")
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
    causes = estimator.causes_ if competing else None
    boot = 0 if oob else n_bootstrap
    return _pe_finish(
        Xe, ye, start, row_ids, predict, H, causes, cause, w, null, alpha, units, unit_names,
        n_repeats, boot, random_state, excluded, st,
    )


def _pe_finish(Xe, ye, start, row_ids, predict, H, causes, cause, w, null, alpha, units, unit_names,
                n_repeats, n_bootstrap, random_state, excluded, st):
    """The shared tail of PE-scoring permutation importance: intact score, the per-unit
    permutation loop, ``share_of_gain`` and the result ``Bunch`` (design §3.1)."""
    ev = _score.Evaluation(Xe, ye, start, row_ids, w, null, alpha, causes, cause, predict)
    intact = _score.score(ev, ye, H, ids=row_ids)
    entropy = _entropy(random_state)
    res = [
        _score.unit_importance(
            ev, H, intact, cols, st, np.random.SeedSequence(entropy, spawn_key=(int(cols.min()),)), n_repeats, n_bootstrap
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


def _pe_landmark(model, df, features, groups, strata, n_strata, conditional_on, n_bins, scoring, windows, alpha,
                  cause, n_repeats, n_bootstrap, random_state, n_jobs, censoring_estimator, g_min, n_times, competing):
    """Landmark dispatch of ``permutation_importance`` (``scoring in {"pe", "brier", "ibs"}``)."""
    from .landmark import _raw_groups

    n_repeats = _strata.check_count(n_repeats, "n_repeats")
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, numbers.Integral) or n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be an integer >= 0, got {n_bootstrap!r}")
    n_strata = _strata.check_count(n_strata, "n_strata")
    n_bins = _strata.check_count(n_bins, "n_bins")
    data = model._landmark_data(df)
    Xe, ye, row_ids, s = data.X, data.y, data.ids, data.s
    names = data.feature_names
    threads = effective_n_jobs(n_jobs)
    raw_groups = _raw_groups(model.history_features)
    units, unit_names = _units.resolve_landmark_units(features, groups, raw_groups, names, Xe.shape[1])
    cond = []
    if conditional_on is not None:
        refs = [conditional_on] if isinstance(conditional_on, (str, numbers.Integral)) else list(conditional_on)
        cond = [_units._column(c, names, Xe.shape[1], None) for c in refs]
    if isinstance(strata, str) and strata in ("time", "landmark"):
        codes = np.unique(s, return_inverse=True)[1]
        st = _score.Strata("user", codes, n_strata, cond, n_bins)
    else:
        raise ValueError(
            f"strata is fixed to the landmark partition for landmark estimators; pass 'time' (the default) "
            f"or 'landmark', got {strata!r}"
        )
    if scoring == "pe":
        _reject_brier_kwargs(censoring_estimator, g_min, n_times, "scoring='pe'")
        w = _windows(model.forest_, windows)
        null = _baseline_at(model.forest_, w)
        inner = model.forest_.forest_
        if competing:
            def predict(Xr, rows):
                return inner.predict_cause_cumhaz(np.ascontiguousarray(Xr), w, threads)
        else:
            def predict(Xr, rows):
                return inner.predict_cumhaz(np.ascontiguousarray(Xr), w, model.forest_.aggregate, threads)
        H = predict(Xe, np.arange(Xe.shape[0]))
        causes = model.forest_.causes_ if competing else None
        start = np.zeros(Xe.shape[0])
        return _pe_finish(
            Xe, ye, start, row_ids, predict, H, causes, cause, w, null, alpha, units, unit_names,
            n_repeats, n_bootstrap, random_state, 0, st,
        )
    n_times = _strata.check_count(n_times, "n_times")
    if scoring == "ibs" and n_times < 2:
        raise ValueError(f"n_times must be an integer >= 2 with scoring='ibs', got {n_times!r}")
    entropy = _entropy(random_state)
    res = _score.landmark_loss_importance(
        model, Xe, ye, row_ids, s, units, scoring, cause, n_repeats, n_bootstrap,
        entropy, n_times, censoring_estimator, g_min, competing, st,
    )
    return Bunch(
        importances=res.importances,
        importances_mean=res.importances_mean,
        importances_std=res.importances_std,
        importances_se=res.importances_se,
        importances_window=None,
        window_edges=None,
        importances_cause=None,
        importances_id=None,
        id_labels=None,
        share_of_gain=None,
        baseline_score=res.baseline_score,
        null_score=None,
        zero_rate_share=None,
        n_events=None,
        n_truncated_events=None,
        n_rows_excluded=None,
        n_unpermuted=None,
        feature_names=np.array(unit_names, dtype=object),
        units=units,
    )


def hazard_effect(estimator, X, y, *, feature, values=None, windows=8, kind="average", ids=None, cause=None):
    """Time-stratified partial dependence of window hazards on one covariate.

    For each scoring window ``W_m`` and grid value ``v``, the estimand is the
    exposure-weighted average of the fixed-profile window hazard
    ``lambda~_m(x_r with x_j = v)`` over the rows at risk in ``W_m`` (weight:
    the row's exposure ``e_rm`` in that window, as in ``metrics.piecewise_exponential_score``).
    A window with no row at risk gives ``NaN`` for every grid value.

    This is valid for internal and external covariates alike (it is
    associational on the hazard scale) and shows non-proportional
    (time-varying) effects directly, unlike a single marginal partial
    dependence curve.

    For a fitted ``LandmarkSurvivalForest`` / ``LandmarkCompetingRisksForest``,
    ``X`` is the raw long frame and ``y``/``ids`` must be ``None``; the windows
    are on the horizon clock, as in ``permutation_importance``'s landmark dispatch.

    Parameters
    ----------
    estimator : fitted SurvivalForestTV, CompetingRisksForestTV, LandmarkSurvivalForest or LandmarkCompetingRisksForest
    X : array-like or DataFrame of shape (n_rows, n_features)
        Held-out counting-process rows, or the raw long frame for a landmark estimator.
    y : survival or competing-risks target of the rows. Must be ``None`` for a landmark estimator.
    feature : str or int
        The covariate to vary. Not the ids column; not ``"landmark"`` for a landmark estimator.
    values : array-like, default=None
        Grid of values; default 20 quantiles of the observed column (deduplicated).
    windows : int or array-like, default=8
        Number of scoring windows (``metrics.event_windows``) or their edges.
    kind : "average" or "individual", default="average"
        ``"individual"`` also returns per-row rates (ICE); they exposure-weighted-average to
        the ``"average"`` result.
    ids : array-like of shape (n_rows,) or str, default=None
        Subject of each row (or a column of a DataFrame ``X``). Must be ``None`` for a landmark estimator.
    cause : int, default=None
        Competing risks: one cause's hazard; ``None`` returns every cause.

    Returns
    -------
    Bunch with ``values``, ``hazard`` (``(n_values, M)`` or CR ``(n_values, J, M)``),
    ``window_edges``, ``support_mask`` (``(n_values, M)``), ``individual``
    (``None`` unless ``kind="individual"``), ``feature_name``.

    ``values`` shadows ``dict.values`` (as ``sklearn.inspection.partial_dependence``'s
    own ``"values"`` entry does): ``result.values`` is the bound ``dict.values`` method,
    not the grid, and calling it (``result.values()``) gives the ordinary dict values,
    not an error. Access the grid as ``result["values"]``.

    A ``hazard_effect`` two-point contrast is only comparable to a linear model's
    (e.g. ``CoxTimeVaryingFitter``) single coefficient sign when ``feature`` is binary
    or genuinely continuous. For a nominal covariate with more than two levels, the
    linear model fits one coefficient across every level while ``hazard_effect`` makes
    no linearity assumption; a sign read off two contrast points need not agree with,
    or disagree with, the linear coefficient in any meaningful sense for that covariate.
    Comparing this contrast against a linear model still fit on the original,
    ordinally-coded column (one coefficient across every level) is not a like-for-like
    comparison. To compare them meaningfully, re-encode the covariate for the linear fit
    as one reference-level indicator per non-reference level (drop a baseline level, one
    dummy per remaining level) and refit; then compare each dummy's coefficient sign
    against ``hazard_effect``'s own two-point contrast for that same level against that
    same reference.

    Examples
    --------
    >>> import numpy as np
    >>> from rftvc import SurvivalForestTV, make_survival_y, inspection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> t = rng.exponential(np.exp(-0.5 * X[:, 0]))
    >>> y = make_survival_y(np.minimum(t, 2.0), t <= 2.0)
    >>> forest = SurvivalForestTV(n_estimators=100, random_state=0).fit(X, y)
    >>> result = inspection.hazard_effect(forest, X, y, feature=0, values=[-1.0, 0.0, 1.0], windows=4)
    >>> result.hazard.shape
    (3, 4)
    """
    if kind not in ("average", "individual"):
        raise ValueError(f"kind must be 'average' or 'individual', got {kind!r}")
    is_landmark, competing = _family(estimator, fn="hazard_effect")
    if cause is not None and not competing:
        raise ValueError("cause is only for competing-risks estimators")
    if is_landmark:
        if y is not None:
            raise ValueError("y must be None for a landmark estimator: outcomes come from X's own event/stop columns")
        if ids is not None:
            raise ValueError("ids must be None for a landmark estimator: ids come from X's own id column (model.id)")
        data = estimator._landmark_data(X)
        Xe, ye, s = data.X, data.y, data.s
        names = data.feature_names
        n_features = Xe.shape[1]
        landmark_idx = n_features - 1
        is_landmark_col = isinstance(feature, numbers.Integral) and not isinstance(feature, (bool, np.bool_)) and int(feature) == landmark_idx
        if feature == "landmark" or is_landmark_col:
            raise ValueError("'landmark' is constant within a landmark stratum and is not a permutable feature")
        feature_idx = _units._column(feature, names, n_features, None)
        w = _windows(estimator.forest_, windows)
        inner = estimator.forest_.forest_
        threads = effective_n_jobs(None)
        if competing:
            def predict(Xr, rows):
                return inner.predict_cause_cumhaz(np.ascontiguousarray(Xr), w, threads)
        else:
            def predict(Xr, rows):
                return inner.predict_cumhaz(np.ascontiguousarray(Xr), w, estimator.forest_.aggregate, threads)
        start = np.zeros(Xe.shape[0])
        if competing:
            stop = competing_risks_labels(ye)[1]
        else:
            stop = check_survival_y(ye, require_events=False)[1]
        cause_idx = None if cause is None else estimator.forest_._cause_index(cause)
    else:
        if y is None:
            raise ValueError("y is required for counting-process estimators")
        w = _windows(estimator, windows)
        threads = effective_n_jobs(None)
        Xe, ye, start, row_ids, predict, _, _ = _heldout(estimator, X, y, ids, competing, w, threads)
        names = getattr(estimator, "feature_names_in_", None)
        ids_column = getattr(estimator, "ids_column_", None)
        feature_idx = _units._column(feature, names, Xe.shape[1], ids_column)
        if competing:
            stop = competing_risks_labels(ye)[1]
        else:
            stop = check_survival_y(ye, require_events=False)[1]
        cause_idx = None if cause is None else estimator._cause_index(cause)
    if values is None:
        values = _effects.default_values(Xe[:, feature_idx])
    else:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("values must be a non-empty 1-D array of finite numbers")
    hazard, support_mask, individual = _effects.hazard_grid(
        Xe, start, stop, predict, w, feature_idx, values, kind, competing, cause_idx
    )
    feature_name = str(names[feature_idx]) if names is not None else str(feature_idx)
    return Bunch(
        values=values,
        hazard=hazard,
        window_edges=w,
        support_mask=support_mask,
        individual=individual,
        feature_name=feature_name,
    )


def path_effect(
    estimator, X, intervals, ids, *, feature, delta, from_time, horizons, origin=None, cause=None, extrapolate="none"
):
    """Prediction contrast along a specified covariate path.

    Compares the supplied path to the same path with ``feature`` shifted by
    ``delta`` from ``from_time`` onward, as the change in
    ``P(T <= h | T > origin, path)`` (or ``F_k(h | origin)``) at each ``h`` in
    ``horizons``. Rows straddling ``from_time`` are split there first
    (covariates copied to both halves).

    This is a prediction under a specified covariate path (Kalbfleisch-Prentice),
    valid when ``feature`` is external. It is **not a causal effect** unless
    ``feature``'s effect on the hazard is unconfounded given the other covariates
    (Keogh & van Geloven 2024).

    A large ``delta`` that shifts the path into a much higher-hazard region can
    **understate** the true risk change: like any random forest, this one
    shrinks its predictions toward the bulk of the training distribution, and
    that shrinkage is stronger where the shifted path's hazard is elevated
    (confirmed on the S3 simulation, `docs/plans/s20-plan.md` T9). This was
    checked on the *mean* over many subjects; an individual subject's own
    estimate can still have the wrong sign, as any per-subject estimate can.

    Not defined for landmark estimators (paths are a counting-process concept):
    ``LandmarkSurvivalForest`` / ``LandmarkCompetingRisksForest`` raise ``TypeError``.

    Parameters
    ----------
    estimator : fitted SurvivalForestTV or CompetingRisksForestTV
    X : array-like or DataFrame of shape (n_rows, n_features)
        Covariate-path rows (one row per ``(subject, interval)``), as ``predict_cumulative_hazard(intervals=...)``.
    intervals : structured array or DataFrame with ``start``, ``stop``
        Each row's interval; grouped into subjects by ``ids``.
    ids : array-like of shape (n_rows,) or str
        Subject of each row (or a column of a DataFrame ``X``). Required (paths need subject grouping).
    feature : str or int
        The covariate to shift.
    delta : float or callable
        The shift from ``from_time`` onward: a constant, or ``f(values, start) -> values``.
    from_time : float
        Absolute analysis time the shift starts at, on the same clock as ``intervals``.
    horizons : array-like of float
        Absolute analysis times to evaluate the contrast at; each must be ``>= from_time``.
    origin : float or array-like, default=None
        Per ``predict_cumulative_hazard``; defaults to each subject's first ``start``.
    cause : int, default=None
        Competing risks: one cause's ``F_k``; ``None`` returns every cause.
    extrapolate : "none" or "locf", default="none"
        Covariate behaviour past a subject's last observed ``stop``, for horizons beyond it.

    Returns
    -------
    Bunch with ``per_subject`` (``(n_subjects, n_horizons)`` or CR
    ``(n_subjects, n_causes, n_horizons)``/``(n_subjects, n_horizons)`` for one
    ``cause``), ``mean``, ``horizons``, ``id_labels``.

    Examples
    --------
    One subject's forward covariate path, held flat, then feature 0 (the one driving
    the hazard here: ``rate = exp(-0.5 * X[:, 0])``, so raising it *lowers* risk)
    shifted by ``delta=1.0`` from a time inside its second row onward — the contrast
    should come out negative at both horizons:

    >>> import numpy as np
    >>> from rftvc import SurvivalForestTV, make_survival_y, inspection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 2))
    >>> t = rng.exponential(np.exp(-0.5 * X[:, 0]))
    >>> y = make_survival_y(np.minimum(t, 5.0), t <= 5.0)
    >>> forest = SurvivalForestTV(n_estimators=100, random_state=0).fit(X, y)
    >>> n_steps = 5
    >>> Xp = np.column_stack([np.zeros(n_steps), np.zeros(n_steps)])
    >>> iv = make_survival_y(
    ...     np.arange(1.0, n_steps + 1), np.zeros(n_steps, dtype=bool), start=np.arange(0.0, n_steps)
    ... )
    >>> ids = np.zeros(n_steps, dtype=int)
    >>> result = inspection.path_effect(forest, Xp, iv, ids, feature=0, delta=1.0, from_time=1.5, horizons=[3.0, 4.0])
    >>> result.per_subject.shape
    (1, 2)
    >>> bool((result.per_subject < 0).all())
    True
    """
    is_landmark, competing = _family(estimator, fn="path_effect")
    if is_landmark:
        raise TypeError("path_effect is not defined for landmark estimators: paths are a counting-process concept")
    if cause is not None and not competing:
        raise ValueError("cause is only for competing-risks estimators")
    Xc, _, idsc = estimator._check_predict(X, None, ids)
    if idsc is None:
        raise ValueError("ids is required (paths are grouped into subjects by id)")
    if not isinstance(from_time, numbers.Real) or isinstance(from_time, (bool, np.bool_)) or not np.isfinite(from_time):
        raise ValueError(f"from_time must be a finite number, got {from_time!r}")
    horizons = np.asarray(horizons, dtype=float)
    if horizons.ndim != 1 or horizons.size == 0:
        raise ValueError("horizons must be a non-empty 1-D array")
    if (horizons < from_time).any():
        raise ValueError(f"horizons must all be >= from_time ({from_time}); got {horizons.tolist()}")
    Xo, start, stop, offsets, origin_r = estimator._path_args(Xc, intervals, idsc, origin, extrapolate)
    start0, stop0 = check_intervals(intervals)
    cp = check_counting_process(start0, stop0, None, idsc)
    id_labels = np.asarray(idsc)[cp.order][cp.offsets[:-1]]
    last_stop = stop[offsets[1:] - 1]
    bad_origin = from_time < origin_r
    if bad_origin.any():
        raise ValueError(f"from_time ({from_time}) is before origin for ids {id_labels[bad_origin].tolist()}")
    unreached = from_time > last_stop
    if unreached.any():
        raise ValueError(f"from_time ({from_time}) is beyond the last stop for ids {id_labels[unreached].tolist()}")
    beyond = horizons.max() > last_stop
    if extrapolate != "locf":
        if beyond.any():
            raise ValueError(
                f"horizons extend beyond the last stop for ids {id_labels[beyond].tolist()}; "
                "pass extrapolate='locf' or supply the future path as extra rows"
            )
    elif beyond.any():
        # from_time <= last_stop is already guaranteed (the `unreached` check above); if it lands
        # exactly at the last stop, no row satisfies `start >= from_time` to carry the shift into
        # the LOCF-extrapolated future, and shifting the last row itself would wrongly change its
        # covariates before from_time too. Ambiguous: reject rather than silently drop the shift.
        at_boundary = last_stop == from_time
        if at_boundary.any():
            raise ValueError(
                f"from_time ({from_time}) equals the last observed stop for ids "
                f"{id_labels[at_boundary & beyond].tolist()}, and a horizon extends beyond it: "
                "extrapolate='locf' cannot apply the shift to the extrapolated future from exactly "
                "the last stop; supply the future path as extra rows instead"
            )
    Xs, start_s, stop_s, offsets_s = _effects.split_at(Xo, start, stop, offsets, from_time)
    feature_idx = _units._column(feature, getattr(estimator, "feature_names_in_", None), Xs.shape[1], None)
    Xshift = _effects.shift_from(Xs, start_s, feature_idx, from_time, delta)
    path_ids = np.repeat(np.arange(offsets_s.size - 1), np.diff(offsets_s))
    iv = make_survival_y(stop_s, np.zeros(stop_s.shape[0], dtype=bool), start=start_s)
    if competing:
        original = estimator.predict_cumulative_incidence(
            Xs, horizons, cause=cause, intervals=iv, ids=path_ids, origin=origin_r, extrapolate=extrapolate
        )
        shifted = estimator.predict_cumulative_incidence(
            Xshift, horizons, cause=cause, intervals=iv, ids=path_ids, origin=origin_r, extrapolate=extrapolate
        )
    else:
        original = 1.0 - estimator.predict_survival_function(
            Xs, horizons, intervals=iv, ids=path_ids, origin=origin_r, extrapolate=extrapolate
        )
        shifted = 1.0 - estimator.predict_survival_function(
            Xshift, horizons, intervals=iv, ids=path_ids, origin=origin_r, extrapolate=extrapolate
        )
    per_subject = shifted - original
    return Bunch(
        per_subject=per_subject,
        mean=per_subject.mean(axis=0),
        horizons=horizons,
        id_labels=id_labels,
    )


def drop_column_importance(
    estimator,
    X,
    y=None,
    *,
    ids=None,
    cv=5,
    features=None,
    groups=None,
    scoring="pe",
    windows=8,
    alpha=0.01,
    cause=None,
    n_seeds=1,
    add_noise_control=False,
    random_state=None,
    n_jobs=None,
    censoring_estimator=None,
    g_min=0.05,
    n_times=10,
):
    """Cross-fitted drop-column (LOCO) importance, scored by the piecewise-exponential log score.

    The importance of a unit (a feature, or a group of columns dropped jointly) is the
    drop in held-out PE score (``metrics.piecewise_exponential_score``, per scored event)
    between the estimator refitted with every feature and the estimator refitted without
    the unit, cross-fitted over ``cv``. Unlike ``permutation_importance``, ``estimator`` is
    a template: every fold clones it (``sklearn.base.clone``) and refits, so a fitted
    estimator's forest is discarded. Correlated units can compensate for each other, so two
    strongly correlated variables can both have low LOCO importance (Hooker et al. 2021;
    Williamson-type VIM, survival version Wolock et al. 2025).

    For each fold: a clone fitted on every feature gives the scoring windows
    (``metrics.event_windows``) and the training null (``baseline_cumhaz_``); a clone per
    dropped unit is fitted on the same training rows and scored on the test rows with that
    windows/null. ``n_seeds > 1`` averages the full and dropped fits over seeds, since
    refits differ by forest randomness even for a unit with no true effect;
    ``add_noise_control=True`` appends a standard-normal column (drawn once, shared by every
    fold) and reports it as unit ``"_noise"``, a noise floor for the other units.

    For a ``LandmarkSurvivalForest`` / ``LandmarkCompetingRisksForest`` **template**, ``X`` is
    the raw long frame (``y``/``ids`` must be ``None``: outcomes and ids come from the
    model's own columns). Dropping a unit removes every ``history_features`` entry built on
    it (a raw column, or the individual derived features a ``groups=`` unit names); folds are
    the landmark partition (``cv.split`` on the stacked ``s``/id grouping), not
    ``model_selection._cv_folds``. ``add_noise_control`` draws one value per **row of the raw
    frame** (not per stacked row) and adds it to every fold's full model as an extra
    ``history_features`` entry (aggregation ``"last"``). ``scoring in {"brier", "ibs"}`` pools
    per-landmark losses as ``permutation_importance`` does, but ``importances_se`` is not
    computed for LOCO in that case (``NaN``; use ``permutation_importance``'s bootstrap SE for
    a loss-scored standard error).

    Parameters
    ----------
    estimator : SurvivalForestTV, CompetingRisksForestTV, LandmarkSurvivalForest or LandmarkCompetingRisksForest
        A template estimator (fitted or not); every fold clones and refits it.
    X : array-like or DataFrame of shape (n_rows, n_features)
        Counting-process rows; for a landmark estimator, the raw long frame.
    y : survival or competing-risks target of the rows (``start``, ``stop``, ``event``).
        Must be ``None`` for a landmark estimator.
    ids : array-like of shape (n_rows,) or str, default=None
        Subject of each row (or a column of a DataFrame ``X``). ``None`` makes every row its
        own subject. Must be ``None`` for a landmark estimator.
    cv : int or a splitter, default=5
        An int gives ``GroupKFold(cv)`` on ``ids`` (new-subject cross-fitting): the
        documented, primary mode. A ``model_selection.RollingOriginSplit`` / ``GroupTimeSplit``
        administratively censors the training rows of each fold at the earliest test row's
        ``start`` (as ``model_selection.landmark_cross_validate`` does for landmark models);
        any other splitter must keep ``ids`` disjoint between train and test. **A time
        splitter is of limited practical use here**: administrative censoring clips every
        training row's ``stop`` at the fold's cutoff, so a fold's own ``event_times_`` never
        extends past it, while every test row's ``stop`` is at or after it by construction —
        and a Nelson-Aalen-based forest's prediction and training null are both flat past
        their own last observed event, so windows entirely beyond the cutoff score a real
        event as ``-inf``, whatever ``alpha``. This is a structural property of the estimator
        family, not fixable here; it does not affect ``cv=int`` (test ids' events share
        training ids' time range) or S19's landmark LOCO (scored on the reset clock).
    features : list of names or indices, default=None
        One unit per feature (dropped alone); default all features.
    groups : dict name -> list of columns, default=None
        Units dropped jointly (e.g. a covariate with its lags); exclusive with ``features``.
    scoring : "pe", default="pe"
        The piecewise-exponential log score (``"brier"`` / ``"ibs"``: landmark models, S19).
    windows : int or array-like, default=8
        Number of scoring windows or their edges, resolved **per fold** from that fold's own
        full-model fit (folds can have different training data, hence different
        ``event_times_``); unlike ``permutation_importance`` this is not checked against any
        single estimator's last event time, and an edge beyond a fold's own last training
        event time extrapolates flatly. Under a time-based ``cv`` (above), an int raises
        ``ValueError``: no single fold's own windows are valid for every fold, so edges must
        be given explicitly (and, per the caveat above, still will not score test events
        that lie beyond the training fold's own last event usefully).
    alpha : float, default=0.01
        Mixture weight of the training null in the scored rate.
    cause : int, default=None
        Competing risks: score one cause only; ``None`` sums causes and reports
        ``importances_cause``.
    n_seeds : int, default=1
        Refits per fold per unit, averaged (fit-noise control).
    add_noise_control : bool, default=False
        Add a standard-normal column and report it as unit ``"_noise"``.
    random_state : int, RandomState instance, Generator, or None, default=None
        Seeds the noise column and every fold/seed's fit, independently of ``n_jobs``.
    n_jobs : int, default=None
        Unlike ``permutation_importance`` (which parallelises only inside the engine's own
        prediction call, since unit-level parallelism on top of engine threads would
        oversubscribe), LOCO's cost is dominated by independent refits: ``n_jobs`` here
        parallelises over the ``(p_units + 1) * n_folds * n_seeds`` fits themselves
        (``joblib.Parallel(prefer="threads")``, since the Rust fit releases the GIL). Each
        clone keeps its own ``n_jobs`` (inherited from ``estimator``); combining a large
        ``n_jobs`` here with a multi-threaded template estimator oversubscribes.
    censoring_estimator, g_min, n_times : default=None, 0.05, 10
        Landmark ``scoring in {"brier", "ibs"}`` only; non-default with ``scoring="pe"`` or a
        counting-process estimator raises (as ``permutation_importance``).

    Returns
    -------
    Bunch with, per unit (in score units per scored event, pooled over folds):

    - ``importances (p, n_folds)`` (each fold's own per-event importance, seed-averaged;
      not expected to average to ``importances_mean`` under unequal fold sizes),
      ``importances_mean``, ``importances_se`` (id-cluster cross-fit SE,
      ``sd(per-id drop) * sqrt(n_ids) / n_events``; ``NaN`` under a time-based ``cv``,
      since test folds are not exchangeable over time);
    - ``importances_window (p, M)`` (sums to ``importances_mean``) and ``window_edges``
      (the first fold's; ``None`` when folds disagree on the number of windows, e.g. heavy
      ties in a small fold);
    - ``importances_cause (p, J)`` (competing risks, ``cause=None``; else ``None``);
    - ``share_of_gain``: ``importances_mean / (baseline_score - null_score)``, an unbounded
      ratio (``NaN`` when the model does not beat the training null on average);
    - ``baseline_score``, ``null_score`` (pooled, per event), ``fold_scores`` (the full
      model's own per-fold per-event PE score, a diagnostic independent of any unit),
      ``n_events``, ``n_ids``, ``n_folds``, ``feature_names``, ``units`` (column indices
      per unit, including the appended ``"_noise"`` column when requested).

    Examples
    --------
    >>> import numpy as np
    >>> from rftvc import SurvivalForestTV, make_survival_y, inspection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> t = rng.exponential(np.exp(-0.5 * X[:, 0]))
    >>> y = make_survival_y(np.minimum(t, 2.0), t <= 2.0)
    >>> forest = SurvivalForestTV(n_estimators=50, random_state=0)
    >>> result = inspection.drop_column_importance(forest, X, y, cv=3, random_state=0)
    >>> result.importances_mean.shape
    (3,)
    """
    is_landmark, competing = _family(estimator, fn="drop_column_importance", fitted=False)
    if is_landmark:
        if y is not None:
            raise ValueError("y must be None for a landmark estimator: outcomes come from X's own event/stop columns")
        if ids is not None:
            raise ValueError("ids must be None for a landmark estimator: ids come from X's own id column (model.id)")
        if scoring not in ("pe", "brier", "ibs"):
            raise ValueError(f"scoring must be 'pe', 'brier' or 'ibs', got {scoring!r}")
        return _loco_landmark(
            estimator, X, cv, features, groups, scoring, windows, alpha, cause, n_seeds, add_noise_control,
            random_state, n_jobs, censoring_estimator, g_min, n_times, competing,
        )
    if y is None:
        raise ValueError("y is required for counting-process estimators")
    if scoring != "pe":
        raise ValueError(
            f"scoring must be 'pe' for counting-process estimators (brier/ibs: landmark models), got {scoring!r}"
        )
    _reject_brier_kwargs(censoring_estimator, g_min, n_times, "counting-process estimator")
    n_seeds = _strata.check_count(n_seeds, "n_seeds")
    if isinstance(cv, numbers.Integral) and not isinstance(cv, (bool, np.bool_)):
        if cv < 2:
            raise ValueError(f"cv must be an integer >= 2 or a splitter, got {cv!r}")
    elif not hasattr(cv, "split"):
        raise ValueError(f"cv must be an integer >= 2 or a splitter with a split method, got {cv!r}")
    if not (isinstance(windows, numbers.Integral) and not isinstance(windows, (bool, np.bool_))):
        windows = _check_windows(windows)
    ids_name = ids if isinstance(ids, str) else None
    Xnum, names, ids_values = split_frame(X, ids)
    ye, _ = _target(y, competing)
    if ye.shape[0] != Xnum.shape[0]:
        raise ValueError(f"X has {Xnum.shape[0]} rows but y has {ye.shape[0]}")
    ids_values = np.arange(Xnum.shape[0]) if ids_values is None else np.asarray(ids_values)
    if ids_values.shape != (Xnum.shape[0],):
        raise ValueError(f"ids must have {Xnum.shape[0]} entries")
    units, unit_names = _units.resolve_units(features, groups, names, Xnum.shape[1], ids_name)
    entropy = _entropy(random_state)
    if add_noise_control:
        Xnum = np.c_[Xnum, _loco.noise_column(entropy, Xnum.shape[0])]
        units = units + [np.array([Xnum.shape[1] - 1], dtype=np.intp)]
        unit_names = unit_names + ["_noise"]
    res = _loco.run(
        estimator, Xnum, ye, ids_values, units, cv, windows, alpha, cause, competing, n_seeds, entropy, n_jobs
    )
    mean = res.importances_mean
    gain = res.baseline_score - res.null_score
    if gain > 1e-12 * max(1.0, abs(res.null_score)):
        share = mean / gain
    else:
        warnings.warn(
            "the model does not beat the training null on these data; share_of_gain is NaN", UserWarning, stacklevel=2
        )
        share = np.full(mean.shape, np.nan)
    return Bunch(
        importances=res.importances,
        importances_mean=mean,
        importances_se=res.importances_se,
        importances_window=res.importances_window,
        window_edges=res.window_edges,
        importances_cause=res.importances_cause,
        share_of_gain=share,
        baseline_score=res.baseline_score,
        null_score=res.null_score,
        fold_scores=res.fold_scores,
        n_events=res.n_events,
        n_ids=res.n_ids,
        n_folds=res.n_folds,
        feature_names=np.array(unit_names, dtype=object),
        units=units,
    )


def _loco_landmark(model, df, cv, features, groups, scoring, windows, alpha, cause, n_seeds, add_noise_control,
                    random_state, n_jobs, censoring_estimator, g_min, n_times, competing):
    """Landmark dispatch of ``drop_column_importance`` (``scoring in {"pe", "brier", "ibs"}``)."""
    import polars as pl
    from sklearn.base import clone

    from ._estimator import SurvivalForestTV
    from .landmark import _as_polars, _raw_groups

    n_seeds = _strata.check_count(n_seeds, "n_seeds")
    if isinstance(cv, numbers.Integral) and not isinstance(cv, (bool, np.bool_)):
        if cv < 2:
            raise ValueError(f"cv must be an integer >= 2 or a splitter, got {cv!r}")
        from sklearn.model_selection import GroupKFold

        cv = GroupKFold(cv)
    elif not hasattr(cv, "split"):
        raise ValueError(f"cv must be an integer >= 2 or a splitter with a split method, got {cv!r}")
    if not (isinstance(windows, numbers.Integral) and not isinstance(windows, (bool, np.bool_))):
        windows = _check_windows(windows)
    entropy = _entropy(random_state)
    default_forest = CompetingRisksForestTV() if competing else SurvivalForestTV()
    stack_template = model if model.forest is not None else clone(model).set_params(forest=default_forest)
    df = _as_polars(df)
    if add_noise_control:
        df = df.with_columns(pl.Series("_noise", _loco.noise_column(entropy, df.height)))
        stack_template = clone(stack_template).set_params(history_features=list(stack_template.history_features) + ["_noise"])
    raw_groups = _raw_groups(model.history_features)  # the user's units, before any noise column
    data = stack_template._landmark_data(df)
    names = data.feature_names
    units, unit_names = _units.resolve_landmark_units(features, groups, raw_groups, names, data.X.shape[1])
    if add_noise_control:
        units = units + [np.array([names.index("_noise")], dtype=np.intp)]
        unit_names = unit_names + ["_noise"]
    if scoring == "pe":
        _reject_brier_kwargs(censoring_estimator, g_min, n_times, "scoring='pe'")
        res = _loco.run_landmark(stack_template, df, data, units, cv, windows, alpha, cause, competing, n_seeds, entropy, n_jobs)
        mean = res.importances_mean
        gain = res.baseline_score - res.null_score
        if gain > 1e-12 * max(1.0, abs(res.null_score)):
            share = mean / gain
        else:
            warnings.warn(
                "the model does not beat the training null on these data; share_of_gain is NaN", UserWarning, stacklevel=2
            )
            share = np.full(mean.shape, np.nan)
        return Bunch(
            importances=res.importances,
            importances_mean=mean,
            importances_se=res.importances_se,
            importances_window=res.importances_window,
            window_edges=res.window_edges,
            importances_cause=res.importances_cause,
            share_of_gain=share,
            baseline_score=res.baseline_score,
            null_score=res.null_score,
            fold_scores=res.fold_scores,
            n_events=res.n_events,
            n_ids=res.n_ids,
            n_folds=res.n_folds,
            feature_names=np.array(unit_names, dtype=object),
            units=units,
        )
    n_times = _strata.check_count(n_times, "n_times")
    if scoring == "ibs" and n_times < 2:
        raise ValueError(f"n_times must be an integer >= 2 with scoring='ibs', got {n_times!r}")
    res = _loco.run_landmark_loss(
        stack_template, df, data, units, cv, scoring, cause, n_seeds, entropy, n_jobs, n_times,
        censoring_estimator, g_min, competing,
    )
    return Bunch(
        importances=res.importances,
        importances_mean=res.importances_mean,
        importances_se=res.importances_se,
        importances_window=None,
        window_edges=None,
        importances_cause=None,
        share_of_gain=None,
        baseline_score=res.baseline_score,
        null_score=None,
        fold_scores=None,
        n_events=None,
        n_ids=None,
        n_folds=res.n_folds,
        feature_names=np.array(unit_names, dtype=object),
        units=units,
    )
