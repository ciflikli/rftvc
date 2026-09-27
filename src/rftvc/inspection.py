"""Model inspection for the survival forests: permutation and drop-column (LOCO) importance."""

import numbers
import warnings

import numpy as np
from joblib import effective_n_jobs
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._competing import CompetingRisksForestTV
from ._estimator import _BaseForestTV
from ._inspection import _loco, _score, _strata, _units
from ._validation import check_survival_y, competing_risks_labels, make_competing_risks_y, make_survival_y, split_frame
from .metrics import _baseline_at, _check_windows, event_windows

__all__ = ["drop_column_importance", "permutation_importance"]


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
    random_state : int, RandomState, Generator or None
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

    Parameters
    ----------
    estimator : SurvivalForestTV or CompetingRisksForestTV
        A template estimator (fitted or not); every fold clones and refits it.
    X : array-like or DataFrame of shape (n_rows, n_features)
        Counting-process rows.
    y : survival or competing-risks target of the rows (``start``, ``stop``, ``event``).
    ids : array-like of shape (n_rows,) or str, default=None
        Subject of each row (or a column of a DataFrame ``X``). ``None`` makes every row its
        own subject.
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
    random_state : int, RandomState, Generator or None
        Seeds the noise column and every fold/seed's fit, independently of ``n_jobs``.
    n_jobs : int, default=None
        Unlike ``permutation_importance`` (which parallelises only inside the engine's own
        prediction call, since unit-level parallelism on top of engine threads would
        oversubscribe), LOCO's cost is dominated by independent refits: ``n_jobs`` here
        parallelises over the ``(p_units + 1) * n_folds * n_seeds`` fits themselves
        (``joblib.Parallel(prefer="threads")``, since the Rust fit releases the GIL). Each
        clone keeps its own ``n_jobs`` (inherited from ``estimator``); combining a large
        ``n_jobs`` here with a multi-threaded template estimator oversubscribes.

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
    """
    is_landmark, competing = _family(estimator, fn="drop_column_importance", fitted=False)
    if is_landmark:
        raise NotImplementedError("drop_column_importance for landmark models lands in S19 T5")
    if y is None:
        raise ValueError("y is required for counting-process estimators")
    if scoring != "pe":
        raise ValueError(
            f"scoring must be 'pe' for counting-process estimators (brier/ibs: landmark models), got {scoring!r}"
        )
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
