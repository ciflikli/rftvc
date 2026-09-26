"""Time-aware cross-validation for survival models.

The error estimate is an explicit choice (design.md):

- **new subjects**: group splits (``sklearn.model_selection.GroupKFold`` with
  ``groups=ids``) or the forest's id-level OOB score;
- **future periods**: ``RollingOriginSplit``, which trains on earlier times
  and tests on later ones separated by a ``gap``;
- **new subjects in future periods**: ``GroupTimeSplit``.

``landmark_cross_validate`` runs these splitters on the landmark grid of a
``LandmarkSurvivalForest`` (optionally nested, for tuning).
"""

import itertools
import numbers

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from ._validation import _as_labels, _check_causes
from .landmark import LandmarkCompetingRisksForest, _as_polars, make_landmark_data
from .metrics import (
    KaplanMeierCensoring,
    UndefinedMetricError,
    _classes,
    brier_landmark,
    cindex_dynamic,
    integrated_brier,
)

__all__ = ["GroupTimeSplit", "RollingOriginSplit", "landmark_cross_validate"]


class RollingOriginSplit:
    """Rolling-origin (forward-chaining) splits on a time axis.

    Test windows of width ``test_size`` are stacked back from the last time:
    fold ``k`` of ``n_splits`` tests the times in
    ``(end - (n_splits - k) * test_size, end - (n_splits - k - 1) * test_size]``.
    Training takes every time ``<= min(test times) - gap``, so train and test
    times are at least ``gap`` apart. With landmark data, ``gap >= horizon``
    keeps every training outcome window ``(s, s + horizon]`` before the test
    landmarks (checked by ``landmark_cross_validate``).

    Parameters
    ----------
    n_splits : int, default=5
    test_size : float
        Width of each test window, in time units.
    gap : float, default=0
        Minimum distance between training and test times, in time units.
    time_col : str or None
        Column of ``X`` (a DataFrame) holding the times; ``None`` means ``X``
        itself is a 1-d array of times.
    """

    def __init__(self, n_splits=5, *, test_size, gap=0.0, time_col=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _check_params(self):
        if not (isinstance(self.n_splits, numbers.Integral) and self.n_splits >= 1):
            raise ValueError("n_splits must be an integer >= 1")
        if not self.test_size > 0:
            raise ValueError("test_size must be positive")
        if not self.gap >= 0:
            raise ValueError("gap must be non-negative")

    def _times(self, X):
        if self.time_col is not None:
            t = np.asarray(_as_polars(X)[self.time_col].to_numpy(), dtype=float)
        else:
            t = np.asarray(X, dtype=float)
            if t.ndim == 2 and t.shape[1] == 1:
                t = t[:, 0]
            if t.ndim != 1:
                raise ValueError("X must be a 1-d array of times, or pass time_col")
        if np.isnan(t).any():
            raise ValueError("times must not contain NaN")
        return t

    def _windows(self, t):
        """``(test_mask, cutoff)`` per fold; training is ``t <= cutoff``."""
        self._check_params()
        end = t.max()
        for k in range(self.n_splits):
            hi = end - (self.n_splits - 1 - k) * self.test_size
            test = (t > hi - self.test_size) & (t <= hi)
            if not test.any():
                raise ValueError(f"fold {k}: no times in test window ({hi - self.test_size}, {hi}]")
            yield test, t[test].min() - self.gap

    def split(self, X, y=None, groups=None):
        t = self._times(X)
        for k, (test, cutoff) in enumerate(self._windows(t)):
            train = t <= cutoff
            if not train.any():
                raise ValueError(f"fold {k}: no training times <= {cutoff}; use fewer splits or a smaller gap")
            yield np.flatnonzero(train), np.flatnonzero(test)


class GroupTimeSplit(RollingOriginSplit):
    """New groups in future periods: group folds crossed with rolling time windows.

    Groups are partitioned into ``n_splits`` folds (``GroupKFold``). Fold ``k``
    tests the rows of group fold ``k`` inside time window ``k`` (as in
    ``RollingOriginSplit``) and trains on the rows of all *other* groups with
    times ``<= min(test times) - gap``. No group is on both sides.
    """

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("GroupTimeSplit needs groups")
        t = self._times(X)
        groups = np.asarray(groups)
        if groups.shape[0] != t.shape[0]:
            raise ValueError("groups and X have different lengths")
        group_folds = [test for _, test in GroupKFold(self.n_splits).split(t, groups=groups)]
        for k, (window, _) in enumerate(self._windows(t)):
            in_fold = np.zeros(t.size, dtype=bool)
            in_fold[group_folds[k]] = True
            test = window & in_fold
            if not test.any():
                raise ValueError(f"fold {k}: no rows of the fold's groups in its test window")
            cutoff = t[test].min() - self.gap
            train = (t <= cutoff) & ~np.isin(groups, groups[test])
            if not train.any():
                raise ValueError(f"fold {k}: no training rows; use fewer splits or a smaller gap")
            yield np.flatnonzero(train), np.flatnonzero(test)


# name -> better direction
_SCORERS = {
    "brier": "min",
    "integrated_brier": "min",
    "cindex_cumulative": "max",
    "cindex_incident": "max",
}


def _censor_at(df, cutoff, *, start, stop, event):
    """Administrative censoring at ``cutoff``: the rows as seen at calendar ``cutoff``.

    An event (or cause label) after ``cutoff`` becomes censoring (0 / False);
    the column keeps its dtype.
    """
    dtype = df.schema[event]
    return df.filter(pl.col(start) < cutoff).with_columns(
        pl.when(pl.col(stop) <= cutoff).then(pl.col(event)).otherwise(pl.lit(0).cast(dtype)).cast(dtype).alias(event),
        pl.min_horizontal(pl.col(stop).cast(pl.Float64), pl.lit(float(cutoff))).alias(stop),
    )


def _landmark_data(model, df, landmarks=None):
    return make_landmark_data(
        df,
        horizon=model.horizon,
        history_features=model.history_features,
        landmarks=model.landmarks if landmarks is None else landmarks,
        step=model.step if landmarks is None else None,
        id=model.id,
        start=model.start,
        stop=model.stop,
        event=model.event,
        measured_at=model.measured_at,
    )


def _score_landmark(curve, y, w, scoring, times, censoring_estimator, g_min, cause=None):
    """Scores of one landmark's test risk set, with ``G_s`` fitted on its outcomes.

    ``curve`` is the predicted survival ``S`` at ``times`` (last column at
    ``w``), or with ``cause=k`` the cumulative incidence ``F_k`` at ``times``.
    """
    stop = y["stop"]
    event = y["event"] if cause is None else _as_labels(y["event"])
    need = any(not np.all(np.logical_or.reduce(_classes(stop, event, t, cause))) for t in times)
    cens = None
    if need:
        cens = KaplanMeierCensoring() if censoring_estimator is None else clone(censoring_estimator)
        cens = cens.fit(y)
    risk = 1.0 - curve[:, -1] if cause is None else curve[:, -1]
    kw = dict(censoring_estimator=cens, g_min=g_min)
    if cause is not None:
        kw["cause"] = cause
    out = {}
    brier, info = brier_landmark(y, risk, w, return_info=True, **kw)
    out.update(n=info["n"], n_cases=info["n_cases"], n_censored=info["n_censored"], n_clipped=info["n_clipped"])
    for name in scoring:
        try:
            if name == "brier":
                out[name] = brier
            elif name == "integrated_brier":
                out[name] = integrated_brier(y, curve, times, **kw)
            elif name == "cindex_cumulative":
                out[name] = cindex_dynamic(y, risk, w, kind="cumulative", **kw)
            elif name == "cindex_incident":
                out[name] = cindex_dynamic(y, risk, w, kind="incident", **kw)
            else:
                out[name] = float(scoring[name](y, risk, w, **kw))
        except UndefinedMetricError:  # e.g. no cases at this landmark; configuration errors propagate
            out[name] = np.nan
    return out


def landmark_cross_validate(
    model,
    df,
    cv,
    scoring=("brier",),
    *,
    horizon=None,
    n_times=10,
    censoring_estimator=None,
    g_min=0.05,
    param_grid=None,
    inner_cv=None,
    refit="brier",
    return_predictions=False,
):
    """Cross-validate a ``LandmarkSurvivalForest`` or ``LandmarkCompetingRisksForest`` over its landmark grid.

    The splitter runs on the stacked landmark rows with the landmark ``s`` as
    the time (and ids as groups). For each fold:

    1. The training frame is the training ids' rows. With a time splitter
       (``RollingOriginSplit``, ``GroupTimeSplit``) it is administratively
       censored at the earliest test landmark, so the fit never sees later
       data, and ``cv.gap >= model.horizon`` is required: training outcomes on
       ``(s, s + horizon]`` then end before every test landmark. Any other
       splitter must keep ids disjoint (new-subject CV, e.g. ``GroupKFold``).
    2. A clone of ``model`` is fitted with ``landmarks`` = the training landmarks.
    3. Each test landmark is scored on its risk set at horizon ``w``
       (``horizon``, default the model's). IPCW uses ``G_s``, a censoring model
       fitted on that landmark's test outcomes only (reverse Kaplan–Meier, or a
       clone of ``censoring_estimator``); it never sees predictions.

    With ``param_grid`` (dict or list of dicts of ``model`` parameters) and
    ``inner_cv``, each outer fold first runs this procedure on its own training
    frame for every candidate and refits the one with the best mean ``refit``
    score (nested CV).

    **Competing risks** (a ``LandmarkCompetingRisksForest``): one cause
    vocabulary (``model.causes``, else the labels in the whole data) and one
    scored cause ``k`` (``model.score_cause``, else the first label) are fixed
    before splitting and set on every fit, so each fold has the same outputs.
    Scores use ``F_k`` with ``cause=k``: the cause-specific IPCW Brier,
    integrated Brier (``integrated_brier`` of ``F_k``) and Wolbers C
    (``cindex_incident``); ``cindex_cumulative`` (competing-risks AUC) is not
    available. Custom scorers receive ``cause=k`` as a keyword.

    Parameters
    ----------
    scoring : sequence of str or dict
        Names among ``"brier"``, ``"integrated_brier"`` (over ``n_times`` equally
        spaced times in ``(0, w]``), ``"cindex_cumulative"``, ``"cindex_incident"``,
        or a dict ``{name: f(y, risk, w, *, censoring_estimator, g_min)}``. A score
        that raises ``metrics.UndefinedMetricError`` at a landmark is NaN there;
        any other error propagates.

    return_predictions : bool, default=False
        Also return the out-of-fold predictions the scores were computed from.

    Returns
    -------
    scores : polars.DataFrame
        One row per (fold, test landmark): ``fold``, ``landmark``, counts
        (``n``, ``n_cases``, ``n_censored``, ``n_clipped``), one column per score,
        and ``params`` (the selected parameters) under nested CV.
    predictions : polars.DataFrame
        Only with ``return_predictions=True``. One row per scored test row (a
        subject in a test landmark's risk set): ``fold``, ``landmark``, ``id``,
        ``time`` and ``event`` (the outcome on the reset clock, time since the
        landmark), ``risk`` (``1 - S(w)``), and ``survival``: a list of ``S`` at
        the ``n_times`` times ``w/n_times, ..., w`` (the ``integrated_brier`` grid).
        For competing risks: ``cause``, ``risk`` (``F_k(w)``) and ``cif`` (a list
        of ``F_k`` at those times) instead of ``survival``.
    """
    df = _as_polars(df)
    if model.horizon is None:
        raise ValueError("model.horizon is required")
    w = model.horizon if horizon is None else horizon
    if not 0 < w <= model.horizon:
        raise ValueError(f"horizon must lie in (0, model.horizon={model.horizon}]")
    time_split = isinstance(cv, RollingOriginSplit)
    if time_split and cv.gap < model.horizon:
        raise ValueError(
            f"cv.gap={cv.gap} must be >= model.horizon={model.horizon}: otherwise training outcome "
            "windows overlap the test landmarks"
        )
    if isinstance(scoring, str):
        scoring = (scoring,)
    if not isinstance(scoring, dict):
        unknown = set(scoring) - set(_SCORERS)
        if unknown:
            raise ValueError(f"unknown scoring {sorted(unknown)}; use {sorted(_SCORERS)} or a dict of callables")
    candidates = _candidates(param_grid, inner_cv)
    if candidates is not None and refit not in scoring:
        raise ValueError("refit must be one of the scoring names")

    if not isinstance(n_times, numbers.Integral) or n_times < (2 if "integrated_brier" in scoring else 1):
        raise ValueError("n_times must be an integer >= 1 (>= 2 with integrated_brier)")
    times = np.linspace(0, w, n_times + 1)[1:]
    data = _landmark_data(model, df)
    ids = data.ids
    cause = None
    if isinstance(model, LandmarkCompetingRisksForest):
        if "cindex_cumulative" in scoring:
            raise ValueError("cindex_cumulative (competing-risks AUC) is not available for competing risks")
        # One vocabulary and one scored cause for every (outer and inner) fit.
        labels = _as_labels(data.y["event"])
        causes = np.unique(labels[labels != 0]) if model.causes is None else _check_causes(model.causes)
        k = model.score_cause
        if k is None:
            k = causes[0]
        elif not (isinstance(k, numbers.Integral) and not isinstance(k, (bool, np.bool_)) and k in causes):
            raise ValueError(f"score_cause={k!r} is not a label in causes={causes.tolist()}")
        cause = int(k)
        model = clone(model).set_params(causes=causes.tolist(), score_cause=cause)
    rows, preds = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(data.s, groups=data.groups)):
        test_s = np.unique(data.s[test_idx])
        train_ids = np.unique(ids[train_idx])
        df_train = df.filter(pl.col(model.id).is_in(train_ids))
        if time_split:
            df_train = _censor_at(df_train, test_s.min(), start=model.start, stop=model.stop, event=model.event)
        elif np.isin(ids[test_idx], train_ids).any():
            raise ValueError(
                "cv must keep ids disjoint (e.g. GroupKFold with groups=ids) or be a "
                "RollingOriginSplit/GroupTimeSplit"
            )
        train_s = np.unique(data.s[train_idx])
        params = None
        if candidates is not None:
            params = _select(model, df_train, train_s, candidates, inner_cv, scoring, refit, w, n_times,
                             censoring_estimator, g_min)
        fitted = clone(model).set_params(landmarks=train_s, step=None, **(params or {})).fit(df_train)
        for s in test_s:
            m = test_idx[data.s[test_idx] == s]
            if cause is None:
                curve = fitted.forest_.predict_survival_function(data.X[m], times)
            else:
                curve = fitted.forest_.predict_cumulative_incidence(data.X[m], times, cause=cause)
            scores = _score_landmark(curve, data.y[m], w, scoring, times, censoring_estimator, g_min, cause)
            if return_predictions:
                base = {
                    "fold": np.full(len(m), fold),
                    "landmark": np.full(len(m), float(s)),
                    "id": ids[m],
                    "time": data.y["stop"][m],
                    "event": data.y["event"][m],
                }
                if cause is None:
                    base.update(risk=1.0 - curve[:, -1], survival=list(curve))
                else:
                    base.update(cause=np.full(len(m), cause), risk=curve[:, -1], cif=list(curve))
                preds.append(pl.DataFrame(base))
            row = {"fold": fold, "landmark": float(s), **scores}
            if candidates is not None:
                row["params"] = repr(params)
            rows.append(row)
    if return_predictions:
        return pl.DataFrame(rows), pl.concat(preds)
    return pl.DataFrame(rows)


def _candidates(param_grid, inner_cv):
    if param_grid is None and inner_cv is None:
        return None
    if param_grid is None or inner_cv is None:
        raise ValueError("nested CV needs both param_grid and inner_cv")
    grids = [param_grid] if isinstance(param_grid, dict) else list(param_grid)
    out = []
    for grid in grids:
        keys = sorted(grid)
        out += [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    if not out:
        raise ValueError("param_grid is empty")
    return out


def _select(model, df_train, train_s, candidates, inner_cv, scoring, refit, w, n_times, censoring_estimator, g_min):
    best, best_score = None, None
    sign = -1.0 if _SCORERS.get(refit, "max") == "min" else 1.0
    for params in candidates:
        inner = clone(model).set_params(landmarks=train_s, step=None, **params)
        res = landmark_cross_validate(
            inner, df_train, inner_cv, scoring,
            horizon=w, n_times=n_times, censoring_estimator=censoring_estimator, g_min=g_min,
        )
        score = sign * float(np.nanmean(res[refit].to_numpy()))
        if best_score is None or score > best_score:
            best, best_score = params, score
    return best
