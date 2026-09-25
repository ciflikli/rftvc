"""Scikit-learn compatible survival forest estimator."""

import numbers

import numpy as np
from joblib import effective_n_jobs
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from . import _core
from ._validation import check_counting_process, check_intervals, check_survival_y


class SurvivalForestTV(BaseEstimator):
    """Random survival forest for counting-process data.

    Data are counting-process rows ``(start, stop, event, X)`` grouped by ``ids``:
    each row's covariates apply on ``(start, stop]``, rows of an id are
    contiguous, and only an id's last row may carry the event. Delayed entry
    (``start > 0`` on an id's first row) is handled as left truncation.
    Resampling, leaf sizes and OOB count ids, not rows.

    Parameters
    ----------
    n_estimators : int, default=500
    max_features : {"sqrt", "log2"}, int, float or None, default="sqrt"
        Features tried per node. ``None`` uses all features.
    max_depth : int or None, default=None
        ``0`` gives single-leaf trees (Nelson–Aalen on each tree's sample).
    min_ids_leaf : int or "auto", default=15
        Minimum ids per child. ``"auto"`` uses ``max(15, floor(sqrt(n_ids)))``.
    min_events_leaf : int, default=3
        Minimum events per child.
    max_bins : int, default=255
        Feature histogram bins, in [2, 256].
    resample_unit : {"id"}, default="id"
        Unit drawn when growing each tree. Only whole ids in v1.
    max_samples : int, float or None, default=None
        Ids drawn per tree: a fraction of ids (float) or a count (int).
        ``None`` is 0.632 without bootstrap and 1.0 with it.
    bootstrap : bool, default=False
        Draw ids with replacement (classic bootstrap). The default subsamples
        without replacement (design.md D10).
    aggregate : {"hazard", "survival"}, default="hazard"
        Ensemble rule: ``exp(-mean Λ_b)`` or ``mean exp(-Λ_b)`` (design.md D11).
    n_jobs : int or None, default=None
        Threads for fitting and prediction; ``-1`` uses all cores.
    random_state : int, RandomState or None, default=None
    """

    def __init__(
        self,
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_ids_leaf=15,
        min_events_leaf=3,
        max_bins=255,
        resample_unit="id",
        max_samples=None,
        bootstrap=False,
        aggregate="hazard",
        n_jobs=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_ids_leaf = min_ids_leaf
        self.min_events_leaf = min_events_leaf
        self.max_bins = max_bins
        self.resample_unit = resample_unit
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.aggregate = aggregate
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, ids=None, *, measured_at=None, gap_policy="error"):
        """Fit on counting-process rows.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Covariates in force on each row's ``(start, stop]``.
        y : structured array with fields ``start``, ``stop``, ``event``
        ids : array-like of shape (n_rows,), default=None
            Subject of each row; ``None`` makes every row its own subject.
        measured_at : array-like of shape (n_rows,), default=None
            When each row's covariates were measured; must be ``<= start``.
        gap_policy : {"error", "split_id"}, default="error"
            How to treat gaps between an id's rows (see ``check_counting_process``).
        """
        X = check_array(X, dtype=np.float64, order="C")
        start, stop, event = check_survival_y(y)
        n = X.shape[0]
        if n != start.shape[0]:
            raise ValueError(f"X has {n} rows but y has {start.shape[0]}")
        cp = check_counting_process(start, stop, event, ids, measured_at=measured_at, gap_policy=gap_policy)
        groups, n_ids = cp.group, cp.n_groups
        self._validate_params()

        self.n_features_in_ = X.shape[1]
        self.n_ids_ = n_ids
        self.min_ids_leaf_ = self._resolve_min_ids_leaf(n_ids)
        self.n_draw_ = self._resolve_n_draw(n_ids)
        rng = check_random_state(self.random_state)
        self.forest_ = _core.fit_forest(
            X,
            start,
            stop,
            event,
            groups,
            n_ids,
            n_trees=self.n_estimators,
            n_draw=self.n_draw_,
            bootstrap=bool(self.bootstrap),
            max_depth=self.max_depth,
            min_ids_leaf=self.min_ids_leaf_,
            min_events_leaf=self.min_events_leaf,
            max_features=self._resolve_max_features(X.shape[1]),
            max_bins=self.max_bins,
            seed=int(rng.randint(np.iinfo(np.int64).max, dtype=np.int64)),
            n_jobs=effective_n_jobs(self.n_jobs),
        )
        self.event_times_ = np.unique(stop[event])
        return self

    def predict_cumulative_hazard(self, X, times=None, *, intervals=None, ids=None, origin=None, extrapolate="none"):
        """Ensemble cumulative hazard.

        Without ``intervals``, each row of ``X`` is a subject whose covariates are
        fixed from time 0 on; the result has shape ``(n_rows, n_times)``.

        With ``intervals`` (a structured array with ``start``, ``stop``), rows are
        a covariate *path*: row ``r``'s covariates apply on ``(start_r, stop_r]``,
        grouped into subjects by ``ids`` (rows contiguous per id, as in ``fit``).
        The result has one row per subject, in order of first appearance, and is
        the conditional cumulative hazard ``Λ(t) - Λ(origin)``:

        - ``origin`` defaults to each subject's first ``start``; a scalar or one
          value per subject may be given, within ``[first start, last stop]``.
        - ``t < origin`` gives NaN.
        - ``t`` beyond the last ``stop`` gives NaN, unless ``extrapolate="locf"``:
          a named scenario in which the last row's covariates stay in force.
          Supplying the future path as extra rows is the alternative, valid for
          external covariates.

        ``times`` defaults to ``event_times_``. Under ``aggregate="survival"`` the
        result is ``-log`` of the averaged (per-tree conditional) survival.
        """
        X, times = self._check_predict(X, times)
        n_jobs = effective_n_jobs(self.n_jobs)
        if intervals is None:
            if ids is not None or origin is not None or extrapolate != "none":
                raise ValueError("ids, origin and extrapolate require intervals")
            return self.forest_.predict_cumhaz(X, times, self.aggregate, n_jobs)
        if extrapolate not in ("none", "locf"):
            raise ValueError(f"extrapolate must be 'none' or 'locf', got {extrapolate!r}")
        start, stop = check_intervals(intervals)
        if start.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but intervals has {start.shape[0]}")
        cp = check_counting_process(start, stop, None, ids)
        o = cp.order
        first_start = start[o][cp.offsets[:-1].astype(np.int64)]
        last_stop = stop[o][cp.offsets[1:].astype(np.int64) - 1]
        if origin is None:
            origin = first_start
        origin = np.broadcast_to(np.asarray(origin, dtype=float), first_start.shape).copy()
        bad = ~np.isfinite(origin) | (origin < first_start) | (origin > last_stop)
        if bad.any():
            raise ValueError("origin must lie within each subject's [first start, last stop]")
        return self.forest_.predict_paths(
            np.ascontiguousarray(X[o]),
            np.ascontiguousarray(start[o]),
            np.ascontiguousarray(stop[o]),
            cp.offsets,
            origin,
            times,
            self.aggregate,
            extrapolate,
            n_jobs,
        )

    def predict_survival_function(self, X, times=None, **path_kwargs):
        """Survival probabilities ``exp(-H)``; see ``predict_cumulative_hazard``."""
        return np.exp(-self.predict_cumulative_hazard(X, times, **path_kwargs))

    def predict_risk(self, X, horizon, **path_kwargs):
        """Event probability by ``horizon``: ``1 - S(horizon)``, one value per subject.

        With ``intervals``, this is ``P(T <= horizon | T > origin, path)``.
        """
        return 1.0 - self.predict_survival_function(X, [horizon], **path_kwargs)[:, 0]

    def apply(self, X):
        """Leaf index per (row, tree), shape ``(n_samples, n_estimators)``."""
        X, _ = self._check_predict(X, None)
        return self.forest_.apply(X, effective_n_jobs(self.n_jobs))

    def _check_predict(self, X, times):
        check_is_fitted(self, "forest_")
        X = check_array(X, dtype=np.float64, order="C")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        times = self.event_times_ if times is None else np.asarray(times, dtype=float).ravel()
        if np.isnan(times).any():
            raise ValueError("times must not contain NaN")
        return X, np.ascontiguousarray(times)

    def _validate_params(self):
        if self.resample_unit != "id":
            raise ValueError(f"resample_unit must be 'id' in v1, got {self.resample_unit!r}")
        if self.aggregate not in ("hazard", "survival"):
            raise ValueError(f"aggregate must be 'hazard' or 'survival', got {self.aggregate!r}")
        self._check_int("n_estimators", minimum=1)
        self._check_int("min_events_leaf", minimum=1)
        if self.min_ids_leaf != "auto":
            self._check_int("min_ids_leaf", minimum=1)
        if not (isinstance(self.max_bins, numbers.Integral) and 2 <= self.max_bins <= 256):
            raise ValueError("max_bins must be an integer in [2, 256]")
        if self.max_depth is not None:
            self._check_int("max_depth", minimum=0)

    def _resolve_min_ids_leaf(self, n_ids):
        if self.min_ids_leaf == "auto":
            return max(15, int(np.floor(np.sqrt(n_ids))))
        return int(self.min_ids_leaf)

    def _resolve_n_draw(self, n_ids):
        ms = self.max_samples
        if ms is None:
            ms = 1.0 if self.bootstrap else 0.632
        if isinstance(ms, (bool, np.bool_)):
            raise ValueError(f"invalid max_samples={ms!r}: use an int count or a float fraction")
        if isinstance(ms, numbers.Integral):
            if ms < 1 or (ms > n_ids and not self.bootstrap):
                raise ValueError(f"max_samples={ms} must be in [1, n_ids={n_ids}] without bootstrap")
            return int(ms)
        if isinstance(ms, numbers.Real) and 0 < ms <= 1:
            return max(1, int(round(ms * n_ids)))
        raise ValueError(f"invalid max_samples={ms!r}")

    def _resolve_max_features(self, p):
        mf = self.max_features
        if mf is None:
            return p
        if mf == "sqrt":
            return max(1, int(np.sqrt(p)))
        if mf == "log2":
            return max(1, int(np.log2(p)))
        if isinstance(mf, numbers.Integral) and 1 <= mf:
            return min(int(mf), p)
        if isinstance(mf, numbers.Real) and 0 < mf <= 1:
            return max(1, int(mf * p))
        raise ValueError(f"invalid max_features={mf!r}")

    def _check_int(self, name, minimum):
        value = getattr(self, name)
        if not isinstance(value, numbers.Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
