"""Scikit-learn compatible survival forest estimator."""

import numbers

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from . import _core
from ._validation import check_survival_y


class SurvivalForestTV(BaseEstimator):
    """Random survival forest for counting-process data.

    Walking-skeleton status (plan.md S1): a single tree on right-censored data
    with fixed covariates (``start == 0``, one row per id). Forests arrive in S2,
    time-varying covariates and delayed entry in S3.

    Parameters
    ----------
    n_estimators : int, default=1
        Number of trees. Only 1 is supported until S2.
    max_features : {"sqrt", "log2"}, int, float or None, default=None
        Features tried per node. ``None`` uses all features.
    max_depth : int or None, default=None
        ``0`` gives a single root leaf (the Nelson–Aalen estimator).
    min_ids_leaf : int, default=15
        Minimum ids per child.
    min_events_leaf : int, default=3
        Minimum events per child.
    max_bins : int, default=255
        Feature histogram bins, in [2, 256].
    random_state : int, RandomState or None, default=None
    """

    def __init__(
        self,
        n_estimators=1,
        max_features=None,
        max_depth=None,
        min_ids_leaf=15,
        min_events_leaf=3,
        max_bins=255,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_ids_leaf = min_ids_leaf
        self.min_events_leaf = min_events_leaf
        self.max_bins = max_bins
        self.random_state = random_state

    def fit(self, X, y, ids=None):
        X = check_array(X, dtype=np.float64, order="C")
        start, stop, event = check_survival_y(y)
        if X.shape[0] != start.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {start.shape[0]}")
        if self.n_estimators != 1:
            raise NotImplementedError("forests (n_estimators > 1) arrive in S2")
        if (start != 0).any():
            raise NotImplementedError("delayed entry / time-varying covariates arrive in S3")
        if ids is not None:
            ids = np.asarray(ids)
            if ids.ndim != 1 or ids.shape[0] != X.shape[0]:
                raise ValueError(f"ids must be 1-d with {X.shape[0]} entries, got shape {ids.shape}")
            if np.unique(ids).shape[0] != ids.shape[0]:
                raise NotImplementedError("multiple rows per id arrive in S3")
        self._check_int("min_ids_leaf", minimum=1)
        self._check_int("min_events_leaf", minimum=1)
        if not (isinstance(self.max_bins, numbers.Integral) and 2 <= self.max_bins <= 256):
            raise ValueError("max_bins must be an integer in [2, 256]")
        if self.max_depth is not None:
            self._check_int("max_depth", minimum=0)

        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        self.tree_ = _core.fit_tree(
            X,
            start,
            stop,
            event,
            max_depth=self.max_depth,
            min_ids_leaf=self.min_ids_leaf,
            min_events_leaf=self.min_events_leaf,
            max_features=self._resolve_max_features(X.shape[1]),
            max_bins=self.max_bins,
            seed=int(rng.randint(np.iinfo(np.int32).max)),
        )
        self.event_times_ = np.unique(stop[event])
        return self

    def predict_cumulative_hazard(self, X, times=None):
        """Cumulative hazard, shape ``(n_samples, n_times)``; ``times`` defaults to ``event_times_``."""
        X, times = self._check_predict(X, times)
        return self.tree_.predict_cumhaz(X, times)

    def predict_survival_function(self, X, times=None):
        """Survival probabilities ``exp(-H(t))``, shape ``(n_samples, n_times)``."""
        return np.exp(-self.predict_cumulative_hazard(X, times))

    def apply(self, X):
        """Leaf index reached by each row."""
        X, _ = self._check_predict(X, None)
        return self.tree_.apply(X)

    def _check_predict(self, X, times):
        check_is_fitted(self, "tree_")
        X = check_array(X, dtype=np.float64, order="C")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        times = self.event_times_ if times is None else np.asarray(times, dtype=float).ravel()
        if np.isnan(times).any():
            raise ValueError("times must not contain NaN")
        return X, np.ascontiguousarray(times)

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
