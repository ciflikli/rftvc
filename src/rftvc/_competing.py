"""Scikit-learn compatible competing-risks forest (cause-specific hazards, Aalen–Johansen CIF)."""

import numbers

import numpy as np
from joblib import effective_n_jobs

from ._estimator import _BaseForestTV
from ._validation import check_competing_risks_y

_S12 = "is not supported yet by CompetingRisksForestTV (planned: S12)"


class CompetingRisksForestTV(_BaseForestTV):
    """Random forest for competing risks on counting-process data.

    Rows are ``(start, stop, event, X)`` as in ``SurvivalForestTV``, but
    ``event`` holds a cause label (0 = censored; see
    ``rftvc.make_competing_risks_y``). The trees are shared by all causes; each
    leaf stores the delayed-entry Nelson–Aalen cumulative hazard of every cause.
    The cumulative incidence ``F_k`` and the event-free survival ``S`` come from
    the discrete Aalen–Johansen estimator on the tree-averaged cause-specific
    hazard increments:

    ``S(t) = prod_{v <= t} (1 - sum_j dL_j(v))``, ``F_k(t) = sum_{v <= t} S(v-) dL_k(v)``,

    so ``sum_k F_k + S = 1``. Like the survival forest, predictions for a
    covariate path are valid for external (or specified) covariates only.

    Parameters
    ----------
    n_estimators, max_features, max_depth, min_ids_leaf, max_bins, max_samples, bootstrap, n_jobs, random_state
        As in ``SurvivalForestTV``.
    min_events_leaf : int, default=3
        Minimum events of any cause per child.
    causes : array-like of int or None, default=None
        Cause label vocabulary. ``None`` uses the labels observed in ``y``.
        Give it in cross-validation, so that every fold has the same outputs: a
        cause without events in a fold then has zero hazard (with a ``UserWarning``).
        A label in ``y`` outside ``causes`` is an error.
    criterion : {"composite"}, default="composite"
        Split rule: the sum over causes of the per-cause LTRC log-rank
        chi-square statistics, which detects a covariate that raises one cause
        and lowers another (the all-cause log-rank does not). With one cause it
        is the survival forest's log-rank.
    split_cause : int or None, default=None
        A cause label: split on that cause's log-rank alone, other causes
        counting as censoring (cause-specific forest). Overrides ``criterion``.
    score_cause : int or None, default=None
        Cause label used by ``predict``; ``None`` is the first label of ``causes_``.
    aggregate : {"hazard"}, default="hazard"
        Average the cause-specific cumulative hazards over trees, then apply
        Aalen–Johansen. (``"cif"``, per-tree Aalen–Johansen then averaging, comes in S12.)
    ntime, resample_unit, block_length, oob_buffer, oob_score
        Reserved: only the defaults (exact grid, id resampling, no OOB) are
        supported so far; other values raise ``NotImplementedError``.

    Attributes
    ----------
    causes_ : ndarray of int
        Cause labels, sorted; the order of the cause axis of every output.
    n_causes_ : int
    event_times_ : ndarray
        Distinct times with an event of any cause.
    n_ids_, n_units_, forest_
        As in ``SurvivalForestTV``.
    """

    _AGGREGATES = ("hazard", "cif")

    def __init__(
        self,
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_ids_leaf=15,
        min_events_leaf=3,
        max_bins=255,
        ntime=None,
        resample_unit="id",
        block_length=None,
        oob_buffer=1,
        max_samples=None,
        bootstrap=False,
        aggregate="hazard",
        oob_score=False,
        n_jobs=None,
        random_state=None,
        causes=None,
        criterion="composite",
        split_cause=None,
        score_cause=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_ids_leaf = min_ids_leaf
        self.min_events_leaf = min_events_leaf
        self.max_bins = max_bins
        self.ntime = ntime
        self.resample_unit = resample_unit
        self.block_length = block_length
        self.oob_buffer = oob_buffer
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.aggregate = aggregate
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.causes = causes
        self.criterion = criterion
        self.split_cause = split_cause
        self.score_cause = score_cause

    def fit(self, X, y, ids=None, *, measured_at=None, gap_policy="error", layout="counting_process"):
        """Fit on counting-process rows with cause-coded events.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_rows, n_features)
        y : structured array or DataFrame with ``start``, ``stop``, ``event``
            ``event`` is a cause label (0 = censored); see ``make_competing_risks_y``.
        ids, measured_at, gap_policy, layout
            As in ``SurvivalForestTV.fit``. Only an id's last row may carry an
            event, whatever its cause.
        """
        return self._fit(X, y, ids, measured_at, gap_policy, layout, None)

    def _check_y(self, y):
        start, stop, codes, causes = check_competing_risks_y(y, self.causes)
        self.causes_ = causes
        self.n_causes_ = int(causes.size)
        # Labels are checked here, before the (possibly long) fit.
        self._split_code = None if self.split_cause is None else self._cause_index(self.split_cause, "split_cause") + 1
        if self.score_cause is not None:
            self._cause_index(self.score_cause, "score_cause")
        return start, stop, codes

    def _engine_kwargs(self):
        return {"n_causes": self.n_causes_, "split_cause": self._split_code}

    def _validate_params(self):
        # S12 options first, so they raise NotImplementedError whatever else is set.
        reserved = [
            ("ntime", None),
            ("resample_unit", "id"),
            ("block_length", None),
            ("oob_buffer", 1),
            ("oob_score", False),
            ("aggregate", "cif"),
        ]
        for name, default in reserved:
            value = getattr(self, name)
            unsupported = value == "cif" if name == "aggregate" else value != default
            if unsupported:
                raise NotImplementedError(f"{name}={value!r} {_S12}")
        super()._validate_params()
        if self.criterion != "composite":
            raise ValueError(f"criterion must be 'composite', got {self.criterion!r}")

    def _cause_index(self, cause, name="cause"):
        """Position of a cause label in ``causes_``."""
        ok = isinstance(cause, numbers.Integral) and not isinstance(cause, (bool, np.bool_))
        pos = np.flatnonzero(self.causes_ == cause) if ok else []
        if len(pos) != 1:
            raise ValueError(f"{name}={cause!r} is not a fitted cause label; causes_ = {self.causes_.tolist()}")
        return int(pos[0])

    def _check_times(self, X, times, intervals, ids, origin, extrapolate):
        if intervals is not None or ids is not None or origin is not None or extrapolate != "none":
            raise NotImplementedError(f"covariate paths (intervals, ids, origin, extrapolate) {_S12}")
        X, times, _ = self._check_predict(X, times)
        return X, times

    def predict_cumulative_incidence(
        self, X, times=None, *, cause=None, intervals=None, ids=None, origin=None, extrapolate="none"
    ):
        """Cumulative incidence ``F_k(t | x)`` of each row, covariates fixed from time 0.

        Returns shape ``(n_rows, n_causes_, n_times)`` with causes in ``causes_``
        order, or ``(n_rows, n_times)`` for one ``cause`` label. ``times``
        defaults to ``event_times_``.
        """
        X, times = self._check_times(X, times, intervals, ids, origin, extrapolate)
        k = None if cause is None else self._cause_index(cause)
        cif, _, _ = self.forest_.predict_cif(X, times, effective_n_jobs(self.n_jobs))
        return cif if k is None else cif[:, k, :]

    def predict_cumulative_hazard(self, X, times=None, *, cause=None):
        """Ensemble cause-specific cumulative hazards, covariates fixed from time 0.

        Returns ``(n_rows, n_causes_, n_times)``; a ``cause`` label gives
        ``(n_rows, n_times)``, and ``cause="all"`` the all-cause hazard (sum over causes).
        """
        X, times, _ = self._check_predict(X, times)
        all_causes = isinstance(cause, str) and cause == "all"
        k = None if cause is None or all_causes else self._cause_index(cause)
        H = self.forest_.predict_cause_cumhaz(X, times, effective_n_jobs(self.n_jobs))
        if all_causes:
            return H.sum(axis=1)
        return H if k is None else H[:, k, :]

    def predict_survival_function(self, X, times=None):
        """Event-free survival ``S(t | x)`` (no event of any cause), shape ``(n_rows, n_times)``.

        This is the Aalen–Johansen product limit ``prod (1 - sum_j dL_j)``,
        so ``S + sum_k F_k = 1``; it is not ``exp(-L)`` as in ``SurvivalForestTV``.
        """
        X, times, _ = self._check_predict(X, times)
        _, surv, _ = self.forest_.predict_cif(X, times, effective_n_jobs(self.n_jobs))
        return surv

    def predict(self, X):
        """Risk score per row: ``F_k`` of ``score_cause`` at the last event time (higher = higher risk)."""
        X, _, _ = self._check_predict(X, None)
        k = 0 if self.score_cause is None else self._cause_index(self.score_cause, "score_cause")
        cif, _, _ = self.forest_.predict_cif(X, self.event_times_[-1:], effective_n_jobs(self.n_jobs))
        return cif[:, k, 0]

    def score(self, X, y, ids=None):
        """Cause-specific concordance (Wolbers); not available yet."""
        raise NotImplementedError(f"score {_S12}")
