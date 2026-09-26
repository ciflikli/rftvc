"""Scikit-learn compatible competing-risks forest (cause-specific hazards, Aalen–Johansen CIF)."""

import numbers
import warnings

import numpy as np
import polars as pl
from joblib import effective_n_jobs

from ._estimator import _BaseForestTV
from ._validation import check_competing_risks_y, competing_risks_labels, make_competing_risks_y


# The S14 bake-off challengers ("quadratic", "ishwaran", "logrank_all") lost and were removed.
_CRITERIA = ("composite",)


class CompetingRisksForestTV(_BaseForestTV):
    """Random forest for competing risks on counting-process data.

    Rows are ``(start, stop, event, X)`` as in ``SurvivalForestTV``, but
    ``event`` holds a cause label (0 = censored; see
    ``rftvc.make_competing_risks_y``). The trees are shared by all causes; each
    leaf stores the delayed-entry Nelson–Aalen cumulative hazard of every cause.
    The cumulative incidence ``F_k`` and the event-free survival ``S`` come from
    the discrete Aalen–Johansen estimator on the cause-specific hazard increments:

    ``S(t) = prod_{v <= t} (1 - sum_j dL_j(v))``, ``F_k(t) = sum_{v <= t} S(v-) dL_k(v)``,

    so ``sum_k F_k + S = 1``. Along a covariate path (``intervals``) each row's
    covariates apply on its ``(start, stop]``; as for the survival forest, this
    is valid for external (or specified) covariates only. For dynamic
    prediction with internal covariates, use landmarking.

    Parameters
    ----------
    n_estimators, max_features, max_depth, min_ids_leaf, max_bins, ntime, resample_unit, block_length, oob_buffer, \
max_samples, bootstrap, n_jobs, random_state
        As in ``SurvivalForestTV``; ``ntime`` coarsening moves each event's cause
        label with it.
    min_events_leaf : int, default=3
        Minimum events of any cause per child.
    aggregate : {"cif", "hazard"}, default="cif"
        ``"cif"``: Aalen–Johansen per tree, then average ``F`` and ``S``.
        ``"hazard"``: average the cause-specific hazard increments over trees,
        then apply Aalen–Johansen. Cumulative hazards are the tree average under
        both. The default comes from the S14 bake-off: ``"cif"`` was better with
        opposing cause effects and not worse elsewhere (``docs/bench/s14-cr.md``).
    oob_score : bool, default=False
        Compute ``oob_prediction_`` and ``oob_score_`` (see ``SurvivalForestTV``
        for what each resampling unit's OOB estimates).
    causes : array-like of int or None, default=None
        Cause label vocabulary. ``None`` uses the labels observed in ``y``.
        Give it in cross-validation, so that every fold has the same outputs: a
        cause without events in a fold then has zero hazard (with a ``UserWarning``).
        A label in ``y`` outside ``causes`` is an error.
    criterion : {"composite"}, default="composite"
        Split rule: the sum over causes of the per-cause LTRC log-rank
        chi-square statistics, which detects a covariate that raises one cause
        and lowers another (the all-cause log-rank, and the equal-weight
        composite of Ishwaran et al. 2014, do not). With one cause it is the
        survival forest's log-rank. Chosen in the S14 bake-off.
    split_cause : int or None, default=None
        A cause label: split on that cause's log-rank alone, other causes
        counting as censoring (cause-specific forest). Overrides ``criterion``.
    min_events_leaf_cause : int or None, default=None
        With ``split_cause``: minimum events of that cause per child (a node
        needs twice as many to be split). A tree whose sample has fewer stays a
        single leaf.
    score_cause : int or None, default=None
        Cause label used by ``predict``, ``score`` and ``oob_score_``; ``None``
        is the first label of ``causes_``.

    Attributes
    ----------
    causes_ : ndarray of int
        Cause labels, sorted; the order of the cause axis of every output.
    n_causes_ : int
    event_times_ : ndarray
        Distinct times with an event of any cause (the coarse grid with ``ntime``).
    oob_prediction_ : ndarray of shape (n_rows, ``n_causes_``)
        Out-of-bag ``F_k`` at the last event time per training row, columns in
        ``causes_`` order (NaN rows: no qualifying tree, or dropped by coarsening).
        Only with ``oob_score=True``.
    oob_n_trees_ : ndarray of shape (n_rows,)
    oob_score_ : float
        ``metrics.concordance_index_cr`` of the ``score_cause`` column.
    n_ids_, n_units_, coarse_grid_, n_coarsen_dropped_rows_, n_coarsen_lost_events_, forest_
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
        aggregate="cif",
        oob_score=False,
        n_jobs=None,
        random_state=None,
        causes=None,
        criterion="composite",
        split_cause=None,
        min_events_leaf_cause=None,
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
        self.min_events_leaf_cause = min_events_leaf_cause
        self.score_cause = score_cause

    def fit(
        self, X, y, ids=None, *, measured_at=None, gap_policy="error", layout="counting_process", block_time=None
    ):
        """Fit on counting-process rows with cause-coded events.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_rows, n_features)
        y : structured array or DataFrame with ``start``, ``stop``, ``event``
            ``event`` is a cause label (0 = censored); see ``make_competing_risks_y``.
        ids, measured_at, gap_policy, layout, block_time
            As in ``SurvivalForestTV.fit``. Only an id's last row may carry an
            event, whatever its cause.
        """
        return self._fit(X, y, ids, measured_at, gap_policy, layout, block_time)

    # --- fitting hooks ---------------------------------------------------------

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
        return {
            "n_causes": self.n_causes_,
            "split_cause": self._split_code,
            "min_events_leaf_cause": self.min_events_leaf_cause,
            "leaf_events": True,
            "criterion": self.criterion,
        }

    def _validate_params(self):
        super()._validate_params()
        if self.criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {self.criterion!r}")
        if self.min_events_leaf_cause is not None:
            self._check_int("min_events_leaf_cause", minimum=1)
            if self.split_cause is None:
                raise ValueError("min_events_leaf_cause requires split_cause")

    def _oob_target(self, stop, event, start):
        """The coarsened target with cause labels (codes mapped back through ``causes_``)."""
        labels = np.where(event > 0, self.causes_[np.maximum(event.astype(np.int64), 1) - 1], 0)
        return make_competing_risks_y(stop, labels, start=start)

    def _compute_oob(self, X, y, groups, oob_set):
        """OOB ``F_j`` at the last event time; sets ``oob_prediction_``, ``oob_n_trees_``, ``oob_score_``."""
        from .metrics import concordance_index_cr

        cif, n_trees = self.forest_.oob_cif(
            X, *oob_set, self.event_times_[-1:], self.aggregate, effective_n_jobs(self.n_jobs)
        )
        pred = cif[:, :, 0]
        self.oob_prediction_ = pred
        self.oob_n_trees_ = n_trees
        ok = np.isfinite(pred).all(axis=1)
        unit = "id" if self.resample_unit == "id" else "block (with its buffer)"
        if not ok.any():
            raise ValueError(f"no {unit} is out of bag in any tree; lower max_samples or add trees")
        if not ok.all():
            warnings.warn(
                f"{int((~ok).sum())} rows have no tree whose bag leaves out their {unit}; "
                "they are left out of oob_score_",
                UserWarning,
            )
        start, stop, labels = competing_risks_labels(y)
        y_ok = make_competing_risks_y(stop[ok], labels[ok], start=start[ok])
        k = self._score_index()
        self.oob_score_ = concordance_index_cr(y_ok, pred[ok, k], cause=int(self.causes_[k]), ids=groups[ok])
        return pred

    # --- prediction --------------------------------------------------------------

    def _cause_index(self, cause, name="cause"):
        """Position of a cause label in ``causes_``."""
        ok = isinstance(cause, numbers.Integral) and not isinstance(cause, (bool, np.bool_))
        pos = np.flatnonzero(self.causes_ == cause) if ok else []
        if len(pos) != 1:
            raise ValueError(f"{name}={cause!r} is not a fitted cause label; causes_ = {self.causes_.tolist()}")
        return int(pos[0])

    def _score_index(self):
        return 0 if self.score_cause is None else self._cause_index(self.score_cause, "score_cause")

    def _aalen_johansen(self, X, times, intervals, ids, origin, extrapolate):
        """``(cif (n, J, T), surv (n, T), cumhaz (n, J, T))`` per row, or per subject along paths."""
        X, times, ids = self._check_predict(X, times, ids)
        n_jobs = effective_n_jobs(self.n_jobs)
        if intervals is None:
            if ids is not None or origin is not None or extrapolate != "none":
                raise ValueError("ids, origin and extrapolate require intervals")
            cif, surv, _ = self.forest_.predict_cif(X, times, n_jobs, self.aggregate)
            return cif, surv, self.forest_.predict_cause_cumhaz(X, times, n_jobs)
        cif, surv, cumhaz, _ = self.forest_.predict_cif_paths(
            *self._path_args(X, intervals, ids, origin, extrapolate), times, self.aggregate, extrapolate, n_jobs
        )
        return cif, surv, cumhaz

    def predict_cumulative_incidence(
        self, X, times=None, *, cause=None, intervals=None, ids=None, origin=None, extrapolate="none"
    ):
        """Cumulative incidence ``F_k(t)`` of each cause.

        Without ``intervals``, each row of ``X`` is a subject whose covariates
        are fixed from time 0. With ``intervals`` (and ``ids``), rows are a
        covariate path per subject, and the result, one row per subject in order
        of first appearance, is conditional on being event-free at ``origin``:
        ``F_k(t | u) = P(T <= t, cause k | T > u, path)``. ``origin``,
        ``extrapolate`` and the NaN rules are those of
        ``SurvivalForestTV.predict_cumulative_hazard``.

        Returns shape ``(n, n_causes_, n_times)`` with causes in ``causes_``
        order, or ``(n, n_times)`` for one ``cause`` label. ``times`` defaults
        to ``event_times_``.
        """
        k = None if cause is None else self._cause_index(cause)
        cif, _, _ = self._aalen_johansen(X, times, intervals, ids, origin, extrapolate)
        return cif if k is None else cif[:, k, :]

    def predict_cumulative_hazard(
        self, X, times=None, *, cause=None, intervals=None, ids=None, origin=None, extrapolate="none"
    ):
        """Ensemble cause-specific cumulative hazards (tree average).

        Returns ``(n, n_causes_, n_times)``; a ``cause`` label gives
        ``(n, n_times)``, and ``cause="all"`` the all-cause hazard (sum over
        causes). Path keywords as in ``predict_cumulative_incidence``: along a
        path the result is ``L_k(t) - L_k(origin)``.
        """
        all_causes = isinstance(cause, str) and cause == "all"
        k = None if cause is None or all_causes else self._cause_index(cause)
        _, _, H = self._aalen_johansen(X, times, intervals, ids, origin, extrapolate)
        if all_causes:
            return H.sum(axis=1)
        return H if k is None else H[:, k, :]

    def predict_survival_function(self, X, times=None, **path_kwargs):
        """Event-free survival ``S(t)`` (no event of any cause), shape ``(n, n_times)``.

        This is the Aalen–Johansen product limit ``prod (1 - sum_j dL_j)``,
        so ``S + sum_k F_k = 1``; it is not ``exp(-L)`` as in ``SurvivalForestTV``.
        Path keywords as in ``predict_cumulative_incidence``.
        """
        _, surv, _ = self._aalen_johansen(
            X,
            times,
            path_kwargs.pop("intervals", None),
            path_kwargs.pop("ids", None),
            path_kwargs.pop("origin", None),
            path_kwargs.pop("extrapolate", "none"),
        )
        if path_kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(path_kwargs)}")
        return surv

    def predict(self, X):
        """Risk score per row: ``F_k`` of ``score_cause`` at the last event time (higher = higher risk)."""
        X, _, _ = self._check_predict(X, None)
        return self._risk(X)

    def _risk(self, X):
        cif, _, _ = self.forest_.predict_cif(X, self.event_times_[-1:], effective_n_jobs(self.n_jobs), self.aggregate)
        return cif[:, self._score_index(), 0]

    def score(self, X, y, ids=None):
        """Cause-specific concordance of ``predict(X)`` with ``y`` for ``score_cause``.

        ``metrics.concordance_index_cr`` (Wolbers et al.): each ``score_cause``
        event row against the rows at risk then and the subjects that already had
        a competing event.
        """
        from .metrics import concordance_index_cr

        X, _, ids = self._check_predict(X, None, ids)
        k = self._score_index()
        return concordance_index_cr(y, self._risk(X), cause=int(self.causes_[k]), ids=ids)

    def leaf_cause_events_summary(self):
        """Per-cause in-bag events per leaf over all trees: ``cause, min, q10, median, max``.

        Rare causes with few events per leaf give unstable incidence jumps;
        ``split_cause`` with ``min_events_leaf_cause`` sets a floor for one cause.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "forest_")
        if not self.forest_.has_leaf_cause_events:
            raise ValueError("this fitted forest stores no per-leaf cause counts (fitted before S12); refit it")
        counts = np.concatenate([self.forest_.leaf_cause_events(b) for b in range(self.forest_.n_trees)])
        q = np.quantile(counts, [0.0, 0.1, 0.5, 1.0], axis=0)
        return pl.DataFrame(
            {"cause": self.causes_, "min": q[0], "q10": q[1], "median": q[2], "max": q[3]}
        )
