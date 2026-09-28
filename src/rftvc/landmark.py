"""Landmark data views and the landmark super-model forest (design.md, D5).

For a landmark ``s`` and horizon ``w`` the prediction target is
``P(T <= s + w | T > s, H(s))``: the event probability within ``w`` for a
subject event-free and under observation at ``s``, given its history up to ``s``.
Each (subject, landmark) pair becomes one row with the clock reset to ``s``, so
the stacked rows have no delayed entry and need no future covariate path.
"""

import warnings
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from ._competing import CompetingRisksForestTV
from ._estimator import SurvivalForestTV
from ._validation import CR_DTYPE, _as_labels, check_counting_process, make_competing_risks_y, make_survival_y

AGGREGATIONS = ("last", "first", "mean", "min", "max", "sum", "count", "slope", "std")


class LandmarkData(NamedTuple):
    """A stacked landmark dataset.

    ``X`` holds the history features plus the landmark ``s`` (last column);
    ``y`` the outcome on the reset clock (``SURV_DTYPE`` for a boolean or {0, 1}
    event column, ``CR_DTYPE`` with cause labels otherwise); ``ids`` the subject of each row;
    ``groups`` the CV grouping (subjects); ``s`` the landmark of each row.
    """

    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    groups: np.ndarray
    s: np.ndarray
    feature_names: list


def _as_polars(df):
    if isinstance(df, pl.DataFrame):
        return df
    try:
        return pl.from_pandas(df)
    except (TypeError, ValueError, ImportError) as exc:
        raise TypeError("df must be a polars or pandas DataFrame") from exc


def _feature_specs(history_features, forbidden):
    """Normalise ``history_features`` to ``[(name, column, agg)]``.

    Each item is a column name (its last value at the landmark) or a
    ``(column, agg)`` pair with ``agg`` in ``AGGREGATIONS``.
    """
    specs = []
    for item in history_features:
        column, agg = (item, "last") if isinstance(item, str) else tuple(item)
        if agg not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {agg!r}; use one of {AGGREGATIONS}")
        if column in forbidden:
            raise ValueError(
                f"history feature on {column!r} would look ahead: the {column!r} column "
                "carries information from after the landmark"
            )
        specs.append((column if agg == "last" else f"{column}_{agg}", column, agg))
    if not specs:
        raise ValueError("history_features must name at least one feature")
    names = [name for name, _, _ in specs]
    if len(set(names)) != len(names) or "landmark" in names:
        raise ValueError(f"duplicate or reserved feature names: {names}")
    return specs


def _raw_groups(history_features):
    """``{raw column -> [derived feature names]}``, first-appearance order of both.

    ``"landmark"`` never appears (it is appended after ``history_features`` is
    parsed, by ``make_landmark_data`` / ``landmark_features``), which is exactly
    the "not a permutable unit" property, for free.
    """
    groups = {}
    for name, column, _ in _feature_specs(history_features, forbidden=set()):
        groups.setdefault(column, []).append(name)
    return groups


def _agg_expr(name, column, agg, start, time):
    col = pl.col(column).cast(pl.Float64)
    t = pl.col(time).cast(pl.Float64)
    exprs = {
        "last": col.sort_by(start).last(),
        "first": col.sort_by(start).first(),
        "mean": col.mean(),
        "min": col.min(),
        "max": col.max(),
        "sum": col.sum(),
        "count": pl.len().cast(pl.Float64),
        # cov(t, col) is pairwise-complete (polars drops a row where either is null), so
        # the denominator must be the variance of t over that same non-null-col subset,
        # not every row's t: otherwise a null col value still contributes its own t to the
        # denominator alone, silently changing the OLS slope.
        "slope": pl.when(t.filter(col.is_not_null()).n_unique() < 2)
        .then(None)
        .otherwise(pl.cov(t, col) / t.filter(col.is_not_null()).var()),
        "std": col.std(),
    }
    return exprs[agg].alias(name)


def _history_features(df, s, specs, *, id, start, measured_at=None):
    """Features from the rows known at ``s``, one row per id.

    A row is known at ``s`` iff ``start <= s``. A later row must not be used even
    if its covariates were measured before ``s``: the row exists only because
    the subject survived to its start, which is not yet known at ``s``. ``slope``
    is the OLS slope on ``measured_at`` when given, else on ``start``.
    """
    time = measured_at if measured_at is not None else start
    hist = df.filter(pl.col(start) <= s)
    return hist.group_by(id, maintain_order=True).agg([_agg_expr(n, c, a, start, time) for n, c, a in specs])


def _landmark_grid(first_start, last_stop, landmarks, step):
    if (landmarks is None) == (step is None):
        raise ValueError("pass exactly one of landmarks or step")
    if landmarks is None:
        if not step > 0:
            raise ValueError("step must be positive")
        landmarks = np.arange(first_start.min(), last_stop.max(), step)
    landmarks = np.unique(np.asarray(landmarks, dtype=float))
    if landmarks.size == 0 or not np.isfinite(landmarks).all():
        raise ValueError("landmarks must be a non-empty set of finite times")
    return landmarks


def make_landmark_data(
    df,
    *,
    horizon,
    history_features,
    landmarks=None,
    step=None,
    id="id",
    start="start",
    stop="stop",
    event="event",
    measured_at=None,
):
    """Stack landmark datasets from counting-process rows.

    For each landmark ``s`` (``landmarks``, or every ``step`` from the earliest
    entry), with ``U`` an id's last ``stop`` (event or censoring time):

    1. Risk set: ids that have entered (first ``start <= s``) and are event-free
       and uncensored at ``s`` (``U > s``).
    2. Row: ``start = 0``, ``stop = min(U, s + horizon) - s``,
       ``event = 1{event and U <= s + horizon}``. With cause labels in the
       event column (0 = censored), the row keeps the id's terminal label
       when ``U <= s + horizon`` and is 0 otherwise.
    3. Features: ``history_features`` computed from the rows known at ``s``
       (``start <= s``; every subject in the risk set has at least one), plus
       ``s`` itself. ``measured_at``, if given, is validated to be ``<= start``.

    Validity rests on censoring being independent of the event given
    ``H(s)`` and ``s``. Using ``s`` as a feature lets the forest learn
    landmark-dependent effects; it does not correct for selection or censoring.

    Estimand: as in the stacked landmark super-model, every (subject, landmark)
    row is an observation in the forest's risk sets and leaf estimates, so a
    subject at risk at many landmarks carries more weight. Only resampling (and
    leaf sizes) are by subject. Thin the grid (``step``) to limit the overlap.
    """
    df = _as_polars(df)
    if not horizon > 0:
        raise ValueError("horizon must be positive")
    required = {id, start, stop, event} | ({measured_at} if measured_at else set())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns {sorted(missing)}")
    specs = _feature_specs(history_features, forbidden={stop, event, id})
    for _, column, _ in specs:
        if column not in df.columns:
            raise ValueError(f"history feature column {column!r} not in df")

    starts = df[start].cast(pl.Float64).to_numpy()
    stops = df[stop].cast(pl.Float64).to_numpy()
    boolean = df[event].dtype == pl.Boolean
    labels = _as_labels(df[event].to_numpy())  # bool -> {0, 1}; rejects negative / non-integer labels
    events = labels != 0
    check_counting_process(
        starts, stops, events, df[id].to_numpy(),
        measured_at=None if measured_at is None else df[measured_at].cast(pl.Float64).to_numpy(),
    )
    subjects = df.group_by(id, maintain_order=True).agg(
        pl.col(start).cast(pl.Float64).min().alias("_entry"),
        pl.col(stop).cast(pl.Float64).max().alias("_U"),
        # Only an id's last row may carry an event (checked above), so max = the terminal label.
        pl.col(event).cast(pl.Int64).max().alias("_label"),
    )
    grid = _landmark_grid(subjects["_entry"].to_numpy(), subjects["_U"].to_numpy(), landmarks, step)

    parts = []
    for s in grid:
        at_risk = subjects.filter((pl.col("_entry") <= s) & (pl.col("_U") > s))
        if at_risk.height == 0:
            continue
        feats = _history_features(df.join(at_risk.select(id), on=id), s, specs, id=id, start=start, measured_at=measured_at)
        part = at_risk.join(feats, on=id, how="inner", maintain_order="left").with_columns(
            pl.lit(s).alias("landmark"),
            ((pl.min_horizontal("_U", pl.lit(s + horizon))) - s).alias("_stop"),
            pl.when(pl.col("_U") <= s + horizon).then(pl.col("_label")).otherwise(0).alias("_lm_event"),
        )
        parts.append(part)
    if not parts:
        raise ValueError("no subject is at risk at any landmark")
    stacked = pl.concat(parts)
    names = [name for name, _, _ in specs] + ["landmark"]
    X = stacked.select(names).to_numpy().astype(np.float64)
    lm_event = stacked["_lm_event"].to_numpy()
    if boolean or labels.max(initial=0) <= 1:
        y = make_survival_y(stacked["_stop"].to_numpy(), lm_event != 0)
    else:
        y = make_competing_risks_y(stacked["_stop"].to_numpy(), lm_event)
    ids = stacked[id].to_numpy()
    s_col = stacked["landmark"].to_numpy()
    return LandmarkData(np.ascontiguousarray(X), y, ids, ids.copy(), s_col, names)


def landmark_features(df, s, *, history_features, id="id", start="start", stop="stop", event=None, measured_at=None):
    """Features at landmark ``s`` for prediction: ``(ids, X)``, X including ``s``.

    The prediction population is subjects event-free and under observation at
    ``s``: entered by ``s`` and observed through ``s`` (last ``stop >= s``). This
    includes a last ``stop`` exactly at ``s``, the usual case for current data.
    (Training needs ``U > s`` only because such a subject would contribute a
    zero-length row, not because it is outside the target population.) If
    ``event`` is given, subjects with an event at or before ``s`` are excluded.
    Rows are known at ``s`` iff ``start <= s`` (see ``make_landmark_data``);
    ``measured_at``, if given, is the OLS time axis for a ``"slope"`` feature.
    """
    df = _as_polars(df)
    specs = _feature_specs(history_features, forbidden={stop, id} | ({event} if event else set()))
    agg = [pl.col(start).cast(pl.Float64).min().alias("_entry"), pl.col(stop).cast(pl.Float64).max().alias("_U")]
    if event is not None:
        agg.append(pl.col(event).cast(pl.Boolean).any().alias("_event"))
    subjects = df.group_by(id, maintain_order=True).agg(agg)
    keep = (pl.col("_entry") <= s) & (pl.col("_U") >= s)
    if event is not None:
        keep = keep & ~(pl.col("_event") & (pl.col("_U") <= s))
    at_risk = subjects.filter(keep)
    feats = _history_features(df.join(at_risk.select(id), on=id), s, specs, id=id, start=start, measured_at=measured_at)
    out = at_risk.join(feats, on=id, how="inner", maintain_order="left").with_columns(pl.lit(float(s)).alias("landmark"))
    names = [name for name, _, _ in specs] + ["landmark"]
    return out[id].to_numpy(), np.ascontiguousarray(out.select(names).to_numpy().astype(np.float64))


class _LandmarkBase(BaseEstimator):
    """Shared stacking, fitting and feature plumbing of the landmark super-models."""

    def _columns(self):
        return dict(id=self.id, start=self.start, stop=self.stop, measured_at=self.measured_at)

    def _landmark_data(self, df):
        return make_landmark_data(
            df,
            horizon=self.horizon,
            history_features=self.history_features,
            landmarks=self.landmarks,
            step=self.step,
            event=self.event,
            **self._columns(),
        )

    def _fit_forest(self, data, forest):
        """Fit ``forest`` on stacked ``data`` (block resampling on the landmark time ``s``)."""
        params = forest.get_params()
        block_time = None
        if params["resample_unit"] == "block":
            block_time = data.s
            reach = params["block_length"] * params["oob_buffer"]
            if params["oob_score"] and reach < self.horizon:
                warnings.warn(
                    f"block OOB leaks: landmarks within horizon={self.horizon:g} share an outcome window, but "
                    f"blocks are only excluded {reach:g} apart (block_length * oob_buffer); use "
                    "block_length >= horizon with oob_buffer=1",
                    UserWarning,
                    stacklevel=3,
                )
        self.forest_ = forest.fit(data.X, data.y, ids=data.ids, layout="stacked", block_time=block_time)
        self.feature_names_ = data.feature_names
        self.landmarks_ = np.unique(data.s)
        self.n_rows_ = data.X.shape[0]
        return self

    def _features(self, df, s):
        check_is_fitted(self, "forest_")
        event = self.event if self.event in _as_polars(df).columns else None
        return landmark_features(
            df, s, history_features=self.history_features, event=event, **self._columns()
        )

    def _check_times(self, times):
        times = np.asarray(times, dtype=float).ravel()
        if (times < 0).any() or (times > self.horizon).any():
            raise ValueError(f"times must lie in [0, horizon={self.horizon}]")
        return times


class LandmarkSurvivalForest(_LandmarkBase):
    """Landmark super-model: a survival forest on stacked landmark data.

    Predicts ``P(T <= s + w | T > s, H(s))`` for ``w <= horizon`` from the
    history up to the landmark ``s``; no future covariate path is needed.
    This is a meta-estimator that owns the joint transformation of rows,
    outcomes and ids; it is not a scikit-learn ``Pipeline``.

    .. seealso:: :doc:`/user_guide/foundation` for what this estimates, its
       assumptions and the validity of each prediction call.

    Parameters
    ----------
    horizon : float
        Prediction window ``w`` (outcomes are administratively censored at ``s + w``).
    history_features : list
        Column names (last value at ``s``) or ``(column, agg)`` pairs, ``agg`` in
        ``"last", "first", "mean", "min", "max", "sum", "count", "slope", "std"``
        (``"slope"``: the OLS slope on ``measured_at``, else ``start``).
    landmarks : array-like or None
        Training landmarks. Exactly one of ``landmarks`` and ``step``.
    step : float or None
        Landmark spacing from the earliest entry.
    forest : SurvivalForestTV or None
        Unfitted forest to clone; defaults to ``SurvivalForestTV()``. Resampling
        is by subject: all landmark rows of a subject enter a tree together.
        With ``resample_unit="block"`` the unit is a subject's landmarks within
        one window of width ``block_length`` on the landmark time ``s``.
    id, start, stop, event, measured_at : str
        Column names in the input frames.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> from rftvc import LandmarkSurvivalForest, SurvivalForestTV
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(60):
    ...     z = rng.normal()
    ...     t = rng.exponential(np.exp(-0.5 * z))
    ...     rows.append((i, 0.0, min(t, 5.0), bool(t <= 5.0), z))
    >>> df = pl.DataFrame(rows, schema=["id", "start", "stop", "event", "z"], orient="row")
    >>> model = LandmarkSurvivalForest(
    ...     horizon=2.0, history_features=["z"], landmarks=[0.0, 1.0, 2.0],
    ...     forest=SurvivalForestTV(n_estimators=100, random_state=0),
    ... ).fit(df)
    >>> model.predict_risk(df, s=0.0).shape
    (60, 3)
    """

    def __init__(
        self,
        horizon=None,
        history_features=(),
        landmarks=None,
        step=None,
        forest=None,
        id="id",
        start="start",
        stop="stop",
        event="event",
        measured_at=None,
    ):
        self.horizon = horizon
        self.history_features = history_features
        self.landmarks = landmarks
        self.step = step
        self.forest = forest
        self.id = id
        self.start = start
        self.stop = stop
        self.event = event
        self.measured_at = measured_at

    def fit(self, df):
        if self.horizon is None:
            raise ValueError("horizon is required")
        data = self._landmark_data(df)
        if data.y.dtype == CR_DTYPE:
            raise ValueError(
                "the event column holds cause labels > 1 (competing risks); use LandmarkCompetingRisksForest"
            )
        forest = SurvivalForestTV() if self.forest is None else clone(self.forest)
        # Block resampling groups an id's landmarks into windows of the landmark time s.
        return self._fit_forest(data, forest)

    def predict_survival_function(self, df, s, times):
        """``P(T > s + t | T > s, H(s))`` for ``t`` in ``times`` (reset clock, ``<= horizon``).

        Returns ``(ids, S)`` with one row of ``S`` per subject at risk at ``s``.
        """
        times = self._check_times(times)
        ids, X = self._features(df, s)
        return ids, self.forest_.predict_survival_function(X, times)

    def predict_risk(self, df, s, horizon=None):
        """``P(T <= s + w | T > s, H(s))`` with ``w = horizon`` (default: the fitted horizon).

        Returns a polars DataFrame with the id column, ``landmark`` and ``risk``.
        """
        w = self.horizon if horizon is None else horizon
        ids, S = self.predict_survival_function(df, s, [w])
        return pl.DataFrame({self.id: ids, "landmark": np.full(len(ids), float(s)), "risk": 1.0 - S[:, 0]})


class LandmarkCompetingRisksForest(_LandmarkBase):
    """Landmark super-model for competing risks: a ``CompetingRisksForestTV`` on stacked landmark data.

    Predicts the cumulative incidence ``F_k(s + w | s, H(s)) = P(T <= s + w,
    cause k | T > s, H(s))`` for ``w <= horizon`` from the history up to the
    landmark ``s``. The clock is reset at ``s``, so the Aalen–Johansen estimate
    in each leaf is a direct estimate of this target. The event column holds
    cause labels (0 = censored).

    .. seealso:: :doc:`/user_guide/foundation` for what this estimates, its
       assumptions and the validity of each prediction call.

    Parameters
    ----------
    horizon, history_features, landmarks, step, id, start, stop, event, measured_at
        As in ``LandmarkSurvivalForest``.
    forest : CompetingRisksForestTV or None
        Unfitted forest to clone; defaults to ``CompetingRisksForestTV()``.
    causes : array-like of int or None
        Cause label vocabulary, set on the forest (keeps the cause axis fixed
        when a training set lacks a cause, e.g. in cross-validation).
    score_cause : int or None
        Default cause of ``predict_risk``, set on the forest; ``None`` is the
        first label of the fitted ``causes_``.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> from rftvc import LandmarkCompetingRisksForest, CompetingRisksForestTV
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     z = rng.normal()
    ...     t = rng.exponential(np.exp(-0.5 * z))
    ...     cause = int(rng.integers(1, 3))
    ...     rows.append((i, 0.0, min(t, 5.0), cause if t <= 5.0 else 0, z))
    >>> df = pl.DataFrame(rows, schema=["id", "start", "stop", "event", "z"], orient="row")
    >>> model = LandmarkCompetingRisksForest(
    ...     horizon=2.0, history_features=["z"], landmarks=[0.0, 1.0, 2.0],
    ...     forest=CompetingRisksForestTV(n_estimators=100, random_state=0),
    ... ).fit(df)
    >>> model.predict_risk(df, s=0.0, cause=1).shape
    (80, 4)
    """

    def __init__(
        self,
        horizon=None,
        history_features=(),
        landmarks=None,
        step=None,
        forest=None,
        id="id",
        start="start",
        stop="stop",
        event="event",
        measured_at=None,
        causes=None,
        score_cause=None,
    ):
        self.horizon = horizon
        self.history_features = history_features
        self.landmarks = landmarks
        self.step = step
        self.forest = forest
        self.id = id
        self.start = start
        self.stop = stop
        self.event = event
        self.measured_at = measured_at
        self.causes = causes
        self.score_cause = score_cause

    def fit(self, df):
        if self.horizon is None:
            raise ValueError("horizon is required")
        forest = CompetingRisksForestTV() if self.forest is None else clone(self.forest)
        if not isinstance(forest, CompetingRisksForestTV):
            raise TypeError(f"forest must be a CompetingRisksForestTV, got {type(forest).__name__}")
        overrides = {k: v for k, v in (("causes", self.causes), ("score_cause", self.score_cause)) if v is not None}
        forest.set_params(**overrides)
        return self._fit_forest(self._landmark_data(df), forest)

    @property
    def causes_(self):
        return self.forest_.causes_

    def _cause(self, cause):
        """The cause label to predict: ``cause``, else the forest's ``score_cause``, else the first label."""
        if cause is not None:
            return cause
        score = self.forest_.score_cause
        return int(self.forest_.causes_[0]) if score is None else score

    def predict_cumulative_incidence(self, df, s, times, cause=None):
        """``F_k(s + t | s, H(s))`` for ``t`` in ``times`` (reset clock, ``<= horizon``).

        Returns ``(ids, F)``: ``F`` has shape ``(n, n_causes, n_times)``
        (causes in ``causes_`` order), or ``(n, n_times)`` for one ``cause`` label.
        """
        times = self._check_times(times)
        ids, X = self._features(df, s)
        return ids, self.forest_.predict_cumulative_incidence(X, times, cause=cause)

    def predict_survival_function(self, df, s, times):
        """Event-free survival ``P(T > s + t | T > s, H(s))``; returns ``(ids, S)``."""
        times = self._check_times(times)
        ids, X = self._features(df, s)
        return ids, self.forest_.predict_survival_function(X, times)

    def predict_risk(self, df, s, horizon=None, cause=None):
        """``F_k(s + w | s, H(s))`` with ``w = horizon`` (default: the fitted horizon).

        ``cause`` defaults to ``score_cause`` (else the first label). Returns a
        polars DataFrame with the id column, ``landmark``, ``cause`` and ``risk``.
        """
        check_is_fitted(self, "forest_")
        w = self.horizon if horizon is None else horizon
        k = self._cause(cause)
        ids, F = self.predict_cumulative_incidence(df, s, [w], cause=k)
        n = len(ids)
        return pl.DataFrame(
            {self.id: ids, "landmark": np.full(n, float(s)), "cause": np.full(n, int(k)), "risk": F[:, 0]}
        )
