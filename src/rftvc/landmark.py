"""Landmark data views and the landmark super-model forest (design.md, D5).

For a landmark ``s`` and horizon ``w`` the prediction target is
``P(T <= s + w | T > s, H(s))``: the event probability within ``w`` for a
subject event-free and under observation at ``s``, given its history up to ``s``.
Each (subject, landmark) pair becomes one row with the clock reset to ``s``, so
the stacked rows have no delayed entry and need no future covariate path.
"""

from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from ._estimator import SurvivalForestTV
from ._validation import check_counting_process, make_survival_y

AGGREGATIONS = ("last", "first", "mean", "min", "max", "sum", "count")


class LandmarkData(NamedTuple):
    """A stacked landmark dataset.

    ``X`` holds the history features plus the landmark ``s`` (last column);
    ``y`` the outcome on the reset clock; ``ids`` the subject of each row;
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


def _agg_expr(name, column, agg, start):
    col = pl.col(column).cast(pl.Float64)
    exprs = {
        "last": col.sort_by(start).last(),
        "first": col.sort_by(start).first(),
        "mean": col.mean(),
        "min": col.min(),
        "max": col.max(),
        "sum": col.sum(),
        "count": pl.len().cast(pl.Float64),
    }
    return exprs[agg].alias(name)


def _history_features(df, s, specs, *, id, start):
    """Features from the rows known at ``s``, one row per id.

    A row is known at ``s`` iff ``start <= s``. A later row must not be used even
    if its covariates were measured before ``s``: the row exists only because
    the subject survived to its start, which is not yet known at ``s``.
    """
    hist = df.filter(pl.col(start) <= s)
    return hist.group_by(id, maintain_order=True).agg([_agg_expr(n, c, a, start) for n, c, a in specs])


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
       ``event = 1{event and U <= s + horizon}``.
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
    events = df[event].cast(pl.Boolean).to_numpy()
    check_counting_process(
        starts, stops, events, df[id].to_numpy(),
        measured_at=None if measured_at is None else df[measured_at].cast(pl.Float64).to_numpy(),
    )
    subjects = df.group_by(id, maintain_order=True).agg(
        pl.col(start).cast(pl.Float64).min().alias("_entry"),
        pl.col(stop).cast(pl.Float64).max().alias("_U"),
        pl.col(event).cast(pl.Boolean).any().alias("_event"),
    )
    grid = _landmark_grid(subjects["_entry"].to_numpy(), subjects["_U"].to_numpy(), landmarks, step)

    parts = []
    for s in grid:
        at_risk = subjects.filter((pl.col("_entry") <= s) & (pl.col("_U") > s))
        if at_risk.height == 0:
            continue
        feats = _history_features(df.join(at_risk.select(id), on=id), s, specs, id=id, start=start)
        part = at_risk.join(feats, on=id, how="inner", maintain_order="left").with_columns(
            pl.lit(s).alias("landmark"),
            ((pl.min_horizontal("_U", pl.lit(s + horizon))) - s).alias("_stop"),
            (pl.col("_event") & (pl.col("_U") <= s + horizon)).alias("_lm_event"),
        )
        parts.append(part)
    if not parts:
        raise ValueError("no subject is at risk at any landmark")
    stacked = pl.concat(parts)
    names = [name for name, _, _ in specs] + ["landmark"]
    X = stacked.select(names).to_numpy().astype(np.float64)
    y = make_survival_y(stacked["_stop"].to_numpy(), stacked["_lm_event"].to_numpy())
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
    ``measured_at`` is accepted for signature symmetry; rows are known at
    ``s`` iff ``start <= s`` (see ``make_landmark_data``).
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
    feats = _history_features(df.join(at_risk.select(id), on=id), s, specs, id=id, start=start)
    out = at_risk.join(feats, on=id, how="inner", maintain_order="left").with_columns(pl.lit(float(s)).alias("landmark"))
    names = [name for name, _, _ in specs] + ["landmark"]
    return out[id].to_numpy(), np.ascontiguousarray(out.select(names).to_numpy().astype(np.float64))


class LandmarkSurvivalForest(BaseEstimator):
    """Landmark super-model: a survival forest on stacked landmark data.

    Predicts ``P(T <= s + w | T > s, H(s))`` for ``w <= horizon`` from the
    history up to the landmark ``s``; no future covariate path is needed.
    This is a meta-estimator that owns the joint transformation of rows,
    outcomes and ids; it is not a scikit-learn ``Pipeline``.

    Parameters
    ----------
    horizon : float
        Prediction window ``w`` (outcomes are administratively censored at ``s + w``).
    history_features : list
        Column names (last value at ``s``) or ``(column, agg)`` pairs, ``agg`` in
        ``"last", "first", "mean", "min", "max", "sum", "count"``.
    landmarks : array-like or None
        Training landmarks. Exactly one of ``landmarks`` and ``step``.
    step : float or None
        Landmark spacing from the earliest entry.
    forest : SurvivalForestTV or None
        Unfitted forest to clone; defaults to ``SurvivalForestTV()``. Resampling
        is by subject: all landmark rows of a subject enter a tree together.
    id, start, stop, event, measured_at : str
        Column names in the input frames.
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

    def _columns(self):
        return dict(id=self.id, start=self.start, stop=self.stop, measured_at=self.measured_at)

    def fit(self, df):
        if self.horizon is None:
            raise ValueError("horizon is required")
        data = make_landmark_data(
            df,
            horizon=self.horizon,
            history_features=self.history_features,
            landmarks=self.landmarks,
            step=self.step,
            event=self.event,
            **self._columns(),
        )
        forest = SurvivalForestTV() if self.forest is None else clone(self.forest)
        self.forest_ = forest.fit(data.X, data.y, ids=data.ids, layout="stacked")
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

    def predict_survival_function(self, df, s, times):
        """``P(T > s + t | T > s, H(s))`` for ``t`` in ``times`` (reset clock, ``<= horizon``).

        Returns ``(ids, S)`` with one row of ``S`` per subject at risk at ``s``.
        """
        times = np.asarray(times, dtype=float).ravel()
        if (times < 0).any() or (times > self.horizon).any():
            raise ValueError(f"times must lie in [0, horizon={self.horizon}]")
        ids, X = self._features(df, s)
        return ids, self.forest_.predict_survival_function(X, times)

    def predict_risk(self, df, s, horizon=None):
        """``P(T <= s + w | T > s, H(s))`` with ``w = horizon`` (default: the fitted horizon).

        Returns a polars DataFrame with the id column, ``landmark`` and ``risk``.
        """
        w = self.horizon if horizon is None else horizon
        ids, S = self.predict_survival_function(df, s, [w])
        return pl.DataFrame({self.id: ids, "landmark": np.full(len(ids), float(s)), "risk": 1.0 - S[:, 0]})
