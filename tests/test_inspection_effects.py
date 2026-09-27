"""S20: ``inspection.hazard_effect`` / ``inspection.path_effect``."""

import copy

import numpy as np
import pytest

from rftvc import (
    CompetingRisksForestTV,
    LandmarkSurvivalForest,
    SurvivalForestTV,
    inspection,
    make_competing_risks_y,
    make_survival_y,
)
from tests.sim import rows, simulate

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*windows are too fine:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
]


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x0, z, U, ev = simulate(n, rng)
    return rows(x0, z, U, ev)


X, Y, IDS = _data(300, 0)
FOREST = SurvivalForestTV(n_estimators=30, random_state=0).fit(X, Y, IDS)


def _stubbed(model, stub):
    m = copy.copy(model)
    m.forest_ = stub
    return m


class HazardOracle:
    """Constant hazard rate ``c * exp(beta * X[:, 1])`` (z is column 1)."""

    def __init__(self, c=0.2, beta=0.6):
        self.c, self.beta = c, beta

    def predict_cumhaz(self, X, times, aggregate, n_jobs):
        rate = self.c * np.exp(self.beta * X[:, 1])
        return rate[:, None] * np.asarray(times, dtype=float)[None, :]


def test_hazard_effect_oracle_gives_exact_constant_rate():
    stub = HazardOracle(c=0.2, beta=0.6)
    m = _stubbed(FOREST, stub)
    values = np.array([-1.0, 0.0, 1.0, 2.0])
    res = inspection.hazard_effect(m, X, Y, feature=1, values=values, windows=4)
    expected = stub.c * np.exp(stub.beta * values)
    np.testing.assert_allclose(res.hazard, expected[:, None] * np.ones((1, res.hazard.shape[1])), rtol=1e-8)


def test_hazard_effect_weights_equal_exposures_hand_check():
    # 3 rows, 2 windows [0,1],(1,2]; row0 at risk in both, row1 only window 1, row2 only window 2.
    X3 = np.array([[0.0], [0.0], [0.0]])
    start = np.array([0.0, 0.0, 1.0])
    stop = np.array([2.0, 1.0, 2.0])
    y3 = make_survival_y(stop, np.array([True, False, True]), start=start)
    m = SurvivalForestTV(n_estimators=5, random_state=0).fit(X3, y3)

    class ConstOracle:
        def predict_cumhaz(self, X, times, aggregate, n_jobs):
            return np.tile(np.asarray(times, dtype=float), (X.shape[0], 1)) * 0.5  # rate = 0.5 for every row

    m = _stubbed(m, ConstOracle())
    res = inspection.hazard_effect(m, X3, y3, feature=0, values=[0.0], windows=np.array([0.0, 1.0, 2.0]))
    # exposure-weighted average of a constant rate is that rate, regardless of weights
    np.testing.assert_allclose(res.hazard[0], [0.5, 0.5], atol=1e-10)


def test_hazard_effect_support_mask_varies_by_window():
    # feature values only overlap [0,10] in window 1 and [100,110] in window 2
    n = 40
    rng = np.random.default_rng(1)
    x0 = np.zeros(n)
    lo = rng.uniform(0, 10, n // 2)
    hi = rng.uniform(100, 110, n - n // 2)
    z = np.concatenate([lo, hi])
    start = np.where(np.arange(n) < n // 2, 0.0, 1.0)
    stop = np.where(np.arange(n) < n // 2, 1.0, 2.0)
    X4 = np.column_stack([x0, z])
    event = np.zeros(n, dtype=bool)
    event[0], event[-1] = True, True
    y4 = make_survival_y(stop, event, start=start)
    m = SurvivalForestTV(n_estimators=5, random_state=0).fit(X4, y4)
    m = _stubbed(m, HazardOracle(c=0.1, beta=0.0))
    res = inspection.hazard_effect(m, X4, y4, feature=1, values=[5.0, 105.0], windows=np.array([0.0, 1.0, 2.0]))
    assert res.support_mask[0, 0] and not res.support_mask[0, 1]
    assert res.support_mask[1, 1] and not res.support_mask[1, 0]


def test_hazard_effect_individual_averages_to_average():
    res = inspection.hazard_effect(FOREST, X, Y, feature=1, values=[0.0, 1.0], windows=6, kind="individual")
    e = np.zeros((X.shape[0], len(res.window_edges) - 1))
    from rftvc.metrics import _window_exposure_1

    for m in range(e.shape[1]):
        starts = Y["start"]
        stops = Y["stop"]
        e[:, m] = _window_exposure_1(starts, stops, res.window_edges[m], res.window_edges[m + 1])
    for i in range(len(res["values"])):
        ind = res.individual[:, i, :]
        at_risk = e > 0
        num = np.where(at_risk, ind * e, 0.0).sum(axis=0)
        den = np.where(at_risk, e, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore"):
            avg = num / den
        np.testing.assert_allclose(avg[den > 0], res.hazard[i][den > 0], rtol=1e-8)


def test_hazard_effect_empty_window_gives_nan():
    # a training-derived window can have zero exposure on held-out evaluation data
    Xtr = np.zeros((5, 2))
    start_tr = np.array([0.0, 0.0, 0.0, 0.0, 2.0])
    stop_tr = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
    event_tr = np.array([True, False, False, False, True])
    ytr = make_survival_y(stop_tr, event_tr, start=start_tr)
    m = SurvivalForestTV(n_estimators=5, random_state=0).fit(Xtr, ytr)
    Xev = np.zeros((4, 2))
    yev = make_survival_y(np.ones(4), np.zeros(4, dtype=bool), start=np.zeros(4))
    w = np.array([0.0, 1.0, 2.0, 3.0])
    res = inspection.hazard_effect(m, Xev, yev, feature=1, values=[0.0], windows=w)
    assert np.isnan(res.hazard[0, 1])
    assert not res.support_mask[0, 1]
    res_i = inspection.hazard_effect(m, Xev, yev, feature=1, values=[0.0], windows=w, kind="individual")
    assert np.isnan(res_i.individual[:, 0, 1]).all()


def test_hazard_effect_competing_risks_cause_selection():
    rng = np.random.default_rng(2)
    x0, z, U, ev = simulate(150, rng)
    Xc, y2, idc = rows(x0, z, U, ev)
    labels = np.where(y2["event"], 1, 0)
    yc = make_competing_risks_y(y2["stop"], labels, start=y2["start"])
    m = CompetingRisksForestTV(n_estimators=20, random_state=0, causes=[1]).fit(Xc, yc, idc)
    res_all = inspection.hazard_effect(m, Xc, yc, feature=1, values=[0.0, 1.0], windows=4)
    res_1 = inspection.hazard_effect(m, Xc, yc, feature=1, values=[0.0, 1.0], windows=4, cause=1)
    assert res_all.hazard.shape == (2, 1, 4)
    np.testing.assert_allclose(res_1.hazard, res_all.hazard[:, 0, :])


def test_hazard_effect_landmark_dispatch_uses_inner_forest(monkeypatch):
    import polars as pl

    rng = np.random.default_rng(3)
    n_ids = 60
    df = pl.DataFrame(
        {
            "id": np.repeat(np.arange(n_ids), 4),
            "start": np.tile([0.0, 1.0, 2.0, 3.0], n_ids),
            "stop": np.tile([1.0, 2.0, 3.0, 4.0], n_ids),
            "event": np.zeros(n_ids * 4, dtype=bool),
            "z": rng.normal(size=n_ids * 4),
        }
    )
    ev_idx = rng.choice(n_ids, size=15, replace=False)
    df = df.with_columns(
        pl.when(pl.col("id").is_in(ev_idx) & (pl.col("stop") == 4.0)).then(True).otherwise(pl.col("event")).alias("event")
    )
    model = LandmarkSurvivalForest(
        id="id", start="start", stop="stop", event="event", landmarks=[1.0, 2.0], horizon=2.0,
        history_features=["z"], forest=SurvivalForestTV(n_estimators=10, random_state=0),
    ).fit(df)
    calls = []
    orig_windows = inspection._windows

    def spy_windows(estimator, windows):
        calls.append(estimator)
        return orig_windows(estimator, windows)

    monkeypatch.setattr(inspection, "_windows", spy_windows)
    res = inspection.hazard_effect(model, df, None, feature="z", values=[0.0, 1.0], windows=3)
    assert calls and calls[0] is model.forest_, "expected the inner (model.forest_) forest to be the windows target"
    assert res.hazard.shape == (2, len(res.window_edges) - 1)
    assert res.window_edges[-1] <= model.horizon
    with pytest.raises(ValueError, match="landmark"):
        inspection.hazard_effect(model, df, None, feature="landmark", values=[0.0])


def test_hazard_effect_errors():
    with pytest.raises(ValueError, match="kind"):
        inspection.hazard_effect(FOREST, X, Y, feature=1, kind="bogus")


# --- path_effect -----------------------------------------------------------------------


class PathOracle:
    """Analytic conditional cumhaz for a piecewise-constant rate ``c * exp(beta * z)``."""

    def __init__(self, c=0.15, beta=0.5, z_col=1):
        self.c, self.beta, self.z_col = c, beta, z_col

    def predict_paths(self, X, start, stop, offsets, origin, times, aggregate, extrapolate, n_jobs):
        times = np.asarray(times, dtype=float)
        rate = self.c * np.exp(self.beta * X[:, self.z_col])
        n_subj = offsets.size - 1
        out = np.zeros((n_subj, times.size))
        for i in range(n_subj):
            lo, hi = offsets[i], offsets[i + 1]
            o = origin[i]
            for j, t in enumerate(times):
                overlap = np.clip(np.minimum(stop[lo:hi], t) - np.maximum(start[lo:hi], o), 0.0, None)
                out[i, j] = (rate[lo:hi] * overlap).sum()
        return out


def _path_fixture():
    K = 8
    n_subj = 6
    ids = np.repeat(np.arange(n_subj), K)
    starts = np.tile(np.arange(K, dtype=float), n_subj)
    stops = starts + 1.0
    rng = np.random.default_rng(5)
    z = rng.normal(size=n_subj * K)
    x0 = np.zeros(n_subj * K)
    Xp = np.column_stack([x0, z])
    iv = make_survival_y(stops, np.zeros(n_subj * K, dtype=bool), start=starts)
    return Xp, iv, ids


def test_path_effect_delta_zero_is_exactly_zero():
    Xp, iv, ids = _path_fixture()
    m = _stubbed(FOREST, PathOracle())
    res = inspection.path_effect(m, Xp, iv, ids, feature=1, delta=0.0, from_time=3.0, horizons=[4.0, 5.0])
    np.testing.assert_array_equal(res.per_subject, 0.0)


def test_path_effect_oracle_matches_analytic_change():
    Xp, iv, ids = _path_fixture()
    oracle = PathOracle(c=0.15, beta=0.5)
    m = _stubbed(FOREST, oracle)
    delta = 1.0
    from_time = 3.0
    horizons = np.array([4.0, 5.0, 6.0])
    res = inspection.path_effect(m, Xp, iv, ids, feature=1, delta=delta, from_time=from_time, horizons=horizons)

    # analytic: for t >= from_time, shifted rate on rows with start >= from_time uses z+delta
    n_subj = 6
    z = Xp[:, 1].reshape(n_subj, -1)
    starts = np.arange(z.shape[1], dtype=float)
    stops = starts + 1.0
    expected = np.zeros((n_subj, len(horizons)))
    for i in range(n_subj):
        rate_orig = oracle.c * np.exp(oracle.beta * z[i])
        rate_shift = oracle.c * np.exp(oracle.beta * (z[i] + delta))
        rate_row = np.where(starts >= from_time, rate_shift, rate_orig)
        for j, t in enumerate(horizons):
            overlap_orig = np.clip(np.minimum(stops, t) - starts, 0.0, None)
            overlap_shift = np.clip(np.minimum(stops, t) - starts, 0.0, None)
            H_orig = (rate_orig * overlap_orig).sum()
            H_shift = (rate_row * overlap_shift).sum()
            expected[i, j] = (1 - np.exp(-H_shift)) - (1 - np.exp(-H_orig))
    np.testing.assert_allclose(res.per_subject, expected, rtol=1e-8, atol=1e-10)


def test_path_effect_straddling_row_is_split_correctly():
    ids = np.array([0, 0])
    start = np.array([0.0, 2.0])
    stop = np.array([2.0, 4.0])
    z = np.array([1.0, 1.0])
    Xp = np.column_stack([np.zeros(2), z])
    iv = make_survival_y(stop, np.zeros(2, dtype=bool), start=start)
    m = _stubbed(FOREST, PathOracle())
    res_straddle = inspection.path_effect(m, Xp, iv, ids, feature=1, delta=1.0, from_time=1.0, horizons=[3.0])

    ids2 = np.array([0, 0, 0])
    start2 = np.array([0.0, 1.0, 2.0])
    stop2 = np.array([1.0, 2.0, 4.0])
    z2 = np.array([1.0, 1.0, 1.0])
    Xp2 = np.column_stack([np.zeros(3), z2])
    iv2 = make_survival_y(stop2, np.zeros(3, dtype=bool), start=start2)
    res_presplit = inspection.path_effect(m, Xp2, iv2, ids2, feature=1, delta=1.0, from_time=1.0, horizons=[3.0])
    np.testing.assert_allclose(res_straddle.per_subject, res_presplit.per_subject)


def test_path_effect_validation_errors():
    Xp, iv, ids = _path_fixture()
    m = _stubbed(FOREST, PathOracle())
    with pytest.raises(ValueError, match="before the first start"):
        inspection.path_effect(m, Xp, iv, ids, feature=1, delta=1.0, from_time=-1.0, horizons=[1.0])
    with pytest.raises(ValueError, match="beyond the last stop"):
        inspection.path_effect(m, Xp, iv, ids, feature=1, delta=1.0, from_time=3.0, horizons=[20.0])
    res = inspection.path_effect(
        m, Xp, iv, ids, feature=1, delta=1.0, from_time=3.0, horizons=[20.0], extrapolate="locf"
    )
    assert np.isfinite(res.per_subject).all()


def test_path_effect_competing_risks_shape_and_cause():
    K = 4
    n_subj = 5
    ids = np.repeat(np.arange(n_subj), K)
    starts = np.tile(np.arange(K, dtype=float), n_subj)
    stops = starts + 1.0
    rng = np.random.default_rng(7)
    x0, z, U, ev = simulate(200, rng)
    Xc, y2, idc = rows(x0, z, U, ev)
    labels = np.where(y2["event"], 1, 0)
    yc = make_competing_risks_y(y2["stop"], labels, start=y2["start"])
    cr = CompetingRisksForestTV(n_estimators=15, random_state=0, causes=[1]).fit(Xc, yc, idc)
    Xp = np.column_stack([np.zeros(n_subj * K), rng.normal(size=n_subj * K)])
    iv = make_survival_y(stops, np.zeros(n_subj * K, dtype=bool), start=starts)
    res_all = inspection.path_effect(cr, Xp, iv, ids, feature=1, delta=1.0, from_time=1.0, horizons=[2.0, 3.0])
    res_1 = inspection.path_effect(cr, Xp, iv, ids, feature=1, delta=1.0, from_time=1.0, horizons=[2.0, 3.0], cause=1)
    assert res_all.per_subject.shape == (n_subj, 1, 2)
    np.testing.assert_allclose(res_1.per_subject, res_all.per_subject[:, 0, :])


def test_path_effect_landmark_raises_type_error():
    import polars as pl

    event = np.zeros(40, dtype=bool)
    event[1::2][:5] = True  # a few ids have an event at stop=2.0
    df = pl.DataFrame(
        {
            "id": np.repeat(np.arange(20), 2),
            "start": np.tile([0.0, 1.0], 20),
            "stop": np.tile([1.0, 2.0], 20),
            "event": event,
            "z": np.random.default_rng(0).normal(size=40),
        }
    )
    model = LandmarkSurvivalForest(
        id="id", start="start", stop="stop", event="event", landmarks=[1.0], horizon=1.0,
        history_features=["z"], forest=SurvivalForestTV(n_estimators=5, random_state=0),
    ).fit(df)
    with pytest.raises(TypeError, match="landmark"):
        inspection.path_effect(model, np.zeros((2, 1)), make_survival_y(np.array([1.0, 1.0]), np.zeros(2, dtype=bool)), [0, 0], feature=0, delta=1.0, from_time=0.0, horizons=[1.0])
