"""S20: ``inspection.hazard_effect`` / ``inspection.path_effect``."""

import copy

import numpy as np
import pandas as pd
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
    """Hazard rate ``c * exp(beta * z + gamma * x0)`` (z is the replaced column 1;
    ``x0``, column 0, is left alone by ``hazard_effect``'s grid substitution, so the
    post-substitution rate genuinely varies row to row -- unlike a rate that depends
    only on the replaced feature, which would be identical across every row and make
    the exposure weighting untestable (every weighting scheme, including a broken
    one, averages a constant to itself)."""

    def __init__(self, c=0.2, beta=0.6, gamma=0.3):
        self.c, self.beta, self.gamma = c, beta, gamma

    def predict_cumhaz(self, X, times, aggregate, n_jobs):
        rate = self.c * np.exp(self.beta * X[:, 1] + self.gamma * X[:, 0])
        return rate[:, None] * np.asarray(times, dtype=float)[None, :]


def test_hazard_effect_oracle_matches_hand_derived_exposure_weighted_rate():
    from rftvc.metrics import _window_exposure_1

    stub = HazardOracle(c=0.2, beta=0.6, gamma=0.3)
    m = _stubbed(FOREST, stub)
    values = np.array([-1.0, 0.0, 1.0, 2.0])
    w = np.array([0.0, 1.0, 3.0, 6.0])
    res = inspection.hazard_effect(m, X, Y, feature=1, values=values, windows=w)
    start, stop, x0 = Y["start"], Y["stop"], X[:, 0]
    for i, v in enumerate(values):
        row_rate = stub.c * np.exp(stub.beta * v + stub.gamma * x0)
        for mi in range(len(w) - 1):
            e = _window_exposure_1(start, stop, w[mi], w[mi + 1])
            at_risk = e > 0
            expected = (e[at_risk] * row_rate[at_risk]).sum() / e[at_risk].sum()
            assert np.isclose(res.hazard[i, mi], expected, rtol=1e-8), (i, mi)


def test_hazard_effect_weighted_average_uses_true_per_row_exposures():
    # 3 rows with unequal, hand-computed per-window exposures and distinct, X-independent
    # per-row rates, so a naive unweighted mean over at-risk rows gives a different (wrong)
    # answer than the correct exposure-weighted one in both windows -- this actually
    # exercises the weighting, unlike a fixture with equal weights or a constant rate.
    X3 = np.array([[0.0], [0.0], [0.0]])
    start = np.array([0.0, 0.0, 1.0])
    stop = np.array([2.0, 0.5, 1.5])
    y3 = make_survival_y(stop, np.array([True, False, False]), start=start)
    m = SurvivalForestTV(n_estimators=5, random_state=0).fit(X3, y3)

    class RowRateOracle:
        RATE = np.array([1.0, 2.0, 4.0])  # by row position; hazard_effect never reorders rows

        def predict_cumhaz(self, X, times, aggregate, n_jobs):
            return self.RATE[:, None] * np.asarray(times, dtype=float)[None, :]

    m = _stubbed(m, RowRateOracle())
    res = inspection.hazard_effect(m, X3, y3, feature=0, values=[0.0], windows=np.array([0.0, 1.0, 2.0]))
    # window (0,1]: exposures [1, 0.5, 0] -> weighted (1*1 + 0.5*2)/1.5 = 4/3; naive mean = 1.5
    # window (1,2]: exposures [1, 0, 0.5] -> weighted (1*1 + 0.5*4)/1.5 = 2.0; naive mean = 2.5
    np.testing.assert_allclose(res.hazard[0], [4.0 / 3.0, 2.0], atol=1e-10)


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
    # two real causes, and cause=2 is NOT the first axis: with a single-cause fixture
    # (as an earlier draft of this test used), selecting "index 0 of a length-1 axis"
    # would pass even if the cause-index lookup were broken (any bug that always picks
    # index 0 is invisible when there is only one cause).
    rng = np.random.default_rng(2)
    x0, z, U, ev = simulate(300, rng)
    Xc, y2, idc = rows(x0, z, U, ev)
    cause_draw = rng.integers(1, 3, size=y2["event"].shape[0])  # 1 or 2
    labels = np.where(y2["event"], cause_draw, 0)
    yc = make_competing_risks_y(y2["stop"], labels, start=y2["start"])
    m = CompetingRisksForestTV(n_estimators=20, random_state=0, causes=[1, 2]).fit(Xc, yc, idc)
    k = int(np.flatnonzero(m.causes_ == 2)[0])
    assert k != 0, "fixture must put cause 2 at a non-zero axis position for this test to be meaningful"
    res_all = inspection.hazard_effect(m, Xc, yc, feature=1, values=[0.0, 1.0], windows=4)
    res_2 = inspection.hazard_effect(m, Xc, yc, feature=1, values=[0.0, 1.0], windows=4, cause=2)
    assert res_all.hazard.shape == (2, 2, 4)
    np.testing.assert_allclose(res_2.hazard, res_all.hazard[:, k, :])


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


def test_hazard_effect_ids_column_is_not_a_feature():
    df = pd.DataFrame({"id": IDS, "x0": X[:, 0], "z": X[:, 1]})
    m = SurvivalForestTV(n_estimators=5, random_state=0).fit(df, Y, ids="id")
    with pytest.raises(ValueError, match="ids column"):
        inspection.hazard_effect(m, df, Y, feature="id")


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


def test_path_effect_nonzero_origin_is_actually_used():
    # a bug that silently ignored the origin= argument (e.g. always using each
    # subject's first start) would make this identical to the default-origin call.
    Xp, iv, ids = _path_fixture()
    m = _stubbed(FOREST, PathOracle(c=0.15, beta=0.5))
    horizons = np.array([4.0, 5.0])
    res_default = inspection.path_effect(m, Xp, iv, ids, feature=1, delta=1.0, from_time=3.0, horizons=horizons)
    res_origin1 = inspection.path_effect(
        m, Xp, iv, ids, feature=1, delta=1.0, from_time=3.0, horizons=horizons, origin=1.0
    )
    assert not np.allclose(res_default.per_subject, res_origin1.per_subject)


def test_path_effect_horizons_before_from_time_raises():
    Xp, iv, ids = _path_fixture()
    m = _stubbed(FOREST, PathOracle())
    with pytest.raises(ValueError, match="from_time"):
        inspection.path_effect(m, Xp, iv, ids, feature=1, delta=1.0, from_time=3.0, horizons=[2.0])


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
    # two real causes, cause=2 selected at a non-zero axis position (see the
    # matching note on test_hazard_effect_competing_risks_cause_selection: a
    # single-cause fixture cannot catch an index-lookup bug that always picks axis 0).
    K = 4
    n_subj = 5
    ids = np.repeat(np.arange(n_subj), K)
    starts = np.tile(np.arange(K, dtype=float), n_subj)
    stops = starts + 1.0
    rng = np.random.default_rng(7)
    x0, z, U, ev = simulate(300, rng)
    Xc, y2, idc = rows(x0, z, U, ev)
    cause_draw = rng.integers(1, 3, size=y2["event"].shape[0])
    labels = np.where(y2["event"], cause_draw, 0)
    yc = make_competing_risks_y(y2["stop"], labels, start=y2["start"])
    cr = CompetingRisksForestTV(n_estimators=15, random_state=0, causes=[1, 2]).fit(Xc, yc, idc)
    k = int(np.flatnonzero(cr.causes_ == 2)[0])
    assert k != 0, "fixture must put cause 2 at a non-zero axis position for this test to be meaningful"
    Xp = np.column_stack([np.zeros(n_subj * K), rng.normal(size=n_subj * K)])
    iv = make_survival_y(stops, np.zeros(n_subj * K, dtype=bool), start=starts)
    res_all = inspection.path_effect(cr, Xp, iv, ids, feature=1, delta=1.0, from_time=1.0, horizons=[2.0, 3.0])
    res_2 = inspection.path_effect(cr, Xp, iv, ids, feature=1, delta=1.0, from_time=1.0, horizons=[2.0, 3.0], cause=2)
    assert res_all.per_subject.shape == (n_subj, 2, 2)
    np.testing.assert_allclose(res_2.per_subject, res_all.per_subject[:, k, :])


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
