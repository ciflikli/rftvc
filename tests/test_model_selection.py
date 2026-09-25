"""S5: time-aware splitters and landmark cross-validation (leakage and spy tests)."""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.model_selection import GroupKFold, KFold

from rftvc import LandmarkSurvivalForest, SurvivalForestTV, make_landmark_data
from rftvc.metrics import KaplanMeierCensoring
from rftvc.model_selection import GroupTimeSplit, RollingOriginSplit, landmark_cross_validate
from tests.sim_panel import simulate_panel

FEATURES = ["x", ("x", "mean"), "z"]


def _model(**kw):
    forest = SurvivalForestTV(n_estimators=20, min_ids_leaf=5, random_state=0)
    return LandmarkSurvivalForest(horizon=6.0, history_features=FEATURES, step=6.0, forest=forest, **kw)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(n_units=150, n_periods=48, seed=1)


# --- splitters -------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    times=st.lists(st.integers(0, 60), min_size=20, max_size=80),
    n_splits=st.integers(1, 4),
    test_size=st.integers(1, 8),
    gap=st.integers(0, 10),
)
def test_rolling_origin_keeps_test_times_beyond_train_window_plus_gap(times, n_splits, test_size, gap):
    t = np.array(times, dtype=float)
    cv = RollingOriginSplit(n_splits, test_size=test_size, gap=gap)
    try:
        folds = list(cv.split(t))
    except ValueError:
        return  # an empty fold is reported, not silently skipped
    end = t.max()
    assert len(folds) == n_splits
    for k, (train, test) in enumerate(folds):
        hi = end - (n_splits - 1 - k) * test_size
        np.testing.assert_array_equal(test, np.flatnonzero((t > hi - test_size) & (t <= hi)))
        assert t[test].min() - t[train].max() >= gap
        # No test time falls inside [train start, train end + gap).
        assert not np.any((t[test] >= t[train].min()) & (t[test] < t[train].max() + gap))


def test_rolling_origin_errors_on_empty_training_window():
    with pytest.raises(ValueError, match="no training"):
        list(RollingOriginSplit(1, test_size=3, gap=8).split(np.arange(10.0)))


def test_group_time_split_separates_groups_and_times():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 40, 500).astype(float)
    g = rng.integers(0, 30, 500)
    for train, test in GroupTimeSplit(3, test_size=6, gap=4).split(t, groups=g):
        assert not np.intersect1d(g[train], g[test]).size
        assert t[test].min() - t[train].max() >= 4


def test_time_col_reads_a_dataframe_column():
    df = pl.DataFrame({"s": np.arange(20.0), "v": np.zeros(20)})
    a = list(RollingOriginSplit(2, test_size=3, gap=1, time_col="s").split(df))
    b = list(RollingOriginSplit(2, test_size=3, gap=1).split(np.arange(20.0)))
    for (tr1, te1), (tr2, te2) in zip(a, b):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


# --- landmark_cross_validate -------------------------------------------------------


def test_gap_below_horizon_raises(panel):
    with pytest.raises(ValueError, match="gap"):
        landmark_cross_validate(_model(), panel, RollingOriginSplit(2, test_size=12, gap=3))


@pytest.mark.filterwarnings("ignore:The groups parameter is ignored")
def test_splitter_sharing_ids_raises(panel):
    with pytest.raises(ValueError, match="disjoint"):
        landmark_cross_validate(_model(), panel, KFold(3))


class SpyForest(LandmarkSurvivalForest):
    seen = []

    def fit(self, df):
        SpyForest.seen.append((df, np.asarray(self.landmarks)))
        return super().fit(df)


class SpyCensoring(KaplanMeierCensoring):
    seen = []

    def fit(self, y):
        SpyCensoring.seen.append(y.copy())
        return super().fit(y)


@pytest.mark.parametrize("cv", [RollingOriginSplit(3, test_size=6, gap=6), GroupTimeSplit(3, test_size=6, gap=6)])
def test_training_never_sees_data_after_the_first_test_landmark(panel, cv):
    SpyForest.seen = []
    base = _model()
    spy = SpyForest(**base.get_params(deep=False))
    res = landmark_cross_validate(spy, panel, cv, scoring=["brier", "cindex_cumulative"])
    data = make_landmark_data(panel, horizon=6.0, step=6.0, history_features=FEATURES)
    for fold, (df_train, train_s) in enumerate(SpyForest.seen):
        test_s = res.filter(pl.col("fold") == fold)["landmark"].to_numpy()
        cutoff = test_s.min()
        assert df_train["stop"].max() <= cutoff
        assert train_s.max() + 6.0 <= cutoff
        if isinstance(cv, GroupTimeSplit):
            assert not set(df_train["id"]) & set(_fold_test_ids(cv, data, fold))
    assert res["n"].min() > 0 and res["brier"].is_finite().all()


def _fold_test_ids(cv, data, fold):
    _, test = list(cv.split(data.s, groups=data.groups))[fold]
    return np.unique(data.ids[test])


def test_censoring_model_is_fitted_on_each_test_landmark_only(panel):
    SpyCensoring.seen = []
    cv = RollingOriginSplit(2, test_size=6, gap=6)
    res = landmark_cross_validate(_model(), panel, cv, censoring_estimator=SpyCensoring())
    data = make_landmark_data(panel, horizon=6.0, step=6.0, history_features=FEATURES)
    ipcw_rows = res.filter(pl.col("n_censored") > 0)
    assert len(SpyCensoring.seen) == ipcw_rows.height > 0
    for y, s in zip(SpyCensoring.seen, ipcw_rows["landmark"]):
        expected = data.y[data.s == s]  # the test risk set at s, nothing else
        np.testing.assert_array_equal(np.sort(y, order=["stop", "event"]), np.sort(expected, order=["stop", "event"]))


def test_group_kfold_is_new_subject_cv(panel):
    data = make_landmark_data(panel, horizon=6.0, step=6.0, history_features=FEATURES)
    res = landmark_cross_validate(_model(), panel, GroupKFold(3), scoring=["brier", "integrated_brier"])
    assert set(res["landmark"]) == set(np.unique(data.s))  # every landmark row is tested once
    assert res["integrated_brier"].is_finite().all()


def test_nested_cv_selects_parameters_on_the_training_frame(panel):
    outer = RollingOriginSplit(2, test_size=6, gap=6)
    inner = RollingOriginSplit(2, test_size=6, gap=6)
    grid = {"forest__min_ids_leaf": [5, 40]}
    res = landmark_cross_validate(_model(), panel, outer, scoring=["brier"], param_grid=grid, inner_cv=inner)
    assert "params" in res.columns and res["brier"].is_finite().all()
