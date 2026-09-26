"""S5: time-aware splitters and landmark cross-validation (leakage and spy tests)."""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from rftvc import LandmarkSurvivalForest, SurvivalForestTV, make_landmark_data
from rftvc.metrics import KaplanMeierCensoring
from rftvc.model_selection import GroupTimeSplit, RollingOriginSplit, _censor_at, landmark_cross_validate
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
    end = t.max()
    windows = [end - (n_splits - 1 - k) * test_size for k in range(n_splits)]
    empty = any(not np.any((t > hi - test_size) & (t <= hi)) for hi in windows)
    if empty or t.min() > t[(t > windows[0] - test_size) & (t <= windows[0])].min() - gap:
        with pytest.raises(ValueError, match="no t"):  # "no times in test window" / "no training"
            list(cv.split(t))
        return
    folds = list(cv.split(t))
    assert len(folds) == n_splits
    for k, (train, test) in enumerate(folds):
        hi = end - (n_splits - 1 - k) * test_size
        np.testing.assert_array_equal(test, np.flatnonzero((t > hi - test_size) & (t <= hi)))
        assert t[test].min() - t[train].max() >= gap
        # No test time falls inside [train start, train end + gap).
        assert not np.any((t[test] >= t[train].min()) & (t[test] < t[train].max() + gap))


def test_rolling_origin_folds_on_a_fixed_grid():
    folds = list(RollingOriginSplit(3, test_size=5, gap=2).split(np.arange(30.0)))
    expected = [(13, range(15, 20)), (18, range(20, 25)), (23, range(25, 30))]
    for (train, test), (train_end, test_range) in zip(folds, expected):
        np.testing.assert_array_equal(train, np.arange(train_end + 1))
        np.testing.assert_array_equal(test, np.array(test_range))


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


@pytest.mark.parametrize("cv", [RollingOriginSplit(2, test_size=6, gap=6), GroupTimeSplit(2, test_size=6, gap=6)])
def test_censoring_model_is_fitted_on_each_test_landmark_only(panel, cv):
    SpyCensoring.seen = []
    res = landmark_cross_validate(_model(), panel, cv, censoring_estimator=SpyCensoring())
    data = make_landmark_data(panel, horizon=6.0, step=6.0, history_features=FEATURES)
    ipcw_rows = res.filter(pl.col("n_censored") > 0)
    assert len(SpyCensoring.seen) == ipcw_rows.height > 0
    tests = [test for _, test in cv.split(data.s, groups=data.groups)]
    for y, fold, s in zip(SpyCensoring.seen, ipcw_rows["fold"], ipcw_rows["landmark"]):
        test = tests[fold]
        expected = data.y[test[data.s[test] == s]]  # this fold's test risk set at s, nothing else
        if isinstance(cv, GroupTimeSplit):
            assert expected.size < (data.s == s).sum()  # other groups at s are excluded
        np.testing.assert_array_equal(np.sort(y, order=["stop", "event"]), np.sort(expected, order=["stop", "event"]))


def test_group_kfold_is_new_subject_cv(panel):
    SpyForest.seen = []
    data = make_landmark_data(panel, horizon=6.0, step=6.0, history_features=FEATURES)
    spy = SpyForest(**_model().get_params(deep=False))
    cv = GroupKFold(3)
    res = landmark_cross_validate(spy, panel, cv, scoring=["brier", "integrated_brier"])
    assert set(res["landmark"]) == set(np.unique(data.s))  # every landmark row is tested once
    assert res["integrated_brier"].is_finite().all()
    for (df_train, train_s), (train, test) in zip(SpyForest.seen, cv.split(data.s, groups=data.groups)):
        train_ids = np.unique(data.ids[train])
        full = panel.filter(pl.col("id").is_in(train_ids))
        # New-subject CV: whole training histories (no censoring), all their landmarks, no test id.
        assert df_train.sort(["id", "start"]).equals(full.sort(["id", "start"]))
        np.testing.assert_array_equal(train_s, np.unique(data.s[train]))
        assert not np.isin(data.ids[test], train_ids).any()


@pytest.mark.parametrize("scoring, n_times", [(["integrated_brier"], 1), (["brier"], 0)])
def test_invalid_time_grid_raises_instead_of_scoring_nan(panel, scoring, n_times):
    with pytest.raises(ValueError, match="n_times"):
        landmark_cross_validate(_model(), panel, RollingOriginSplit(2, test_size=6, gap=6), scoring, n_times=n_times)


def test_scorer_callables_and_shorter_horizon(panel):
    cv = RollingOriginSplit(2, test_size=6, gap=6)
    seen = {}

    def capture(tag):
        def scorer(y, risk, w, *, censoring_estimator, g_min):
            seen.setdefault(tag, []).append((w, np.asarray(risk).copy()))
            return 42.0
        return scorer

    short = landmark_cross_validate(_model(), panel, cv, scoring={"s": capture(3.0)}, horizon=3.0)
    full = landmark_cross_validate(_model(), panel, cv, scoring={"s": capture(6.0)})
    assert (short["s"] == 42.0).all() and short["n"].equals(full["n"])
    assert {w for w, _ in seen[3.0]} == {3.0} and {w for w, _ in seen[6.0]} == {6.0}
    r3 = np.concatenate([r for _, r in seen[3.0]])
    r6 = np.concatenate([r for _, r in seen[6.0]])
    assert np.all(r3 <= r6 + 1e-12) and np.any(r3 < r6)  # risk by w=3 is risk at 3, not at 6
    with pytest.raises(ValueError, match="horizon"):
        landmark_cross_validate(_model(), panel, cv, horizon=7.0)


@pytest.mark.parametrize("refit, best", [("brier", np.argmin), ("cindex_cumulative", np.argmax)])
def test_nested_cv_selects_the_best_inner_candidate(panel, refit, best):
    outer = RollingOriginSplit(1, test_size=6, gap=6)
    inner = RollingOriginSplit(2, test_size=6, gap=6)
    candidates = [{"forest__max_depth": 0}, {"forest__max_depth": 3}]
    grid = {"forest__max_depth": [0, 3]}
    scoring = ["brier", "cindex_cumulative"]
    model = _model()
    res = landmark_cross_validate(model, panel, outer, scoring=scoring, param_grid=grid, inner_cv=inner, refit=refit)

    # Independent replay of the inner loop on the outer training frame.
    data = make_landmark_data(panel, horizon=6.0, step=6.0, history_features=FEATURES)
    ((train, test),) = outer.split(data.s)
    df_train = _censor_at(
        panel.filter(pl.col("id").is_in(np.unique(data.ids[train]))), data.s[test].min(),
        start="start", stop="stop", event="event",
    )
    means = []
    for params in candidates:
        m = clone(model).set_params(landmarks=np.unique(data.s[train]), step=None, **params)
        inner_res = landmark_cross_validate(m, df_train, inner, scoring=scoring)
        means.append(float(np.nanmean(inner_res[refit].to_numpy())))
    assert means[0] != means[1]
    assert set(res["params"]) == {repr(candidates[int(best(means))])}


# --- out-of-fold predictions (S8) -------------------------------------------------


def test_returned_predictions_reproduce_every_fold_landmark_score(panel):
    from rftvc import make_survival_y
    from rftvc.metrics import brier_landmark, integrated_brier

    cv = RollingOriginSplit(2, test_size=6, gap=6)
    scoring = ["brier", "integrated_brier"]
    scores, preds = landmark_cross_validate(_model(), panel, cv, scoring=scoring, n_times=4,
                                            return_predictions=True)
    times = np.linspace(0, 6.0, 5)[1:]
    assert len(scores) > 1
    for row in scores.iter_rows(named=True):
        p = preds.filter((pl.col("fold") == row["fold"]) & (pl.col("landmark") == row["landmark"]))
        assert len(p) == row["n"]
        y = make_survival_y(p["time"].to_numpy(), p["event"].to_numpy())
        cens = KaplanMeierCensoring().fit(y)
        S = np.vstack(p["survival"].to_list())
        np.testing.assert_allclose(p["risk"].to_numpy(), 1.0 - S[:, -1])
        assert brier_landmark(y, p["risk"].to_numpy(), 6.0, censoring_estimator=cens) == pytest.approx(row["brier"])
        assert integrated_brier(y, S, times, censoring_estimator=cens) == pytest.approx(row["integrated_brier"])


def test_new_subject_cv_predicts_every_landmark_row_once(panel):
    data = make_landmark_data(panel, horizon=6.0, step=6.0, history_features=FEATURES)
    scores, preds = landmark_cross_validate(_model(), panel, GroupKFold(3), return_predictions=True)
    got = sorted(zip(preds["landmark"].to_list(), preds["id"].to_list()))
    assert got == sorted(zip(data.s.tolist(), data.ids.tolist()))
    # Each id is predicted in exactly one fold.
    assert preds.group_by("id").agg(pl.col("fold").n_unique())["fold"].max() == 1
    # Outcomes are on the reset clock.
    np.testing.assert_array_equal(np.sort(preds["time"].to_numpy()), np.sort(data.y["stop"]))


def test_default_return_is_unchanged_and_nested_cv_returns_predictions(panel):
    cv = RollingOriginSplit(1, test_size=6, gap=6)
    plain = landmark_cross_validate(_model(), panel, cv)
    assert isinstance(plain, pl.DataFrame)
    scores, preds = landmark_cross_validate(
        _model(), panel, cv, param_grid={"forest__max_depth": [0, 3]},
        inner_cv=RollingOriginSplit(2, test_size=6, gap=6), return_predictions=True,
    )
    assert "params" in scores.columns and len(preds) == int(scores["n"].sum())
    assert scores.drop("params").columns == plain.columns
