"""S13: landmark competing risks (stacks with cause labels, wrapper, cross-validation)."""

import pickle
import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from rftvc import (
    CR_DTYPE,
    SURV_DTYPE,
    CompetingRisksForestTV,
    LandmarkCompetingRisksForest,
    LandmarkSurvivalForest,
    SurvivalForestTV,
    landmark_features,
    make_landmark_data,
)
from rftvc.metrics import UndefinedMetricError, brier_landmark, cindex_dynamic
from rftvc.model_selection import RollingOriginSplit, _censor_at, landmark_cross_validate
from tests.fixtures.pbcseq import pbcseq_competing_risks, pbcseq_counting_process

H = 730.5
LANDMARKS = [365.25, 730.5, 1095.75]
FEATURES = ["log_bili", "albumin", ("log_bili", "max")]


@pytest.fixture(scope="module")
def pbc():
    return pbcseq_competing_risks()


def _forest(**kw):
    return CompetingRisksForestTV(n_estimators=8, min_ids_leaf=5, random_state=0, **kw)


def _model(**kw):
    return LandmarkCompetingRisksForest(horizon=H, landmarks=LANDMARKS, history_features=FEATURES,
                                        forest=_forest(), **kw)


# --- stacks ------------------------------------------------------------------------


def test_boolean_and_binary_integer_columns_give_todays_stacks():
    df = pbcseq_counting_process()
    kw = dict(horizon=H, landmarks=LANDMARKS, history_features=FEATURES)
    a = make_landmark_data(df, **kw)
    b = make_landmark_data(df.with_columns(pl.col("event").cast(pl.Int64)), **kw)
    assert a.y.dtype == b.y.dtype == SURV_DTYPE
    for x, z in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(z))


def test_cause_labels_on_terminal_rows_within_the_horizon(pbc):
    d = make_landmark_data(pbc, horizon=H, landmarks=LANDMARKS, history_features=FEATURES)
    assert d.y.dtype == CR_DTYPE
    subj = pbc.group_by("id").agg(pl.col("stop").max().alias("U"), pl.col("event").max().alias("label"))
    U = dict(zip(subj["id"].to_list(), subj["U"].to_list()))
    label = dict(zip(subj["id"].to_list(), subj["label"].to_list()))
    want = np.array([label[i] if U[i] <= s + H else 0 for i, s in zip(d.ids, d.s)])
    np.testing.assert_array_equal(d.y["event"], want)
    np.testing.assert_array_equal(d.y["stop"], [min(U[i], s + H) - s for i, s in zip(d.ids, d.s)])
    assert set(np.unique(d.y["event"])) == {0, 1, 2}
    # A cause-2 death after s + w is censored at the horizon.
    late = np.array([label[i] == 2 and U[i] > s + H for i, s in zip(d.ids, d.s)])
    assert late.any() and (d.y["event"][late] == 0).all() and np.allclose(d.y["stop"][late], H)


def test_bad_labels_in_the_event_column_raise(pbc):
    kw = dict(horizon=H, landmarks=LANDMARKS, history_features=["albumin"])
    first_row = pl.int_range(pl.len()).over("id") == 0
    multi = pbc.filter(pl.len().over("id") > 1)
    with pytest.raises(ValueError, match="not the id's last row"):
        make_landmark_data(multi.with_columns(pl.when(first_row).then(2).otherwise(pl.col("event")).alias("event")),
                           **kw)
    with pytest.raises(ValueError, match="non-negative"):
        make_landmark_data(pbc.with_columns(pl.col("event") - 1), **kw)


def test_survival_wrapper_rejects_cause_labels(pbc):
    with pytest.raises(ValueError, match="LandmarkCompetingRisksForest"):
        LandmarkSurvivalForest(horizon=H, landmarks=LANDMARKS, history_features=FEATURES,
                               forest=SurvivalForestTV(n_estimators=2)).fit(pbc)


# --- wrapper ---------------------------------------------------------------------------


def test_wrapper_predictions_are_the_forests_incidence(pbc):
    m = _model(score_cause=2).fit(pbc)
    np.testing.assert_array_equal(m.causes_, [1, 2])
    s = 730.5
    ids, X = landmark_features(pbc, s, history_features=FEATURES, event="event")
    r = m.predict_risk(pbc, s)
    np.testing.assert_array_equal(r["id"].to_numpy(), ids)
    assert r["cause"].to_list() == [2] * len(ids)
    np.testing.assert_array_equal(r["risk"].to_numpy(), m.forest_.predict_cumulative_incidence(X, [H], cause=2)[:, 0])
    r1 = m.predict_risk(pbc, s, horizon=365.0, cause=1)
    np.testing.assert_array_equal(r1["risk"].to_numpy(),
                                  m.forest_.predict_cumulative_incidence(X, [365.0], cause=1)[:, 0])
    _, F = m.predict_cumulative_incidence(pbc, s, [100.0, H])
    _, S = m.predict_survival_function(pbc, s, [100.0, H])
    assert F.shape == (len(ids), 2, 2)
    np.testing.assert_allclose(F.sum(axis=1) + S, 1.0, atol=1e-12)
    with pytest.raises(ValueError, match="horizon"):
        m.predict_cumulative_incidence(pbc, s, [H + 1])


def test_vocabulary_keeps_the_cause_axis_without_a_cause(pbc):
    transplanted = pbc.filter(pl.col("event") == 1)["id"].unique()
    no_transplant = pbc.filter(~pl.col("id").is_in(transplanted))
    with pytest.warns(UserWarning, match=r"causes \[1\] have no events"):
        m = _model(causes=[1, 2]).fit(no_transplant)
    assert m.forest_.n_causes_ == 2
    _, F = m.predict_cumulative_incidence(no_transplant, 730.5, [H])
    np.testing.assert_array_equal(F[:, 0], 0.0)


def test_wrapper_clone_params_and_pickle(pbc):
    m = _model(causes=[1, 2])
    c = clone(m)
    assert c.get_params()["forest__n_estimators"] == 8 and c.get_params()["causes"] == [1, 2]
    m.fit(pbc)
    r = m.predict_risk(pbc, 730.5)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(m)).predict_risk(pbc, 730.5)["risk"], r["risk"])
    with pytest.raises(TypeError, match="CompetingRisksForestTV"):
        LandmarkCompetingRisksForest(horizon=H, landmarks=LANDMARKS, history_features=FEATURES,
                                     forest=SurvivalForestTV()).fit(pbc)


# --- cross-validation -------------------------------------------------------------------------


def _manual_group_cv(model, df, n_splits, w, n_times):
    """The CV loop by hand: fixed vocabulary + cause, per-fold fit, F_k, cause-specific metrics."""
    data = make_landmark_data(df, horizon=model.horizon, landmarks=model.landmarks,
                              history_features=model.history_features)
    labels = data.y["event"]
    causes = sorted(int(c) for c in np.unique(labels[labels != 0]))
    k = causes[0] if model.score_cause is None else model.score_cause
    times = np.linspace(0, w, n_times + 1)[1:]
    out = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits).split(data.s, groups=data.groups)):
        train_ids = np.unique(data.ids[tr])
        fitted = clone(model).set_params(landmarks=np.unique(data.s[tr]), step=None, causes=causes,
                                         score_cause=k).fit(df.filter(pl.col("id").is_in(train_ids)))
        for s in np.unique(data.s[te]):
            m = te[data.s[te] == s]
            F = fitted.forest_.predict_cumulative_incidence(data.X[m], times, cause=k)
            y = data.y[m]
            brier = brier_landmark(y, F[:, -1], w, cause=k, y_censor=y)
            c = {}
            for kind in ("incident", "cumulative"):
                try:
                    c[kind] = cindex_dynamic(y, F[:, -1], w, kind=kind, cause=k, y_censor=y)
                except UndefinedMetricError:
                    c[kind] = np.nan
            out.append((fold, float(s), brier, c["incident"], c["cumulative"]))
    return out


def test_cv_equals_the_manual_loop(pbc):
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res, preds = landmark_cross_validate(model, pbc, GroupKFold(3),
                                             ("brier", "cindex_incident", "cindex_cumulative"),
                                             n_times=4, return_predictions=True)
        manual = _manual_group_cv(model, pbc, 3, H, 4)
    got = list(zip(res["fold"], res["landmark"], res["brier"], res["cindex_incident"], res["cindex_cumulative"]))
    assert len(got) == len(manual)
    for g, m in zip(got, manual):
        assert g[:2] == m[:2]
        assert g[2] == pytest.approx(m[2], abs=1e-15)
        for a, b in zip(g[3:], m[3:]):
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-15)
    assert np.isfinite(res["cindex_cumulative"].to_numpy()).any()
    assert set(preds.columns) >= {"cause", "risk", "cif"} and "survival" not in preds.columns
    assert preds["cause"].unique().to_list() == [1]
    assert all(len(v) == 4 for v in preds["cif"].to_list())
    assert np.isfinite(res["brier"].to_numpy()).all()


class _Fixed:
    """A splitter with given (train, test) folds over stacked rows, chosen by id."""

    def __init__(self, folds):
        self.folds = folds

    def split(self, s, groups=None):
        for test_ids in self.folds:
            test = np.isin(groups, test_ids)
            yield np.flatnonzero(~test), np.flatnonzero(test)


def test_folds_lacking_a_cause_keep_the_scored_cause_and_shape(pbc):
    transplanted = pbc.filter(pl.col("event") == 1)["id"].unique().to_numpy()
    others = np.setdiff1d(pbc["id"].unique().to_numpy(), transplanted)
    # Fold 0 tests every transplanted id, so its training data has no cause 1.
    cv = _Fixed([transplanted, others[: len(others) // 2]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res, preds = landmark_cross_validate(_model(), pbc, cv, ("brier",), n_times=3, return_predictions=True)
    assert preds["cause"].unique().to_list() == [1]
    fold0 = preds.filter(pl.col("fold") == 0)
    np.testing.assert_array_equal(np.vstack(fold0["cif"].to_list()), 0.0)  # cause 1 unseen in training
    assert np.isfinite(res["brier"].to_numpy()).all()


def test_time_split_and_nested_selection_run(pbc):
    model = _model(score_cause=2)
    wide = clone(model).set_params(landmarks=[365.25, 730.5, 1095.75, 1461.0, 1826.25])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Test landmark 1826.25; training landmarks <= 1826.25 - horizon.
        res = landmark_cross_validate(wide, pbc, RollingOriginSplit(n_splits=1, test_size=100.0, gap=H),
                                      ("brier", "integrated_brier"), n_times=3)
        nested = landmark_cross_validate(model, pbc, GroupKFold(2), ("brier",), n_times=2,
                                         param_grid={"forest__max_depth": [1, None]}, inner_cv=GroupKFold(2))
    assert res.height >= 1 and np.isfinite(res["brier"].to_numpy()).all()
    assert nested["params"].str.contains("max_depth").all()


def test_censor_at_keeps_cause_labels():
    df = pl.DataFrame({"id": [1, 1, 2, 3], "start": [0.0, 1.0, 0.0, 0.0], "stop": [1.0, 3.0, 2.0, 5.0],
                       "event": [0, 2, 1, 1]})
    out = _censor_at(df, 2.5, start="start", stop="stop", event="event")
    assert out["event"].dtype == df["event"].dtype
    assert out["event"].to_list() == [0, 0, 1, 0] and out["stop"].to_list() == [1.0, 2.5, 2.0, 2.5]
    b = _censor_at(df.with_columns(pl.col("event") > 0), 2.5, start="start", stop="stop", event="event")
    assert b["event"].dtype == pl.Boolean and b["event"].to_list() == [False, False, True, False]


@pytest.mark.parametrize("kw, match", [({"score_cause": 1.5}, "score_cause"), ({"score_cause": 3}, "score_cause"),
                                        ({"causes": [1.5, 2]}, "integers")])
def test_cv_rejects_invalid_cause_labels(pbc, kw, match):
    with pytest.raises(ValueError, match=match):
        landmark_cross_validate(_model(**kw), pbc, GroupKFold(2), ("brier",))
