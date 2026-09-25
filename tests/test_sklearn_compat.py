"""S7: scikit-learn compatibility matrix.

``check_estimator`` runs every sklearn check. Checks that generate a plain
numeric ``y`` cannot fit a survival model (``y`` must be a structured
``(start, stop, event)`` target), so they are expected to fail; each one is
covered below by a survival-adapted test with the same intent.
"""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
import sklearn
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from rftvc import LandmarkSurvivalForest, SurvivalForestTV, make_survival_y
from tests.test_tvc import _cp_data

Y_REASON = "y must be a structured survival target; covered by {}"
# sklearn check -> the survival-adapted test in this file that covers it
EXPECTED_FAILED = {
    "check_dict_unchanged": "test_predict_leaves_state_unchanged",
    "check_dont_overwrite_parameters": "test_fit_returns_self_and_keeps_params",
    "check_estimators_overwrite_params": "test_fit_returns_self_and_keeps_params",
    "check_estimators_fit_returns_self": "test_fit_returns_self_and_keeps_params",
    "check_dtype_object": "test_object_and_integer_dtypes",
    "check_estimators_dtypes": "test_object_and_integer_dtypes",
    "check_estimators_nan_inf": "test_nan_and_inf_are_rejected",
    "check_estimators_pickle": "test_pickle_and_clone",
    "check_f_contiguous_array_estimator": "test_memory_layouts",
    "check_readonly_memmap_input": "test_memory_layouts",
    "check_fit2d_1feature": "test_degenerate_shapes",
    "check_fit2d_1sample": "test_degenerate_shapes",
    "check_fit2d_predict1d": "test_degenerate_shapes",
    "check_fit_check_is_fitted": "test_unfitted_raises",
    "check_fit_idempotent": "test_fit_is_idempotent",
    "check_methods_sample_order_invariance": "test_sample_order_and_subset_invariance",
    "check_methods_subset_invariance": "test_sample_order_and_subset_invariance",
    "check_n_features_in": "test_n_features_and_feature_names",
    "check_n_features_in_after_fitting": "test_n_features_and_feature_names",
    "check_pipeline_consistency": "test_pipeline_and_cross_validate_with_routed_ids",
    "check_fit_score_takes_y": "test_pipeline_and_cross_validate_with_routed_ids",
    "check_requires_y_none": "test_requires_y",
    "check_positive_only_tag_during_fit": "test_object_and_integer_dtypes",
}


def _est(**kw):
    return SurvivalForestTV(n_estimators=8, min_ids_leaf=3, random_state=0, **kw)


@parametrize_with_checks(
    [_est()],
    expected_failed_checks=lambda est: {k: Y_REASON.format(v) for k, v in EXPECTED_FAILED.items()},
)
def test_sklearn_check_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def data():
    X, y, ids = _cp_data(60, seed=2)
    return X, y, ids


def _fitted(data, **kw):
    X, y, ids = data
    return _est(**kw).fit(X, y, ids)


def test_fit_returns_self_and_keeps_params(data):
    X, y, ids = data
    est = _est(max_features=None)
    before = est.get_params(deep=True)
    assert est.fit(X, y, ids) is est
    after = est.get_params(deep=True)
    assert before.keys() == after.keys() and all(before[k] is after[k] or before[k] == after[k] for k in before)
    # No public attribute other than fitted ones (trailing underscore) is added by fit.
    assert all(k.endswith("_") or k in before for k in vars(est) if not k.startswith("_"))


def test_predict_leaves_state_unchanged(data):
    est = _fitted(data)
    X = data[0]
    state = {k: v for k, v in vars(est).items() if k != "forest_"}
    est.predict(X)
    est.predict_survival_function(X[:3])
    est.score(X, data[1], data[2])
    for k, v in state.items():
        np.testing.assert_equal(vars(est)[k], v)


def test_fit_is_idempotent(data):
    X, y, ids = data
    est = _est()
    a = est.fit(X, y, ids).predict(X)
    b = est.fit(X, y, ids).predict(X)
    np.testing.assert_array_equal(a, b)


def test_sample_order_and_subset_invariance(data):
    est = _fitted(data)
    X = data[0]
    perm = np.random.default_rng(0).permutation(X.shape[0])
    full = est.predict_cumulative_hazard(X)
    np.testing.assert_allclose(est.predict_cumulative_hazard(X[perm]), full[perm], rtol=1e-12)
    np.testing.assert_allclose(est.predict(X[:7]), est.predict(X)[:7], rtol=1e-12)


def test_object_and_integer_dtypes(data):
    X, y, ids = data
    ref = _est().fit(X, y, ids).predict(X)
    np.testing.assert_allclose(_est().fit(X.astype(object), y, ids).predict(X.astype(object)), ref)
    Xi = np.round(X * 10).astype(np.int64)
    assert np.isfinite(_est().fit(Xi, y, ids).predict(Xi)).all()
    assert np.isfinite(_est().fit(-np.abs(X), y, ids).predict(X)).all()  # negative values are fine
    with pytest.raises((TypeError, ValueError)):
        _est().fit(np.full(X.shape, "a", dtype=object), y, ids)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_and_inf_are_rejected(data, bad):
    X, y, ids = data
    Xb = X.copy()
    Xb[0, 0] = bad
    with pytest.raises(ValueError):
        _est().fit(Xb, y, ids)
    with pytest.raises(ValueError):
        _fitted(data).predict(Xb)


def test_pickle_and_clone(data):
    est = _fitted(data)
    X = data[0]
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(est)).predict(X), est.predict(X))
    c = clone(est)
    assert c.get_params() == est.get_params() and not hasattr(c, "forest_")


def test_memory_layouts(data, tmp_path):
    X, y, ids = data
    ref = _est().fit(X, y, ids).predict(X)
    np.testing.assert_allclose(_est().fit(np.asfortranarray(X), y, ids).predict(np.asfortranarray(X)), ref)
    mm = np.memmap(tmp_path / "x.dat", dtype=np.float64, mode="w+", shape=X.shape)
    mm[:] = X
    mm.flush()
    ro = np.memmap(tmp_path / "x.dat", dtype=np.float64, mode="r", shape=X.shape)
    np.testing.assert_allclose(_est().fit(ro, y, ids).predict(ro), ref)


def test_degenerate_shapes(data):
    X, y, ids = data
    one = _est().fit(X[:, :1], y, ids)
    assert one.n_features_in_ == 1 and one.predict(X[:, :1]).shape == (X.shape[0],)
    single = make_survival_y([2.0], [True])
    assert _est().fit(X[:1], single).predict(X[:1]).shape == (1,)  # one subject: a single Nelson–Aalen leaf
    with pytest.raises(ValueError, match="no events"):
        _est().fit(X[:1], make_survival_y([2.0], [False]))
    with pytest.raises(ValueError):
        _fitted(data).predict(X[0])  # 1-d X


def test_unfitted_raises(data):
    with pytest.raises(NotFittedError):
        _est().predict(data[0])


def test_requires_y(data):
    assert _est().__sklearn_tags__().target_tags.required
    with pytest.raises((TypeError, ValueError)):
        _est().fit(data[0], None)


def test_n_features_and_feature_names(data):
    X, y, ids = data
    est = _est().fit(X, y, ids)
    assert est.n_features_in_ == X.shape[1] and not hasattr(est, "feature_names_in_")
    with pytest.raises(ValueError, match="features"):
        est.predict(X[:, :1])
    for frame in (pd.DataFrame(X, columns=["z", "w"]), pl.DataFrame(X, schema=["z", "w"])):
        est = _est().fit(frame, y, ids)
        assert list(est.feature_names_in_) == ["z", "w"]
        np.testing.assert_allclose(est.predict(frame), _est().fit(X, y, ids).predict(X))
    with pytest.raises(ValueError, match="feature names"):
        est.predict(pl.DataFrame(X, schema=["w", "z"]))
    est.fit(X, y, ids)
    assert not hasattr(est, "feature_names_in_")  # refit on an array clears the names


def test_dataframe_ids_column_and_dataframe_y(data):
    X, y, ids = data
    ref = _est().fit(X, y, ids)
    df = pd.DataFrame({"z": X[:, 0], "w": X[:, 1], "subject": ids})
    ydf = pl.DataFrame({"start": y["start"], "stop": y["stop"], "event": y["event"]})
    est = _est().fit(df, ydf, ids="subject")
    assert list(est.feature_names_in_) == ["z", "w"] and est.ids_column_ == "subject"
    np.testing.assert_allclose(est.predict(df), ref.predict(X))  # the id column is dropped at predict
    iv = pd.DataFrame({"start": y["start"], "stop": y["stop"]})
    np.testing.assert_allclose(
        est.predict_cumulative_hazard(df, [1.0, 2.0], intervals=iv, ids="subject"),
        ref.predict_cumulative_hazard(X, [1.0, 2.0], intervals=y, ids=ids),
    )
    # ids as an array with the id column still present: the column is dropped, the array groups paths.
    np.testing.assert_allclose(
        est.predict_cumulative_hazard(df, [1.0, 2.0], intervals=iv, ids=ids),
        ref.predict_cumulative_hazard(X, [1.0, 2.0], intervals=y, ids=ids),
    )
    with pytest.raises(ValueError, match="not in X"):
        est.predict_cumulative_hazard(df, [1.0], intervals=iv, ids="missing")
    nullable = pd.DataFrame({"z": pd.array([1, None, 2], dtype="Int64"), "w": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="NaN"):
        est.predict(nullable)


class _IdsSpy(SurvivalForestTV):
    seen = []

    def fit(self, X, y, ids=None, **kw):
        _IdsSpy.seen.append(np.asarray(ids).copy())
        return super().fit(X, y, ids, **kw)


def test_pipeline_and_cross_validate_with_routed_ids(data):
    X, y, ids = data
    pipe = make_pipeline(StandardScaler(), _est())
    pipe.fit(X, y)
    assert 0 <= pipe.score(X, y) <= 1 and pipe.predict(X).shape == (X.shape[0],)
    _IdsSpy.seen = []
    with sklearn.config_context(enable_metadata_routing=True):
        spy = _IdsSpy(n_estimators=8, min_ids_leaf=3, random_state=0).set_fit_request(ids=True).set_score_request(ids=True)
        cv = GroupKFold(3)
        res = cross_validate(spy, X, y, cv=cv, params={"ids": ids, "groups": ids})
    assert np.all((res["test_score"] > 0) & (res["test_score"] < 1))
    for seen, (train, _) in zip(_IdsSpy.seen, cv.split(X, y, groups=ids)):
        np.testing.assert_array_equal(seen, ids[train])  # ids reach fit, sliced to the fold


def test_landmark_forest_clone_params_and_pickle():
    from tests.fixtures.pbcseq import pbcseq_counting_process

    m = LandmarkSurvivalForest(horizon=730.0, history_features=["log_bili", "albumin"], step=365.0,
                               forest=_est())
    c = clone(m)
    assert c.get_params(deep=False).keys() == m.get_params(deep=False).keys()
    assert c.get_params()["forest__n_estimators"] == 8
    df = pbcseq_counting_process()
    m.fit(df)
    r = m.predict_risk(df, 730.0)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(m)).predict_risk(df, 730.0)["risk"], r["risk"])
