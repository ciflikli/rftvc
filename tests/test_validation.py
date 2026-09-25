import numpy as np
import pytest

from rftvc import SurvivalForestTV, check_survival_y, make_survival_y


def test_make_and_check_roundtrip():
    y = make_survival_y([2.0, 3.0], [1, 0])
    start, stop, event = check_survival_y(y)
    np.testing.assert_array_equal(start, [0, 0])
    np.testing.assert_array_equal(stop, [2, 3])
    np.testing.assert_array_equal(event, [True, False])


def test_rejects_start_not_before_stop():
    with pytest.raises(ValueError, match="start < stop"):
        check_survival_y(make_survival_y([2.0, 3.0], [1, 0], start=[0.0, 3.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        check_survival_y(make_survival_y([2.0, bad], [1, 0]))


def test_rejects_unstructured_or_missing_fields():
    with pytest.raises(TypeError):
        check_survival_y(np.array([1.0, 2.0]))
    y = np.zeros(2, dtype=[("time", "f8"), ("event", "?")])
    with pytest.raises(TypeError, match="start"):
        check_survival_y(y)


def test_rejects_non_binary_event():
    y = np.zeros(2, dtype=[("start", "f8"), ("stop", "f8"), ("event", "i8")])
    y["stop"], y["event"] = [1, 2], [1, 2]
    with pytest.raises(TypeError, match="event"):
        check_survival_y(y)


def test_rejects_no_events():
    with pytest.raises(ValueError, match="no events"):
        check_survival_y(make_survival_y([1.0, 2.0], [0, 0]))


def test_s1_scope_guards():
    X = np.zeros((3, 1))
    y = make_survival_y([1.0, 2.0, 3.0], [1, 1, 0])
    with pytest.raises(NotImplementedError):
        SurvivalForestTV(n_estimators=2).fit(X, y)
    with pytest.raises(NotImplementedError):
        SurvivalForestTV().fit(X, make_survival_y([1.0, 2.0, 3.0], [1, 1, 0], start=[0.5, 0, 0]))
    with pytest.raises(NotImplementedError):
        SurvivalForestTV().fit(X, y, ids=[1, 1, 2])


@pytest.mark.parametrize("param", [{"min_ids_leaf": 0}, {"max_bins": 300}, {"max_features": "bogus"}, {"max_depth": -1}])
def test_invalid_params(param):
    X = np.zeros((3, 1))
    with pytest.raises(ValueError):
        SurvivalForestTV(**param).fit(X, make_survival_y([1.0, 2.0, 3.0], [1, 1, 0]))


def test_rejects_bad_ids_shape():
    X = np.zeros((3, 1))
    y = make_survival_y([1.0, 2.0, 3.0], [1, 1, 0])
    with pytest.raises(ValueError, match="ids"):
        SurvivalForestTV().fit(X, y, ids=[1, 2])
    with pytest.raises(ValueError, match="ids"):
        SurvivalForestTV().fit(X, y, ids=[[1, 2, 3]])


def test_rejects_nan_prediction_times():
    X = np.zeros((3, 1))
    model = SurvivalForestTV(max_depth=0).fit(X, make_survival_y([1.0, 2.0, 3.0], [1, 1, 0]))
    with pytest.raises(ValueError, match="NaN"):
        model.predict_cumulative_hazard(X, [1.0, np.nan])
