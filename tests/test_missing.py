"""End-to-end missing-value routing through both public forest estimators."""

import pickle

import numpy as np
import pytest

from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y, make_survival_y
from rftvc.viz import plot_tree


def _data(competing=False):
    n = 40
    x = np.full((n, 2), 7.0)
    x[: n // 2, 0] = np.nan
    x[:, 1] = np.nan  # a feature with no observed values cannot offer a split
    stop = np.r_[np.arange(1, 21), np.arange(21, 41)].astype(float)
    if competing:
        codes = np.where(np.arange(n) % 3 == 0, 2, 1)
        y = make_competing_risks_y(stop, codes)
        est = CompetingRisksForestTV
    else:
        y = make_survival_y(stop, np.ones(n, dtype=bool))
        est = SurvivalForestTV
    return x, y, est


@pytest.mark.parametrize("competing", [False, True])
def test_fit_predict_pickle_and_missing_split(competing):
    x, y, cls = _data(competing)
    model = cls(n_estimators=1, max_features=None, max_depth=1,
                min_ids_leaf=2, min_events_leaf=1, max_samples=1.0,
                random_state=0).fit(x, y)
    et = model.export_tree(0)
    assert et.node_count == 3
    assert et.feature[0] == 0
    assert np.isnan(et.threshold[0])
    assert not et.missing_goes_right[0]
    leaves = model.apply(x)[:, 0]
    assert np.unique(leaves[:20]).size == np.unique(leaves[20:]).size == 1
    assert leaves[0] != leaves[-1]
    assert "missing?" in plot_tree(et)

    hazard = model.predict_cumulative_hazard(x)
    survival = model.predict_survival_function(x)
    assert np.isfinite(hazard).all() and np.isfinite(survival).all()
    assert not np.array_equal(hazard[0], hazard[-1])
    if competing:
        assert np.isfinite(model.predict_cumulative_incidence(x)).all()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.apply(x), model.apply(x))
    np.testing.assert_array_equal(restored.predict_cumulative_hazard(x), hazard)


@pytest.mark.parametrize("competing", [False, True])
def test_nan_at_predict_time_and_infinity_still_rejected(competing):
    x, y, cls = _data(competing)
    observed = x.copy()
    observed[:, 0] = np.arange(len(x))
    model = cls(n_estimators=3, max_features=None, max_depth=2,
                min_ids_leaf=2, min_events_leaf=1, random_state=0).fit(observed, y)
    assert np.isfinite(model.predict(x)).all()
    assert model.__sklearn_tags__().input_tags.allow_nan
    bad = x.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        model.predict(bad)
    with pytest.raises(ValueError):
        cls(n_estimators=1, random_state=0).fit(bad, y)
