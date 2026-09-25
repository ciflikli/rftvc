import pickle

import numpy as np
import pytest
from sklearn.base import clone

from rftvc import SurvivalForestTV, make_survival_y


def _data(n=300, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    t = rng.exponential(np.exp(-0.7 * X[:, 0] + 0.5 * (X[:, 1] > 0)))
    e = rng.random(n) < 0.75
    return X, make_survival_y(t, e)


def test_subsample_draws_whole_distinct_ids():
    X, y = _data()
    ids = np.arange(300) * 7 + 3  # arbitrary labels
    model = SurvivalForestTV(n_estimators=10, random_state=0).fit(X, y, ids=ids)
    assert model.n_draw_ == round(0.632 * 300)
    for tree in range(10):
        bag = model.forest_.in_bag_ids(tree)
        assert len(bag) == model.n_draw_
        assert len(np.unique(bag)) == len(bag)
        assert bag.max() < 300


def test_bootstrap_draws_with_replacement():
    X, y = _data()
    model = SurvivalForestTV(n_estimators=5, bootstrap=True, random_state=0).fit(X, y)
    bag = model.forest_.in_bag_ids(0)
    assert len(bag) == 300
    assert len(np.unique(bag)) < 300


def test_deterministic_across_n_jobs():
    X, y = _data()
    preds = [
        SurvivalForestTV(n_estimators=30, random_state=3, n_jobs=j).fit(X, y).predict_cumulative_hazard(X)
        for j in (1, 4)
    ]
    np.testing.assert_array_equal(*preds)


def test_hazard_aggregation_is_below_survival_aggregation():
    """Jensen: exp(-mean Λ_b) <= mean exp(-Λ_b) pointwise."""
    X, y = _data()
    params = dict(n_estimators=50, random_state=1)
    s_h = SurvivalForestTV(aggregate="hazard", **params).fit(X, y).predict_survival_function(X)
    s_s = SurvivalForestTV(aggregate="survival", **params).fit(X, y).predict_survival_function(X)
    assert (s_h <= s_s + 1e-12).all()
    assert (s_h < s_s - 1e-6).any()


def test_hazard_aggregation_is_mean_of_tree_hazards():
    X, y = _data()
    forest = SurvivalForestTV(n_estimators=8, random_state=2).fit(X, y)
    times = forest.event_times_
    per_tree = []
    for b in range(8):
        leaves = forest.apply(X)[:, b]
        h = np.empty((len(X), len(times)))
        for i, leaf in enumerate(leaves):
            lt, *_, cum = map(np.asarray, forest.forest_.leaf_profile(b, int(leaf)))
            h[i] = np.r_[0.0, cum][np.searchsorted(lt, times, side="right")]
        per_tree.append(h)
    np.testing.assert_allclose(forest.predict_cumulative_hazard(X), np.mean(per_tree, axis=0), atol=1e-12)


@pytest.mark.parametrize(("n_ids", "expected"), [(224, 15), (225, 15), (226, 15), (400, 20), (1000, 31)])
def test_min_ids_leaf_auto(n_ids, expected):
    X, y = _data(n=n_ids)
    model = SurvivalForestTV(n_estimators=1, min_ids_leaf="auto", random_state=0).fit(X, y)
    assert model.min_ids_leaf_ == expected


def test_resample_unit_other_than_id_raises():
    X, y = _data()
    with pytest.raises(ValueError, match="resample_unit"):
        SurvivalForestTV(n_estimators=1, resample_unit="row").fit(X, y)


@pytest.mark.parametrize("max_samples", [0, 1.5, 301, -0.1, "x"])
def test_invalid_max_samples(max_samples):
    X, y = _data()
    with pytest.raises(ValueError, match="max_samples"):
        SurvivalForestTV(n_estimators=1, max_samples=max_samples).fit(X, y)


def test_pickle_roundtrip_and_clone():
    X, y = _data()
    model = SurvivalForestTV(n_estimators=20, random_state=0).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict_cumulative_hazard(X), restored.predict_cumulative_hazard(X))
    np.testing.assert_array_equal(model.forest_.in_bag_ids(3), restored.forest_.in_bag_ids(3))
    assert clone(model).get_params() == model.get_params()


def test_predict_risk_matches_survival():
    X, y = _data()
    model = SurvivalForestTV(n_estimators=20, random_state=0).fit(X, y)
    h = float(np.median(y["stop"]))
    np.testing.assert_allclose(model.predict_risk(X, h), 1 - model.predict_survival_function(X, [h])[:, 0])


def _harrell_c(t, e, risk):
    conc = tot = 0.0
    for i in np.flatnonzero(e):
        later = t > t[i]
        tot += later.sum()
        conc += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return conc / tot


def test_forest_discrimination_close_to_true_risk_score():
    X, y = _data(n=600, seed=4)
    model = SurvivalForestTV(n_estimators=100, random_state=0).fit(X[:400], y[:400])
    risk = model.predict_risk(X[400:], float(np.median(y["stop"])))
    t, e = y["stop"][400:], y["event"][400:]
    true_score = 0.7 * X[400:, 0] - 0.5 * (X[400:, 1] > 0)
    assert _harrell_c(t, e, risk) > _harrell_c(t, e, true_score) - 0.03
    assert _harrell_c(t, e, risk) > 0.58


def test_predict_rejects_wrong_width():
    X, y = _data()
    model = SurvivalForestTV(n_estimators=2, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict_cumulative_hazard(X[:, :2])
