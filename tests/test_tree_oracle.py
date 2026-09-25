import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lifelines import NelsonAalenFitter

from rftvc import SurvivalForestTV, _core, make_survival_y
from tests.ref.logrank_ref import logrank_ref

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "survdiff.json").read_text())


def _data(n, p, seed, ties=True):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, (n, p)).astype(float)
    t = rng.exponential(1 + X[:, 0] / 4)
    t = np.round(t, 1) + 0.1 if ties else t + 1e-3
    e = rng.random(n) < 0.7
    e[0] = True
    return X, t, e


@pytest.mark.parametrize("ties", [True, False])
def test_root_leaf_matches_nelson_aalen(ties):
    X, t, e = _data(300, 3, seed=1, ties=ties)
    model = SurvivalForestTV(max_depth=0).fit(X, make_survival_y(t, e))
    assert model.tree_.n_leaves == 1
    times, d, y, cumhaz = model.tree_.leaf_profile(0)
    naf = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(t, e, timeline=times)
    np.testing.assert_allclose(cumhaz, naf.cumulative_hazard_.to_numpy().ravel(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.predict_cumulative_hazard(X[:5], times), np.tile(cumhaz, (5, 1)), atol=1e-10)


def test_cumulative_hazard_is_right_continuous_step():
    X, t, e = _data(100, 1, seed=2)
    model = SurvivalForestTV(max_depth=0).fit(X, make_survival_y(t, e))
    times, *_, cumhaz = map(np.asarray, model.tree_.leaf_profile(0))
    before = model.predict_cumulative_hazard(X[:1], times - 1e-9)[0]
    at = model.predict_cumulative_hazard(X[:1], times)[0]
    np.testing.assert_allclose(at, cumhaz, atol=1e-12)
    np.testing.assert_allclose(before[1:], cumhaz[:-1], atol=1e-12)
    assert before[0] == 0.0


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_logrank_matches_survdiff_and_reference(name):
    f = FIXTURES[name]
    t, e, g = np.array(f["time"], float), np.array(f["event"], bool), np.array(f["group"], bool)
    start = np.zeros_like(t)
    score = _core.logrank_score(start, t, e, g)
    assert score == pytest.approx(f["chisq"], rel=1e-10)
    assert score == pytest.approx(logrank_ref(start, t, e, g), rel=1e-10)


def _brute_force_best(X, t, e, min_leaf, min_events):
    start = np.zeros_like(t)
    best = 0.0
    for f in range(X.shape[1]):
        for v in np.unique(X[:, f])[:-1]:
            left = X[:, f] <= v
            nl, el = left.sum(), (e & left).sum()
            if min(nl, len(t) - nl) < min_leaf or min(el, e.sum() - el) < min_events:
                continue
            best = max(best, logrank_ref(start, t, e, left))
    return best


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(12, 60),
    p=st.integers(1, 3),
    seed=st.integers(0, 10_000),
    min_leaf=st.integers(1, 5),
    min_events=st.integers(1, 2),
)
def test_histogram_split_matches_brute_force(n, p, seed, min_leaf, min_events):
    X, t, e = _data(n, p, seed)
    start = np.zeros_like(t)
    expected = _brute_force_best(X, t, e, min_leaf, min_events)
    got = _core.best_split(X, start, t, e, min_ids_leaf=min_leaf, min_events_leaf=min_events)
    if expected == 0.0:
        assert got is None
        return
    feature, threshold, score, mask = got
    mask = np.asarray(mask)
    assert score == pytest.approx(expected, rel=1e-9)
    assert score == pytest.approx(logrank_ref(start, t, e, mask), rel=1e-9)
    np.testing.assert_array_equal(mask, X[:, feature] <= threshold)


def test_tree_predictions_follow_leaf_assignment():
    X, t, e = _data(400, 3, seed=3)
    model = SurvivalForestTV(min_ids_leaf=20, random_state=0).fit(X, make_survival_y(t, e))
    assert model.tree_.n_leaves > 1
    leaves = model.apply(X)
    H = model.predict_cumulative_hazard(X)
    for leaf in np.unique(leaves):
        rows = H[leaves == leaf]
        np.testing.assert_array_equal(rows, np.tile(rows[0], (rows.shape[0], 1)))


def test_min_leaf_constraints_hold():
    X, t, e = _data(400, 3, seed=4)
    model = SurvivalForestTV(min_ids_leaf=25, min_events_leaf=5, random_state=0).fit(X, make_survival_y(t, e))
    leaves = model.apply(X)
    for leaf in np.unique(leaves):
        in_leaf = leaves == leaf
        assert in_leaf.sum() >= 25
        assert (e & in_leaf).sum() >= 5


def test_random_state_is_deterministic():
    X, t, e = _data(300, 4, seed=5)
    y = make_survival_y(t, e)
    a = SurvivalForestTV(max_features=2, random_state=7).fit(X, y).predict_cumulative_hazard(X)
    b = SurvivalForestTV(max_features=2, random_state=7).fit(X, y).predict_cumulative_hazard(X)
    np.testing.assert_array_equal(a, b)


def test_split_on_extreme_feature_values():
    m = np.finfo(float).max
    X = np.r_[np.full(10, -m), np.full(10, m)][:, None]
    t = np.r_[np.full(10, 1.0), np.full(10, 5.0)] + np.arange(20) * 0.01
    got = _core.best_split(X, np.zeros(20), t, np.ones(20, bool), min_ids_leaf=2, min_events_leaf=1)
    assert got is not None
    np.testing.assert_array_equal(got[3], X[:, 0] < 0)
