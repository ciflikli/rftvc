import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lifelines import NelsonAalenFitter

from rftvc import SurvivalForestTV, _core, make_survival_y
from tests.ref.logrank_ref import logrank_ref, nelson_aalen_ref

def _tree(**kw):
    """One tree on all ids: the S1 single-tree setting."""
    return SurvivalForestTV(**{"n_estimators": 1, "max_samples": 1.0, "max_features": None, **kw})


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
    model = _tree(max_depth=0).fit(X, make_survival_y(t, e))
    assert model.forest_.n_leaves(0) == 1
    times, cumhaz = model.forest_.leaf_profile(0, 0)
    naf = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(t, e, timeline=times)
    np.testing.assert_allclose(cumhaz, naf.cumulative_hazard_.to_numpy().ravel(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.predict_cumulative_hazard(X[:5], times), np.tile(cumhaz, (5, 1)), atol=1e-10)


def test_cumulative_hazard_is_right_continuous_step():
    X, t, e = _data(100, 1, seed=2)
    model = _tree(max_depth=0).fit(X, make_survival_y(t, e))
    times, cumhaz = map(np.asarray, model.forest_.leaf_profile(0, 0))
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
    feature, threshold, score, mask, *_ = got
    mask = np.asarray(mask)
    assert score == pytest.approx(expected, rel=1e-9)
    assert score == pytest.approx(logrank_ref(start, t, e, mask), rel=1e-9)
    np.testing.assert_array_equal(mask, X[:, feature] <= threshold)


def test_tree_predictions_follow_leaf_assignment():
    X, t, e = _data(400, 3, seed=3)
    model = _tree(min_ids_leaf=20, random_state=0).fit(X, make_survival_y(t, e))
    assert model.forest_.n_leaves(0) > 1
    leaves = model.apply(X)[:, 0]
    H = model.predict_cumulative_hazard(X)
    for leaf in np.unique(leaves):
        rows = H[leaves == leaf]
        np.testing.assert_array_equal(rows, np.tile(rows[0], (rows.shape[0], 1)))


def test_min_leaf_constraints_hold():
    X, t, e = _data(400, 3, seed=4)
    model = _tree(min_ids_leaf=25, min_events_leaf=5, random_state=0).fit(X, make_survival_y(t, e))
    leaves = model.apply(X)[:, 0]
    for leaf in np.unique(leaves):
        in_leaf = leaves == leaf
        assert in_leaf.sum() >= 25
        assert (e & in_leaf).sum() >= 5


def test_random_state_is_deterministic():
    X, t, e = _data(300, 4, seed=5)
    y = make_survival_y(t, e)
    a = _tree(max_features=2, random_state=7).fit(X, y).predict_cumulative_hazard(X)
    b = _tree(max_features=2, random_state=7).fit(X, y).predict_cumulative_hazard(X)
    np.testing.assert_array_equal(a, b)


def test_split_on_extreme_feature_values():
    m = np.finfo(float).max
    X = np.r_[np.full(10, -m), np.full(10, m)][:, None]
    t = np.r_[np.full(10, 1.0), np.full(10, 5.0)] + np.arange(20) * 0.01
    got = _core.best_split(X, np.zeros(20), t, np.ones(20, bool), min_ids_leaf=2, min_events_leaf=1)
    assert got is not None
    np.testing.assert_array_equal(got[3], X[:, 0] < 0)


def test_multi_leaf_predictions_match_per_leaf_nelson_aalen():
    X, t, e = _data(500, 3, seed=6)
    model = _tree(min_ids_leaf=20, random_state=0).fit(X, make_survival_y(t, e))
    leaves = model.apply(X)[:, 0]
    assert len(np.unique(leaves)) >= 3
    times = np.unique(t)
    H = model.predict_cumulative_hazard(X, times)
    for leaf in np.unique(leaves):
        rows = leaves == leaf
        ref_times, ref_cum = nelson_aalen_ref(np.zeros(rows.sum()), t[rows], e[rows])
        expected = np.r_[0.0, ref_cum][np.searchsorted(ref_times, times, side="right")]
        np.testing.assert_allclose(H[rows], np.tile(expected, (rows.sum(), 1)), atol=1e-12)


def _reference_thresholds(col, max_bins):
    """Documented binning rule, reimplemented: raw thresholds (x <= thr goes left)."""
    uniq = np.unique(col)
    if uniq.size <= max_bins:
        return uniq[:-1]
    s = np.sort(col)
    edges = np.unique([s[q * len(s) // max_bins] for q in range(1, max_bins)])
    return edges[edges < s[-1]]


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(30, 400),
    seed=st.integers(0, 10_000),
    max_bins=st.sampled_from([2, 3, 16, 64, 256]),
    min_leaf=st.integers(1, 10),
)
def test_quantile_binned_split_matches_reference(n, seed, max_bins, min_leaf):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    X[:, 1] = np.round(X[:, 1], 1)  # a tied column alongside a continuous one
    t = rng.exponential(1 + (X[:, 0] > 0))
    e = rng.random(n) < 0.7
    e[0] = True
    start = np.zeros(n)
    best = 0.0
    for f in range(2):
        for thr in _reference_thresholds(X[:, f], max_bins):
            left = X[:, f] <= thr
            nl, el = left.sum(), (e & left).sum()
            if min(nl, n - nl) < min_leaf or min(el, e.sum() - el) < 1:
                continue
            best = max(best, logrank_ref(start, t, e, left))
    got = _core.best_split(X, start, t, e, min_ids_leaf=min_leaf, min_events_leaf=1, max_bins=max_bins)
    if best == 0.0:
        assert got is None
        return
    feature, threshold, score, mask, *_ = got
    mask = np.asarray(mask)
    assert score == pytest.approx(best, rel=1e-9)
    np.testing.assert_array_equal(mask, X[:, feature] <= threshold)
    assert score == pytest.approx(logrank_ref(start, t, e, mask), rel=1e-9)


def test_core_rejects_mismatched_lengths():
    X = np.zeros((2, 1))
    zero, one, ev = np.zeros(1), np.ones(1), np.ones(1, bool)
    with pytest.raises(ValueError, match="start"):
        _core.best_split(X, zero, one, ev, min_ids_leaf=1, min_events_leaf=1)
    with pytest.raises(ValueError, match="start"):
        _core.fit_forest(X, zero, one, ev, np.zeros(1, np.uint32), 1, n_trees=1, n_draw=1,
                         bootstrap=False, max_depth=None, min_ids_leaf=1, min_events_leaf=1,
                         max_features=1, max_bins=255, seed=0, n_jobs=1)
    with pytest.raises(ValueError, match="left"):
        _core.logrank_score(zero, one, ev, np.ones(2, bool))
