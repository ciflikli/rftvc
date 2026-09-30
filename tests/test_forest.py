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


def test_random_state_accepts_generator():
    X, y = _data()
    fits = [
        SurvivalForestTV(n_estimators=10, random_state=np.random.default_rng(0))
        .fit(X, y)
        .predict_cumulative_hazard(X)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(*fits)


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
            lt, cum = map(np.asarray, forest.forest_.leaf_profile(b, int(leaf)))
            cum = cum[:, 0]  # one cause
            h[i] = np.r_[0.0, cum][np.searchsorted(lt, times, side="right")]
        per_tree.append(h)
    np.testing.assert_allclose(forest.predict_cumulative_hazard(X), np.mean(per_tree, axis=0), atol=1e-12)


@pytest.mark.parametrize("aggregate,ntime", [("hazard", None), ("hazard", 30), ("survival", None)])
def test_direct_mortality_matches_full_curve(aggregate, ntime):
    X, y = _data()
    forest = SurvivalForestTV(
        n_estimators=35, aggregate=aggregate, ntime=ntime, random_state=3, n_jobs=2
    ).fit(X, y)
    for times in (forest.event_times_, forest.event_times_[::7], forest.event_times_[::-7]):
        times = np.ascontiguousarray(times)
        direct = forest.forest_.predict_mortality(X[:25], times, aggregate, 2)
        full = forest.forest_.predict_cumhaz(X[:25], times, aggregate, 2).sum(axis=1)
        np.testing.assert_allclose(direct, full, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(forest.predict(X[:25]), forest.predict_cumulative_hazard(X[:25]).sum(axis=1),
                               rtol=1e-12, atol=1e-12)


def test_export_tree_structure_matches_apply():
    """Walking children_left/children_right/feature/threshold by hand reaches the same
    leaf as ``apply`` for every row, and every leaf index shows up in the tree."""
    X, y = _data()
    forest = SurvivalForestTV(n_estimators=4, random_state=0).fit(X, y)
    applied = forest.apply(X)
    for b in range(4):
        et = forest.export_tree(b)
        assert et.node_count == len(et.children_left) == len(et.feature) == len(et.threshold) == len(et.leaf)
        assert et.n_leaves == forest.forest_.n_leaves(b)
        assert sorted(et.leaf[et.leaf >= 0]) == list(range(et.n_leaves))
        for row, x in enumerate(X):
            node = 0
            while et.leaf[node] < 0:
                node = et.children_left[node] if x[et.feature[node]] <= et.threshold[node] else et.children_right[node]
            assert et.leaf[node] == applied[row, b]


def test_export_tree_leaf_sentinels_match_sklearn():
    """sklearn's own TREE_LEAF (-1, children) / TREE_UNDEFINED (-2, feature/threshold)."""
    X, y = _data()
    forest = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, y)
    et = forest.export_tree(0)
    is_leaf = et.leaf >= 0
    assert (et.children_left[is_leaf] == -1).all()
    assert (et.children_right[is_leaf] == -1).all()
    assert (et.feature[is_leaf] == -2).all()
    assert (et.threshold[is_leaf] == -2.0).all()
    assert (et.feature[~is_leaf] >= 0).all()
    assert (et.children_left[~is_leaf] >= 0).all()
    assert (et.children_right[~is_leaf] >= 0).all()


def test_export_tree_root_leaf():
    """max_depth=0 forces a single-node tree: node 0 is a leaf with no children."""
    X, y = _data()
    forest = SurvivalForestTV(n_estimators=1, max_depth=0, random_state=0).fit(X, y)
    et = forest.export_tree(0)
    assert et.node_count == 1
    assert et.n_leaves == 1
    assert et.leaf[0] == 0
    assert et.children_left[0] == et.children_right[0] == -1
    assert et.feature[0] == -2
    assert et.threshold[0] == -2.0


def test_export_tree_uses_fitted_tree_count_not_n_estimators_param():
    """set_params after fit must not desync tree-index validation from the actual forest."""
    X, y = _data()
    forest = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, y)
    forest.set_params(n_estimators=10)
    et = forest.export_tree(2)  # still valid: the fitted forest has 3 trees
    assert et.node_count > 0
    with pytest.raises(ValueError, match=r"tree must be in \[0, 3\)"):
        forest.export_tree(5)  # would be valid for n_estimators=10, not for the fitted forest


def test_export_tree_leaf_hazard_matches_leaf_profile():
    X, y = _data()
    forest = SurvivalForestTV(n_estimators=3, random_state=1).fit(X, y)
    et = forest.export_tree(0)
    leaf = int(et.leaf[et.leaf >= 0][0])
    times, cumhaz = forest.forest_.leaf_profile(0, leaf)
    assert cumhaz.shape == (len(times), 1)


def test_export_tree_feature_names():
    pd = pytest.importorskip("pandas")
    X, y = _data(p=3)
    Xdf = pd.DataFrame(X, columns=["a", "b", "c"])
    forest = SurvivalForestTV(n_estimators=2, random_state=0).fit(Xdf, y)
    assert list(forest.export_tree(0).feature_names) == ["a", "b", "c"]
    assert SurvivalForestTV(n_estimators=2, random_state=0).fit(X, y).export_tree(0).feature_names is None


def test_export_tree_invalid_tree_index():
    X, y = _data()
    forest = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="tree must be in"):
        forest.export_tree(3)
    with pytest.raises(ValueError, match="tree must be in"):
        forest.export_tree(-1)
    with pytest.raises(TypeError, match="tree must be an int"):
        forest.export_tree(1.5)


@pytest.mark.parametrize(("n_ids", "expected"), [(224, 15), (225, 15), (226, 15), (400, 20), (1000, 31)])
def test_min_ids_leaf_auto(n_ids, expected):
    X, y = _data(n=n_ids)
    model = SurvivalForestTV(n_estimators=1, min_ids_leaf="auto", random_state=0).fit(X, y)
    assert model.min_ids_leaf_ == expected


def test_resample_unit_other_than_id_raises():
    X, y = _data()
    with pytest.raises(ValueError, match="resample_unit"):
        SurvivalForestTV(n_estimators=1, resample_unit="row").fit(X, y)


@pytest.mark.parametrize("max_samples", [0, 1.5, 301, -0.1])
def test_invalid_max_samples_value(max_samples):
    X, y = _data()
    with pytest.raises(ValueError, match="max_samples"):
        SurvivalForestTV(n_estimators=1, max_samples=max_samples).fit(X, y)


@pytest.mark.parametrize("max_samples", ["x", True, False, np.True_])
def test_invalid_max_samples_type(max_samples):
    X, y = _data()
    with pytest.raises(TypeError, match="max_samples"):
        SurvivalForestTV(n_estimators=1, max_samples=max_samples).fit(X, y)


@pytest.mark.parametrize("max_features", [0, 1.5, -1, 0.0])
def test_invalid_max_features_value(max_features):
    X, y = _data()
    with pytest.raises(ValueError, match="max_features"):
        SurvivalForestTV(n_estimators=1, max_features=max_features).fit(X, y)


@pytest.mark.parametrize("max_features", ["bogus", [1, 2]])
def test_invalid_max_features_type(max_features):
    X, y = _data()
    with pytest.raises(TypeError, match="max_features"):
        SurvivalForestTV(n_estimators=1, max_features=max_features).fit(X, y)


@pytest.mark.parametrize("name,kw", [
    ("n_estimators", {"n_estimators": "5"}),
    ("n_estimators", {"n_estimators": 2.5}),
    ("min_events_leaf", {"n_estimators": 1, "min_events_leaf": None}),
    ("min_ids_leaf", {"n_estimators": 1, "min_ids_leaf": 2.5}),
    ("max_depth", {"n_estimators": 1, "max_depth": "3"}),
    ("ntime", {"n_estimators": 1, "ntime": 1.5}),
    ("oob_buffer", {"n_estimators": 1, "oob_buffer": 1.5}),
])
def test_check_int_wrong_type_raises_type_error(name, kw):
    X, y = _data()
    with pytest.raises(TypeError, match=name):
        SurvivalForestTV(**kw).fit(X, y)


@pytest.mark.parametrize("name,kw", [
    ("n_estimators", {"n_estimators": 0}),
    ("min_events_leaf", {"n_estimators": 1, "min_events_leaf": 0}),
    ("min_ids_leaf", {"n_estimators": 1, "min_ids_leaf": 0}),
    ("max_depth", {"n_estimators": 1, "max_depth": -1}),
    ("ntime", {"n_estimators": 1, "ntime": 0}),
])
def test_check_int_below_minimum_raises_value_error(name, kw):
    X, y = _data()
    with pytest.raises(ValueError, match=name):
        SurvivalForestTV(**kw).fit(X, y)


def test_pickle_roundtrip_and_clone():
    X, y = _data()
    model = SurvivalForestTV(n_estimators=20, random_state=0).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict_cumulative_hazard(X), restored.predict_cumulative_hazard(X))
    np.testing.assert_array_equal(model.forest_.in_bag_ids(3), restored.forest_.in_bag_ids(3))
    assert clone(model).get_params() == model.get_params()


def _state(model):
    return dict(model.forest_.__reduce__()[1][0])


def test_pre_s9_state_is_rejected():
    X, y = _data()
    model = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, y)
    state = _state(model)
    type(model.forest_)._from_state(state)  # the current state loads
    old = {k: v for k, v in state.items() if k not in ("format_version", "cumhaz")}
    n = len(state["event_idx"])
    old.update(d=np.ones(n), y=np.full(n, 2.0))
    with pytest.raises(ValueError, match="older rftvc build"):
        type(model.forest_)._from_state(old)


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.update(format_version=5),
        lambda s: s.update(n_features=0),
        lambda s: s.update(grid=s["grid"][::-1].copy()),
        lambda s: s.update(cumhaz=-s["cumhaz"]),
    ],
    ids=["future version", "no features", "unsorted grid", "negative cumhaz"],
)
def test_corrupt_state_raises_value_error(edit):
    X, y = _data()
    model = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, y)
    state = _state(model)
    edit(state)
    with pytest.raises(ValueError):
        type(model.forest_)._from_state(state)


def test_nbytes_counts_only_hazards_and_event_indices():
    # Per leaf entry: a u32 grid index and an f64 cumulative hazard (no d / y);
    # per leaf a u32 offset; per node a 24-byte enum; one copy of the grid. The
    # arrays are shrunk to fit, so allocated capacity equals this count.
    X, y = _data()
    model = SurvivalForestTV(n_estimators=5, random_state=0).fit(X, y)
    s = _state(model)
    n_trees, n_nodes = len(s["tree_seeds"]), len(s["node_feature"])
    n_leaves, n_entries = len(s["event_offsets"]) - 1, len(s["event_idx"])
    expected = 24 * n_nodes + 4 * (n_leaves + n_trees) + 12 * n_entries + 8 * len(s["grid"])
    assert n_entries > n_leaves > n_trees
    assert model.forest_.nbytes == expected


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
