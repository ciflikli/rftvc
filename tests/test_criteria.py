"""S8 candidate split criteria: parity with naive references, edge cases and invariants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lifelines import KaplanMeierFitter

from rftvc import _core
from tests.ref.criteria_ref import grouped_lik_ref, km_gini_ref, km_ref, poisson_ref
from tests.ref.logrank_ref import logrank_ref


def _rows(n, seed, ties=True):
    """Delayed-entry counting-process rows; ties on a 0.5 grid unless ``ties=False``."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, 4, n) * 0.5 if ties else rng.uniform(0, 2, n)
    stop = start + (rng.integers(1, 8, n) * 0.5 if ties else rng.uniform(0.1, 4, n))
    event = rng.random(n) < 0.6
    event[0] = True
    left = rng.random(n) < 0.5
    left[:2] = [True, False]
    units = np.sort(rng.integers(0, max(n // 2, 1), n)).astype(np.uint32)
    return start, stop, event, left, units


def _score(name, start, stop, event, left, **kw):
    return _core.criterion_score(start, stop, event, left, name, **kw)


@settings(max_examples=80, deadline=None)
@given(n=st.integers(2, 60), seed=st.integers(0, 10_000), ties=st.booleans(),
       horizon=st.sampled_from([0.25, 1.0, 1.5, 2.5, 4.0, 99.0]))
def test_criteria_match_naive_references(n, seed, ties, horizon):
    start, stop, event, left, units = _rows(n, seed, ties)
    args = (start, stop, event, left)
    assert _score("logrank", *args) == pytest.approx(logrank_ref(*args), rel=1e-9, abs=1e-12)
    assert _score("grouped_lik", *args) == pytest.approx(grouped_lik_ref(*args), rel=1e-9, abs=1e-9)
    assert _score("poisson", *args) == pytest.approx(poisson_ref(*args), rel=1e-9, abs=1e-9)
    got = _score("km_gini", *args, horizon=horizon, units=units)
    assert got == pytest.approx(km_gini_ref(*args, horizon, units), rel=1e-9, abs=1e-12)


@settings(max_examples=80, deadline=None)
@given(n=st.integers(2, 60), seed=st.integers(0, 10_000), ties=st.booleans())
def test_likelihood_gains_are_nonnegative_and_side_symmetric(n, seed, ties):
    start, stop, event, left, units = _rows(n, seed, ties)
    for name, kw in [("grouped_lik", {}), ("poisson", {}), ("km_gini", {"horizon": 2.0, "units": units})]:
        a = _score(name, start, stop, event, left, **kw)
        b = _score(name, start, stop, event, ~left, **kw)
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12)
        if name != "km_gini":  # the Gini decrease can be negative (straddling units)
            assert a >= -1e-9


@pytest.mark.parametrize("name,kw", [("grouped_lik", {}), ("poisson", {}), ("km_gini", {"horizon": 2.0})])
def test_identical_children_give_zero_gain(name, kw):
    start, stop, event, _, _ = _rows(30, seed=4)
    two = lambda a: np.concatenate([a, a])  # noqa: E731
    left = np.repeat([True, False], 30)
    got = _score(name, two(start), two(stop), two(event), left, **kw)
    assert got == pytest.approx(0.0, abs=1e-10)


def test_all_event_and_tied_nodes_use_zero_log_zero():
    # Every row fails, all at the same two times: d == y at the last time.
    start, stop = np.zeros(6), np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    event, left = np.ones(6, bool), np.array([1, 0, 1, 0, 1, 0], bool)
    for name, ref in [("grouped_lik", grouped_lik_ref), ("poisson", poisson_ref)]:
        got = _score(name, start, stop, event, left)
        assert np.isfinite(got)
        assert got == pytest.approx(ref(start, stop, event, left), abs=1e-12)
    # An eventless left child.
    left = np.array([0, 0, 0, 0, 0, 1], bool)
    event = np.array([1, 1, 1, 1, 1, 0], bool)
    for name, ref in [("grouped_lik", grouped_lik_ref), ("poisson", poisson_ref)]:
        assert _score(name, start, stop, event, left) == pytest.approx(ref(start, stop, event, left), abs=1e-12)


def test_km_gini_horizon_edges():
    # Event times 1, 2 (tied: two events at 2), 3.
    start = np.array([0.0, 0.0, 0.5, 0.0, 1.5, 0.0])
    stop = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
    event = np.array([1, 1, 1, 1, 0, 0], bool)
    left = np.array([1, 1, 0, 0, 1, 0], bool)
    score = lambda h: _score("km_gini", start, stop, event, left, horizon=h)  # noqa: E731
    ref = lambda h: km_gini_ref(start, stop, event, left, h)  # noqa: E731
    assert score(0.5) == 0.0  # before the first event: S = 1 everywhere
    # At a tie the events at tau count (right-continuous).
    assert score(2.0) == pytest.approx(ref(2.0), abs=1e-12)
    assert score(2.0) != pytest.approx(score(2.0 - 1e-9), abs=1e-6)
    # After the last event the score no longer changes.
    assert score(3.0) == pytest.approx(ref(3.0), abs=1e-12)
    assert score(50.0) == score(3.0)


def test_km_gini_counts_straddling_units_in_both_children():
    start, stop = np.array([0.0, 1.0, 0.0, 0.0]), np.array([1.0, 2.0, 2.5, 3.0])
    event, left = np.array([0, 1, 1, 1], bool), np.array([1, 0, 1, 0], bool)
    units = np.array([0, 0, 1, 2], np.uint32)  # unit 0 has a row on each side
    got = _score("km_gini", start, stop, event, left, horizon=2.5, units=units)
    assert got == pytest.approx(km_gini_ref(start, stop, event, left, 2.5, units), abs=1e-12)
    assert got != pytest.approx(_score("km_gini", start, stop, event, left, horizon=2.5), abs=1e-6)


def test_km_reference_matches_lifelines_with_delayed_entry():
    start, stop, event, *_ = _rows(80, seed=9, ties=False)
    km = KaplanMeierFitter().fit(stop, event, entry=start)
    for h in [0.5, 1.5, 3.0]:
        want = float(km.survival_function_at_times(h).iloc[0])
        assert km_ref(start, stop, event, h) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("name,kw,match", [
    ("gini", {}, "split_criterion must be"),
    ("km_gini", {}, "requires criterion_horizon"),
    ("km_gini", {"horizon": 0.0}, "finite and > 0"),
    ("km_gini", {"horizon": float("nan")}, "finite and > 0"),
    ("logrank", {"horizon": 1.0}, "only used by"),
])
def test_criterion_score_validates_name_and_horizon(name, kw, match):
    start, stop, event, left, _ = _rows(5, seed=0)
    with pytest.raises(ValueError, match=match):
        _score(name, start, stop, event, left, **kw)


def test_criterion_score_logrank_equals_logrank_score():
    start, stop, event, left, _ = _rows(40, seed=3)
    assert _score("logrank", start, stop, event, left) == _core.logrank_score(start, stop, event, left)


# --- estimator wiring -------------------------------------------------------------

from sklearn.base import clone  # noqa: E402

from rftvc import LandmarkSurvivalForest, SurvivalForestTV  # noqa: E402
from rftvc.model_selection import RollingOriginSplit, landmark_cross_validate  # noqa: E402
from tests.sim_panel import simulate_panel  # noqa: E402
from tests.test_tvc import _cp_data  # noqa: E402

CRITERIA = [("logrank", {}), ("grouped_lik", {}), ("poisson", {}), ("km_gini", {"horizon": 1.5})]
REFS = {"logrank": logrank_ref, "grouped_lik": grouped_lik_ref, "poisson": poisson_ref}


def _ref_score(name, kw, start, stop, event, left, units):
    if name == "km_gini":
        return km_gini_ref(start, stop, event, left, kw["horizon"], units)
    return REFS[name](start, stop, event, left)


@settings(max_examples=40, deadline=None)
@given(n_ids=st.integers(6, 30), seed=st.integers(0, 10_000), which=st.integers(0, 3))
def test_best_split_is_the_argmax_of_each_criterion(n_ids, seed, which):
    """The splitter (with its per-bin exposure and unit counts) picks the brute-force best."""
    name, kw = CRITERIA[which]
    X, y, ids = _cp_data(n_ids, seed)
    start, stop, event = (np.ascontiguousarray(y[k]) for k in ("start", "stop", "event"))
    units = ids.astype(np.uint32)
    min_ids, min_ev = 2, 1
    best = 0.0
    for f in range(X.shape[1]):
        for v in np.unique(X[:, f])[:-1]:
            left = X[:, f] <= v
            if (min(len(np.unique(units[left])), len(np.unique(units[~left]))) < min_ids
                    or min(event[left].sum(), event[~left].sum()) < min_ev):
                continue
            best = max(best, _ref_score(name, kw, start, stop, event, left, units))
    got = _core.best_split(X, start, stop, event, min_ids_leaf=min_ids, min_events_leaf=min_ev,
                           units=units, criterion=name, **kw)
    if got is None:
        assert best <= 1e-9
        return
    _, _, score, mask, _, _ = got
    assert score == pytest.approx(best, rel=1e-9, abs=1e-12)
    assert _ref_score(name, kw, start, stop, event, np.asarray(mask), units) == pytest.approx(score, rel=1e-9, abs=1e-12)


def _fit(**kw):
    X, y, ids = _cp_data(120, seed=5)
    model = SurvivalForestTV(n_estimators=10, min_ids_leaf=5, random_state=0, **kw).fit(X, y, ids=ids)
    return model, X


@pytest.mark.parametrize("kw,match", [
    ({"split_criterion": "gini"}, "split_criterion must be one of"),
    ({"split_criterion": "km_gini"}, "requires criterion_horizon"),
    ({"split_criterion": "km_gini", "criterion_horizon": -1.0}, "finite and > 0"),
    ({"split_criterion": "km_gini", "criterion_horizon": True}, "finite and > 0"),
    ({"split_criterion": "km_gini", "criterion_horizon": "2"}, "finite and > 0"),
    ({"split_criterion": "poisson", "criterion_horizon": 2.0}, "only used by"),
])
def test_estimator_validates_criterion_params(kw, match):
    with pytest.raises(ValueError, match=match):
        _fit(**kw)


def test_each_criterion_grows_its_own_trees():
    base, X = _fit()
    t = np.linspace(0.2, 5.0, 25)
    ref = base.predict_survival_function(X, t)
    for name, kw in CRITERIA[1:]:
        params = {"split_criterion": name}
        if "horizon" in kw:
            params["criterion_horizon"] = kw["horizon"]
        a, _ = _fit(**params)
        b, _ = _fit(**params)
        sa = a.predict_survival_function(X, t)
        assert np.all((sa >= 0) & (sa <= 1))
        np.testing.assert_array_equal(sa, b.predict_survival_function(X, t))  # deterministic
        assert not np.array_equal(sa, ref), name  # a different split rule, different trees


def test_criterion_params_survive_clone_and_pickle():
    import pickle

    model, X = _fit(split_criterion="km_gini", criterion_horizon=1.5)
    c = clone(model)
    assert (c.split_criterion, c.criterion_horizon) == ("km_gini", 1.5)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.split_criterion == "km_gini"
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    lm = LandmarkSurvivalForest(horizon=6.0, forest=SurvivalForestTV())
    lm.set_params(forest__split_criterion="poisson")
    params = clone(lm).get_params(deep=True)
    assert params["forest__split_criterion"] == "poisson"
    assert params["forest__criterion_horizon"] is None


def test_nested_cv_tunes_the_split_criterion():
    panel = simulate_panel(n_units=100, n_periods=36, seed=2)
    forest = SurvivalForestTV(n_estimators=10, min_ids_leaf=5, random_state=0)
    model = LandmarkSurvivalForest(horizon=6.0, history_features=["x", "z"], step=6.0, forest=forest)
    grid = [
        {"forest__split_criterion": ["logrank", "poisson"]},
        {"forest__split_criterion": ["km_gini"], "forest__criterion_horizon": [6.0]},
    ]
    outer, inner = RollingOriginSplit(1, test_size=6, gap=6), RollingOriginSplit(2, test_size=6, gap=6)
    res = landmark_cross_validate(model, panel, outer, scoring=["brier"], param_grid=grid, inner_cv=inner,
                                  refit="brier")
    allowed = {repr({"forest__split_criterion": c}) for c in ("logrank", "poisson")}
    allowed.add(repr({"forest__split_criterion": "km_gini", "forest__criterion_horizon": 6.0}))
    assert set(res["params"]) <= allowed and np.isfinite(res["brier"].to_numpy()).all()
