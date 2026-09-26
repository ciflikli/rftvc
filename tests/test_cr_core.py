"""S11: competing-risks core (cause-coded y, criteria, per-cause leaves, Aalen–Johansen CIF)."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.base import clone

from rftvc import (
    CR_DTYPE,
    CompetingRisksForestTV,
    SurvivalForestTV,
    _core,
    check_competing_risks_y,
    check_survival_y,
    make_competing_risks_y,
)
from tests.ref.cr_ref import aalen_johansen_ref, composite_ref
from tests.ref.logrank_ref import logrank_ref
from tests.test_tvc import _cp_data

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "aj_survfit.json").read_text())


def _cr_data(n_ids=150, seed=0, n_causes=2, labels=None):
    """`_cp_data` rows (ids, delayed entry, ties) with each event given a random cause label."""
    X, y, ids = _cp_data(n_ids, seed)
    rng = np.random.default_rng(seed + 100)
    labels = np.arange(1, n_causes + 1) if labels is None else np.asarray(labels)
    ev = np.where(y["event"], labels[rng.integers(0, len(labels), len(y))], 0)
    return X, make_competing_risks_y(y["stop"], ev, start=y["start"]), ids


def _single_leaf(**kw):
    """One tree, one leaf, grown on every id: the leaf is the whole-sample estimator."""
    return CompetingRisksForestTV(n_estimators=1, max_depth=0, max_samples=1.0, random_state=0, **kw)


def _state(model):
    return dict(model.forest_.__reduce__()[1][0])


# --- leaf oracle -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_single_leaf_equals_survfit_aalen_johansen(name):
    fx = FIXTURES[name]
    y = make_competing_risks_y(fx["stop"], fx["cause"], start=fx["start"])
    X = np.zeros((len(y), 1))
    m = _single_leaf().fit(X, y, ids=fx["id"])
    t = np.array(fx["time"])
    got = np.column_stack([m.predict_survival_function(X[:1], t)[0], m.predict_cumulative_incidence(X[:1], t)[0].T])
    np.testing.assert_allclose(got, np.array(fx["pstate"]), rtol=0, atol=1e-10)
    # The naive reference agrees, and the leaf holds the per-cause Nelson–Aalen.
    ref_t, ref_h, ref_f, ref_s = aalen_johansen_ref(fx["start"], fx["stop"], fx["cause"], fx["n_causes"])
    np.testing.assert_array_equal(m.event_times_, ref_t)
    np.testing.assert_allclose(m.predict_cumulative_incidence(X[:1])[0].T, ref_f, atol=1e-12)
    np.testing.assert_allclose(m.predict_survival_function(X[:1])[0], ref_s, atol=1e-12)
    times, cumhaz = m.forest_.leaf_profile(0, 0)
    np.testing.assert_array_equal(times, ref_t)
    np.testing.assert_allclose(cumhaz, ref_h, atol=1e-12)
    np.testing.assert_allclose(m.predict_cumulative_hazard(X[:1])[0].T, ref_h, atol=1e-12)


def test_time_with_only_cause_two_is_a_grid_point():
    fx = FIXTURES["tiny"]  # 1.5 is a cause-2 event time and nothing else
    y = make_competing_risks_y(fx["stop"], fx["cause"], start=fx["start"])
    stop, cause = np.array(fx["stop"]), np.array(fx["cause"])
    assert set(cause[stop == 1.5]) == {2}
    X = np.zeros((len(y), 1))
    m = _single_leaf().fit(X, y, ids=fx["id"])
    assert 1.5 in m.event_times_
    F = m.predict_cumulative_incidence(X[:1], [1.5 - 1e-9, 1.5])[0]
    assert F[1, 1] > F[1, 0]  # cause 2 jumps at 1.5
    assert F[0, 1] == F[0, 0]  # cause 1 does not


# --- J = 1 identity ----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("max_features", [1, None])
@pytest.mark.parametrize("rule", [{}, {"split_cause": 1}], ids=["composite", "split_cause"])
def test_one_cause_is_the_survival_forest(seed, max_features, rule):
    X, y, ids = _cp_data(120, seed)
    kw = dict(n_estimators=10, min_ids_leaf=3, max_features=max_features, random_state=seed)
    sf = SurvivalForestTV(**kw).fit(X, y, ids)
    y_cr = make_competing_risks_y(y["stop"], y["event"].astype(int), start=y["start"])
    cr = CompetingRisksForestTV(aggregate="hazard", **kw, **rule).fit(X, y_cr, ids)  # the hazard transform
    a, b = _state(sf), _state(cr)
    assert "leaf_cause_events" not in a  # SF pickles are unchanged; CR adds diagnostics
    assert a.keys() == b.keys() - {"leaf_cause_events"}
    for k in a:  # node arrays, leaf arrays, seeds, grid: bit-identical
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)
    H = sf.predict_cumulative_hazard(X)
    np.testing.assert_array_equal(cr.predict_cumulative_hazard(X, cause=1), H)
    np.testing.assert_array_equal(cr.apply(X), sf.apply(X))
    # Its CIF is the product-limit transform of the same hazard, not 1 - exp(-H).
    cif = cr.predict_cumulative_incidence(X, cause=1)
    want = 1 - np.cumprod(1 - np.diff(H, axis=1, prepend=0.0), axis=1)
    np.testing.assert_allclose(cif, want, rtol=0, atol=1e-12)
    assert not np.allclose(cif, 1 - np.exp(-H), atol=1e-6)


# --- criteria ----------------------------------------------------------------


@st.composite
def _split_data(draw):
    n = draw(st.integers(2, 25))
    n_causes = draw(st.integers(2, 3))
    start = np.array(draw(st.lists(st.integers(0, 4), min_size=n, max_size=n)), float) * 0.5
    stop = start + 0.5 * np.array(draw(st.lists(st.integers(1, 5), min_size=n, max_size=n)))
    codes = np.array(draw(st.lists(st.integers(0, n_causes), min_size=n, max_size=n)), np.uint8)
    left = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))
    return start, stop, codes, left, n_causes


@settings(max_examples=300, deadline=None)
@given(_split_data())
def test_scores_match_naive_references(data):
    start, stop, codes, left, n_causes = data
    got = _core.cause_score(start, stop, codes, left, n_causes)
    assert got == pytest.approx(composite_ref(start, stop, codes, left, n_causes), rel=1e-9, abs=1e-12)
    for k in range(1, n_causes + 1):
        single = _core.cause_score(start, stop, codes, left, n_causes, split_cause=k)
        assert single == pytest.approx(logrank_ref(start, stop, codes == k, left), rel=1e-9, abs=1e-12)


def test_composite_detects_opposing_effects_the_all_cause_log_rank_misses():
    rng = np.random.default_rng(3)
    n = 600
    x = rng.integers(0, 2, n).astype(bool)
    # Cause 1 is 4x faster with x, cause 2 4x slower; the all-cause hazard is the same.
    t1 = rng.exponential(1 / np.where(x, 2.0, 0.5))
    t2 = rng.exponential(1 / np.where(x, 0.5, 2.0))
    stop = np.minimum(t1, t2)
    codes = np.where(t1 < t2, 1, 2).astype(np.uint8)
    start = np.zeros(n)
    composite = _core.cause_score(start, stop, codes, x, 2)
    all_cause = _core.logrank_score(start, stop, codes != 0, x)
    assert composite > 100
    assert all_cause < 4  # chi-square(1) noise


# --- structure, invariants ---------------------------------------------------


@pytest.mark.parametrize("label", [1, 2])
def test_event_on_a_non_terminal_row_raises(label):
    y = make_competing_risks_y([1.0, 2.0, 3.0], [label, 0, 1], start=[0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="not the id's last row"):
        CompetingRisksForestTV(n_estimators=2).fit(np.zeros((3, 1)), y, ids=[0, 0, 1])


def test_aalen_johansen_invariants():
    X, y, ids = _cr_data(200, seed=4, n_causes=3)
    m = CompetingRisksForestTV(n_estimators=30, min_ids_leaf=3, random_state=0).fit(X, y, ids)
    F = m.predict_cumulative_incidence(X)
    S = m.predict_survival_function(X)
    np.testing.assert_allclose(F.sum(axis=1) + S, 1.0, rtol=0, atol=1e-12)
    assert (np.diff(F, axis=2) >= 0).all() and (np.diff(S, axis=1) <= 0).all()
    _, _, clamped = m.forest_.predict_cif(X, m.event_times_, 1)
    assert clamped == 0
    # Before the first event nothing has happened; unsorted times are fine.
    t = np.array([m.event_times_[-1], -1.0, m.event_times_[0]])
    F2 = m.predict_cumulative_incidence(X, t)
    np.testing.assert_array_equal(F2[:, :, 1], 0.0)
    np.testing.assert_array_equal(F2[:, :, 0], F[:, :, -1])
    np.testing.assert_array_equal(F2[:, :, 2], F[:, :, 0])


def test_hazard_by_cause_and_all_causes():
    X, y, ids = _cr_data(120, seed=5)
    m = CompetingRisksForestTV(n_estimators=8, min_ids_leaf=3, aggregate="hazard", random_state=0).fit(X, y, ids)
    H = m.predict_cumulative_hazard(X)
    assert H.shape == (len(X), 2, len(m.event_times_))
    np.testing.assert_array_equal(m.predict_cumulative_hazard(X, cause=2), H[:, 1])
    np.testing.assert_allclose(m.predict_cumulative_hazard(X, cause="all"), H.sum(axis=1))
    # Increments of the ensemble hazard drive the CIF (hazard aggregation).
    dH = np.diff(H, axis=2, prepend=0.0)
    S = np.cumprod(1 - dH.sum(axis=1), axis=1)
    S_before = np.concatenate([np.ones((len(X), 1)), S[:, :-1]], axis=1)
    np.testing.assert_allclose(m.predict_cumulative_incidence(X), np.cumsum(S_before[:, None] * dH, axis=2), atol=1e-12)
    np.testing.assert_allclose(m.predict_survival_function(X), S, atol=1e-12)


# --- labels ------------------------------------------------------------------


@pytest.mark.parametrize("labels", [[2, 5], [7, 300]])
@pytest.mark.parametrize("vocab", [False, True])
def test_non_contiguous_labels_index_the_cause_axis(labels, vocab):
    X, y, ids = _cr_data(120, seed=6, labels=labels)
    kw = dict(causes=labels) if vocab else {}
    m = CompetingRisksForestTV(n_estimators=6, min_ids_leaf=3, random_state=0, **kw).fit(X, y, ids)
    np.testing.assert_array_equal(m.causes_, labels)
    assert m.n_causes_ == 2 and m.forest_.n_causes == 2
    # The same data with labels 1, 2 gives the same forest: labels only name the axis.
    ref_y = y.copy()
    ref_y["event"] = np.searchsorted(labels, y["event"]) + 1
    ref_y["event"][y["event"] == 0] = 0
    ref = CompetingRisksForestTV(n_estimators=6, min_ids_leaf=3, random_state=0).fit(X, ref_y, ids)
    F = m.predict_cumulative_incidence(X)
    np.testing.assert_array_equal(F, ref.predict_cumulative_incidence(X))
    np.testing.assert_array_equal(m.predict_cumulative_incidence(X, cause=labels[1]), F[:, 1])
    m2 = clone(m).set_params(score_cause=labels[1]).fit(X, y, ids)
    np.testing.assert_array_equal(m.predict(X), F[:, 0, -1])
    np.testing.assert_array_equal(m2.predict(X), F[:, 1, -1])


def test_vocabulary_cause_without_events_gets_zero_hazard():
    X, y, ids = _cr_data(100, seed=7)
    with pytest.warns(UserWarning, match=r"causes \[3\] have no events"):
        m = CompetingRisksForestTV(n_estimators=5, min_ids_leaf=3, causes=[1, 2, 3], random_state=0).fit(X, y, ids)
    assert m.n_causes_ == 3
    np.testing.assert_array_equal(m.predict_cumulative_hazard(X, cause=3), 0.0)
    np.testing.assert_array_equal(m.predict_cumulative_incidence(X, cause=3), 0.0)
    ref = CompetingRisksForestTV(n_estimators=5, min_ids_leaf=3, random_state=0).fit(X, y, ids)
    np.testing.assert_allclose(m.predict_cumulative_incidence(X)[:, :2], ref.predict_cumulative_incidence(X), atol=1e-12)


def test_bad_labels_raise():
    X, y, ids = _cr_data(80, seed=8)
    m = CompetingRisksForestTV(n_estimators=3, min_ids_leaf=3, random_state=0).fit(X, y, ids)
    for cause in [3, 0, True, 1.0, "1"]:
        with pytest.raises(ValueError, match="not a fitted cause label"):
            m.predict_cumulative_incidence(X, cause=cause)
    for kw, name in [({"split_cause": 9}, "split_cause"), ({"score_cause": 9}, "score_cause")]:
        with pytest.raises(ValueError, match=name):
            clone(m).set_params(**kw).fit(X, y, ids)
    with pytest.raises(ValueError, match=r"\[2\] are not in causes"):
        clone(m).set_params(causes=[1, 3]).fit(X, y, ids)
    with pytest.raises(ValueError, match="non-negative"):
        make_competing_risks_y([1.0, 2.0], [-1, 1])
    with pytest.raises(ValueError, match="integers"):
        make_competing_risks_y([1.0, 2.0], [1.5, 1])
    for causes, match in [([0, 1], "positive"), ([1, 1], "repeat"), ([], "non-empty"), (range(1, 257), "at most 255")]:
        with pytest.raises(ValueError, match=match):
            check_competing_risks_y(y, causes=list(causes))
    many = make_competing_risks_y(np.arange(1.0, 258.0), np.arange(1, 258))
    with pytest.raises(ValueError, match="at most 255"):
        check_competing_risks_y(many)


@pytest.mark.parametrize("split_label, feature", [(3, 0), (8, 1)])
def test_split_cause_splits_on_that_causes_signal(split_label, feature):
    # Cause 3 depends only on x0, cause 8 only on x1 (labels map to codes 1, 2).
    rng = np.random.default_rng(12)
    n = 800
    x = rng.integers(0, 2, (n, 2)).astype(float)
    t3 = rng.exponential(1 / np.where(x[:, 0] == 1, 3.0, 0.3))
    t8 = rng.exponential(1 / np.where(x[:, 1] == 1, 3.0, 0.3))
    y = make_competing_risks_y(np.minimum(t3, t8), np.where(t3 < t8, 3, 8))
    m = CompetingRisksForestTV(
        n_estimators=1, max_depth=1, max_features=None, max_samples=1.0, split_cause=split_label, random_state=0
    ).fit(x, y)
    assert _state(m)["node_feature"][0] == feature


# --- parameter validation ------------------------------------------------------


def test_invalid_parameters_raise():
    X, y, ids = _cr_data(40, seed=1)
    for kw, match in [
        ({"criterion": "gray"}, "criterion"),
        ({"aggregate": "survival"}, "aggregate"),
        ({"min_events_leaf_cause": 2}, "requires split_cause"),
        ({"min_events_leaf_cause": 0, "split_cause": 1}, "min_events_leaf_cause"),
    ]:
        with pytest.raises(ValueError, match=match):
            CompetingRisksForestTV(n_estimators=2, **kw).fit(X, y, ids)


# --- pickle ------------------------------------------------------------------


def test_pickle_round_trip_and_v2_states():
    X, y, ids = _cr_data(120, seed=10)
    m = CompetingRisksForestTV(n_estimators=6, min_ids_leaf=3, random_state=0).fit(X, y, ids)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.predict_cumulative_incidence(X), m.predict_cumulative_incidence(X))
    np.testing.assert_array_equal(m2.causes_, m.causes_)
    state = _state(m)
    assert (state["format_version"], state["n_causes"]) == (3, 2)
    Forest = type(m.forest_)
    # A v2 state (before S11): single event, no `n_causes`.
    X1, y1, ids1 = _cp_data(80, 0)
    sf = SurvivalForestTV(n_estimators=4, min_ids_leaf=3, random_state=0).fit(X1, y1, ids1)
    v2 = {k: v for k, v in _state(sf).items() if k != "n_causes"}
    v2["format_version"] = 2
    loaded = Forest._from_state(v2)
    assert loaded.n_causes == 1
    np.testing.assert_array_equal(loaded.predict_cumhaz(X1, sf.event_times_, "hazard", 1), sf.predict_cumulative_hazard(X1))
    # Corrupt states raise ValueError, never panic.
    decreasing = state["cumhaz"].copy()
    leaf0 = int(state["event_offsets"][1])
    assert leaf0 >= 2
    decreasing[1] = decreasing[3] + 1.0  # cause 2 of entry 0 above entry 1
    for edit in [
        {"n_causes": 0},
        {"n_causes": 3},
        {"cumhaz": state["cumhaz"][:-1].copy()},
        {"cumhaz": decreasing},
        {"format_version": 2},  # a v2 state cannot hold two causes
    ]:
        with pytest.raises(ValueError):
            Forest._from_state({**state, **edit})
    v2_cr = {k: v for k, v in state.items() if k != "n_causes"} | {"format_version": 2}
    with pytest.raises(ValueError):
        Forest._from_state(v2_cr)


def test_leaf_profile_is_two_dimensional():
    X, y, ids = _cp_data(60, 0)
    sf = SurvivalForestTV(n_estimators=1, max_depth=0, max_samples=1.0, random_state=0).fit(X, y, ids)
    times, cumhaz = sf.forest_.leaf_profile(0, 0)
    assert cumhaz.shape == (len(times), 1)
    np.testing.assert_array_equal(cumhaz[:, 0], sf.predict_cumulative_hazard(X[:1], times)[0])


# --- validation --------------------------------------------------------------


def test_make_and_check_competing_risks_y():
    y = make_competing_risks_y([1.0, 2.0, 3.0], np.array([0.0, 2.0, 5.0]), start=[0.0, 0.5, 1.0])
    assert y.dtype == CR_DTYPE
    start, stop, codes, causes = check_competing_risks_y(y)
    np.testing.assert_array_equal(causes, [2, 5])
    np.testing.assert_array_equal(codes, [0, 1, 2])
    assert codes.dtype == np.uint8
    # A DataFrame y (no start column: start = 0) and a bool event.
    df = pd.DataFrame({"stop": [1.0, 2.0], "event": [True, False]})
    start, _, codes, causes = check_competing_risks_y(df)
    np.testing.assert_array_equal(start, 0.0)
    np.testing.assert_array_equal(causes, [1])
    np.testing.assert_array_equal(codes, [1, 0])
    with pytest.raises(ValueError, match="no events"):
        check_competing_risks_y(make_competing_risks_y([1.0], [0]))


def test_survival_y_rejects_cause_labels_with_a_pointer():
    with pytest.raises(TypeError, match="CompetingRisksForestTV"):
        check_survival_y(np.array([(0.0, 1.0, 2)], dtype=CR_DTYPE))


def test_core_rejects_malformed_inputs_without_panicking():
    X = np.zeros((2, 1))
    ev = np.array([1, 1], np.uint8)
    kw = dict(n_trees=1, n_draw=2, bootstrap=False, max_depth=None, min_ids_leaf=1, min_events_leaf=1,
              max_features=1, max_bins=255, seed=0, n_jobs=1, n_causes=1)
    groups = np.array([0, 1], np.uint32)
    for start, stop in [([0.0, 0.0], [1.0, np.nan]), ([0.0, np.inf], [1.0, 2.0]), ([0.0, 2.0], [1.0, 2.0])]:
        with pytest.raises(ValueError, match="finite with start < stop"):
            _core.fit_forest(X, np.array(start), np.array(stop), ev, groups, 2, **kw)
    with pytest.raises(ValueError, match="at least one feature"):
        _core.fit_forest(np.zeros((2, 0)), np.zeros(2), np.ones(2), ev, groups, 2, **kw)
    with pytest.raises(ValueError, match="event codes"):
        _core.fit_forest(X, np.zeros(2), np.ones(2), np.array([0, 2], np.uint8), groups, 2, **kw)
    with pytest.raises(ValueError, match="split_cause"):
        _core.fit_forest(X, np.zeros(2), np.ones(2), ev, groups, 2, **{**kw, "split_cause": 2})
