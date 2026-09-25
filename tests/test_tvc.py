"""S3: counting-process data, delayed entry, straddling ids, path prediction."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lifelines import NelsonAalenFitter

from rftvc import SurvivalForestTV, _core, check_counting_process, make_survival_y
from tests.ref.logrank_ref import logrank_ref, nelson_aalen_ref


def _cp_data(n_ids, seed, max_rows=4, delayed=True, round_to=1):
    """Contiguous rows per id, a covariate that changes per row, optional delayed entry."""
    rng = np.random.default_rng(seed)
    X, start, stop, event, ids = [], [], [], [], []
    for i in range(n_ids):
        t = float(np.round(rng.uniform(0, 2), round_to)) if delayed else 0.0
        k = int(rng.integers(1, max_rows + 1))
        for j in range(k):
            dur = float(np.round(rng.exponential(1.0), round_to)) + 10.0 ** -round_to
            z = rng.normal()
            X.append([z, rng.integers(0, 4)])
            start.append(t)
            stop.append(t + dur)
            event.append(j == k - 1 and rng.random() < 0.7)
            ids.append(i)
            t += dur
    X, ids = np.array(X, float), np.array(ids)
    y = make_survival_y(np.array(stop), np.array(event), start=np.array(start))
    y["event"][np.argmax(y["stop"])] = True  # at least one event
    return X, y, ids


def _cols(y):
    """Contiguous (start, stop, event); _core rejects strided structured-array fields."""
    return tuple(np.ascontiguousarray(y[k]) for k in ("start", "stop", "event"))


def _one_tree(**kw):
    return SurvivalForestTV(**{"n_estimators": 1, "max_samples": 1.0, "max_features": None, **kw})


# ---------------------------------------------------------------- estimators with delayed entry


def test_root_leaf_matches_delayed_entry_nelson_aalen():
    X, y, ids = _cp_data(200, seed=1)
    model = _one_tree(max_depth=0).fit(X, y, ids=ids)
    times, _, _, cumhaz = map(np.asarray, model.forest_.leaf_profile(0, 0))
    naf = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(y["stop"], y["event"], entry=y["start"], timeline=times)
    np.testing.assert_allclose(cumhaz, naf.cumulative_hazard_.to_numpy().ravel(), atol=1e-10)
    ref_t, ref_c = nelson_aalen_ref(y["start"], y["stop"], y["event"])
    np.testing.assert_allclose(times, ref_t)
    np.testing.assert_allclose(cumhaz, ref_c, atol=1e-12)


def test_boundary_entry_and_exit_on_event_times():
    # Entries exactly at event times are not at risk at that time; exits are.
    start = np.array([0.0, 1.0, 1.0, 2.0, 0.0])
    stop = np.array([1.0, 2.0, 3.0, 3.0, 2.0])
    event = np.array([True, True, False, True, False])
    y = make_survival_y(stop, event, start=start)
    model = _one_tree(max_depth=0).fit(np.zeros((5, 1)), y)
    times, d, at_risk, cumhaz = map(np.asarray, model.forest_.leaf_profile(0, 0))
    ref_t, ref_c = nelson_aalen_ref(start, stop, event)
    np.testing.assert_array_equal(times, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(at_risk, [2.0, 3.0, 2.0])
    np.testing.assert_allclose(cumhaz, ref_c, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_logrank_matches_reference_with_delayed_entry_and_ties(seed):
    X, y, _ = _cp_data(150, seed=seed)
    left = X[:, 0] <= 0
    got = _core.logrank_score(*_cols(y), left)
    assert got == pytest.approx(logrank_ref(y["start"], y["stop"], y["event"], left), rel=1e-10)


# ---------------------------------------------------------------- straddling ids


def _brute_force_ids(X, y, units, min_leaf, min_events):
    """Best admissible split, counting distinct units with any row on each side."""
    best, best_counts = 0.0, None
    e = y["event"]
    for f in range(X.shape[1]):
        for v in np.unique(X[:, f])[:-1]:
            left = X[:, f] <= v
            ids_l, ids_r = len(np.unique(units[left])), len(np.unique(units[~left]))
            el = (e & left).sum()
            if min(ids_l, ids_r) < min_leaf or min(el, e.sum() - el) < min_events:
                continue
            score = logrank_ref(y["start"], y["stop"], e, left)
            if score > best:
                best, best_counts = score, (ids_l, ids_r)
    return best, best_counts


@settings(max_examples=60, deadline=None)
@given(n_ids=st.integers(8, 40), seed=st.integers(0, 10_000), min_leaf=st.integers(1, 8))
def test_split_counts_straddling_ids_in_both_children(n_ids, seed, min_leaf):
    X, y, ids = _cp_data(n_ids, seed)
    X[:, 0] = np.round(X[:, 0], 1)
    order = np.argsort(ids, kind="stable")  # units must be contiguous
    X, y, ids = X[order], y[order], ids[order].astype(np.uint32)
    best, counts = _brute_force_ids(X, y, ids, min_leaf, 1)
    got = _core.best_split(X, *_cols(y), min_ids_leaf=min_leaf, min_events_leaf=1, units=ids)
    if best == 0.0:
        assert got is None
        return
    feature, threshold, score, mask, ids_l, ids_r = got
    mask = np.asarray(mask)
    assert score == pytest.approx(best, rel=1e-9)
    assert (ids_l, ids_r) == (len(np.unique(ids[mask])), len(np.unique(ids[~mask])))
    assert min(ids_l, ids_r) >= min_leaf


def test_straddling_id_admitted_at_boundary_and_rejected_below():
    # id 0 has rows on both sides of x <= 0.5; ids 1..4 are one side each.
    X = np.array([[0.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
    ids = np.array([0, 0, 1, 2, 3, 4], np.uint32)
    y = make_survival_y([1.0, 2.0, 1.5, 2.5, 3.0, 3.5], [0, 1, 1, 1, 1, 1], start=[0.0, 1.0, 0, 0, 0, 0])
    args = (X, *_cols(y))
    got = _core.best_split(*args, min_ids_leaf=3, min_events_leaf=1, units=ids)
    assert got is not None and got[4:] == (3, 3)  # id 0 counted on both sides
    assert _core.best_split(*args, min_ids_leaf=4, min_events_leaf=1, units=ids) is None


def test_fitted_leaves_respect_min_ids_leaf():
    X, y, ids = _cp_data(300, seed=5)
    model = _one_tree(min_ids_leaf=20, min_events_leaf=2, random_state=0).fit(X, y, ids=ids)
    leaves = model.apply(X)[:, 0]
    assert len(np.unique(leaves)) > 1
    for leaf in np.unique(leaves):
        assert len(np.unique(ids[leaves == leaf])) >= 20


def test_whole_ids_are_drawn_with_multiple_rows():
    X, y, ids = _cp_data(200, seed=6)
    model = SurvivalForestTV(n_estimators=5, random_state=0).fit(X, y, ids=ids)
    assert model.n_ids_ == 200
    assert model.n_draw_ == round(0.632 * 200)
    assert len(np.unique(model.forest_.in_bag_ids(0))) == model.n_draw_


# ---------------------------------------------------------------- path prediction


def _paths(seed=7):
    X, y, ids = _cp_data(300, seed=seed, delayed=False)
    model = SurvivalForestTV(n_estimators=30, min_ids_leaf=10, random_state=0).fit(X, y, ids=ids)
    return model, X, y, ids


def test_single_row_path_from_zero_equals_fixed_covariate_prediction():
    model, X, *_ = _paths()
    Xq = X[:5]
    times = np.linspace(0.1, 3, 12)
    intervals = make_survival_y(np.full(5, 10.0), np.zeros(5, bool))
    np.testing.assert_allclose(
        model.predict_cumulative_hazard(Xq, times, intervals=intervals),
        model.predict_cumulative_hazard(Xq, times),
        atol=1e-12,
    )


def test_path_hazard_is_sum_over_rows_of_row_hazards():
    model, *_ = _paths()
    rng = np.random.default_rng(0)
    Xp = rng.normal(size=(3, 2))
    iv = make_survival_y([1.0, 2.5, 4.0], np.zeros(3, bool), start=[0.0, 1.0, 2.5])
    times = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    H = model.predict_cumulative_hazard(Xp, times, intervals=iv, ids=[0, 0, 0])[0]
    fixed = model.predict_cumulative_hazard(Xp, np.r_[times, 1.0, 2.5])
    f = lambda r, t: model.predict_cumulative_hazard(Xp[r : r + 1], [t])[0, 0]  # noqa: E731
    expected = [
        f(0, 0.5),
        f(0, 1.0),
        f(0, 1.0) + f(1, 2.0) - f(1, 1.0),
        f(0, 1.0) + f(1, 2.5) - f(1, 1.0) + f(2, 3.0) - f(2, 2.5),
        f(0, 1.0) + f(1, 2.5) - f(1, 1.0) + f(2, 4.0) - f(2, 2.5),
    ]
    assert fixed.shape == (3, 7)
    np.testing.assert_allclose(H, expected, atol=1e-12)


def test_origin_default_and_conditional_survival():
    model, X, y, ids = _paths()
    sel = np.isin(ids, [3, 4])
    Xp, iv, pid = X[sel], y[sel], ids[sel]
    times = np.linspace(0.0, 2.0, 9)
    first = [iv["start"][pid == i].min() for i in (3, 4)]
    np.testing.assert_array_equal(
        model.predict_cumulative_hazard(Xp, times, intervals=iv, ids=pid),
        model.predict_cumulative_hazard(Xp, times, intervals=iv, ids=pid, origin=first),
    )
    u = 0.5
    last = min(iv["stop"][pid == i].max() for i in (3, 4))
    t = times[(times >= u) & (times <= last)]
    s0 = model.predict_survival_function(Xp, np.r_[u, t], intervals=iv, ids=pid)
    su = model.predict_survival_function(Xp, t, intervals=iv, ids=pid, origin=u)
    np.testing.assert_allclose(su, s0[:, 1:] / s0[:, :1], rtol=1e-10)
    assert np.isnan(model.predict_cumulative_hazard(Xp, [0.25], intervals=iv, ids=pid, origin=u)).all()


def test_extrapolation_none_is_nan_and_locf_equals_appended_row():
    model, *_ = _paths()
    Xp = np.array([[0.3, 1.0], [-1.2, 2.0]])
    iv = make_survival_y([1.0, 2.0], [False, False], start=[0.0, 1.0])
    times = np.array([0.5, 1.5, 2.0, 3.0, 5.0])
    none = model.predict_cumulative_hazard(Xp, times, intervals=iv, ids=[0, 0])[0]
    assert np.isnan(none[3:]).all() and np.isfinite(none[:3]).all()
    locf = model.predict_cumulative_hazard(Xp, times, intervals=iv, ids=[0, 0], extrapolate="locf")[0]
    appended = model.predict_cumulative_hazard(
        np.vstack([Xp, Xp[-1:]]),
        times,
        intervals=make_survival_y([1.0, 2.0, 5.0], [0, 0, 0], start=[0.0, 1.0, 2.0]),
        ids=[0, 0, 0],
    )[0]
    np.testing.assert_allclose(locf, appended, atol=1e-12)


def test_scenario_rows_change_prediction_only_after_last_observed_stop():
    model, *_ = _paths()
    observed_X = np.array([[0.3, 1.0], [-1.2, 2.0]])
    observed = make_survival_y([1.0, 2.0], [0, 0], start=[0.0, 1.0])
    times = np.array([0.5, 1.5, 2.0, 2.5, 3.5])
    preds = []
    for future in ([[2.5, 3.0]], [[-2.5, 0.0]]):
        Xs = np.vstack([observed_X, future])
        iv = make_survival_y([1.0, 2.0, 4.0], [0, 0, 0], start=[0.0, 1.0, 2.0])
        preds.append(model.predict_cumulative_hazard(Xs, times, intervals=iv, ids=[0, 0, 0])[0])
    np.testing.assert_array_equal(preds[0][:3], preds[1][:3])
    assert not np.allclose(preds[0][3:], preds[1][3:])


def test_paths_are_returned_in_first_appearance_order():
    model, *_ = _paths()
    Xp = np.array([[1.0, 0.0], [-1.0, 3.0]])
    iv = make_survival_y([2.0, 2.0], [0, 0])
    a = model.predict_cumulative_hazard(Xp, [1.0], intervals=iv, ids=["b", "a"])
    b = model.predict_cumulative_hazard(Xp, [1.0])
    np.testing.assert_array_equal(a, b)


def test_origin_outside_path_raises():
    model, *_ = _paths()
    Xp = np.zeros((1, 2))
    iv = make_survival_y([2.0], [0], start=[1.0])
    for bad in (0.5, 2.5, np.inf, np.nan):
        for extrapolate in ("none", "locf"):
            with pytest.raises(ValueError, match="origin"):
                model.predict_cumulative_hazard(Xp, [1.5], intervals=iv, origin=bad, extrapolate=extrapolate)


# ---------------------------------------------------------------- counting-process checks


def test_check_counting_process_errors():
    s, t = np.array([0.0, 1.0]), np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="gap"):
        check_counting_process(s, np.array([0.5, 2.0]), None, [0, 0])
    with pytest.raises(ValueError, match="overlap"):
        check_counting_process(s, np.array([1.5, 2.0]), None, [0, 0])
    with pytest.raises(ValueError, match="last row"):
        check_counting_process(s, t, np.array([True, False]), [0, 0])
    with pytest.raises(ValueError, match="look-ahead"):
        check_counting_process(s, t, None, [0, 0], measured_at=[0.0, 1.5])
    check_counting_process(s, t, np.array([False, True]), [0, 0], measured_at=[0.0, 1.0])


def test_split_id_policy_makes_segments_separate_groups():
    s, t = np.array([0.0, 2.0, 0.0]), np.array([1.0, 3.0, 1.0])
    cp = check_counting_process(s, t, None, ["a", "a", "b"], gap_policy="split_id")
    assert cp.n_groups == 3
    assert len(set(cp.group.tolist())) == 3


def test_fit_rejects_invalid_counting_process_and_accepts_split_id():
    X, y, ids = _cp_data(50, seed=8, max_rows=3, delayed=False)
    multi = np.flatnonzero(np.bincount(ids) > 1)
    assert multi.size
    first_row = np.flatnonzero(ids == multi[0])[0]
    y_gap = y.copy()
    y_gap["stop"][first_row] -= 0.05
    with pytest.raises(ValueError, match="gap"):
        SurvivalForestTV(n_estimators=2).fit(X, y_gap, ids=ids)
    model = SurvivalForestTV(n_estimators=2).fit(X, y_gap, ids=ids, gap_policy="split_id")
    assert model.n_ids_ == 51


def test_core_rejects_strided_inputs():
    X, y, _ = _cp_data(20, seed=9)
    with pytest.raises(ValueError, match="contiguous"):
        _core.logrank_score(y["start"], y["stop"], y["event"], X[:, 0] <= 0)
    with pytest.raises(ValueError, match="contiguous"):
        _core.best_split(np.asfortranarray(X), *_cols(y), min_ids_leaf=1, min_events_leaf=1)


def test_non_finite_measured_at_rejected():
    with pytest.raises(ValueError, match="finite"):
        check_counting_process(np.array([0.0]), np.array([1.0]), np.array([False]), [7], measured_at=[np.nan])


@pytest.mark.parametrize("ids", [[1, "1"], [1, "a"], np.array([1, "1"], dtype=object)])
def test_mixed_type_ids_rejected(ids):
    with pytest.raises(ValueError, match="label types"):
        check_counting_process(np.array([0.0, 1.0]), np.array([1.0, 2.0]), None, ids)


def test_homogeneous_string_and_object_ids_accepted():
    s, t = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    assert check_counting_process(s, t, None, ["1", "2"]).n_groups == 2
    assert check_counting_process(s, t, None, np.array(["a", "b"], dtype=object)).n_groups == 2
    assert check_counting_process(s, t, None, [1, 2.0]).n_groups == 2


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="no rows"):
        check_counting_process(np.array([]), np.array([]), None, None)


def test_best_split_rejects_non_contiguous_units():
    X, y, _ = _cp_data(10, seed=10, max_rows=1, delayed=False)
    units = np.zeros(len(X), np.uint32)
    units[1] = 1  # unit 0 appears in two runs
    with pytest.raises(ValueError, match="contiguous"):
        _core.best_split(X, *_cols(y), min_ids_leaf=1, min_events_leaf=1, units=units)
