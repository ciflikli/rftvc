"""S12: competing-risks paths, origins, aggregation, Approach-B equivalence, coarsening, blocks."""

import numpy as np
import pytest

from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y, make_survival_y
from rftvc._blocks import split_at_blocks
from tests.ref.coarsen_ref import coarsen_ref
from tests.test_cr_core import _cr_data, _state


def _aj(v, dL):
    """Aalen–Johansen on increments ``dL (V, J)`` at sorted times ``v``: ``(F, S, H)`` just after each."""
    S_after = np.cumprod(1.0 - dL.sum(axis=1))
    S_before = np.r_[1.0, S_after[:-1]]
    return np.cumsum(S_before[:, None] * dL, axis=0), S_after, np.cumsum(dL, axis=0)


def _at(v, values, t, before):
    """Right-continuous step through ``(v, values)`` at ``t``; ``before`` for ``t < v[0]``."""
    i = np.searchsorted(v, t, side="right")
    return before if i == 0 else values[i - 1]


def _ref_paths(model, X, start, stop, ids, origin, times, aggregate, extrapolate="none"):
    """Route each row, take its leaf's increments on ``(max(start, u), stop]``, then AJ (hazard or per tree)."""
    core, J = model.forest_, model.n_causes_
    leaves = model.apply(X)
    subjects = list(dict.fromkeys(ids.tolist()))
    F_out = np.full((len(subjects), J, len(times)), np.nan)
    S_out = np.full((len(subjects), len(times)), np.nan)
    H_out = np.full((len(subjects), J, len(times)), np.nan)
    for p, sid in enumerate(subjects):
        rows = np.flatnonzero(ids == sid)
        rows = rows[np.argsort(start[rows])]
        u = origin[p]
        per_tree = []
        for b in range(core.n_trees):
            inc = {}
            for i, r in enumerate(rows):
                lt, ch = core.leaf_profile(b, int(leaves[r, b]))
                d = np.diff(np.asarray(ch), axis=0, prepend=0.0)
                hi = np.inf if (i == len(rows) - 1 and extrapolate == "locf") else stop[r]
                for t, dv in zip(lt, d):
                    if max(start[r], u) < t <= hi:
                        inc[t] = inc.get(t, 0.0) + dv
            per_tree.append(inc)
        grid = np.array(sorted(set().union(*per_tree)))
        dense = np.array([[inc.get(t, np.zeros(J)) for t in grid] for inc in per_tree]).reshape(len(per_tree), -1, J)
        if aggregate == "hazard":
            F, S, H = _aj(grid, dense.mean(axis=0))
            curves = [(F, S, H)]
        else:
            curves = [_aj(grid, dense[b]) for b in range(len(per_tree))]
        last = stop[rows[-1]]
        for ti, t in enumerate(times):
            if t < u or (t > last and extrapolate == "none"):
                continue
            vals = [
                (_at(grid, F, t, np.zeros(J)), _at(grid, S, t, 1.0), _at(grid, H, t, np.zeros(J)))
                for F, S, H in curves
            ]
            F_out[p, :, ti] = np.mean([v[0] for v in vals], axis=0)
            S_out[p, ti] = np.mean([v[1] for v in vals])
            H_out[p, :, ti] = np.mean([v[2] for v in vals], axis=0)
    return F_out, S_out, H_out


def _intervals(y):
    return np.array(list(zip(y["start"], y["stop"])), dtype=[("start", float), ("stop", float)])


@pytest.fixture(scope="module")
def fitted():
    X, y, ids = _cr_data(120, seed=21, n_causes=2)
    models = {
        agg: CompetingRisksForestTV(n_estimators=6, min_ids_leaf=3, aggregate=agg, random_state=0).fit(X, y, ids)
        for agg in ("hazard", "cif")
    }
    return X, y, ids, models


@pytest.mark.parametrize("aggregate", ["hazard", "cif"])
@pytest.mark.parametrize("extrapolate", ["none", "locf"])
@pytest.mark.parametrize("mid_origin", [False, True])
def test_paths_match_reference(fitted, aggregate, extrapolate, mid_origin):
    X, y, ids, models = fitted
    m = models[aggregate]
    sel = ids < 25  # 25 subjects, several rows each
    Xs, ys, ids_s = X[sel], y[sel], ids[sel]
    start, stop = ys["start"], ys["stop"]
    subjects = list(dict.fromkeys(ids_s.tolist()))
    first = np.array([start[ids_s == s].min() for s in subjects])
    last = np.array([stop[ids_s == s].max() for s in subjects])
    origin = (first + last) / 2 if mid_origin else first
    grid = m.event_times_
    times = np.r_[grid[::5], grid[::7] + 1e-3, -1.0, last.max() + 1.0, np.median(origin)]
    kw = dict(intervals=_intervals(ys), ids=ids_s, origin=origin if mid_origin else None, extrapolate=extrapolate)
    F = m.predict_cumulative_incidence(Xs, times, **kw)
    S = m.predict_survival_function(Xs, times, **kw)
    H = m.predict_cumulative_hazard(Xs, times, **kw)
    rF, rS, rH = _ref_paths(m, Xs, start, stop, ids_s, origin, times, aggregate, extrapolate)
    np.testing.assert_allclose(F, rF, rtol=0, atol=1e-12)
    np.testing.assert_allclose(S, rS, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H, rH, rtol=0, atol=1e-12)
    ok = ~np.isnan(S)
    assert ok.any() and (~ok).any()  # both defined and NaN (before origin / after the path) values
    np.testing.assert_allclose((F.sum(axis=1) + S)[ok], 1.0, atol=1e-12)
    if mid_origin:  # conditional on being event-free at the origin
        at_u = m.predict_cumulative_incidence(Xs, origin[:1], **{**kw, "ids": ids_s})[0, :, 0]
        np.testing.assert_array_equal(at_u, 0.0)


@pytest.mark.parametrize("aggregate", ["hazard", "cif"])
def test_one_row_path_before_every_event_is_the_row_prediction(fitted, aggregate):
    X, y, ids, models = fitted
    m = models[aggregate]
    times = np.r_[m.event_times_, m.event_times_[-1] + 1]
    n = 10
    iv = np.zeros(n, dtype=[("start", float), ("stop", float)])
    iv["start"], iv["stop"] = m.event_times_[0] - 1.0, times[-1]
    kw = dict(intervals=iv, ids=np.arange(n))
    np.testing.assert_array_equal(
        m.predict_cumulative_incidence(X[:n], times, **kw), m.predict_cumulative_incidence(X[:n], times)
    )
    np.testing.assert_array_equal(m.predict_survival_function(X[:n], times, **kw), m.predict_survival_function(X[:n], times))
    np.testing.assert_allclose(
        m.predict_cumulative_hazard(X[:n], times, **kw), m.predict_cumulative_hazard(X[:n], times), atol=1e-12
    )


def test_cif_aggregation_is_the_mean_of_per_tree_aalen_johansen(fitted):
    X, y, ids, models = fitted
    m = models["cif"]
    n = 12
    times = m.event_times_[::3]
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    rF, rS, _ = _ref_paths(m, X[:n], lo, hi, np.arange(n), lo, times, "cif")
    np.testing.assert_allclose(m.predict_cumulative_incidence(X[:n], times), rF, atol=1e-12)
    np.testing.assert_allclose(m.predict_survival_function(X[:n], times), rS, atol=1e-12)
    # It differs from hazard aggregation on the same trees.
    assert not np.allclose(models["hazard"].predict_cumulative_incidence(X[:n], times), rF, atol=1e-6)


def test_one_cause_path_hazard_equals_the_survival_forest():
    X, y, ids = _cr_data(100, seed=3, n_causes=1)
    kw = dict(n_estimators=6, min_ids_leaf=3, random_state=1)
    y_sf = make_survival_y(y["stop"], y["event"] > 0, start=y["start"])
    sf = SurvivalForestTV(**kw).fit(X, y_sf, ids)
    cr = CompetingRisksForestTV(**kw).fit(X, y, ids)
    for extrapolate in ["none", "locf"]:
        pk = dict(intervals=_intervals(y), ids=ids, extrapolate=extrapolate)
        times = np.r_[sf.event_times_, sf.event_times_[-1] + 5]
        np.testing.assert_allclose(
            cr.predict_cumulative_hazard(X, times, cause=1, **pk), sf.predict_cumulative_hazard(X, times, **pk),
            rtol=0, atol=1e-12,
        )


# --- Approach B, cause floor ------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("m", [1, 3])
def test_split_cause_with_floor_is_the_cause_specific_forest(k, m):
    X, y, ids = _cr_data(160, seed=5, n_causes=2)
    kw = dict(n_estimators=8, min_ids_leaf=3, max_features=None, random_state=2)
    sf = SurvivalForestTV(min_events_leaf=m, **kw).fit(
        X, make_survival_y(y["stop"], y["event"] == k, start=y["start"]), ids
    )
    cr = CompetingRisksForestTV(split_cause=k, min_events_leaf=1, min_events_leaf_cause=m, **kw).fit(X, y, ids)
    a, b = _state(sf), _state(cr)
    for key in ["tree_seeds", "node_offsets", "node_feature", "node_threshold", "node_left", "node_right", "leaf_offsets"]:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert b["node_offsets"][-1] > 8  # the trees split
    for tree in range(8):
        for leaf in range(sf.forest_.n_leaves(tree)):
            t_sf, h_sf = sf.forest_.leaf_profile(tree, leaf)
            t_cr, h_cr = cr.forest_.leaf_profile(tree, leaf)
            on = np.isin(t_cr, t_sf)
            np.testing.assert_array_equal(np.asarray(t_cr)[on], t_sf)
            np.testing.assert_array_equal(h_cr[on, k - 1], h_sf[:, 0])  # bit-for-bit
            # Off the cause-k times, cause k adds exact zeros.
            np.testing.assert_array_equal(np.diff(h_cr[:, k - 1], prepend=0.0)[~on], 0.0)


def _brute_leaf_counts(model, X, ids, codes, tree):
    """In-bag events per (leaf, cause) by routing each in-bag row (bootstrap copies count)."""
    core = model.forest_
    leaves = model.apply(X)[:, tree]
    _, group = np.unique(ids, return_inverse=True)  # _cr_data ids are 0..n-1 by first appearance
    counts = np.zeros((core.n_leaves(tree), model.n_causes_), np.int64)
    for g in core.in_bag_ids(tree):
        for r in np.flatnonzero((group == g) & (codes > 0)):
            counts[leaves[r], codes[r] - 1] += 1
    return counts


@pytest.mark.parametrize("bootstrap", [False, True])
def test_leaf_cause_counts_and_cause_floor(bootstrap):
    X, y, ids = _cr_data(200, seed=8, n_causes=2)
    m_floor = 4
    model = CompetingRisksForestTV(
        n_estimators=10, min_ids_leaf=2, min_events_leaf=1, split_cause=2, min_events_leaf_cause=m_floor,
        bootstrap=bootstrap, random_state=3,
    ).fit(X, y, ids)
    codes = y["event"].astype(np.int64)
    split_trees = 0
    for tree in range(10):
        counts = model.forest_.leaf_cause_events(tree)
        np.testing.assert_array_equal(counts, _brute_leaf_counts(model, X, ids, codes, tree))
        if model.forest_.n_leaves(tree) > 1:  # leaves from admitted splits meet the floor
            split_trees += 1
            assert counts[:, 1].min() >= m_floor
    assert split_trees > 0
    summary = model.leaf_cause_events_summary()
    assert summary["cause"].to_list() == [1, 2]
    all_counts = np.concatenate([model.forest_.leaf_cause_events(b) for b in range(10)])
    assert summary["min"].to_list() == all_counts.min(axis=0).tolist()
    assert summary["max"].to_list() == all_counts.max(axis=0).tolist()


def test_root_below_the_floor_stays_a_leaf():
    X, y, ids = _cr_data(60, seed=9, n_causes=2)
    n2 = int((y["event"] == 2).sum())
    model = CompetingRisksForestTV(
        n_estimators=3, min_ids_leaf=1, min_events_leaf=1, split_cause=2, min_events_leaf_cause=n2, random_state=0
    ).fit(X, y, ids)  # every bag has < 2 * n2 cause-2 events
    # The pre-RNG gate's effect on feature draws is pinned by the Approach-B test.
    assert all(model.forest_.n_leaves(b) == 1 for b in range(3))


# --- coarsening, blocks -------------------------------------------------------------


@pytest.mark.parametrize("ntime", [4, 15])
def test_coarse_fit_equals_fit_on_coarsened_rows(ntime):
    X, y, ids = _cr_data(120, seed=11, labels=[2, 5])
    codes = np.searchsorted([2, 5], y["event"]) + 1
    codes[y["event"] == 0] = 0
    kept, s, t, e, grid, lost = coarsen_ref(y["start"], y["stop"], codes, ids, ntime)
    labels = np.where(e > 0, np.array([2, 5])[np.maximum(e, 1) - 1], 0)
    kw = dict(n_estimators=6, min_ids_leaf=3, random_state=4)
    coarse = CompetingRisksForestTV(ntime=ntime, **kw).fit(X, y, ids)
    exact = CompetingRisksForestTV(**kw).fit(X[kept], make_competing_risks_y(t, labels, start=s), ids[kept])
    np.testing.assert_array_equal(coarse.coarse_grid_, grid)
    assert (coarse.n_coarsen_dropped_rows_, coarse.n_coarsen_lost_events_) == (len(y) - kept.size, lost)
    times = exact.event_times_
    np.testing.assert_array_equal(coarse.predict_cumulative_incidence(X, times), exact.predict_cumulative_incidence(X, times))
    moved = (e > 0) & (codes[kept] == 0)  # a dropped row's event moved to the previous row
    if ntime == 4:
        assert moved.any()


def test_block_pieces_keep_the_cause_code():
    start, stop = np.array([0.0, 0.5]), np.array([2.5, 1.0])
    row, block, p_start, p_stop, p_event = split_at_blocks(start, stop, np.array([5, 0], np.int64), 1.0)
    assert p_event.dtype == np.int64
    np.testing.assert_array_equal(row, [0, 0, 0, 1])
    np.testing.assert_array_equal(p_event, [0, 0, 5, 0])
    b = split_at_blocks(start, stop, np.array([True, False]), 1.0)[4]
    assert b.dtype == bool and b.tolist() == [False, False, True, False]


def test_one_block_per_id_is_id_resampling():
    X, y, ids = _cr_data(80, seed=12)
    kw = dict(n_estimators=6, min_ids_leaf=3, random_state=5, oob_score=True)
    by_id = CompetingRisksForestTV(**kw).fit(X, y, ids)
    by_block = CompetingRisksForestTV(resample_unit="block", block_length=1e6, oob_buffer=0, **kw).fit(X, y, ids)
    np.testing.assert_array_equal(by_block.predict_cumulative_incidence(X), by_id.predict_cumulative_incidence(X))
    np.testing.assert_array_equal(by_block.oob_prediction_, by_id.oob_prediction_)
    assert by_block.oob_score_ == by_id.oob_score_


def test_leaf_count_key_in_pickled_states():
    X, y, ids = _cr_data(80, seed=16)
    m = CompetingRisksForestTV(n_estimators=4, min_ids_leaf=3, random_state=0).fit(X, y, ids)
    state = _state(m)
    Forest = type(m.forest_)
    n_leaves = int(state["leaf_offsets"][-1])
    assert state["leaf_cause_events"].shape == (n_leaves * 2,)
    with pytest.raises(ValueError):
        Forest._from_state({**state, "leaf_cause_events": state["leaf_cause_events"][:-1].copy()})
    with pytest.raises(ValueError, match="empty"):
        Forest._from_state({**state, "leaf_cause_events": np.zeros(0, np.uint32)})
    # An S11 state (no key) loads; predictions are unchanged, diagnostics are unavailable.
    old = {k: v for k, v in state.items() if k != "leaf_cause_events"}
    m.forest_ = Forest._from_state(old)
    assert not m.forest_.has_leaf_cause_events
    np.testing.assert_array_equal(m.predict_cumulative_incidence(X), CompetingRisksForestTV(
        n_estimators=4, min_ids_leaf=3, random_state=0).fit(X, y, ids).predict_cumulative_incidence(X))
    with pytest.raises(ValueError, match="no per-leaf cause counts"):
        m.leaf_cause_events_summary()
    with pytest.raises(ValueError, match="no per-leaf cause counts"):
        m.forest_.leaf_cause_events(0)


@pytest.mark.parametrize("aggregate", ["hazard", "cif"])
def test_infinite_time_gives_the_final_state(fitted, aggregate):
    X, y, ids, models = fitted
    m = models[aggregate]
    last = m.event_times_[-1]
    np.testing.assert_array_equal(
        m.predict_cumulative_incidence(X[:5], [np.inf, last]), m.predict_cumulative_incidence(X[:5], [last, last])
    )
    sel = ids < 5
    kw = dict(intervals=_intervals(y[sel]), ids=ids[sel], extrapolate="locf")
    F = m.predict_cumulative_incidence(X[sel], [np.inf, last + 1e6], **kw)
    np.testing.assert_array_equal(F[:, :, 0], F[:, :, 1])
    assert (F[:, :, 0].sum(axis=1) > 0).all()
