"""S6: coarse grid mode (design.md D8): table-driven fixtures and independent references."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lifelines import NelsonAalenFitter

from rftvc import SurvivalForestTV, _core, check_counting_process, make_survival_y
from tests.ref.coarsen_ref import coarsen_ref
from tests.ref.logrank_ref import nelson_aalen_ref
from tests.test_tvc import _cp_data

# Event times 1.5, 1.8, 2, 2, 4, 4 with ntime=2 give the grid {2, 4}
# (inverse-CDF ranks 3 and 6); the origin is the earliest start, 0.
TABLE = [
    # id,  start, stop, event
    ("A", 0.0, 1.5, False),  # A: collapsed event on a non-first row -> event moves to (0, 2]
    ("A", 1.5, 1.8, True),
    ("B", 0.5, 1.5, True),   # B: entered and failed inside (0, 2] -> dropped, event lost
    ("C", 0.5, 3.0, False),  # C: in-bin entry -> (2, 4]: at risk from 4 on
    ("D", 0.0, 3.0, False),  # D: in-bin censoring -> (0, 4]
    ("E", 0.0, 6.0, False),  # E: beyond the last event -> clamped to (0, 4]
    ("F", 5.0, 7.0, False),  # F: entirely after the last event -> dropped (no event lost)
    ("G", 0.0, 2.0, True),
    ("H", 0.0, 2.0, True),
    ("J", 0.0, 4.0, True),
    ("K", 0.0, 4.0, True),
]
EXPECTED = {
    ("A", 0.0, 2.0, True),
    ("C", 2.0, 4.0, False),
    ("D", 0.0, 4.0, False),
    ("E", 0.0, 4.0, False),
    ("G", 0.0, 2.0, True),
    ("H", 0.0, 2.0, True),
    ("J", 0.0, 4.0, True),
    ("K", 0.0, 4.0, True),
}


def _arrays(table):
    ids = np.array([r[0] for r in table])
    start, stop = (np.array([r[i] for r in table], float) for i in (1, 2))
    event = np.array([r[3] for r in table], bool)
    return ids, start, stop, event


def _core_coarsen(start, stop, event, ids, ntime, stacked=False):
    n = start.size
    if stacked:
        order, offsets = np.arange(n, dtype=np.uint32), np.arange(n + 1, dtype=np.uint64)
    else:
        cp = check_counting_process(start, stop, event, ids)
        order, offsets = cp.order.astype(np.uint32), cp.offsets
    return _core.coarsen(start, stop, event, order, offsets, ntime)


def test_d8_table_rows_and_lost_event_count():
    ids, start, stop, event = _arrays(TABLE)
    kept, s, t, e, grid, lost = _core_coarsen(start, stop, event, ids, 2)
    np.testing.assert_array_equal(grid, [2.0, 4.0])
    assert {(ids[k], a, b, c) for k, a, b, c in zip(kept, s, t, e)} == EXPECTED
    assert lost == 1  # B
    assert kept.size == len(TABLE) - 3  # A's second row, B, F


def test_stacked_rows_are_separate_chains():
    # Same id, two stacked observations: the collapsed event row is lost on its
    # own; the id's other row stays (in counting-process layout it would carry it).
    ids = np.array(["S", "S", "G", "H", "J", "K"])
    start = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    stop = np.array([1.5, 3.0, 2.0, 2.0, 4.0, 4.0])
    event = np.array([True, False, True, True, True, True])
    kept, s, t, e, _, lost = _core_coarsen(start, stop, event, ids, 2, stacked=True)
    assert lost == 1 and 0 not in kept and 1 in kept
    assert (s[kept == 1][0], t[kept == 1][0], e[kept == 1][0]) == (0.0, 4.0, False)


def test_single_node_matches_nelson_aalen_on_hand_coarsened_rows():
    ids, start, stop, event = _arrays(TABLE)
    X = np.zeros((len(TABLE), 1))
    f = SurvivalForestTV(n_estimators=1, max_depth=0, max_samples=1.0, ntime=2, random_state=0)
    f.fit(X, make_survival_y(stop, event, start=start), ids)
    rows = sorted(EXPECTED)
    s, t, e = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    times, ref = nelson_aalen_ref(s, t, e.astype(bool))
    got = f.predict_cumulative_hazard(np.zeros((1, 1)), times)[0]
    np.testing.assert_allclose(got, ref, atol=1e-12)
    naf = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(t, e, entry=s, timeline=times)
    np.testing.assert_allclose(got, naf.cumulative_hazard_.to_numpy().ravel(), atol=1e-12)
    np.testing.assert_array_equal(f.coarse_grid_, [2.0, 4.0])
    assert (f.n_coarsen_dropped_rows_, f.n_coarsen_lost_events_) == (3, 1)
    assert f.n_ids_ == 8  # B and F dropped: resampling units counted after coarsening


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), ntime=st.integers(1, 12), stacked=st.booleans())
def test_coarsen_matches_naive_reference(seed, ntime, stacked):
    X, y, ids = _cp_data(15, seed=seed, max_rows=3)
    start, stop, event = (np.ascontiguousarray(y[f]) for f in ("start", "stop", "event"))
    got = _core_coarsen(start, stop, event, ids, ntime, stacked=stacked)
    ref = coarsen_ref(start, stop, event, ids, ntime, stacked=stacked)
    np.testing.assert_array_equal(got[4], ref[4])
    assert got[5] == ref[5]
    rows = lambda out: sorted(zip(out[0].tolist(), out[1].tolist(), out[2].tolist(), out[3].tolist()))
    assert rows(got) == rows(ref)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coarse_forest_equals_exact_forest_on_reference_coarsened_rows(seed):
    X, y, ids = _cp_data(80, seed=seed, max_rows=4)
    kw = dict(n_estimators=10, min_ids_leaf=3, random_state=seed)
    coarse = SurvivalForestTV(ntime=8, **kw).fit(X, y, ids)
    kept, s, t, e, grid, _ = coarsen_ref(y["start"], y["stop"], y["event"], ids, 8)
    exact = SurvivalForestTV(**kw).fit(X[kept], make_survival_y(t, e, start=s), ids[kept])
    Xq = np.random.default_rng(seed).normal(size=(30, X.shape[1]))
    times = np.linspace(0, t.max(), 25)
    np.testing.assert_allclose(
        coarse.predict_cumulative_hazard(Xq, times), exact.predict_cumulative_hazard(Xq, times), atol=1e-12
    )
    np.testing.assert_array_equal(coarse.event_times_, exact.event_times_)


def test_oob_in_coarse_mode_keeps_original_rows():
    X, y, ids = _cp_data(60, seed=4, max_rows=4)
    f = SurvivalForestTV(n_estimators=20, min_ids_leaf=3, ntime=5, oob_score=True, random_state=0)
    f.fit(X, y, ids)
    kept = coarsen_ref(y["start"], y["stop"], y["event"], ids, 5)[0]
    dropped = np.setdiff1d(np.arange(X.shape[0]), kept)
    assert f.oob_prediction_.shape == (X.shape[0],)
    assert np.isnan(f.oob_prediction_[dropped]).all() and np.isfinite(f.oob_prediction_[kept]).all()
    assert 0 < f.oob_score_ < 1


def test_invalid_ntime():
    X, y, ids = _cp_data(10, seed=0)
    with pytest.raises(ValueError, match="ntime"):
        SurvivalForestTV(ntime=0).fit(X, y, ids)
