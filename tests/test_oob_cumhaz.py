"""Out-of-bag cumulative hazards (S16): the engine calls behind OOB importance."""

import numpy as np
import pytest

from bench.perf_fit import synth
from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y, make_survival_y

X, Y, IDS = synth(1200, rows_per_id=4, seed=1)

MODES = {
    "id": {},
    "survival_agg": {"aggregate": "survival"},
    "coarse": {"ntime": 30},
    "block0": {"resample_unit": "block", "block_length": 2.0, "oob_buffer": 0},
    "block1": {"resample_unit": "block", "block_length": 2.0, "oob_buffer": 1},
    "bootstrap": {"bootstrap": True},
}


@pytest.mark.parametrize("mode", MODES)
def test_row_sum_equals_oob_mortality_bitwise(mode):
    m = SurvivalForestTV(n_estimators=25, oob_score=True, random_state=0, **MODES[mode]).fit(X, Y, IDS)
    d = m._rebuild_design(X, Y, IDS)
    H, n_trees = m.forest_.oob_cumhaz(d.X, *d.oob_set, m.event_times_, m.aggregate, 2)
    ref = m.oob_prediction_ if d.kept is None else m.oob_prediction_[d.kept]
    ref_n = m.oob_n_trees_ if d.kept is None else m.oob_n_trees_[d.kept]
    # oob_mortality sums the row buffer left to right; cumsum is sequential too
    assert np.array_equal(np.cumsum(H, axis=1)[:, -1], ref, equal_nan=True)
    np.testing.assert_array_equal(n_trees, ref_n)
    np.testing.assert_array_equal(np.isnan(H).all(axis=1), n_trees == 0)


def test_each_column_is_its_own_time():
    """Row sums cannot see values moved between time columns; single-time calls can."""
    m = SurvivalForestTV(n_estimators=25, random_state=0).fit(X, Y, IDS)
    d = m._rebuild_design(X, Y, IDS)
    times = m.event_times_[[0, 5, 40, -1]]
    H, _ = m.forest_.oob_cumhaz(d.X, *d.oob_set, times, m.aggregate, 1)
    for j, t in enumerate(times):
        mort, _ = m.forest_.oob_mortality(d.X, *d.oob_set, np.array([t]), m.aggregate, 1)
        assert np.array_equal(H[:, j], mort, equal_nan=True)
    ok = np.isfinite(H).all(axis=1)
    assert (np.diff(H[ok], axis=1) >= 0).all()  # a cumulative hazard does not decrease


def test_stacked_coarse_block_rebuild_matches_the_fit():
    """The rebuilt design reproduces the fit's OOB predictions bit-for-bit (independent of _fit_design)."""
    y = make_survival_y(Y["stop"] - Y["start"], Y["event"], start=np.zeros(len(Y)))
    bt = Y["start"].copy()
    m = SurvivalForestTV(n_estimators=25, ntime=20, resample_unit="block", block_length=2.0, oob_score=True,
                         random_state=0).fit(X, y, IDS, layout="stacked", block_time=bt)
    d = m._rebuild_design(X, y, IDS, block_time=bt)
    H, _ = m.forest_.oob_cumhaz(d.X, *d.oob_set, m.event_times_, m.aggregate, 1)
    assert np.array_equal(np.cumsum(H, axis=1)[:, -1], m.oob_prediction_[d.kept], equal_nan=True)


def test_rows_without_an_oob_tree_are_nan():
    m = SurvivalForestTV(n_estimators=2, max_samples=0.95, random_state=3).fit(X, Y, IDS)
    d = m._rebuild_design(X, Y, IDS)
    H, n_trees = m.forest_.oob_cumhaz(d.X, *d.oob_set, m.event_times_[:5], m.aggregate, 1)
    assert (n_trees == 0).any() and (n_trees > 0).any()
    assert np.isnan(H[n_trees == 0]).all() and np.isfinite(H[n_trees > 0]).all()


def test_full_bag_gives_all_nan():
    m = SurvivalForestTV(n_estimators=3, max_samples=1.0, random_state=0).fit(X, Y, IDS)
    d = m._rebuild_design(X, Y, IDS)
    H, n_trees = m.forest_.oob_cumhaz(d.X, *d.oob_set, m.event_times_[:3], m.aggregate, 1)
    assert (n_trees == 0).all() and np.isnan(H).all()


def _aj(dH):
    """Discrete Aalen–Johansen from per-grid cause increments ``(n, J, K)``, with the engine's clamp."""
    n, J, K = dH.shape
    S = np.ones(n)
    F = np.zeros((n, J, K))
    clamped = 0
    for k in range(K):
        step = 1.0 - dH[:, :, k].sum(axis=1)
        clamped += int((step < 0).sum())
        F[:, :, k] = (F[:, :, k - 1] if k else 0.0) + S[:, None] * dH[:, :, k]
        S = S * np.maximum(step, 0.0)
    return F, clamped


def test_cause_cumhaz_is_consistent_with_oob_cif():
    rng = np.random.default_rng(3)
    labels = np.where(Y["event"], rng.integers(1, 3, len(Y)), 0)
    y = make_competing_risks_y(Y["stop"], labels, start=Y["start"])
    m = CompetingRisksForestTV(n_estimators=25, oob_score=True, random_state=0).fit(X, y, IDS)
    d = m._rebuild_design(X, y, IDS)
    grid = m.event_times_
    H, n_trees = m.forest_.oob_cause_cumhaz(d.X, *d.oob_set, grid, 1)
    cif, n_cif = m.forest_.oob_cif(d.X, *d.oob_set, grid, "hazard", 1)
    np.testing.assert_array_equal(n_trees, n_cif)
    ok = n_trees > 0
    dH = np.diff(np.concatenate([np.zeros(H.shape[:2] + (1,)), H], axis=2), axis=2)
    F, clamped = _aj(dH[ok])
    assert clamped == 0  # this fixture does not exercise the clamp path
    np.testing.assert_allclose(F, cif[ok], rtol=0, atol=1e-12)
    assert np.isnan(H[~ok]).all()


def test_empty_times_still_count_oob_trees():
    m = SurvivalForestTV(n_estimators=10, random_state=0).fit(X, Y, IDS)
    d = m._rebuild_design(X, Y, IDS)
    empty = np.array([], dtype=float)
    H, n = m.forest_.oob_cumhaz(d.X, *d.oob_set, empty, m.aggregate, 1)
    _, n_ref = m.forest_.oob_mortality(d.X, *d.oob_set, m.event_times_[:1], m.aggregate, 1)
    assert H.shape == (len(d.X), 0)
    np.testing.assert_array_equal(n, n_ref)
    rng = np.random.default_rng(0)
    y = make_competing_risks_y(Y["stop"], np.where(Y["event"], rng.integers(1, 3, len(Y)), 0), start=Y["start"])
    mc = CompetingRisksForestTV(n_estimators=10, random_state=0).fit(X, y, IDS)
    dc = mc._rebuild_design(X, y, IDS)
    Hc, nc = mc.forest_.oob_cause_cumhaz(dc.X, *dc.oob_set, empty, 1)
    _, nc_ref = mc.forest_.oob_cause_cumhaz(dc.X, *dc.oob_set, mc.event_times_[:1], 1)
    assert Hc.shape == (len(dc.X), 2, 0)
    np.testing.assert_array_equal(nc, nc_ref)
