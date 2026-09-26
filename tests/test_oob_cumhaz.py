"""Out-of-bag cumulative hazards (S16): the engine calls behind OOB importance."""

import numpy as np
import pytest

from bench.perf_fit import synth
from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y

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
