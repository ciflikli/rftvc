"""Slice 12: bench/cr_rfsrc_truth_check.py's closed-form truth, and the
competing-risks-vs-randomForestSRC parity gate itself.

Pass rule for ``test_rftvc_ise_within_epsilon_of_rfsrc_fixture``, declared before this gate was
run, using an epsilon calibrated from an out-of-band pilot (see
bench/cr_rfsrc_truth_check.py's module docstring): over R=10 replications (seeds 0-9), per-cause
``mean(rftvc_ISE) <= rfsrc_fixture_ISE + EPSILON``.

Deliberately NOT marked ``slow`` -- 500-tree competing-risks fits at this DGP's scale are cheap
(same order as Slice 5's cr_forest_truth_check.py gate), default tier per Slices 5-11's discipline.
"""

import numpy as np
import pytest
from scipy import integrate

from bench.cr_rfsrc_truth_check import EPSILON, GRID, run, rfsrc_ise_from_fixture, simulate, true_cif


def test_true_cif_matches_numerical_integration():
    rng = np.random.default_rng(0)
    n = 5
    x = rng.normal(size=n)
    from bench.cr_rfsrc_truth_check import _lam

    F = true_cif(x, GRID)
    for i in range(n):
        lam1, lam2, lam_tot = (v[0] for v in _lam(x[[i]]))

        def surv(s):
            return np.exp(-lam_tot * s)

        for cause, lam in ((0, lam1), (1, lam2)):
            for m, t in enumerate(GRID):
                if t == 0:
                    assert F[i, cause, m] == 0.0
                    continue
                val, _ = integrate.quad(lambda s: lam * surv(s), 0, t, limit=200)
                assert F[i, cause, m] == pytest.approx(val, abs=1e-6)


def test_true_cif_causes_sum_below_one_and_nondecreasing():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20)
    F = true_cif(x, GRID)
    assert np.all(F >= -1e-12)
    assert np.all(F.sum(axis=1) <= 1.0 + 1e-9)
    assert np.all(np.diff(F, axis=2) >= -1e-9)


def test_simulate_rows_are_well_formed():
    rng = np.random.default_rng(0)
    x, time, status = simulate(300, rng)
    assert x.shape == (300,)
    assert (time > 0).all()
    assert set(np.unique(status)) <= {0, 1, 2}


def test_replicate_smoke():
    from bench.cr_rfsrc_truth_check import replicate

    res = replicate(0, n_train=100, n_test=50, n_estimators=15)
    assert res.shape == (2,)
    assert np.all(np.isfinite(res))


def test_rfsrc_ise_from_fixture_finite():
    res = rfsrc_ise_from_fixture()
    assert res.shape == (2,)
    assert np.all(np.isfinite(res))
    assert np.all(res >= 0.0)


def test_rftvc_ise_within_epsilon_of_rfsrc_fixture():
    res = run()
    assert np.all(np.isfinite(res))
    rfsrc_ise = rfsrc_ise_from_fixture()
    gap = res.mean(axis=0) - rfsrc_ise
    assert np.all(gap <= EPSILON), (
        f"rftvc per-cause ISE exceeded rfsrc fixture ISE by more than epsilon: "
        f"gap={gap}, EPSILON={EPSILON}"
    )
