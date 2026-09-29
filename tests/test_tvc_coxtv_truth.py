"""Slice 11 (docs/plans/plan.md): bench/tvc_coxtv_truth_check.py's closed-form truth, and the
TVC-vs-CoxTimeVaryingFitter parity gate itself.

Pass rule for ``test_rftvc_ise_within_epsilon_of_cox``, declared before this gate was run, using
an epsilon calibrated from an out-of-band pilot (see bench/tvc_coxtv_truth_check.py's module
docstring): over R=10 replications (seeds 0-9), ``mean(rftvc_ISE) <= mean(cox_ISE) + 0.18``.

Deliberately NOT marked ``slow`` -- ~4-5s at this scale, well within the default tier per Slices
5-10's discipline (Slice 2's analogous static-case gate predates that lesson and stayed slow).
"""

import numpy as np
import pytest
from scipy import integrate

from bench.tvc_coxtv_truth_check import EPSILON, K, RATE0, BETA, run, simulate, true_cumhaz

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*extrapolate:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
]


def test_true_cumhaz_matches_numerical_integration():
    rng = np.random.default_rng(0)
    n = 5
    z = rng.normal(size=(n, K))
    t_grid = np.linspace(0.0, 6.0, 13)
    H = true_cumhaz(z, t_grid)
    for i in range(n):

        def rate(s, i=i):
            k = min(int(np.floor(s)), K - 1)
            return RATE0 * np.exp(BETA * z[i, k])

        for m, t in enumerate(t_grid):
            if t == 0:
                assert H[i, m] == 0.0
                continue
            val, _ = integrate.quad(rate, 0, t, limit=200)
            assert H[i, m] == pytest.approx(val, abs=1e-6)


def test_simulate_rows_are_well_formed():
    rng = np.random.default_rng(0)
    z, U, event = simulate(300, rng)
    assert z.shape == (300, K)
    assert (U > 0).all()
    assert event.dtype == bool


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_replicate_smoke():
    from bench.tvc_coxtv_truth_check import replicate

    rftvc_ise, cox_ise = replicate(0, n_train=100, n_test=50, n_estimators=15)
    assert np.isfinite(rftvc_ise)
    assert np.isfinite(cox_ise)


def test_rftvc_ise_within_epsilon_of_cox():
    res = run()
    assert np.all(np.isfinite(res))
    gap = res[:, 0].mean() - res[:, 1].mean()
    assert gap <= EPSILON, f"rftvc ISE exceeded cox ISE by more than epsilon: gap={gap}, EPSILON={EPSILON}"
