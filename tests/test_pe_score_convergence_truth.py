"""Slice 1 (docs/plans/plan.md): bench/pe_score_convergence_sim.py's closed-form
truth, and the convergence gate itself.

Pass rule for ``test_pe_score_mean_gap_shrinks_with_n``, declared here before
the sim code was run: over R=10 replications at ``n_values=(200, 5000)``, the
lower bound of the one-sided 95% CI on the mean paired difference (gap at
n=200 minus gap at n=5000) is > 0.
"""

import numpy as np
import pytest
from scipy import integrate

from bench.pe_score_convergence_sim import WINDOWS, run, true_cumhaz
from tests.sim import K, hazard


def test_true_cumhaz_matches_numerical_integration():
    rng = np.random.default_rng(0)
    n = 5
    x0 = rng.normal(size=n)
    z = rng.normal(size=(n, K))
    h = true_cumhaz(x0, z, WINDOWS)
    for i in range(n):

        def integrand(s, i=i):
            k = min(int(np.floor(s)), K - 1)
            return hazard(x0[i : i + 1], z[i : i + 1])[0, k]

        for m, t in enumerate(WINDOWS):
            if t == 0:
                assert h[i, m] == 0.0
                continue
            val, _ = integrate.quad(integrand, 0, t, limit=200)
            assert h[i, m] == pytest.approx(val, abs=1e-6)


@pytest.mark.slow
def test_pe_score_mean_gap_shrinks_with_n():
    _, _, lower = run(n_values=(200, 5000), n_reps=10)
    assert lower > 0
