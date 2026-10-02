"""Slice 2: bench/lifelines_truth_check.py's closed-form
truth, static-covariate structure, and the lifelines cross-check gate.

Pass rule for ``test_rftvc_ise_within_tolerance_of_coxph``, declared before
this gate was run, epsilon calibrated from an out-of-band pilot (seeds
>= 10_000, not these seeds) — see bench/lifelines_truth_check.py's module
docstring: over R=10 replications (seeds 0-9), ``mean(rftvc_ISE) <=
mean(cox_ISE) + 0.08``.
"""

import numpy as np
import pytest
from scipy import integrate

from bench.lifelines_truth_check import BETA, GRID, LAMBDA0, EPSILON, run, simulate, true_survival


def test_covariate_is_static_per_subject():
    """simulate() draws x once per subject and never modifies it — the DGP has
    exactly one (x, U, event) triple per subject, not a redrawn-per-interval
    covariate subset to one column."""
    rng = np.random.default_rng(0)
    x, U, event = simulate(50, rng)
    assert x.shape == U.shape == event.shape == (50,)


def test_true_survival_matches_numerical_integration():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5)

    def hazard(t, i):
        return LAMBDA0 * np.exp(BETA * x[i])  # constant in t: exponential baseline

    for i in range(5):
        for t in GRID[1:]:
            val, _ = integrate.quad(hazard, 0, t, args=(i,), limit=200)
            expected = np.exp(-val)
            assert true_survival(x[i : i + 1], np.array([t]))[0, 0] == pytest.approx(expected, abs=1e-9)


@pytest.mark.slow
def test_rftvc_ise_within_tolerance_of_coxph():
    res = run(n_reps=10)
    rftvc_ise, cox_ise = res[:, 0].mean(), res[:, 1].mean()
    assert rftvc_ise <= cox_ise + EPSILON
