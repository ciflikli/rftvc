"""S14: the competing-risks simulation's closed-form truth (bench/s14_cr_sim.py)."""

import numpy as np
import pytest

from bench.s14_cr_sim import GRID, covariates, event_times, hazards, true_cif


@pytest.mark.parametrize("scenario", ["A", "B", "C"])
def test_closed_form_equals_numerical_integration(scenario):
    rng = np.random.default_rng(0)
    x0, x1, _, z = covariates(5, rng)
    lam = hazards(scenario, x0, x1, z)
    F = true_cif(lam, GRID)
    dt = 1e-4
    t = np.arange(0.0, GRID[-1] + dt / 2, dt)
    rate = lam[:, np.minimum(np.floor(t).astype(int), 7)]  # (n, len(t), J)
    S, Fn, out = np.ones(5), np.zeros((5, lam.shape[2])), []
    for i in range(t.size):
        out.append(Fn.copy())
        # Exact within the step (piecewise-constant rate), so only the grid alignment errs.
        h = rate[:, i].sum(axis=1)
        dS = S * (1 - np.exp(-h * dt))
        Fn = Fn + rate[:, i] / h[:, None] * dS[:, None]
        S = S - dS
    Fnum = np.stack(out, axis=2)[:, :, np.round(GRID / dt).astype(int)]
    np.testing.assert_allclose(F, Fnum, atol=1e-9)


@pytest.mark.parametrize("scenario", ["A", "B", "C"])
def test_simulated_event_times_follow_the_truth(scenario):
    rng = np.random.default_rng(1)
    n = 40_000
    x0, x1, _, z = covariates(n, rng)
    lam = hazards(scenario, x0, x1, z)
    t, cause = event_times(lam, rng)
    F = true_cif(lam, GRID).mean(axis=0)  # (J, len(GRID))
    for k in range(lam.shape[2]):
        emp = ((t[:, None] <= GRID) & (cause[:, None] == k + 1)).mean(axis=0)
        np.testing.assert_allclose(emp, F[k], atol=4 * np.sqrt(0.25 / n))
