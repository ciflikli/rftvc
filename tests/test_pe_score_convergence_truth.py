"""Slice 1 (docs/plans/plan.md): bench/pe_score_convergence_sim.py's closed-form
truth, and the convergence gate itself.

Slice 10 (docs/plans/plan.md, docs/plans/n-sweep-gate-{questions,research,design}.md):
replaces the original ``slow``-tier, R=10, 2-point gate with a default-tier
version at R=20 plus a third, reported-only point. Research found the
original R=10 rule is **not** robust across seed ranges it was never tested
against — its one-sided lower bound goes negative (fails) at an out-of-band
batch (seeds 20010-20029); R=20 restores a real, if sometimes thin, positive
margin (0.014-0.070 across 3 independent batches). Research also found that
a genuine 3+ point "decreases at every step" sweep is not honestly
achievable at any scale tested: the mid-to-large step's one-sided lower
bound is negative in every batch tried — the oracle-to-model gap plateaus
hard after the first jump, a real property of this score/DGP, not a
rep-count problem. So the middle point (n=1000) is included in the sweep
and reported, but only the original 200-vs-5000 outer pair is gated.

Accepted cost (user-confirmed 2026-09-29): ~30s for this one test, the
heaviest of any default-tier gate in this codebase — R=20 at
``n_estimators=200`` fitting forests up to n=5000, three times per
replication. Scaling this down further was tried in research and found to
reintroduce the same seed-range fragility it exists to fix.
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


N_VALUES = (200, 1000, 5000)
N_ESTIMATORS = 200
N_REPS = 20


def test_pe_score_gap_shrinks_from_n200_to_n5000():
    """The design's real, unmodified 200-vs-5000 endpoint claim (one-sided 95% lower bound on
    the paired gap difference > 0), at R=20 instead of the original R=10 (see module docstring
    for why). The n=1000 middle point is reported via its mean in the assertion message but is
    not itself gated -- only checked for finiteness."""
    res, diff, lower = run(n_values=N_VALUES, n_estimators=N_ESTIMATORS, n_reps=N_REPS)
    means = res.mean(axis=0)
    assert np.all(np.isfinite(means))
    assert lower > 0, f"200-vs-5000 gap did not shrink with a significant margin: lower={lower}, means={means}"
