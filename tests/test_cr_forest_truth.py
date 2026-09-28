"""Slice 5 (docs/plans/plan.md): CompetingRisksForestTV fitted-model-vs-truth gate.

Closes the gap flagged in docs/plans/simulation-validation-findings.md row 2:
bench/s14_cr_sim.py's run() computes no pass/fail decision, so no CI-gated
check exists that CompetingRisksForestTV's own predictions are close to a
known competing-risks truth. This test adds exactly that, on top of the
DGP's own closed-form truth (already checked by tests/test_cr_sim_truth.py,
untouched here).

Pass rule for test_forest_ise_within_tolerance_of_truth, declared before this
gate was run, threshold calibrated from an out-of-band pilot (seeds
10_000..10_004, not these seeds; see bench/cr_forest_truth_check.py's module
docstring): over R=10 replications (seeds 0-9) on scenario A, mean per-cause
ISE <= 0.10.
"""

import numpy as np
import pytest

from bench.cr_forest_truth_check import ISE_THRESHOLD, replicate, run


def test_replicate_smoke():
    err = replicate(0, n_train=100, n_test=50, n_estimators=20)
    assert err.shape == (2,)
    assert np.all(np.isfinite(err))


@pytest.mark.slow
def test_forest_ise_within_tolerance_of_truth():
    res = run(n_reps=10)
    assert res.mean() <= ISE_THRESHOLD
