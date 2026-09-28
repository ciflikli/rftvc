"""Slice 6 (docs/plans/plan.md): trend-scenario permutation-importance CI gate.

Closes part of the gap flagged in docs/plans/simulation-validation-findings.md
row 3 (S17's real R=50 gate never runs in CI). Pass rule for
test_trend_importance_within_real_margin_of_oracle, declared before this
gate was run, thresholds calibrated from an out-of-band pilot (seeds
10000-10009, not these seeds; see bench/trend_importance_truth_check.py's
module docstring): over R=15 replications (seeds 0-14), mean(imp_z1) >=
0.45 * oz1 and abs(mean(imp_z2)) <= 0.15, where (oz1, oz2) is the closed-form
trend oracle.

Deliberately NOT marked ``slow``: the whole point of this test is a check
that actually runs in the default merge-gate CI tier.
"""

import numpy as np
import pytest

from bench.trend_importance_truth_check import Z1_RATIO, Z2_ABS, run
from bench.tvc_perm_sim import trend_oracle

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*extrapolate:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*strata=None.*:UserWarning"),
]


def test_replicate_smoke():
    res = run(n_reps=1, n_estimators=15)
    assert res.shape == (1, 2)
    assert np.all(np.isfinite(res))


def test_trend_importance_within_real_margin_of_oracle():
    oz1, oz2 = trend_oracle()
    assert oz2 == 0.0
    res = run(n_reps=15)
    assert res[:, 0].mean() >= Z1_RATIO * oz1
    assert abs(res[:, 1].mean()) <= Z2_ABS
