"""Slice 6 (docs/plans/plan.md): trend-scenario permutation-importance CI gate.

Closes part of the gap flagged in `docs/plans/simulation-validation-findings.md`
row 3: S17's declared R=50 statistical gate for `permutation_importance` only
ever ran manually (`bench.tvc_perm_sim run()`); the only CI-gated version
(`tests/test_tvc_sim_truth.py`'s `slow`-marked pilot) uses **loosened**
margins (>= 25% of oracle instead of the design's >= 50%) and — per Slice
5's own lesson — doesn't run in CI's default tier at all, since it's
`slow`-marked. This adds a check that **does** run by default, with margins
freshly and honestly calibrated for its own (smaller) scale — not the
original design's 0.5x/0.1x-of-oracle numbers, which were calibrated for
`n_train=n_eval=1000, n_estimators=200` and are not directly portable to a
faster setting without their own recalibration.

Reuses `bench.tvc_perm_sim`'s existing, already-truth-checked trend scenario
(`trend_data`/`trend_replicate`/`trend_oracle`/`TrendOracle`) unchanged — no
new DGP. Only the replication *scale* is reduced for CI speed
(`n_train=300, n_eval=300, n_estimators=60`, vs. the design's
`1000/1000/200`), and the pass thresholds are freshly calibrated at that
reduced scale via an out-of-band pilot, not reused from the design's
full-scale numbers.

**Scope: trend scenario only** (relevant feature z1 vs irrelevant feature
z2). S17's timing-window and competing-risks oracle scenarios, and all of
S19's landmark-importance scenarios, remain open — the same manual-only gap,
explicit future work, not closed by this slice.

Pilot (seeds 10000-10009, 10 replications, this module's own reduced scale;
not the gate's own seeds 0-14): `imp_z1` mean ≈ 0.380 (oracle `oz1` =
0.6146, ratio ≈ 62%), std ≈ 0.067, min 0.276, max 0.515; `imp_z2` mean ≈
0.017 (oracle `oz2` = 0.0 exactly), std ≈ 0.043.

Pass rule (declared here, before the gate's own R=15 replications were run):
over R=15 replications (seeds 0-14), `mean(imp_z1) >= 0.45 * oz1` (≈ 0.277,
well under the pilot mean 0.380 but meaningfully above the noise floor) and
`abs(mean(imp_z2)) <= 0.15` (well above the pilot mean ≈ 0.017 but far below
`oz1`'s scale, so it still catches a real "irrelevant feature looks
relevant" regression).
"""

import numpy as np

from bench.tvc_perm_sim import trend_replicate  # noqa: F401 (re-exported for callers)

N_TRAIN = 300
N_EVAL = 300
N_ESTIMATORS = 60
Z1_RATIO = 0.45
Z2_ABS = 0.15


def run(n_reps=15, seed0=0, n_train=N_TRAIN, n_eval=N_EVAL, n_estimators=N_ESTIMATORS):
    """``(n_reps, 2)`` array of ``(imp_z1, imp_z2)``, seeds ``seed0..seed0+n_reps-1``."""
    rows = [
        trend_replicate(seed0 + r, n_train=n_train, n_eval=n_eval, n_estimators=n_estimators)
        for r in range(n_reps)
    ]
    return np.array([[r["imp_z1"], r["imp_z2"]] for r in rows])
