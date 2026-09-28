"""Slice 8 (docs/plans/plan.md): competing-risks permutation-importance CI gate.

Closes the last part of the gap flagged in
`docs/plans/simulation-validation-findings.md` row 3: S17's declared R=50
statistical gate for the competing-risks oracle scenario only ever ran
manually (`bench.tvc_perm_sim run()`); the only CI-gated version
(`tests/test_tvc_sim_truth.py`'s `slow`-marked pilot) applies a loosened
0.3×-of-oracle bound instead of the design's real 0.1×.

Reuses `bench.tvc_perm_sim`'s existing, already-truth-checked competing-risks
scenario (`cr_data`/`cr_replicate`/`cr_oracle`/`CROracle`) and its statistics
helpers (`holm_reject`/`one_sided_t`/`upper_bound`) unchanged — no new DGP,
no new statistical machinery. As with Slice 7 (timing), this gate ports the
design's **real, unmodified rule**:

- (a) Holm–Bonferroni-corrected one-sided t-tests across `[d_s1, d_s2]`
  (cause-1's importance detected on cause-1's true driver, cause-2's
  spillover onto cause-1's driver not) reject on `d_s1` (position 0).
- (b) `d_s2`'s one-sided 95% upper confidence bound is
  <= `delta1 = 0.1 * oracle(cause 1)` (cause-2's importance for a feature
  that's only cause-1's driver doesn't leak above 10% of cause-1's own
  oracle scale).

Only the replication *scale* is reduced for CI speed (`n_train=n_eval=300,
n_estimators=60`, vs. the design's `1000/1000/200`) — verified empirically
across three independent seed ranges (pilots at 10000-10009 and
20010-20024, and a preview at the gate's own seeds 0-14) that the real rule
holds with real margin at this scale (`ub_s2` ≈ 0.017-0.026 vs.
`delta1` ≈ 0.061-0.062).

**Scope: this closes S17 entirely** (trend: Slice 6; timing: Slice 7;
competing risks: this slice). All of S19's landmark-importance scenarios
remain manual-only — a separate, explicit future gap.
"""

import numpy as np

from bench.tvc_perm_sim import cr_oracle, cr_replicate, holm_reject, one_sided_t, upper_bound

N_TRAIN = 300
N_EVAL = 300
N_ESTIMATORS = 60
N_REPS = 15


def run(n_reps=N_REPS, seed0=0, n_train=N_TRAIN, n_eval=N_EVAL, n_estimators=N_ESTIMATORS):
    """``(n_reps, 2)`` array of ``(d_s1, d_s2)``, seeds ``seed0..seed0+n_reps-1``."""
    rows = [
        cr_replicate(seed0 + r, n_train=n_train, n_eval=n_eval, n_estimators=n_estimators)
        for r in range(n_reps)
    ]
    return np.array([[r["d_s1"], r["d_s2"]] for r in rows])


def check(res):
    """``(pass_cause1, pass_cause2, ub_s2, delta1)`` for a ``(n_reps, 2)`` array from ``run``.

    ``pass_cause1``: Holm-corrected one-sided test on ``d_s1`` rejects.
    ``pass_cause2``: ``d_s2``'s Bonferroni-free (single-test, per the design)
    upper 95% bound is <= ``delta1``.
    """
    o1 = cr_oracle()
    delta1 = 0.1 * o1
    d_s1, d_s2 = res[:, 0], res[:, 1]
    reject = holm_reject([one_sided_t(d_s1), one_sided_t(d_s2)])
    ub_s2 = upper_bound(d_s2, 0.05)
    return bool(reject[0]), bool(ub_s2 <= delta1), ub_s2, delta1
