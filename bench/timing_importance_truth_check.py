"""Slice 7: timing-window permutation-importance CI gate.

Closes another part of the gap flagged in
the validation audit row 3: S17's declared R=50
statistical gate for the timing scenario only ever ran manually
(`bench.tvc_perm_sim run()`); the only CI-gated version
(`tests/test_tvc_sim_truth.py`'s `slow`-marked pilot) applies a Holm
correction across only the 2 inside windows and **never checks the
outside-window near-zero bound against a fitted forest at all** — only the
oracle's own exact-zero fact is checked there, a different claim.

Reuses `bench.tvc_perm_sim`'s existing, already-truth-checked timing
scenario (`timing_data`/`timing_replicate`/`timing_oracle`/
`TIMING_INSIDE`/`TIMING_OUTSIDE`) and its statistics helpers
(`holm_reject`/`one_sided_t`/`upper_bound`) unchanged — no new DGP, no new
statistical machinery. Unlike Slice 6 (trend), this gate ports the design's
**real, unmodified rule** — both halves of it:

- (a) Holm–Bonferroni-corrected one-sided t-tests across all 6 windows
  reject in at least one of the 2 inside windows (the signal is detected).
- (b) each of the 4 outside windows' one-sided 95% upper confidence bound
  (Bonferroni-corrected across those 4) is <= `delta = 0.1 * oracle-inside-sum`
  (the signal doesn't leak outside its true window).

Only the replication *scale* is reduced for CI speed (`n_train=n_eval=300,
n_estimators=60`, vs. the design's `1000/1000/200`) — verified empirically
(not assumed) to still support the real rule: an out-of-band pilot at this
scale (seeds 10000-10009, then a second independent check at seeds
20010-20024 and at the gate's own seeds 0-14) passed both (a) and (b) in
every check, with real margin (outside upper bounds ~0.008-0.026 vs.
delta ~0.037).

**Scope: timing scenario only.** The competing-risks oracle scenario (S17's
third), and all of S19's landmark-importance scenarios, remain manual-only —
same gap, explicit future work.
"""

import numpy as np

from bench.tvc_perm_sim import (
    TIMING_INSIDE,
    TIMING_OUTSIDE,
    holm_reject,
    one_sided_t,
    timing_oracle,
    timing_replicate,
    upper_bound,
)

N_TRAIN = 300
N_EVAL = 300
N_ESTIMATORS = 60
N_REPS = 15
OUTSIDE_ALPHA = 0.05 / len(TIMING_OUTSIDE)


def run(n_reps=N_REPS, seed0=0, n_train=N_TRAIN, n_eval=N_EVAL, n_estimators=N_ESTIMATORS):
    """``(n_reps, 6)`` array of per-window importance, seeds ``seed0..seed0+n_reps-1``."""
    rows = [
        timing_replicate(seed0 + r, n_train=n_train, n_eval=n_eval, n_estimators=n_estimators)
        for r in range(n_reps)
    ]
    return np.array([[r[f"w{m}"] for m in range(6)] for r in rows])


def check(windows):
    """``(pass_inside, pass_outside, outside_upper_bounds, delta)`` for a ``(n_reps, 6)`` window array.

    ``pass_inside``: Holm-corrected one-sided tests across all 6 windows reject
    in >= 1 of the 2 inside windows. ``pass_outside``: every outside window's
    Bonferroni-corrected upper 95% bound is <= ``delta``.
    """
    ow = timing_oracle()
    delta = 0.1 * ow[TIMING_INSIDE].sum()
    cols = [windows[:, m] for m in range(6)]
    reject = holm_reject([one_sided_t(c) for c in cols])
    pass_inside = bool(np.any([reject[m] for m in TIMING_INSIDE]))
    outside_ub = [upper_bound(cols[m], OUTSIDE_ALPHA) for m in TIMING_OUTSIDE]
    pass_outside = all(u <= delta for u in outside_ub)
    return pass_inside, pass_outside, outside_ub, delta
