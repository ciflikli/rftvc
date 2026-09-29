# Design: n-sweep beyond Slice 1's two points (findings.md row 5)

## Executive summary

User-confirmed approach (2026-09-29): move the existing `slow`-tier 200-vs-5000 endpoint claim to
the default CI tier, fixing the R=10 fragility research found (bumps to R=20), and add a third
point (n=1000) as a **reported, not gated** diagnostic — matching the S19 history-scenario
precedent of asserting only what's robust rather than a full 3-point "decreases everywhere" claim,
which research found isn't honestly achievable at any scale. This replaces the existing slow-tier
test in place (same scale/claim, just more reps and one extra diagnostic point), rather than adding
a second, redundant test — unlike Slice 9's landmark gates, the old and new tests here would assert
the literal same underlying claim at the literal same scale, just with a different R.

## What changes

- `bench/pe_score_convergence_sim.py`: **no changes** — `replicate`/`run` already accept arbitrary
  `n_values` and already return per-n gaps for however many points are passed.
- `tests/test_pe_score_convergence_truth.py`: `test_pe_score_mean_gap_shrinks_with_n` is replaced
  by `test_pe_score_gap_shrinks_from_n200_to_n5000` — same scale (`n_estimators=200`), same gated
  claim (one-sided 95% lower bound on the paired 200-minus-5000 gap difference > 0), but:
  - R=10 → **R=20** (research: R=10's lower bound is negative — i.e., the claim fails outright — at
    an out-of-band seed batch (20010-20029); R=20 restores a positive bound, margin 0.014-0.070
    across the 3 batches tested).
  - `n_values=(200, 1000, 5000)` instead of `(200, 5000)` — the middle point's mean is reported in
    the assertion message (and could be written to a CSV under `docs/bench/`, matching other
    slices' convention) but is **not** part of any pass/fail condition: research found the
    mid-to-large step is never significant at any scale tested (the gap plateaus after the first
    jump — real, not a rep-count problem), so gating it would either be a no-op (a trivially loose
    bound) or dishonest (claiming significance that isn't there).
  - Marker: **not `slow`** — moved to the default tier, per this session's user decision.

## Cost accepted

~30s for this one test (R=20 at `n_estimators=200`, fitting 3 forests — sizes 200/1000/5000 — per
replication), vs. 0.4-9s for every gate closed in Slices 5-9. This is a one-time, explicit trade-off
the user accepted for closing the last real statistical-validity blind spot on the punchlist, not an
oversight.

## Test shape

```python
# tests/test_pe_score_convergence_truth.py
N_VALUES = (200, 1000, 5000)
N_ESTIMATORS = 200
N_REPS = 20

def test_pe_score_gap_shrinks_from_n200_to_n5000():
    """Real, unmodified 200-vs-5000 endpoint claim from Slice 1 (docs/plans/plan.md), at R=20
    instead of the original R=10 -- research found R=10's lower bound goes negative at an
    out-of-band seed batch (20010-20029), i.e. the original rep count isn't robust; R=20 restores
    a real (if sometimes thin) positive margin across 3 independent batches.

    The middle point (n=1000) is reported, not gated: the mid-to-large step's decrease is never
    statistically significant at any scale tested (docs/plans/n-sweep-gate-research.md) -- a real
    plateau in the oracle-to-model gap, not a rep-count problem -- so asserting it would either be
    a trivially loose no-op bound or an overclaim.
    """
    res, diff, lower = run(n_values=N_VALUES, n_estimators=N_ESTIMATORS, n_reps=N_REPS)
    means = res.mean(axis=0)
    assert lower > 0, f"200-vs-5000 gap did not shrink with a significant margin: lower={lower}"
    assert np.all(np.isfinite(means))  # n=1000's mean is reported here, not asserted on
```

Note: `run()`'s existing `diff`/`lower` are already computed from `res[:, 0] - res[:, -1]` (first
vs. last entry of `n_values`), so passing a 3-tuple automatically keeps the gated claim on the
outer pair without any change to `bench/pe_score_convergence_sim.py`.

## Docs updates

- `docs/plans/simulation-validation-findings.md` row 5: update "CI tier" from `slow` to `not slow`
  (with the new ~30s runtime noted), and its blind-spot text from "only two n values checked" to
  "three n values, but the sweep-shape claim (monotonic decrease at every step) is only gated at
  the outer pair — the middle point is diagnostic only, since research found the mid-to-large step
  never reaches significance at any tested scale."
- Remove or update the closing "No n-sweep / asymptotic study" bullet to reflect partial closure
  (an outer-pair claim now runs in the default tier with a 3rd reported point, not a full
  monotonic-everywhere sweep).

## Acceptance criteria

- New test runs in the plain `pytest` invocation (no `-m` flag), verified by actually running it.
- R=20's robustness verified (already done in research) across >= 2 out-of-band seed batches
  distinct from the gate's own (0-19).
- Findings doc accurately scopes what's closed (outer-pair claim, now default-tier, 3rd point
  reported) vs. what remains inherently unclaimed (a true multi-step monotonic sweep — not a gap to
  close later, since research established the underlying plateau is real, not a scale artifact).
- One `codex:rescue` diff review before merge.
