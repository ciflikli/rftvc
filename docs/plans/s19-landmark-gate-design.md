# Design: S19 landmark-importance CI gates (findings.md row 4)

## Executive summary

Closes the last open item on the manual-gate punchlist: S19's three landmark-importance scenarios
(`bench/tvc_landmark_sim.py`) get real default-tier `pytest` gates, following the Slice 5-8 recipe
(reuse the DGP unchanged, verify the rule at reduced scale via out-of-band pilots, land in the
default tier). Two of the three scenarios (copies, censoring) port cleanly, same as Slices 7-8.
The third (history/Markov) cannot be gated the way the others were: the design's real 0.25×-of-oracle
history rule already fails at full scale and is worse-than-coin-flip at any reduced scale tested —
research ruled out recalibrating it, matching the "don't threshold-shop a known-failing rule"
concern flagged going in. The recommended gate instead asserts the two claims that *are* robust for
that scenario: the Markov control's real (unmodified) bound, and a new, weaker-but-honest
significance claim for history ("history-given-level is significantly positive across R=30 reps",
not "clears 0.25× of oracle"). Recommend **one PR** covering all three scenarios, matching the
existing single-file/single-test-file structure (Slice 5's shape, not Slices 6-8's).

## Open question 1: how to gate the history/Markov scenario

### Approach A: gate the Markov control only; leave history as an uncounted smoke assertion

Add a default-tier test for `mark.mean() <= 0.05 * oracle(markov)` (real rule, passes with large
margin at every scale tested) and leave history at its current weak `hist.mean() > 0` smoke check
(already in the `slow`-tier test, could be duplicated as a fast smoke test instead).

- **Pro:** never asserts anything about history beyond what's already known to be true; simplest,
  least likely to become a future false positive/negative.
- **Con:** doesn't really close "S19's landmark-importance scenarios" for scenario 1 — the row-4
  gap's whole point (a real statistical claim running in CI) stays open for half the scenario.
  Row 4 would need to stay annotated as "history: no real claim gated" indefinitely.

### Approach B: recalibrate a new magnitude threshold for history, like Slice 6 did for trend

Fit a new ratio-of-oracle threshold at reduced scale from an out-of-band pilot, the way Slice 6
recalibrated `Z1_RATIO`/`Z2_ABS` for the trend scenario.

- **Con — ruled out by research:** Slice 6's recalibration works because the trend effect is
  strong and unambiguous; picking a new number only changes which reduced-scale magnitude is
  required. History's effect is not scale-limited — it's a real, narrow, *already failing* effect
  at full R=50 scale (94% of the design's own bound, confirmed not noise at R=100). Any threshold
  low enough to "pass" at reduced scale would need to sit below where the real R=50 result already
  landed, which is indistinguishable from silently overriding a user-approved deviation and giving
  it a passing checkmark it doesn't have at the scale that matters. This is exactly the
  "tuning-the-fixture-to-the-result" pattern the S19 plan explicitly declined to do (line 130) and
  that the questions file flagged as a risk to avoid. Not recommended.

### Approach C (recommended): keep the Markov bound (as in A), and replace history's magnitude
claim with a weaker but statistically honest significance claim

Assert two separate, real things for scenario 1, at different rep counts:
- **Markov control:** the design's real `mark.mean() <= 0.05 * oracle(markov-level)` bound, R=15
  (matches Slices 7-8's convention) — passes comfortably in every batch tested (margin 3-7×).
- **History:** *not* the 0.25×-of-oracle magnitude claim (research shows this doesn't hold
  reliably even at R=15, reduced scale: 43%-101% of the bound depending on seed batch). Instead, a
  one-sided test that `history-given-level`'s population mean is significantly positive
  (`H0: mean <= 0` vs `H1: mean > 0`), at **R=30** (not R=15 — research found R=15 gives p=0.138 in
  the primary seed batch, not significant at alpha=0.05; R=30 is robust: p = 0.0004, 1.5e-6,
  7.3e-5 across three independent batches). This is honest about what's actually true: history
  *does* matter (a real, signed, statistically detectable effect, confirmed by 3 independent
  batches) even though the model's proxy feature doesn't recover the design's originally-declared
  *magnitude* of that effect (the deviation already on record in `s19-plan.md`).

```python
# bench/landmark_importance_truth_check.py (illustrative — see plan.md for exact signatures)
N_REPS_HISTORY = 30   # needs more reps than the others: the real effect is narrow
N_REPS_OTHER = 15     # matches Slices 6-8's convention

def check_level_history(hist, mark, oracle_markov):
    """``hist``/``mark``: (R,) arrays from ``replicate``. Returns
    (p_history_positive, pass_markov, markov_bound)."""
    p_history = one_sided_t(hist)  # H0: mean <= 0 (from bench.tvc_perm_sim, reused unchanged)
    bound = 0.05 * oracle_markov
    pass_markov = mark.mean() <= bound
    return p_history, pass_markov, bound
```

- **Pro:** every assertion in the gate is something research actually verified holds robustly
  across independent seed batches; nothing is a repackaged version of a claim already known to be
  marginal-or-failing.
- **Con:** this is *not* the design's originally-declared 0.25× rule — it's a strictly weaker claim
  invented for this gate. Must be documented clearly as such (row 4's findings-doc entry needs to
  say plainly "history's magnitude claim is not gated; only sign+significance is" — the same
  discipline Slice 6 used to flag "recalibrated, not the original numbers").

**Recommendation: Approach C.** It's the only one of the three that (a) actually adds a new,
non-trivial, CI-enforced statistical claim for the history scenario (unlike A) and (b) doesn't
misrepresent a known-marginal effect as passing a bar it doesn't reliably clear (unlike B).

## Open question 2: one PR or three

### Approach A: one PR for all three scenarios

New `bench/landmark_importance_truth_check.py` (or extend `bench/tvc_landmark_sim.py` in place —
see below) with `check_*` functions per scenario, one new/extended test file, one findings-doc row
update.

### Approach B: three PRs, one per scenario (mirroring Slices 6-8)

Split copies, censoring, and history/Markov into separate branches/PRs.

**Recommendation: Approach A (one PR).** Research (Q5) found the Slices 6-8 split tracked a real
code-structure boundary (three separate DGP files) that doesn't exist here — all three scenarios
already share one bench file and one existing test file, and were delivered/reviewed as a single
unit in S19 itself. Splitting into three PRs here would be splitting for its own sake, not along an
existing seam; it also triples the Codex-review overhead for three small, mechanically similar
changes. This matches Slice 5's shape (one gate, one PR) rather than Slices 6-8's.

## Recommended implementation shape

- **New file `bench/landmark_importance_truth_check.py`**, reusing `bench.tvc_landmark_sim`'s
  `level_history_data`/`replicate`/`oracle`/`copies_replicate`/`censoring_data`/
  `censoring_replicate` and `bench.tvc_perm_sim`'s `one_sided_t` unchanged. Three `check_*`
  functions (one per scenario), each taking pre-computed replicate arrays and returning a
  pass/fail tuple plus the numbers needed for a clear assertion message — same shape as
  `bench/timing_importance_truth_check.py`'s `check()`.
- **Extend `tests/test_landmark_sim_truth.py`** (not a new test file — it already imports
  everything needed and already has the fast smoke tests) with three new default-tier test
  functions: `test_markov_control_and_history_significance`, `test_copies_se_does_not_shrink`,
  `test_censoring_pe_and_brier_rankings_agree`. Leave the existing `slow`-tier
  `test_pilot_pass_rules_point_the_right_way` as-is (still a useful looser cross-check at yet
  another scale) or trim it if it becomes fully redundant — a plan-stage decision.
- **Reduced scales** (from research, each verified across >= 2 independent out-of-band batches):
  - History/Markov: `n_train=n_eval=200, n_estimators=40`, R=15 for Markov, **R=30 for history**.
  - Copies: `n_train=n_eval=100, n_estimators=20`, R=15 for both `step` arms.
  - Censoring: `n_train=n_eval=150, n_estimators=25`, R=15.
- **Findings-doc update:** `docs/plans/simulation-validation-findings.md` row 4 gets a new row 4b
  (or is edited in place) documenting exactly what's now closed — explicitly noting the history
  scenario's gate asserts a *weaker* claim (sign + significance, R=30) than the design's original
  magnitude rule (0.25× oracle), which remains a documented, user-approved deviation, not something
  this gate silently resolves. Also re-check every other row's cross-reference text that currently
  says "all of S19's landmark-importance scenarios remain manual-only" (rows 3b/3c/3d and the
  closing summary), per the standing lesson in [[rftvc-dev-workflow]] about re-grepping the whole
  doc, not just the diff, after each edit.

## Acceptance criteria carried into the plan stage

- All three new default-tier tests run in the plain `pytest` invocation (no `-m` flag), verified
  by actually running it, not assumed.
- Every threshold/rep-count in the gate is one that was locally verified (this design doc's tables)
  across at least 2 independent out-of-band seed batches distinct from the gate's own seeds.
- The history scenario's gate is explicit in its own docstring/comments that it asserts a weaker
  claim than the design's declared rule, and why (mirroring Slice 6's "not the original numbers"
  framing).
- `simulation-validation-findings.md` row 4 (and every stale cross-reference to it) is updated in
  the same PR, not left for a follow-up.
- One `codex:rescue` diff review before merge.
