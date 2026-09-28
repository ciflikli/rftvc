# Research: S19 landmark-importance CI gates (findings.md row 4)

Answers to `docs/plans/s19-landmark-gate-questions.md`. Facts and local pilot-run results only —
no recommended approach here (that's the design stage).

## Q1: precedent for a design rule known to fail at full scale

No existing slice (`test_tvc_sim_truth.py`, `test_trend_importance_truth.py`,
`test_timing_importance_truth.py`, `test_cr_importance_truth.py`) gates on a rule that is known to
fail at its full/declared scale:

- **Slice 6 (trend)** recalibrates a *new* threshold (`Z1_RATIO`/`Z2_ABS` in
  `bench/trend_importance_truth_check.py`) from an out-of-band pilot — but this is a genuinely new,
  honest threshold for a *different* (reduced) scale, not a loosening of a rule that fails on its
  own terms. The trend effect itself is strong and unambiguous at both scales.
- **Slices 7-8 (timing, CR)** port the design's real, unmodified rule and verify via out-of-band
  pilots that it holds with real margin at reduced scale (outside upper bounds ~0.008-0.026 vs.
  delta ~0.037 for timing; `ub_s2` ~0.017-0.026 vs. `delta1` ~0.061-0.062 for CR).
- **`test_tvc_sim_truth.py`'s own `slow`-tier pilot** (10 reps) only "loosely checks the declared
  pass rules point the right way" per its own module docstring — never claims to validate the real
  rule, and is `slow`-marked (i.e., does not run in default CI per the row-2/Slice-5 lesson).

**`test_landmark_sim_truth.py`'s existing `slow`-tier `test_pilot_pass_rules_point_the_right_way`
(lines 84-96)** is the closest analogue and confirms the concern in the questions file: for the
history scenario it asserts only `hist.mean() > 0` (line 91) — a much weaker directional claim than
the real 0.25×-of-oracle rule. It never encodes the real rule at all, loosened or otherwise, and
runs only under `-m slow` (so it never executes in default CI even in its weak form). The Markov
control in the same test *does* encode a loosened form of the real rule (line 92:
`mark.mean() <= 0.05 * ol + 3 * mc_se`, i.e., the real bound plus 3 MC-SE slack) — because the
Markov control passes comfortably at full scale, unlike history.

**Conclusion of fact-finding:** there is no precedent anywhere in this codebase for shipping a
default-tier gate that asserts a rule known to be marginal/failing. Every existing default-tier
gate (Slices 5-8) only exists because the underlying effect is real and passes with margin — reduced
scale changes the *numbers*, not whether the effect clears its bar.

## Q2: copies scenario at reduced scale

Ran `copies_replicate` locally (`.venv`, `PATH` prefixed with `~/.cargo/bin`, per
[[rftvc-dev-workflow]]) at three candidate scales, R=15, across 3 independent seed batches (the
gate's own seeds 0-14, plus out-of-band 10000-10014 and 20010-20024 — the same convention as Slices
6-8):

| n_train=n_eval | n_estimators | seeds 0-14 ratio | seeds 10000-10014 | seeds 20010-20024 | runtime (both `step`s, 15 reps each) |
|---|---|---|---|---|---|
| 200 | 40 | 0.586 | 0.729 | 0.704 | ~15s |
| 150 | 25 | 0.558 | 0.573 | (not run) | ~7.7s |
| 100 | 20 | 0.598 | 0.608 | 0.671 | ~4.6s |

The `>= 0.5` ratio rule (design's real, unmodified rule) **holds with real margin at all three
scales tested, across all batches run** — smaller than the full design scale (500/500/100) but
never close to the 0.5 boundary (closest observed: 0.558). The smallest scale tested (100/100/20)
gives the fastest runtime (~4.6s per batch) with margin comparable to the larger scales (0.60-0.67
vs. 0.56-0.73) — no evidence that shrinking further from 200→100 degrades the effect.

## Q3: censoring scenario at reduced scale, and rank-concordance vs. a significance test

Two formulations tested at n_train=n_eval=300, n_estimators=50, R=15 (3 seed batches):

**Raw mean comparison (the design's actual rule — `pe_z1.mean() > pe_z2.mean()`, same for
Brier):** held in every batch, both scoring families, all 3 out-of-band batches.

**One-sided t-test (`bench.tvc_perm_sim.one_sided_t`, reused unchanged, on the per-replicate
`z1 - z2` difference) as a stronger alternative:** PE path is robustly significant in all 3 batches
(p = 4.3e-6, 4.1e-4, 3.1e-4). **Brier path is not reliably significant at this scale** — p = 0.109
in the primary seeds-0-14 batch (the two other batches were significant: p = 7.6e-4, 3.7e-3). A
gate built on the t-test formulation would be scale- and seed-sensitive for the Brier half in a way
the raw mean-comparison rule is not.

Re-tested the **raw mean-comparison rule** (the actual scenario's pass condition, `pass4` in
`bench/tvc_landmark_sim.py`'s `run()`) at smaller scales:

| n_train=n_eval | n_estimators | seeds 0-14 | seeds 10000-10014 | seeds 20010-20024 | runtime (15 reps) |
|---|---|---|---|---|---|
| 300 | 50 | agree | agree | agree | ~0.5s |
| 150 | 25 | agree | agree | agree | ~0.4s |
| 100 | 20 | **fails**: `UndefinedMetricError: no events in (0, windows[-1]] to score` | — | — | — |

At n=100 the heavy-censoring generator (`censoring_data`) sometimes produces an evaluation fold
with no scorable events in the PE window, raising inside `permutation_importance`'s PE path — this
scale is too small for this scenario specifically (unlike the copies scenario, which tolerated
n=100 fine). n_train=n_eval=150, n_estimators=25 works cleanly and is already very cheap (~0.4s per
15-rep batch, both scoring families).

## Q4: replication count, seeds, runtime summary

Consistent with Slices 6-8's own convention (R=15, seeds 0-14 primary, out-of-band pilots at
10000-10009/10014 and 20010-20024 or nearby):

- **Copies:** R=15 holds with real margin down to n_train=n_eval=100, n_estimators=20 (~4.6s for
  both `step` arms combined per batch — the single most expensive of the three scenarios to gate,
  since `copies_replicate` runs `n_bootstrap=50` internally).
- **Censoring:** R=15 holds at n_train=n_eval=150, n_estimators=25 (~0.4s per batch) — needs at
  least this much data; n=100 breaks the generator's event-availability assumption.
- **History/Markov:** see below — the real rule does not hold reliably at any scale tested.

## Q5: one slice or three?

`docs/plans/plan.md`'s Slice 6/7/8 section states the trend/timing/CR split explicitly:
"*Scoped to the trend scenario only, matching Slice 5's narrow-slice precedent*" (Slice 6), each
with its own new `bench/*_truth_check.py` file. That split reflects the underlying code structure:
S17's three oracle scenarios (`trend_data`/`timing_data`/`cr_data`) already lived in **separate
functions returning separate truth_check modules and separate test files**, so splitting by PR
tracked the natural code boundary and let one deviation-prone scenario (e.g., a Codex-caught bug)
be reviewed and merged independently of the others.

S19's three landmark scenarios are structurally different: all three (`level_history_data`/
`replicate`/`oracle`, `copies_replicate`, `censoring_data`/`censoring_replicate`) already live in
**one existing file, `bench/tvc_landmark_sim.py`**, with **one existing test file**,
`tests/test_landmark_sim_truth.py`, and were already delivered, documented, and reviewed as one
unit in S19 itself ("S19 done" is a single results block covering all three, `s19-plan.md` lines
115-125). This matches **Slice 5's** shape (`CompetingRisksForestTV` gate: one new
`bench/cr_forest_truth_check.py`, one new `tests/test_cr_forest_truth.py`, one PR) more closely than
Slices 6-8's shape. No fact here dictates one PR must equal one bug-risk boundary; this is a
scoping call for the design stage, not something the research turned up a hard constraint against
either way.

## Findings that affect design directly

1. **The history scenario cannot be gated the way Slices 6-8 gate their scenarios.** At full scale
   (R=50) it already fails (94% of threshold, confirmed not noise at R=100, per `s19-plan.md`). At
   reduced scale (R=15, n=200/200/40) it is worse than a coin flip against its own bar: seeds 0-14
   give 0.0108 vs. a required >= 0.0250 (43% of threshold); seeds 10000-10014 give 0.0255 vs. the
   same 0.0250 (barely over, by less than 1 MC-SE). There is no reduced scale at which this
   specific 0.25×-of-oracle rule can be asserted as a reliable pass — the effect itself is
   real-but-narrow (documented deviation), not a rule that merely needs recalibrating.
2. **The Markov control passes very comfortably at every scale tested** (mark.mean() ≈ -0.003 to
   -0.004, vs. a required <= 0.0086-0.0087 — several multiples of margin), matching its full-scale
   behavior (-0.0099 vs. <= 0.0089). This half of the scenario behaves exactly like Slices 6-8's
   passing rules.
3. **Copies and censoring both behave like Slices 6-8's scenarios**: the design's real, unmodified
   rule (not a recalibrated threshold) holds with real margin at a reduced scale, verified across
   3 independent seed batches for copies and 3 for censoring (raw mean-comparison formulation).
4. **A stricter significance-test formulation for censoring is available but not more honest here**
   — it would introduce a seed-sensitive failure mode (Brier path, p=0.109 in the primary batch)
   that the actual design rule (mean comparison) does not have.
5. **Environment note:** `export PATH="$HOME/.cargo/bin:$PATH"` + `.venv` activation is required to
   import `rftvc` locally, consistent with [[rftvc-dev-workflow]].
