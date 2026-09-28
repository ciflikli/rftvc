# Research questions: S19 landmark-importance CI gates (findings.md row 4)

Closes the last open item on the manual-gate punchlist ([[rftvc-roadmap]] "Still open": *"S19's three
landmark-importance scenarios (same manual-only gap, not yet closed)"*). Same recipe as Slices 5-8
(`docs/plans/simulation-validation-findings.md` rows 2b/3b/3c/3d): reuse the existing DGP/oracle in
`bench/tvc_landmark_sim.py` unchanged, verify — don't assume — the design's rule transfers to a
CI-fast reduced scale, land in the default (non-`slow`) pytest tier, one `codex:rescue` diff review
per slice. Three scenarios, all already implemented and manually run at R=50
(`docs/plans/s19-plan.md` §"Simulations", "S19 done" results block):

1. **History/Markov** (`level_history_data`/`replicate`/`oracle`, §7.2): history-given-level
   importance vs. 0.25× oracle(history); Markov control vs. 0.05× oracle(level).
2. **Copies bootstrap-SE** (`copies_replicate`, §7.5b): `se(step=0.5)/se(step=4.0) >= 0.5`.
3. **Censoring PE-vs-Brier** (`censoring_replicate`, §7.5b): PE and Brier importance rankings agree
   on which of `z1`/`z2` is larger.

## Complication specific to this row (unlike rows 2b/3b/3c/3d)

The history scenario's real R=50 rule **already failed** at full scale and was accepted as a
documented deviation, not fixed (`s19-plan.md` lines 121, 127-130): mean history-given-level =
0.0251 vs. required >= 0.0263 (94% of the threshold), confirmed not sampling noise at R=100. The
Markov control passed comfortably (-0.0099 vs. required <= 0.0089). Any new default-tier gate for
this scenario must not silently threshold-shop a number that makes the known-failing check pass —
the S19 plan explicitly flagged that risk ("tuning the fixture to the result") and declined to do
it even when investigating the shortfall.

## Questions

1. What CI-tier precedent exists for a design rule that is *known to fail* at declared scale — does
   any of `test_tvc_sim_truth.py`, `test_trend_importance_truth.py`,
   `test_timing_importance_truth.py`, or `test_cr_importance_truth.py` (Slices 6-8) encode a
   loosened-but-honest version of a failing rule, or do they all gate only on rules that already
   pass at full scale? What does `test_landmark_sim_truth.py`'s existing `slow`-tier
   `test_pilot_pass_rules_point_the_right_way` (lines 84-96) currently assert for the history
   scenario specifically, and is that assertion itself honest about the known shortfall or does it
   only check `hist.mean() > 0` (a much weaker claim than the 0.25× rule)?

2. For the copies scenario, does the `>= 0.5` ratio rule reproduce with real margin at a reduced
   scale (smaller `n_train`/`n_eval`/`n_estimators` than the design's 500/500/100, and fewer than
   10 reps per `step` value) across at least two independent out-of-band seed batches, following
   the same verification discipline as Slices 7-8? What reduced scale and rep count keeps total
   runtime in the same ballpark as `test_timing_importance_truth.py`/`test_cr_importance_truth.py`
   (2.5-3.8s)?

3. For the censoring scenario, does the rank-concordance rule (PE and Brier both rank `z1 > z2`)
   reproduce with real margin at a reduced scale across independent out-of-band seed batches? Is a
   raw mean comparison over R replications sufficient, or does the existing statistical toolkit
   (`holm_reject`/`one_sided_t`/`upper_bound` from `bench/tvc_perm_sim.py`, already reused by
   Slices 7-8) give a stronger, still-honest formulation (e.g., a one-sided test that each scoring
   family's `z1` importance exceeds its `z2` importance) worth porting here instead of a bare mean
   comparison?

4. What replication count, seed ranges, and runtime should the new gate(s) target to stay
   consistent with the row-4-closing slices' own convention (R=15, seeds 0-14 for the primary run,
   two independent out-of-band pilot batches such as 10000-10009 and 20010-20024)? Confirm via
   actual local runs of `bench/tvc_landmark_sim.py`'s existing functions at candidate scales, not
   by assumption.

5. Should this be one slice (one PR covering all three scenarios, since they share one bench file
   and one existing test file) or three, matching the one-scenario-per-PR pattern of Slices 6/7/8?
   What does `docs/plans/plan.md`'s Slice 5-8 section say about why those were split, and does that
   reasoning apply here given all three scenarios already live in a single `bench/tvc_landmark_sim.py`
   / `tests/test_landmark_sim_truth.py` pair (unlike the trend/timing/CR scenarios, which lived in
   separate files per scenario)?

## Codebase references

- `bench/tvc_landmark_sim.py` — all three scenarios' DGP, oracle, and `run()`/`pilot()` (manual gate).
- `tests/test_landmark_sim_truth.py` — existing fast smoke tests + one `slow`-tier loosened pilot.
- `docs/plans/s19-plan.md` lines 99-130 — predeclared rules, R=50 results, the history deviation.
- `bench/timing_importance_truth_check.py` + `tests/test_timing_importance_truth.py` — template for
  porting a design rule unmodified (Slice 7 pattern).
- `bench/trend_importance_truth_check.py` — template for recalibrating a fresh threshold at reduced
  scale instead (Slice 6 pattern) — relevant if question 1's answer says the history scenario needs
  this treatment rather than a direct port.
- `docs/plans/simulation-validation-findings.md` row 4 and its cross-references in rows 3b/3c/3d —
  current documented state of this gap.
- `docs/plans/plan.md` — Slice 5-8 section, for the one-scenario-per-PR precedent and reasoning.
