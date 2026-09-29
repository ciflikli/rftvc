# Implementation Plan: Simulation-Based Statistical Validation + Conformal Prediction Spike

Stage 4 of CRISPI. Input: `docs/plans/design.md` (Approach 2 for Part A, Approach 3 for Part B — both user-confirmed 2026-09-28), `docs/plans/research.md`. **Revised 2026-09-28 after `codex:rescue` plan review** — see "Codex review findings applied" at the end of each slice.

## Status
- [x] Slice 1: PE-score-oracle-gap convergence check (PR #52)
- [x] Slice 2: lifelines cross-check against known truth (genuinely static DGP) (PR #53)
- [x] Slice 3: Validation inventory doc (evidence vs. claims) (PR #54)
- [x] Slice 4: Conformal-coverage spike (empirical-only, no theoretical guarantee claimed) — inconclusive, see docs/plans/conformal-prediction-investigation.md
- [x] Slice 5: CompetingRisksForestTV fitted-model-vs-truth CI gate (closes the row-2 gap from Slice 3's inventory) (PR #57)
- [x] Slice 6: trend-scenario permutation-importance CI gate (closes part of the row-3 gap — trend only, not timing/CR/landmark) (PR #58)
- [x] Slice 7: timing-window permutation-importance CI gate (closes another part of the row-3 gap — ports the design's real unmodified rule, not a recalibrated threshold) (PR #59)
- [x] Slice 8: competing-risks permutation-importance CI gate (completes S17's row-3 coverage — trend/timing/CR all closed)
- [x] Slice 9: S19 landmark-importance CI gates (closes row 4 — history/Markov, copies bootstrap-SE, censoring PE-vs-Brier) (PR #61)
- [x] Slice 10: n-sweep beyond Slice 1's two points (closes row 5's blind spot) (PR #64)
- [ ] Slice 11: TVC vs. `CoxTimeVaryingFitter` on a known-truth DGP (closes the TVC half of the external-tool-parity gap)  <-- CURRENT

## Slice 7: timing-window permutation-importance CI gate

**Why:** continuing Slice 6's pattern down the row-3 gap list. Unlike trend, the timing scenario's real R=50 design rule (Holm-reject inside windows + Bonferroni upper-bound outside windows) was verified, via an out-of-band pilot, to transfer directly to the reduced scale — so this slice ports the *real* rule unmodified rather than recalibrating a new threshold like Slice 6 had to.

**Files:**
- `bench/timing_importance_truth_check.py` (new) — reuses `bench.tvc_perm_sim`'s timing scenario and its `holm_reject`/`one_sided_t`/`upper_bound` stats helpers unchanged, at the same reduced scale as Slice 6 (`n_train=n_eval=300, n_estimators=60`). `run(n_reps) -> (n_reps, 6)` window importances; `check(windows) -> (pass_inside, pass_outside, outside_ub, delta)`.
- `tests/test_timing_importance_truth.py` (new) — fast smoke test, plus the gate itself: both halves of the design's real rule, R=15, not `slow`-marked.

**Explicit scope note:** timing scenario only. Competing-risks oracle scenario (S17's third) and all of S19's landmark-importance scenarios remain manual-only.

**Docs updated:** `docs/plans/simulation-validation-findings.md` — new row 3c, row 3/"does not establish" text updated.

**Acceptance criteria:**
- Gate runs in the default merge-gate suite.
- Rule verified (not assumed) to hold at reduced scale via two independent out-of-band pilots before locking in as a CI gate.
- Findings doc distinguishes 3c's "real rule verified at smaller scale" from 3b's "new threshold calibrated at smaller scale."

## Slice 6: trend-scenario permutation-importance CI gate

**Why:** user follow-up after Slice 5, prioritizing "manual-only R=50 statistical gates never run in CI" as the highest-value remaining gap (S17/S19 importance measures). Scoped to the trend scenario only, matching Slice 5's narrow-slice precedent.

**Files:**
- `bench/trend_importance_truth_check.py` (new) — reuses `bench.tvc_perm_sim`'s trend scenario (`trend_data`/`trend_replicate`/`trend_oracle`) unchanged, at a reduced scale (`n_train=n_eval=300, n_estimators=60` vs. the design's `1000/1000/200`) for CI speed. `run(n_reps) -> (n_reps, 2)` of `(imp_z1, imp_z2)`.
- `tests/test_trend_importance_truth.py` (new) — fast smoke test, plus a gate with thresholds freshly calibrated for the reduced scale via an out-of-band pilot (seeds 10000-10009) — **not** the original design's 50%/10%-of-oracle numbers, which were calibrated for the full 1000/1000/200 scale and aren't portable without their own recalibration. Deliberately not `slow`-marked (~5.5s), per Slice 5's lesson that a `slow`-marked gate never actually runs in CI.

**Explicit scope note:** trend scenario only. S17's timing-window and competing-risks oracle scenarios, and all of S19's landmark-importance scenarios, remain manual-only — explicit future work (Slice 7+), not closed here.

**Docs updated:** `docs/plans/simulation-validation-findings.md` — new row 3b, row 3 and "does not establish" text updated.

**Acceptance criteria:**
- Gate runs in the default merge-gate suite (no `-m` flag needed).
- Thresholds calibrated from a seed range distinct from the gate's own, stated before the gate was run.
- Findings doc scopes exactly what's closed (trend, reduced scale) vs. still open (timing, CR-perm, landmark, and the design's original full-scale numbers).

## Slice 5: CompetingRisksForestTV fitted-model-vs-truth CI gate

**Why:** user follow-up after reviewing Slice 3's inventory (`docs/plans/simulation-validation-findings.md`) — competing risks had no CI-gated check that a fitted `CompetingRisksForestTV` gets close to known truth; only the DGP's own closed-form truth was checked in CI, and the actual forest-vs-truth comparison (`bench/s14_cr_sim.py run()`) computes no pass/fail decision at all, ever.

**Files:**
- `bench/cr_forest_truth_check.py` (new) — reuses `bench.s14_cr_sim`'s scenario-A DGP and `training_rows`/`test_paths`/`true_cif`/`ise` helpers unchanged. Fits `CompetingRisksForestTV` at its actual defaults (`criterion="composite"`, `aggregate="cif"`, `split_cause=None` — one of the bake-off's own arms, not a new comparison), `replicate(seed) -> (2,)` per-cause ISE, `run(n_reps) -> (n_reps, 2)`.
- `tests/test_cr_forest_truth.py` (new) — fast smoke test (small n, finite-value check), `@pytest.mark.slow` gate with a predeclared absolute ISE threshold (0.10, calibrated from an out-of-band pilot, seeds 10000-10004, distinct from the gate's own seeds 0-9).

**Explicit scope note:** scenario A only, default hyperparameters only — does not re-run the S14 bake-off's arms sweep (still manual-only) or cover scenarios B/C. Closes the narrower "does the shipped default get close to truth at all" gap, not the full bake-off question.

**Docs updated:** `docs/plans/simulation-validation-findings.md` — new row 2b, and row 2/"does not establish" text updated to reflect the narrower (not full) closure.

**Acceptance criteria:**
- Fast smoke test passes in the default merge-gate suite.
- Slow gate's threshold stated before the sim was run, calibrated from a seed range distinct from the gate's own.
- Findings doc accurately scopes what's now covered vs. still open (full bake-off, scenarios B/C, external-tool cross-check for competing risks all remain open).

Each slice: own branch/PR, Codex plan review before code (per CLAUDE.md skill triggers / [[rftvc-dev-workflow]]), Codex diff review before merge.

---

## Slice 1: PE-score-oracle-gap convergence check

**Why:** research Q3 — no existing test checks whether `PEScore` approaches an oracle bound as n grows. This is the most direct "is the model statistically correct" check available.

**Correction from plan review:** `PEScore.total` is a rate mixed `(1-alpha)` with a training-set null (default `alpha=0.01`, `metrics.py:715`) — an *unmixed* true-hazard log-likelihood is **not** the ceiling of this specific quantity, and there is no guarantee the oracle score upper-bounds every finite realization. The check must compare against an **oracle computed under the same mixing/reduction convention**, on a **fixed, independent evaluation set** (not the training fold), and treat convergence as a **replication-level mean-gap trend**, not a per-sample bound.

**Files:**
- `bench/pe_score_convergence_sim.py` (new):
  - `oracle_pe_score(hazard_fn, windows, ids, times, alpha) -> PEScore` — computes the *same* `piecewise_exponential_score` mixing/reduction (same `alpha`, same null-mixing formula in `metrics.py:715-783`) but feeding the **true** per-window rate from `hazard_fn` in place of the model's predicted `cumhaz`. This is the oracle under matching convention, not a raw unmixed log-likelihood.
  - `run(n_values, n_estimators, n_eval, seed) -> pd.DataFrame` — for each `n` in `n_values`: draw a training set of size `n`, fit `SurvivalForestTV`; draw a **separate, fixed-size (`n_eval`), fixed-seed evaluation set reused identically across all `n`**; score the fitted model and the oracle on that same evaluation set with the **same `windows`/`event_windows` grid, fixed across all `n`** (compute the grid once from the eval set, not per training fit — closes the `zero_rate_share`-confounding risk research §3 flagged). Record `gap = oracle_score - model_score` per `n` per replication.
- `tests/test_pe_score_convergence_truth.py` (new):
  - `test_oracle_pe_score_matches_numerical_integration()` (fast, merge-gate tier) — `oracle_pe_score`'s per-window rate vs `scipy.integrate` numerical check, tolerance `1e-9`, pattern from `tests/test_cr_sim_truth.py`.
  - `test_pe_score_mean_gap_shrinks_with_n()` (`@pytest.mark.slow`) — predeclared *before* running: mean `gap` over R=10 replications at the largest `n` < mean `gap` at the smallest `n`, one-sided t-test on the paired-replication differences (matching `tests/sim.py:95-99`'s convention). State the rule in the test docstring before the sim code is written. **Superseded by Slice 10** (below): this R=10 rule was found not robust across seed ranges it was never tested against; the test now runs at R=20 in the default tier under a new name, `test_pe_score_gap_shrinks_from_n200_to_n5000`.

**Acceptance criteria:**
- Fast test passes in default merge-gate suite.
- Oracle and model are scored on the *same* held-out eval set and *same* window grid at every `n` — verified by an assertion in `run()` that windows are identical across `n` values, not just "the same code path."
- Slow test's pass rule is stated in the docstring before the sim is implemented, not fit to results afterward.

**Codex review findings applied:** oracle now uses matching alpha-mixing convention (not raw unmixed log-likelihood); scoring moved to a fixed independent eval set reused across `n`; windows/event_windows fixed across the sweep instead of re-derived per `n`; "ceiling" reframed as "oracle under matching convention," not an upper bound claim.

---

## Slice 2: lifelines cross-check against known truth (genuinely static DGP)

**Why:** research Q5 — `lifelines` is already a dev dependency used for real-data comparisons, never checked against a known-hazard DGP.

**Correction from plan review:** `tests/sim.py`'s DGP has `z` redrawn every interval (`tests/sim.py:31-33`, an AR/time-varying covariate) — restricting the *model* to use only `x0` does not make the DGP's own conditional survival truth static, because `z`'s path still drives the actual hazard. A genuinely static comparison needs a **new DGP with a covariate fixed at baseline for each subject**, not a subsetted TVC one. Also corrected: `KaplanMeierFitter` ignores covariates entirely (not a fair comparator against a covariate-driven truth); `CoxTimeVaryingFitter` *does* handle TVC (contrary to the original plan's rationale for avoiding it) — the right static-case comparator is `lifelines.CoxPHFitter`.

**Files:**
- `bench/lifelines_truth_check.py` (new):
  - A new closed-form hazard `hazard(x) = lambda0 * exp(beta * x)` (Weibull or exponential baseline, single **static** covariate `x` fixed per-subject at t=0 — no time-varying component), with `true_survival(t, x)` computed in closed form (reuse `tests/sim.py:64`'s `S(t) = exp(-Λ(t))` pattern, but for the static hazard).
  - Predeclare, in the module docstring, before any results: train/test split sizes, the evaluation time grid, `SurvivalForestTV`/`CoxPHFitter` hyperparameters, number of replications, and the pass rule.
  - Fit `SurvivalForestTV` and `lifelines.CoxPHFitter` on the same static-covariate data; compare each to `true_survival` via ISE (same metric convention as `tests/sim.py`). To avoid instability from dividing by a near-zero best-observed ISE (plan-review finding), use an **additive tolerance band** (`rftvc_ISE <= best_ISE + fixed_epsilon`, epsilon predeclared and justified relative to typical ISE magnitude at the chosen n) rather than a multiplicative ratio.
- `tests/test_lifelines_truth.py` (new, `@pytest.mark.slow`) — asserts the predeclared additive-tolerance rule over R replications with a fixed seed.

**Explicit scope note (module docstring):** single-event, genuinely static-covariate case only. Does not exercise TVC or competing risks — those stay covered by Slice 1 and the existing `bench/*_sim.py` suites. `CoxPHFitter`, not `CoxTimeVaryingFitter`, is the correct comparator here specifically *because* the DGP is static; a TVC cross-check against `CoxTimeVaryingFitter` is explicit future work, not this slice.

**Acceptance criteria:**
- DGP's covariate is fixed per-subject at simulation start — verified by an assertion/test that no covariate value changes across a subject's rows.
- Pass rule (additive tolerance, its epsilon, and why) stated in the module docstring before the comparison code.

**Codex review findings applied:** replaced the TVC-subsetted DGP with a genuinely static one; swapped `KaplanMeierFitter` (no covariates) for `CoxPHFitter` (correct static comparator); corrected the mischaracterization of `CoxTimeVaryingFitter`; replaced the ratio-to-best-ISE rule with an additive-tolerance rule to avoid near-zero-denominator instability.

---

## Slice 3: Validation inventory doc

**Why:** research §1/§7 — assemble what's already validated in one place, and now, per plan review, be explicit about what each check does and doesn't prove.

**Correction from plan review:** `tests/test_sim.py` checks `SurvivalForestTV` against a *fixed-covariate baseline*'s relative performance, not an independent truth-check of the generator itself (the original plan's Slice 3 draft overstated this). The inventory must state each check's **actual assertion**, not a general "validates X" claim, and note blind spots (manual-only bench runs that aren't gated in CI at all, scenarios never simulated).

**Files:**
- `docs/plans/simulation-validation-plan.md` (new) — structured like `rc-validation-plan.md`: scope framing, numbered Decisions, Tasks checklist tied to Slices 1-2.
- `docs/plans/simulation-validation-findings.md` (new) — one row per existing + new check, with columns: **what it actually asserts** (verbatim-close to the test's real assertion, not a paraphrase upward), DGP used, pass rule, fast/slow/manual-bench tier, and **known blind spots** (e.g. "manual-only, not CI-gated," "single scenario, not swept over censoring rates").

**Acceptance criteria:**
- Every existing `bench/*_sim.py` + `tests/test_*_sim_truth.py` pair listed with its actual assertion (verified against the test's real code, not summarized from memory) and at least one blind spot.
- Document does **not** claim to prove general statistical validity — it states, per the plan-review finding, that passing every listed check does not rule out bugs outside the tested scenarios, and lists what scenarios are *not* covered.

**Codex review findings applied:** corrected `test_sim.py`'s claimed assertion; replaced the original "reader can answer what would fail if the Rust core had a bug" acceptance criterion (too broad, per review) with an explicit per-check blind-spots column and an explicit non-completeness statement.

---

## Slice 4: Conformal-coverage spike (empirical investigation only, no shipped API, no theoretical guarantee claimed)

**Why:** design.md Part B Approach 3 (user-confirmed) — investigate, don't ship.

**Correction from plan review (substantial — original spike was underspecified on exactly the points that matter for a conformal method):**
1. **Target must be named explicitly.** This spike targets **P(event by horizon | X)**, the true conditional risk at a fixed horizon — not survival time itself and not the raw censored event indicator. State this in the module docstring.
2. **OOB is not an independent split-conformal calibration set**, and the plan must not claim a formal split-conformal coverage guarantee. `oob_cumhaz`'s internal call path (`inspection.py:120-133`) goes through `estimator._rebuild_design(X, y, ids, ...)`, which requires the **original fit-time data** (it fingerprint-checks against it) and returns OOB predictions **for the fitted training rows**, not for an arbitrary external `X_calib`. So: calibration in this spike happens **on the training set's own OOB rows** (one calibration observation per training subject, using each subject's `oob_cumhaz` at `horizon`), not on a separately-passed calibration split. Rows with `oob_n_trees_ == 0` are excluded from calibration (mirrors how `oob_score_` already excludes them, `_estimator.py:639-648`). A genuinely fresh, non-training test set is then scored with ordinary `predict_risk(X_test, horizon)` (not OOB) to check coverage.
3. **Censoring-adjustment weighting rule must be stated concretely**, not just "IPCW-style": use `rftvc.metrics.KaplanMeierCensoring` (reverse-KM censoring-survival estimator, `metrics.py:51-78`) to weight calibration residuals by `1/Ĝ(min(T_i, horizon))`, following the general-right-censoring weighting scheme research Q8 identified across the literature (Candès/Lei/Wasserman 2021 and its 2025 general-right-censoring extension) — this is a documented, specific rule, not left to implementation-time invention.
4. **Coverage check must use repeated simulations with reported uncertainty**, not a single seed. `check_coverage` runs R≥30 independent replications at each `alpha`, reports mean empirical coverage **and** a Monte Carlo confidence interval on that mean (e.g. normal approximation over replications), not a single point estimate.

**Files:**
- `bench/conformal_coverage_spike.py` (new, exploratory, `src/rftvc/` untouched):
  - Module docstring states up front: target is `P(event by horizon | X)`; calibration set is training-set OOB rows (not an external split); this is an **empirical coverage investigation**, not a claim of a formal split-conformal guarantee (OOB rows are not i.i.d.-exchangeable with a fresh calibration draw in the classical sense); single-event `SurvivalForestTV` only, non-TVC (reuse Slice 2's static DGP so the target risk has a clean closed form to validate against); a "coverage does not hold" or "inconclusive" result is an acceptable, reportable outcome, not a failure of the slice.
  - `weighted_conformal_risk_interval(fitted_estimator, horizon, alpha) -> (lower, upper) per test row` — calibrates on `fitted_estimator`'s own training-set OOB rows internally (per point 2 above; no `X_calib` argument, since there is no separable calibration split in this design), applies `KaplanMeierCensoring`-based weighting (per point 3), returns bounds for `predict_risk(X_test, horizon)` rows passed in.
  - `check_coverage(n_train, n_test, alpha, n_reps=30, seed=0) -> dict` — R independent replications on Slice 2's static known-hazard DGP; each replication: fit, calibrate on OOB, predict+bound on fresh test draw, record empirical coverage against the DGP's true `P(event by horizon | X)`; returns mean coverage, MC confidence interval, and per-replication raw values (for later inspection, not just the summary).
- No changes to `src/rftvc/`.

**Acceptance criteria:**
- `check_coverage` reports mean coverage ± an explicit Monte Carlo CI at 2-3 `alpha` values, R≥30 replications each — not a single-seed point estimate.
- Findings are written up regardless of outcome (coverage holds, coverage fails, or inconclusive) — per design.md's investigate-only framing, "it doesn't work at this n" is a valid, useful result.
- Module docstring explicitly disclaims a formal split-conformal guarantee and states the exact target, calibration-set definition, and weighting rule (points 1-3 above) before any code that uses them.

**Codex review findings applied:** named the calibration target explicitly; replaced the unworkable `X_calib`-based signature with OOB-on-training-rows calibration matching how `_rebuild_design`/`oob_cumhaz` actually work; specified the exact IPCW-style weighting rule instead of leaving it to invent at implementation time; replaced the single-seed `±0.05 at n=2000` criterion with repeated-replication coverage + explicit uncertainty reporting; removed the implied split-conformal theoretical guarantee.

---

## Slice 8: competing-risks permutation-importance CI gate

**Why:** completes the S17 row-3 sweep started in Slices 6-7. Same pattern as Slice 7: the real design rule was verified to transfer to the reduced scale, so it's ported unmodified rather than recalibrated.

**Files:**
- `bench/cr_importance_truth_check.py` (new) — reuses `bench.tvc_perm_sim`'s competing-risks oracle scenario and its stats helpers unchanged, same reduced scale as Slices 6-7. `run(n_reps) -> (n_reps, 2)` of `(d_s1, d_s2)`; `check(res) -> (pass_cause1, pass_cause2, ub_s2, delta1)`.
- `tests/test_cr_importance_truth.py` (new) — fast smoke test, plus the gate: Holm-reject on `d_s1`, `d_s2`'s upper bound <= `0.1 * oracle(cause 1)` (the design's real 0.1×, not the old pilot's loosened 0.3×). R=15, not `slow`-marked.

**Explicit scope note:** competing-risks importance scenario only — this finishes S17 (all three of row 3's oracle scenarios now closed: trend/3b, timing/3c, competing risks/3d). S19's landmark-importance scenarios remain a separate, unrelated gap.

**Docs updated:** `docs/plans/simulation-validation-findings.md` — new row 3d, rows 3/3b/3c and "does not establish" text updated to reflect S17 being fully covered.

**Acceptance criteria:**
- Gate runs in the default merge-gate suite.
- Rule verified to hold at reduced scale via multiple independent out-of-band seed ranges before locking in.
- Findings doc accurately states S17 is now fully closed while S19 remains open.

---

## Slice 9: S19 landmark-importance CI gates

**Why:** the last open item on the manual-gate punchlist ([[rftvc-roadmap]] "Still open"). Ran a
full CRISPI pass for this slice specifically (`docs/plans/s19-landmark-gate-{questions,research,design}.md`)
because it has a real wrinkle Slices 6-8 didn't: one of the three scenarios' design rule is *known
to already fail* at full R=50 scale (a documented, user-approved deviation in `s19-plan.md`, not a
bug), so it can't just be ported or recalibrated the way trend/timing/CR were.

**Scenarios (all three, one PR — see design doc Q5: unlike S17's trio, all three already share one
bench file and one test file, so splitting by PR would not track a real code boundary):**

1. **History/Markov** (`level_history_data`/`replicate`/`oracle` in `bench/tvc_landmark_sim.py`,
   §7.2): the design's real `hist.mean() >= 0.25 * oracle(history)` rule does **not** get gated —
   research verified it's worse than a coin flip at any reduced scale tested (43%-101% of the bound
   across seed batches, matching the full-scale 94%-of-threshold near-miss already on record).
   Instead: (a) the Markov control's real, unmodified bound
   `mark.mean() <= 0.05 * oracle(markov)`, R=15 — passes with 3-7x margin at every scale tested; (b)
   a new, explicitly weaker claim for history — a one-sided t-test that `hist.mean() > 0` is
   statistically significant, **R=30** (R=15 is not robust enough: p=0.138 in the primary seed
   batch; R=30 gives p=0.0004/1.5e-6/7.3e-5 at this gate's own seeds 0-29 and two independent
   out-of-band batches, 10000-10029 and 20010-20039). Reduced scale `n_train=n_eval=200,
   n_estimators=40` was separately confirmed (at R=15, before the R=30 bump) across seeds
   0-14/10000-10014/20010-20024 for the Markov control, which needs no rep-count increase.
2. **Copies bootstrap-SE** (`copies_replicate`, §7.5b): the design's real
   `se(step=0.5)/se(step=4.0) >= 0.5` ratio ported unmodified, R=15 both `step` arms, at
   `n_train=n_eval=100, n_estimators=20` — verified across 3 independent seed batches (margins
   0.56-0.73, never near the 0.5 boundary).
3. **Censoring PE-vs-Brier** (`censoring_data`/`censoring_replicate`, §7.5b): the design's real
   rank-*concordance* rule ported unmodified from `bench.tvc_landmark_sim.run()`'s own `pass4` —
   `(pe_z1.mean() > pe_z2.mean()) == (brier_z1.mean() > brier_z2.mean())`, i.e. the two scoring
   families' rankings agree, not "both rank z1 above z2" (a stricter claim the design never
   declared) — not the stricter one-sided-t alternative research tested and rejected either
   (Brier's per-replicate significance is seed-sensitive at this scale, p=0.109 in one batch,
   while the raw mean comparison is robust in all 3), R=15, at `n_train=n_eval=150,
   n_estimators=25` (n=100 breaks the generator: `UndefinedMetricError: no events in (0,
   windows[-1]] to score`).

**Files:**
- `bench/landmark_importance_truth_check.py` (new) — reuses `bench.tvc_landmark_sim`'s
  `level_history_data`/`replicate`/`oracle`/`copies_replicate`/`censoring_data`/
  `censoring_replicate` and `bench.tvc_perm_sim`'s `one_sided_t` unchanged.
  - `run_level_history(n_reps_history=30, n_reps_markov=15, n_train=200, n_eval=200, n_estimators=40, seed0=0) -> (hist, mark)` — two `(n_reps,)` arrays.
  - `check_level_history(hist, mark, oracle_history, oracle_markov) -> (p_history_positive, pass_markov, markov_bound)`.
  - `run_copies(n_reps=15, n_train=100, n_eval=100, n_estimators=20, seed0=0) -> (small_se, large_se)` — two `(n_reps,)` arrays (`step=0.5`/`step=4.0`).
  - `check_copies(small_se, large_se) -> (pass_ratio, ratio)`.
  - `run_censoring(n_reps=15, n_train=150, n_eval=150, n_estimators=25, seed0=0) -> pd.DataFrame` (columns `pe_z1, pe_z2, brier_z1, brier_z2`, as `censoring_replicate` already returns per-row).
  - `check_censoring(df) -> (pass_agree, pe_rank_z1_larger, brier_rank_z1_larger)`.
- `tests/test_landmark_sim_truth.py` (extend, not new — already imports the needed generators) —
  three new default-tier (not `slow`) tests: `test_markov_control_and_history_significance`,
  `test_copies_se_does_not_shrink_with_more_landmarks`, `test_censoring_pe_and_brier_rankings_agree`.
  Each states its pass rule and the out-of-band verification (seed ranges, margins) in its own
  docstring, matching `test_timing_importance_truth.py`'s style. The existing `slow`-tier
  `test_pilot_pass_rules_point_the_right_way` stays as-is (a looser cross-check at yet another
  scale/seed range — not fully redundant with the new tests, since it also smoke-tests
  `copies_replicate`/`censoring_replicate` at their own different reduced scale).

**Explicit scope note:** closes findings-doc row 4 entirely, but the history scenario's gate is
**not** the design's originally-declared magnitude claim (0.25x oracle) — only a weaker sign +
significance claim. The 0.25x rule's full-scale failure (`s19-plan.md` lines 121, 127-130) remains
the user-approved deviation on record; this slice does not attempt to resolve or re-litigate it.

**Docs updated:** `docs/plans/simulation-validation-findings.md` — new row 4 entry (or edited in
place) documenting exactly what's closed, with the history caveat stated plainly; re-grep the whole
file (not just the diff) for stale "S19's landmark-importance scenarios remain manual-only"
cross-references (rows 3b/3c/3d, the closing summary) per [[rftvc-dev-workflow]]'s standing lesson.

**Acceptance criteria:**
- All three new tests run in the plain `pytest` invocation (no `-m` flag) — verified by actually
  running it.
- Every threshold/rep-count matches a value this slice's own design doc verified across >= 2
  independent out-of-band seed batches distinct from the gate's own seeds (0-14, or 0-29 for
  history).
- History gate's docstring explicitly states it asserts a weaker claim than the design's declared
  rule, and why — mirroring Slice 6's "not the original numbers" framing.
- `simulation-validation-findings.md` and every stale cross-reference to row 4 updated in this PR.
- One `codex:rescue` diff review before merge.

---

## Slice 10: n-sweep beyond Slice 1's two points

**Why:** closes `docs/plans/simulation-validation-findings.md` row 5's declared blind spot ("only
two `n` values checked, not a full sweep"), the last real statistical-validity item on the
manual-gate punchlist after Slice 9. Full CRISPI pass
(`docs/plans/n-sweep-gate-{questions,research,design}.md`). Research found this item is
structurally different from Slices 5-9: (a) the existing `slow`-tier 2-point claim (R=10) was
never checked against any seed range beyond its own — it actually **fails** at an out-of-band batch
(seeds 20010-20029, lower bound -0.015 at the original 2-point config); R=20 restores robustness
(re-verified under the exact 3-point config this slice ships: lower bounds 0.050, 0.058, 0.022
across seeds 0-19, 10000-10019, 20010-20029 — note the middle-point draw shifts the shared RNG
stream, so these aren't numerically identical to the 2-point pilot numbers above, only confirming
the same conclusion holds). (b) A genuine 3+ point "decreases at every step" sweep is not honestly achievable at any
scale tested — the oracle-to-model gap plateaus hard after the first jump (mid-to-large step's
one-sided lower bound is negative in every batch tried, at every scale). (c) this is the most
expensive gate closed this session: R=20 at the necessary scale (`n_estimators=200`, `n` up to
5000) costs ~30s, vs. 0.4-9s for every Slice 5-9 gate — user confirmed this cost is acceptable
(2026-09-29) rather than shrinking scale further (which research also tried and found reintroduces
the same robustness problem it's trying to fix).

**Files:**
- `bench/pe_score_convergence_sim.py` — **unchanged**. `replicate`/`run` already accept an
  arbitrary `n_values` tuple and already gate on `res[:, 0] - res[:, -1]` (first vs. last), so a
  3-tuple keeps the existing outer-pair claim with zero code changes.
- `tests/test_pe_score_convergence_truth.py` — replace `test_pe_score_mean_gap_shrinks_with_n`
  (the old `slow`-tier, R=10, 2-point test) with `test_pe_score_gap_shrinks_from_n200_to_n5000`:
  same scale and same gated claim (`lower > 0` on the 200-vs-5000 paired difference), but R=10 →
  R=20, `n_values=(200, 5000)` → `(200, 1000, 5000)` with the middle point's mean asserted only for
  finiteness — not gated, and not otherwise surfaced on a passing run (see design doc for why
  gating it would be dishonest) — and **not** `slow`-marked.

**Explicit scope note:** this does not establish a general convergence-rate claim across many
points — only that the design's original endpoint pair (200 vs 5000) holds robustly, now in the
default tier, plus one additional reported data point. The mid-to-large plateau is a documented,
real property of this specific score/DGP, not something a future slice should try to "fix" by
finding yet another scale — research already ruled that out.

**Docs updated:** `docs/plans/simulation-validation-findings.md` row 5 — CI tier `slow` → `not
slow` (~30s), blind-spot text updated to describe the 3-point/outer-pair-only shape; closing
"No n-sweep / asymptotic study" bullet updated to reflect partial (not full) closure.

**Acceptance criteria:**
- New test runs in the plain `pytest` invocation (no `-m` flag) — verified by actually running it.
- R=20's robustness already verified (research) across >= 2 out-of-band seed batches distinct from
  the gate's own (0-19).
- Findings doc accurately scopes what's closed vs. what's inherently not closeable further here.
- One `codex:rescue` diff review before merge.

---

## Slice 11: TVC vs. `CoxTimeVaryingFitter` on a known-truth DGP

**Why:** closes `docs/plans/simulation-validation-findings.md`'s closing-summary bullet — the TVC
case had no external-tool cross-check against any known-truth DGP at all (only real-data,
no-ground-truth comparisons via the Rossi rc-validation pass). Full CRISPI pass
(`docs/plans/tvc-external-tool-{questions,research,design}.md`). Mirrors Slice 2's exact pattern
(new, genuinely simple DGP that exactly matches the external tool's assumptions) but for TVC: a
single-covariate, log-linear external-TVC hazard, since `tests/sim.py`'s existing DGP has a
threshold non-linearity that would leave `CoxTimeVaryingFitter` misspecified too (not a fair "home
turf" comparator, per the same reasoning Slice 2's own plan-review correction already established).

Research caught and fixed a real implementation bug before it shipped: naively assuming
`CoxTimeVaryingFitter.baseline_cumulative_hazard_` lands on exact integer times (as some other
unit-interval fixtures in this codebase do) gives silent `NaN`s here, because this DGP's event
times are continuous, not boundary-aligned — fixed with a proper step-function `searchsorted`
lookup instead of `reindex`.

**Files:**
- `bench/tvc_coxtv_truth_check.py` (new) — `true_cumhaz`/`simulate`/`rows` (same event-time-
  inversion pattern as `tests/sim.py`, `RATE0=0.15, BETA=0.8`, single covariate `z`), `replicate`/
  `run` (same shape as `bench/lifelines_truth_check.py`), `_baseline_step_at` (the searchsorted
  fix above), `EPSILON = 0.18` (~2x the observed gap, calibrated from out-of-band seeds
  10000-10004 and 20010-20014, distinct from the gate's own 0-9).
- `tests/test_tvc_coxtv_truth.py` (new) — fast closed-form check (`true_cumhaz` vs. numerical
  integration) and smoke test; the gate itself (`res[:,0].mean() <= res[:,1].mean() + EPSILON`),
  R=10, **not** `slow`-marked (~4-5s, well within the default tier per Slices 5-10's discipline —
  Slice 2's original gate predates that lesson and stayed `slow`).

**Explicit scope note:** single-event TVC only, one covariate. Does not cover competing risks
(separate gap, needs `randomForestSRC`) or landmark estimators (no external tool exists for that
shape at all — stays open).

**Docs updated:** `docs/plans/simulation-validation-findings.md` — new row 8; closing summary's "no
landmark-estimator or TVC cross-check against an external tool exists at all" bullet updated to
reflect TVC is now covered (landmark remains open).

**Acceptance criteria:**
- New test runs in the plain `pytest` invocation (no `-m` flag) — verified by actually running it.
- `EPSILON` calibrated from seeds distinct from the gate's own, stated before the gate ran, verified
  across >= 2 independent out-of-band batches.
- `true_cumhaz` verified against numerical integration to `atol=1e-6`.
- One `codex:rescue` diff review before merge.

---

## Notes for implementation-stage context recovery

- Every slice reuses existing building blocks (`KaplanMeierCensoring`, the `slow`/`network` pytest marker convention, `oob_cumhaz`'s existing private call pattern via `_rebuild_design`) — but Slice 2 and Slice 4 now require **new, genuinely static DGPs** (not reuse of `tests/sim.py`'s TVC one), per the plan-review correction. Don't silently fall back to the TVC DGP because it's already there — that was the mistake this revision fixes.
- Slice 4 explicitly does not touch `src/rftvc/` and does not claim a formal conformal guarantee — if implementation drifts toward either, that's a plan deviation requiring a note and a check-in with the user.
- This plan went through one round of `codex:rescue` plan review (2026-09-28) before any code was written; all five findings from that review are applied above, each tagged "Codex review findings applied" in its slice.
