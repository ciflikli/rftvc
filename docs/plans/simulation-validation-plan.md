# Simulation-based statistical validation plan

Slice 3 of `docs/plans/plan.md`. Parent: user question 2026-09-28, "how can we ensure our implementation is statistically valid?" — distinct from `rc-validation-plan.md` (real-data plausibility/direction checks against `lifelines`, no known ground truth) and from any single slice's own closed-form-DGP sim (S11/S14/S17/S19), which each validate one feature in isolation. This plan's job is to **assemble** what those already validate, state each check's actual assertion and blind spots (not an upward paraphrase), and add the two genuinely-missing checks: does `PEScore` converge to a matching-convention oracle as training size grows (Slice 1), and does `SurvivalForestTV` land in the right ballpark against an external reference tool on a DGP with known closed-form truth, not just real data (Slice 2). Went through one round of `codex:rescue` plan review before Slices 1-2 were coded (see `docs/plans/plan.md`'s per-slice "Codex review findings applied" notes); this document and `simulation-validation-findings.md` are Slice 3, not separately plan-reviewed before writing (a docs-assembly slice, lower risk than the code slices it inventories).

## Why these checks, not more

Four prior slices (S11 TVC-vs-baseline, S14 competing risks, S17 permutation-importance oracles, S19 landmark-importance oracles) already built closed-form-hazard DGPs with truth-checked generators and predeclared statistical pass rules, each scoped to the one feature that slice shipped. Nobody had assembled them into a single answer to "is this library statistically valid," and two genuine gaps existed: `PEScore` itself (the metric several of these importance measures are built on) had no convergence check against its own theoretical target, and no DGP-with-known-truth had ever been checked against an external reference implementation (only real-data plausibility checks existed for that, in `rc-validation-findings.md`). This plan does not attempt a from-scratch unified simulation framework (design.md's rejected Approach 1) or a full n-sweep asymptotic convergence study across every estimator/censoring-rate combination (design.md's Approach 3, deferred) — see `docs/plans/design.md` for why.

## Decisions

1. **Inventory, don't reprove.** `simulation-validation-findings.md` lists what each existing `bench/*_sim.py` + `tests/test_*_sim_truth.py` pair actually asserts, verified against the real test code (not summarized from a plan file's prose, per the Codex plan-review correction to this slice's original draft, which had mischaracterized `tests/test_sim.py`'s assertion). Existing checks are not rerun or modified by this slice.

2. **State blind spots explicitly, per check.** Every row in the findings doc has a "known blind spots" entry: whether the check is CI-gated (fast merge-gate tier) or manual-only (`bench/*_sim.py run()`, no pytest assertion at all — true for S14's R=30 gate and S17/S19's R=50 gates, which only ever ran by hand and are recorded as CSVs under `docs/bench/`), and what scenarios it does not cover.

3. **No completeness claim.** Per the plan-review correction to this slice's original acceptance criterion ("what would fail if the Rust core had a real statistical bug" — too broad): this document does not claim that passing every listed check rules out a statistical bug outside the tested scenarios. It states, in its own closing section, which scenarios are simulated and which are not.

4. **One documented, accepted, real deviation stays documented, not buried.** `tests/test_tvc_sim_truth.py`'s `path_effect` pilot check (S20 T9) is a known case where the *design's* tight pass rule fails even at the full R=50 run — not a code defect, a genuine finite-sample forest bias, checked against an exact-hazard stub separately and user-accepted as a documented deviation (see that test's own inline comment). The inventory carries this forward rather than letting it look, out of context, like every check simply "passed."

## Tasks

- [x] Read every existing `bench/*_sim.py` + `tests/test_*_sim_truth.py` pair's actual code (not just its module docstring) to extract real assertions.
- [x] Write `simulation-validation-findings.md`: one row per check (existing + Slices 1-2), with DGP, actual assertion, pass rule, CI tier, and blind spots.
- [x] Cross-check Slice 1 (`bench/pe_score_convergence_sim.py`) and Slice 2 (`bench/lifelines_truth_check.py`) are represented with the same rigor as the pre-existing four.
- [x] State the non-completeness boundary explicitly in the findings doc's closing section.

## Acceptance

- Every existing `bench/*_sim.py` + `tests/test_*_sim_truth.py` pair is listed in `simulation-validation-findings.md` with its actual assertion (verified against the test file, not paraphrased upward) and at least one blind spot.
- Slices 1-2's checks are listed in the same document, not a separate one.
- The document does not claim general statistical validity beyond what its own listed checks establish, and states what is not covered.
