# Research Questions: Competing-risks vs. randomForestSRC on a known-truth DGP

Closes `docs/plans/simulation-validation-findings.md`'s last open external-tool-parity gap
(competing risks) per `[[rftvc-roadmap]]`'s "still open" list. Mirrors Slice 2
(`bench/lifelines_truth_check.py`) and Slice 11 (`bench/tvc_coxtv_truth_check.py`): a new,
genuinely simple DGP whose true CIF is closed-form, `CompetingRisksForestTV` vs.
`randomForestSRC::rfsrc` (competing-risks mode) both compared to that ground truth, not to
each other on real data (S14's `bench/s14_cr_parity.{R,py}` already did the real-data,
no-ground-truth comparison on `survival::pbc`).

## Questions

1. What DGP shape does `randomForestSRC`'s competing-risks mode actually assume/handle well —
   does it need a genuinely static (non-TVC) DGP the way Slice 2/11 needed simple covariates, or
   can it fairly handle the same threshold-non-linearity DGP already in `tests/sim.py`
   (`bench/s14_cr_sim.py`, `bench/cr_forest_truth_check.py`)? Check `bench/cr_rfsrc_check.R` (2026-09-26
   spike) and `bench/s14_cr_parity.R` for the exact `rfsrc(Surv(...) ~ ., data, ...)` call shape,
   supported `splitrule`s for competing risks, and what `predict()` returns (`$cif`,
   `$time.interest`) plus its indexing/shape.
2. Is there already a closed-form-CIF competing-risks DGP with a known true `F_k(t | x)` in this
   repo? Check `bench/cr_forest_truth_check.py` and `bench/s14_cr_sim.py` for `true_`-prefixed
   functions and how `CompetingRisksForestTV`-vs-truth gates already verify against numerical
   integration (Slice 5, PR #57).
3. What is the R/randomForestSRC scratch-install toolchain exactly — R version, package version,
   scratch lib path, `Rscript` invocation, and what did `[[rftvc-dev-workflow]]`'s "lost on reboot"
   note mean in practice (does it need reinstalling this session)? Check `bench/cr_rfsrc_check.R`'s
   header comment and whether `RL=<path>` currently resolves to anything.
4. What does `randomForestSRC::rfsrc` require for start-stop (left-truncated) rows vs. plain
   right-censored rows in competing-risks mode, and does the chosen DGP need start-stop shape at
   all, or can it use plain `(time, status)` rows like S14's pbc parity did (simpler, avoids the
   `Surv(id, start, stop, event)` formula's quirks noted in `cr_rfsrc_check.R`)?
5. What pass/fail rule and epsilon-calibration discipline did Slice 2/Slice 11 use (both closed-
   form vs. estimator-vs-truth, both default-tier, not `slow`), and does it transfer directly to
   competing risks (two causes → two epsilons, or one pooled metric)? Check
   `bench/lifelines_truth_check.py` and `bench/tvc_coxtv_truth_check.py`'s `EPSILON` calibration
   comments plus `tests/test_tvc_coxtv_truth.py`.
6. Is there a Python `rpy2`-free pattern already established for shelling out to `Rscript` from a
   pytest test (not just a manual `bench/*.py` driver script) — does S14/Slice 11's design run R
   inside the actual CI-gated test, or only in a manual `bench/*.py` script with R output as a
   fixture? This determines whether the new default-tier test can call R directly or needs a
   pre-computed/cached R output checked into the repo (R is not guaranteed present in CI).

## Codebase References
- `bench/cr_rfsrc_check.R` — 2026-09-26 spike: rfsrc formula shape, start-stop, event coding
- `bench/s14_cr_parity.R` / `bench/s14_cr_parity.py` — real-data (pbc) rfsrc comparison, no ground truth
- `bench/cr_forest_truth_check.py` — existing `CompetingRisksForestTV`-vs-truth gate (Slice 5, PR #57)
- `bench/s14_cr_sim.py` — competing-risks DGP used across S14/S11/Slice 5-8 sims
- `bench/tvc_coxtv_truth_check.py`, `tests/test_tvc_coxtv_truth.py` — Slice 11's exact template to mirror
- `docs/plans/tvc-external-tool-{questions,research,design}.md` — Slice 11's own CRISPI trail, closest analog
- `docs/plans/simulation-validation-findings.md` — row to update on completion
