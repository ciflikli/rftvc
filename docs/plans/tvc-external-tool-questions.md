# Research questions: TVC vs. an external tool on a known-truth DGP

Closes `docs/plans/simulation-validation-findings.md`'s closing-summary bullet: "no landmark-
estimator or TVC cross-check against an external tool exists at all... no equivalent check exists
for the TVC case against, e.g., `CoxTimeVaryingFitter` on a *known-truth* (as opposed to real-data)
DGP." (Rows 6/7/7b are all TVC-free by construction — `CoxPHFitter`/`sksurv` can't handle TVC.) The
existing `CoxTimeVaryingFitter`-vs-`rftvc` comparison (`docs/plans/rc-validation-{plan,findings}.md`,
the Rossi dataset) is real data only — no known ground truth to check either model against.

`lifelines.CoxTimeVaryingFitter` does handle counting-process (start, stop, event) TVC data directly
— same row shape `rftvc` already uses internally. Slice 2 (`bench/lifelines_truth_check.py`)
already established the pattern for this kind of check in the static case: a genuinely simple DGP
that exactly matches the external tool's assumptions (so the tool is a strong, correctly-specified
baseline), comparing both models' ISE against the closed-form truth, checking `rftvc` isn't a
*statistical outlier* rather than expecting it to beat a correctly-specified parametric model on its
own home turf.

## Questions

1. Slice 2's own docstring records a plan-review correction: "subsetting a time-varying DGP to one
   covariate does not make its conditional survival truth static, since the redrawn covariate still
   drives the actual hazard" — i.e. Slice 2 deliberately built a *new*, simpler DGP rather than
   reusing `tests/sim.py`'s existing external-TVC DGP. Does `tests/sim.py`'s hazard
   (`0.15 * exp(0.8z + 0.8*1{z>1} + 0.4x0)`) exactly match `CoxTimeVaryingFitter`'s assumptions (a
   log-linear proportional-hazards model in the covariates the model is given), or does its
   `0.8*1{z>1}` threshold term make it non-log-linear in `[x0, z]` — meaning `CoxTimeVaryingFitter`
   would be *misspecified* too, not a strong "home turf" comparator the way Slice 2's DGP made
   `CoxPHFitter`? Read `tests/sim.py`'s `rows()` to confirm exactly which columns are handed to the
   fitted model (raw `[x0, z_k]`, or does it also expose the indicator).
2. What exact DataFrame shape does `CoxTimeVaryingFitter.fit()` need (columns, `id_col`,
   `start_col`/`stop_col`, `event_col`), and what does its prediction API offer for reconstructing
   `S(t | covariate path)` at arbitrary times for a *specific* test subject's known future
   covariate trajectory (the same "predict along a known external path" setup `tests/sim.py`'s own
   pass rule uses for `rftvc`)? Check `lifelines.CoxTimeVaryingFitter`'s actual methods
   (`predict_partial_hazard`, `predict_log_partial_hazard`, `baseline_cumulative_hazard_`) — there
   is no built-in `predict_survival_function` for time-varying covariates, so confirm exactly how
   to combine `baseline_cumulative_hazard_` with per-interval partial hazards to get a cumulative
   hazard over a path, analogous to what `rftvc`'s own `predict_cumulative_hazard(..., intervals=)`
   already does internally (see `bench/pe_score_convergence_sim.py`'s `_eval_set` for the existing
   pattern of building a per-interval prediction frame).
3. If a new, simpler DGP is needed (per Q1's likely answer): what is the simplest genuinely
   log-linear external-TVC hazard that both (a) gives `CoxTimeVaryingFitter` a correctly-specified
   home turf and (b) is different enough from `tests/sim.py`'s existing DGP to not just be a
   trivial copy (e.g., drop the threshold indicator term, keep everything else the same shape:
   `hazard = rate0 * exp(beta * z_k)`, external `z_k ~ N(0,1)` redrawn per interval, no `x0`)?
4. At what training/test sizes and `n_estimators` does a real, out-of-band-verified ISE gap between
   `rftvc` and a correctly-specified `CoxTimeVaryingFitter` emerge — mirroring Slice 2's exact
   discipline (an out-of-band pilot at seeds >= 10000, an additive epsilon tolerance roughly double
   the observed gap, not a ratio, since the reference ISE may be small)? Verify by actually running
   both models, not assuming a similar gap to Slice 2's static case.
5. Runtime: `CoxTimeVaryingFitter` fits an iterative Newton-Raphson on stacked counting-process
   rows — confirm it isn't meaningfully slower than `CoxPHFitter` at the row counts this DGP would
   produce (each subject contributes ~1-8 rows, vs. Slice 2's exactly 1 row per subject), so this
   stays in the same runtime ballpark as Slice 2's gate (`slow`, ~2s) or the newer default-tier
   gates from Slices 5-10.

## Codebase references

- `bench/lifelines_truth_check.py` (Slice 2) — the template: DGP, `replicate`/`run` shape,
  pass-rule discipline (additive epsilon from an out-of-band pilot).
- `tests/sim.py` — existing external-TVC DGP, for comparison/reuse-or-not.
- `docs/plans/rc-validation-{plan,findings}.md` — the existing real-data (Rossi)
  `CoxTimeVaryingFitter`-vs-`rftvc` comparison, for what's already been learned about using this
  fitter in this codebase (API quirks, footguns).
- `docs/plans/simulation-validation-findings.md` rows 1, 5, 6 and the closing summary — current
  documented state of this gap.
