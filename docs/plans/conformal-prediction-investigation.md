# Conformal prediction investigation: findings

Slice 4 of `docs/plans/plan.md`, design.md Part B Approach 3 (investigate/spike only, user-confirmed 2026-09-28). Deliverable: `bench/conformal_coverage_spike.py` (exploratory, no public API, `src/rftvc/` untouched) plus this write-up.

## What was built

An OOB-calibrated, IPCW-weighted interval around `SurvivalForestTV.predict_risk(X, horizon)`:
- **Calibration set**: the fitted estimator's own training-set OOB rows (`estimator._rebuild_design` + `forest_.oob_cumhaz`, the same private internal call `rftvc.inspection` already relies on) — not an external calibration split, since that machinery only returns predictions for fitted training rows.
- **Weighting**: reuses `rftvc.metrics._ipcw`, the same case/control/IPCW-weight convention `brier_landmark`/`integrated_brier` already build on — `1/G(stop-)` for an observed event-by-horizon row, `1/G(horizon-)` for a row observed event-free through the horizon, weight 0 (excluded) for a row censored before the horizon.
- **Interval**: a single symmetric half-width, the weighted `(1-alpha)` quantile of `|case_i - risk_hat_oob_i|` over resolved calibration rows — deliberately the simplest defensible construction, not the most sophisticated one in the literature.
- **Target checked**: the DGP's true conditional probability `P(event by horizon | X)` (known exactly, since `check_coverage` uses `bench.lifelines_truth_check`'s closed-form static DGP) — a *different*, non-standard target only checkable because ground truth is known here (there is no general sense in which it is "stronger" than the standard target — they are different quantities); the standard conformal guarantee is coverage of the realized 0/1 outcome, which this spike does not separately check.

## Result: mechanism runs correctly, but the intervals are uninformative at this DGP/model combination

`check_coverage(alpha, n_reps=30)` at `alpha ∈ {0.1, 0.2, 0.3}` (nominal coverage 90%/80%/70%):

| alpha | nominal | mean empirical coverage | MC 95% CI |
|---|---|---|---|
| 0.1 | 0.90 | 1.00 | (1.00, 1.00) — degenerate |
| 0.2 | 0.80 | 1.00 | (1.00, 1.00) — degenerate |
| 0.3 | 0.70 | 1.00 | (1.00, 1.00) — degenerate |

Every one of the 30 replications scored exactly 1.00 at every alpha, so the normal-approximation Monte Carlo CI collapses to a point — it reflects zero observed variance *within these 30 runs*, not a tight, well-supported claim that coverage would remain exactly 1.00 under a different risk distribution or sample size; a CI this narrow after 30 degenerate observations is itself a symptom of the intervals being too wide to ever miss, not a strong statistical result to lean on.

100% coverage at every alpha looks like success but is not: separately measured mean interval width (5 replications, `n_train=400`, `n_test=300`) is **0.92 at alpha=0.1, 0.85 at alpha=0.2, 0.76 at alpha=0.3** — on a `[0, 1]` probability scale, these intervals span nearly the whole range even at the loosest nominal level checked. At this DGP's risk distribution (OOB risk predictions ranged ≈0.28-0.96, see below), an interval this wide will contain nearly any true probability the DGP actually produces — that is a property of this specific run's risk range, not a universal claim about all possible `(0, 1)` probabilities.

**Why the intervals are this wide (mechanism, not a bug):** the nonconformity score `|case_i - risk_hat_oob_i|` compares a point risk prediction to a *realized binary outcome*, not to another probability. Even a perfectly-calibrated model's residual for a case with true risk 0.5 has an expected magnitude around 0.5 — binary-outcome nonconformity scores are inherently high-variance, and this DGP's baseline risk is itself high (`SurvivalForestTV`'s OOB risk predictions on the training data ranged ≈0.28-0.96 at `horizon=6` in a sample run), pushing typical residuals well above 0.5 at the quantiles the pass rule needs. This is a known, general property of using an absolute-residual nonconformity score against a binary label — not specific to the OOB-calibration or IPCW-weighting choices made here.

## Verdict: inconclusive at this construction — not "coverage fails," not "coverage holds usefully"

Per design.md's investigate-only framing, this is a reportable, acceptable outcome. Concretely: the OOB-calibration and IPCW-weighting mechanism (the two things this spike actually set out to test the plumbing for, per `docs/plans/research.md` Q9-Q10) work end-to-end without error, on real fitted forests, real censored data. But the specific nonconformity-score choice made here (absolute residual vs. a realized binary outcome) is too crude to produce informative intervals at this DGP's risk level — this says more about the score choice than about whether OOB-based conformal calibration is viable for this library in general.

## What a real follow-up would need to change (not built here — out of scope for a spike)

- **A different nonconformity score.** The literature's actual proposals (Candès/Lei/Wasserman 2021 and successors, per `docs/plans/research.md` Q8) generally score against a *survival-time* residual or a rank-based/CDF-based statistic, not an absolute risk-vs-binary-outcome difference — this spike deliberately used the simplest option, and the result suggests it is too simple to be useful.
- **A DGP/scenario sweep**, not just one static DGP — whether this specific poor result is DGP-specific (high baseline risk) or general needs checking at lower base risk, different horizons, different n.
- **Competing risks and TVC remain explicitly out of scope**, per the original plan decision (no settled competing-risks conformal-survival method exists in the literature) — unaffected by this spike's result either way.

## Recommendation

Do not proceed to a shipped `rftvc.calibration` module (design.md's Approach 2) on the strength of this result. If conformal prediction for this library is pursued further, the next step is a fresh, narrower research pass on nonconformity-score choice specifically (not a build), before any public API commitment — the OOB/IPCW plumbing this spike validated is reusable regardless of which score is eventually chosen.
