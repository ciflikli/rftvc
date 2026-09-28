# Research Questions: Simulation-Based Statistical Validation + Conformal Prediction Feasibility

## Part A — Simulation-based validation

1. What data-generating processes (DGPs) does the codebase already have for testing (`tests/fixtures/`, `tests/ref/`, `examples/`)? Do any of them draw from a *known* hazard function, or are all existing fixtures either real data (Rossi/EBMT4) or opaque synthetic data with no closed-form ground truth?
2. What outputs does the library expose that a simulation study could score against ground truth — cumulative hazard, survival function, PE score/deviance, `hazard_effect`/`path_effect` contrasts, importance measures? For each, what would "ground truth" mean (a known hazard function, a known true effect size, a known true relevant/irrelevant feature set)?
3. How does `rftvc.metrics.PEScore` (the α-mixed piecewise-exponential log score) work, and is it itself suitable as a convergence metric (i.e., does a correctly-specified simulation let us assert PE score approaches its theoretical optimum as n grows)?
4. What do `tests/ref/cr_ref.py` and `tests/ref/logrank_ref.py` already validate, and against what reference (R survival package? lifelines? hand-derived)? Is there a pattern there worth reusing for simulation-based ground-truth comparisons?
5. Are there existing R/Python reference packages already used or vendored in this repo (`lifelines`, `CoxTimeVaryingFitter`, anything from `randomForestSRC`) that could serve as a second implementation to compare against on synthetic data with known truth?
6. What's the runtime cost of `SurvivalForestTV`/`CompetingRisksForestTV` fits at realistic simulation scale (multiple n, multiple replications) — is a full simulation study (varying n, censoring rate, TVC vs. static, competing risks) computationally realistic in CI, or does it need to be a separate slow/manual suite?
7. How does the existing `docs/plans/rc-validation-plan.md` structure its plan (decisions, scope boundaries, deliverables) — what should the simulation-validation plan reuse vs. depart from?

## Part B — Conformal prediction feasibility

8. What does "conformal prediction for survival analysis" mean in the literature given censoring (e.g., Candès et al. 2021 conformalized survival analysis, weighted conformal methods for right-censored data), and which variant(s) would fit this library's outputs (cumulative hazard / survival function / risk-at-horizon)?
9. Does the current OOB (out-of-bag) machinery in `_estimator.py`/`_competing.py` already produce per-observation OOB predictions that a split-conformal or OOB-conformal calibration step could consume, or would this require new plumbing?
10. What would the public API surface look like — a new `rftvc.calibration` (or similar) module producing calibrated prediction intervals/sets, or a parameter on existing `predict_*` methods? What do comparable sklearn-ecosystem conformal libraries (e.g. MAPIE) do for their API shape, and is that a reasonable model to follow?
11. Given [[rftvc-general-purpose]] (keep the library general-purpose, not overfit to one domain), what would a general-purpose conformal calibration feature need to support (arbitrary coverage level, TVC and competing-risks cases, both static-horizon and full-curve predictions)?

## Codebase References
- `src/rftvc/_estimator.py` — `SurvivalForestTV`, predict methods (line ~654-730)
- `src/rftvc/_competing.py` — `CompetingRisksForestTV`, predict methods (line ~274-331)
- `src/rftvc/metrics.py` — `PEScore`
- `src/rftvc/inspection.py` — `hazard_effect`, `path_effect`, importance functions
- `tests/ref/` — existing reference-implementation validation
- `tests/fixtures/` — existing test data generators/fixtures
- `docs/plans/rc-validation-plan.md`, `docs/plans/rc-validation-findings.md` — prior real-data validation pass, structural precedent
