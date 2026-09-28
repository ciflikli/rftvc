# Design: simulation-based statistical validation + conformal prediction feasibility

Stage 3 of CRISPI. Input: `docs/plans/questions.md`, `docs/plans/research.md`.

## Executive summary

Simulation-based validation does **not** need new infrastructure — the repo already has a four-times-repeated closed-form-DGP / fast-smoke-test / slow-pilot-test / manual-`bench`-run pattern (`tests/sim.py`, `bench/{s14_cr_sim,tvc_perm_sim,tvc_landmark_sim}.py`). The open question was always scope, not mechanism. Recommended approach: **generalize, don't rebuild** — add the two ground-truth checks that don't exist yet (PE-score-converges-to-known-ceiling; a `lifelines`/`scikit-survival`-vs-closed-truth cross-check) as new slices in the existing pattern, and write a single library-wide plan document (mirroring `rc-validation-plan.md`'s Decisions format) that inventories what's already validated so it's visible in one place, rather than scattered across four slice-specific plan files.

Conformal prediction is **feasible for the single-event, horizon-based case**, materially easier than expected because the OOB full-curve engine (`oob_cumhaz`/`oob_cause_cumhaz`) already exists in Rust and is unit-tested — it just isn't public API. It is **not** feasible to ship as a general-purpose, competing-risks-covered feature right now: the literature has no settled competing-risks conformal-survival method, and [[rftvc-general-purpose]] treats competing risks as load-bearing, not optional. Recommendation: **investigate/prototype only in this pass** — a design note and a follow-up slice for single-event horizon-based conformal risk intervals using OOB calibration, explicitly out-of-scope for competing risks and full-curve calibration until the field settles further.

---

## Part A — Simulation-based validation

### Approach 1: One big new "sim-validation" test suite from scratch

Write a new, unified simulation harness from first principles, covering all estimators/outputs in one plan.

- **Against**: ignores that `tests/sim.py` + three `bench/*_sim.py` families already do this per-slice, well, with predeclared pass rules. Duplicates work, risks drifting from the established fast/slow/bench three-tier convention, and the existing files already have Codex-reviewed methodology (`docs/plans/s18-plan.md`'s VIM-bias note, `rc-validation-plan.md`'s relevance-vs-direction lesson).

### Approach 2: Generalize the existing per-slice pattern, fill the two real gaps

Treat the four existing sim suites as already-done validation; write a plan that (a) documents them as a coherent whole for the first time, (b) adds the two genuinely-missing checks research surfaced — PE score converging to its theoretical ceiling as n grows (Q3), and a closed-truth cross-check against `lifelines`/`scikit-survival` in addition to `rftvc` itself (Q5) — using the same fast-smoke/slow-pilot/manual-bench convention.

- **For**: minimal new surface area, reuses a pattern that's already been through Codex review four times, keeps CI cost near-zero (fast smoke tests only in the merge gate, exactly like today).
- **For**: directly answers the "is our implementation statistically valid" question with an inventory plus two new checks, rather than a from-scratch rebuild whose main value would be organizational, not statistical.
- **Against**: less "impressive" as a single new deliverable; requires reading/understanding four existing files before writing anything new.

### Approach 3: Approach 2, plus a convergence sweep (n → ∞) as a new capability

Same as Approach 2, but add a genuinely new capability: a parametrized sweep (vary n, censoring rate) that plots/asserts monotone improvement in ISE/PE-score-gap, not just a single-n pass/fail. This is closer to a real "does the estimator converge" study than existing suites (which mostly run one n with R replications, not a range of n).

- **For**: this is the strongest form of "statistically valid" — asymptotic behavior, not just single-n plausibility.
- **Against**: real new computational cost (multiple n × multiple reps × multiple estimators); must live entirely in `bench/` (manual-run), not `slow`-marked pytest, per research's finding that even single-n R=10-20 sims are already `slow`-tier.

**Recommendation: Approach 2 now, Approach 3's convergence sweep as an explicit follow-up slice inside the same plan, deferred/optional.** It gets the real statistical-validity gaps closed with low risk and reuses proven methodology; the convergence sweep is valuable but is genuinely new work (not "already exists, just needs assembling") and shouldn't block landing the cheaper wins.

### Representative shape (PE-score-ceiling check, new)

```python
# bench/pe_score_convergence_sim.py (new, mirrors bench/s14_cr_sim.py's shape)
def true_rate_score(hazard_fn, windows, ids, times) -> PEScore:
    """Known-hazard PEScore ceiling: log-likelihood of the *true* piecewise rate
    on the same windows the fitted model is scored on."""
    ...

def run(n_values=(200, 1000, 5000), n_estimators=500, seed=0) -> pd.DataFrame:
    """For each n: fit SurvivalForestTV, compute PEScore vs true_rate_score's
    ceiling, return gap-to-ceiling per n. Mirrors tests/sim.py's ISE pattern
    but for PEScore instead of survival-curve ISE."""
    ...
```

```python
# tests/test_pe_score_convergence_truth.py (fast smoke tier)
def test_true_rate_score_matches_numerical_integration():
    """Closed-form true_rate_score vs numerical integration of the known
    hazard — same pattern as test_cr_sim_truth.py, tolerance 1e-9."""

@pytest.mark.slow
def test_pe_score_gap_shrinks_with_n():
    """One-sided: gap-to-ceiling at n=5000 < gap at n=200, R=10 reps."""
```

---

## Part B — Conformal prediction

### Approach 1: Full scope now — single-event + competing risks, horizon + full-curve

- **Against**: research found no settled competing-risks conformal-survival method in the literature (Q8/Q11) — this would mean either inventing a novel method (out of scope for a library feature, that's a paper) or shipping something not properly validated, which directly contradicts the point of this whole exercise (statistical validity). Reject.

### Approach 2: Single-event, horizon-based only, OOB-calibrated — ship now

Expose `oob_cumhaz`/`oob_cause_cumhaz`-style per-row OOB curves as public API (or reuse the private call pattern `inspection.py` already relies on), add a `rftvc.calibration` module implementing weighted-conformal (IPCW-style, reusing the existing `KaplanMeierCensoring` building block per Q10) calibrated risk-at-horizon intervals for `SurvivalForestTV` only.

- **For**: matches the maturity of the literature (single-event general-right-censoring conformal survival is a converging, multi-paper-validated area per Q8) and the maturity of the codebase's own internals (OOB full-curve engine already unit-tested).
- **For**: `predict_risk(X, horizon)` already exists as the exact shape to calibrate — no new prediction concept, just an interval around an existing one.
- **Against**: doesn't cover competing risks or TVC-landmark estimators — a visible gap against [[rftvc-general-purpose]]'s "don't ship partial generality" lesson. Must be documented explicitly as a scoped-down v1, not silently partial.

### Approach 3: Prototype/investigate only, no shipped code this pass

Write the design note (this document) plus a short spike script under `bench/` proving the OOB-calibration mechanism works numerically on a known-coverage synthetic case, but land no public API change yet — treat competing-risks and full-curve calibration as blocking a "v1" release of the feature until the literature or a follow-up research pass resolves them.

- **For**: matches what the user actually asked ("investigate whether we can include") rather than assuming a build decision was already made.
- **Against**: less immediately useful; defers value.

**Recommendation: Approach 3 for this pass (the user asked to investigate, not build), with Approach 2 as the concretely-scoped next step if the user wants to proceed.** Concretely: this design doc + a coverage-validation spike script (using the Part A simulation infra — a known-hazard DGP is exactly what's needed to check conformal coverage claims empirically) is the deliverable now. A `plan.md` slice for Approach 2 can be written next if you confirm you want to build it, following the same Codex-review-before-code discipline as every other slice.

### Representative shape (spike only, not shipped API)

```python
# bench/conformal_coverage_spike.py (new, exploratory — proves the mechanism,
# not shipped API)
def weighted_conformal_risk_interval(estimator, X_calib, calib_oob_cumhaz,
                                       censoring_km, horizon, alpha=0.1):
    """IPCW-weighted split-conformal lower/upper bound on risk-at-horizon,
    calibrated on OOB rows (estimator.forest_.oob_cumhaz(...) internal call,
    per research Q9). Candès/Lei/Wasserman 2021 + Davidov et al. 2025
    general-right-censoring weighting."""
    ...

def check_coverage(n=2000, alpha=0.1, seed=0) -> float:
    """Fit on tests/sim.py's known-hazard DGP, calibrate via OOB, check
    empirical coverage on held-out data ≈ 1-alpha. This is the empirical
    validity check for the conformal method itself, using Part A's
    infrastructure."""
    ...
```

---

## Open decisions for the user before Stage 4 (plan.md)

1. Part A: confirm Approach 2 (generalize + fill two gaps), with the n-sweep as optional/deferred — or do you want the sweep in scope now?
2. Part B: confirm Approach 3 (investigate/spike only, no shipped API this pass) — or do you already want to commit to building Approach 2's scoped single-event feature?
3. Should `plan.md` (Stage 4) cover both parts, or just Part A (with Part B's spike as a smaller standalone follow-up once you decide on question 2)?
