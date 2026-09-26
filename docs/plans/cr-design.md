# Design: competing risks in `rftvc`

Status: **v1 draft** (2026-09-26), for user alignment and Codex review. Inputs: `cr-questions.md`, `cr-research.md` (Codex-corrected), `design.md` v2.
Notation: `J` = number of causes (event types 1..J, 0 = censored); `K` = number of grid (event) times, as in `design.md`.

## Executive summary
- Competing risks enters as **J cause-specific hazards on the existing counting-process rows**. Risk sets do not change; only event counts gain a cause dimension. Left truncation, TVCs, id/block resampling and coarsening carry over unchanged.
- A new estimator `CompetingRisksForestTV` shares the engine and the Python base with `SurvivalForestTV`. `SurvivalForestTV` stays bit-identical (J = 1).
- Leaves store per-cause Nelson–Aalen increments. The **cumulative incidence** (CIF) is derived by the discrete Aalen–Johansen product, along covariate paths and from any origin. It is valid for external / specified paths, as for survival today (D4).
- Default split rule: a **composite cause-specific LTRC log-rank**; a single-cause rule is an option. Gray-type (subdistribution) splitting is **not** offered on counting-process rows, and deferred on landmark stacks.
- The landmark workflow gets a cause-coded outcome, so `F_k(s+w | s, H(s))` is a direct target. Metrics gain a cause-specific IPCW Brier score and Wolbers concordance.

## Prerequisite checked this session: prior art in R
`randomForestSRC` 3.9.0 (CRAN), tested with a synthetic start–stop dataset (`bench/cr_rfsrc_check.R`):
- `rfsrc(Surv(id, start, stop, event) ~ .)` with causes {1, 2} is **accepted** (family `surv-CR`, returns `cif`), but **only with `splitrule="random"`**.
- The start–stop mode requires `splitrule="tdc.gradient"`, which is missing from the package's own split-rule list, so the default and every named rule fail (`argument is of length zero` / "Must specify tdc.gradient"). The single-cause start–stop control fails the same way.
- **Conclusion:** no released forest (R or Python) found with informative competing-risks splitting on counting-process rows. rfsrc stays a right-censored parity reference.
- `survival::survfit(Surv(start, stop, cause_factor) ~ 1, id=id)` **does** return the Aalen–Johansen `pstate` for start–stop data: this is the single-node leaf oracle (resolves `cr-research.md` §8.5).

## Decisions
| # | Decision | Status |
|---|---|---|
| C1 | Target on counting-process rows = cause-specific hazards; CIF derived by Aalen–Johansen; no subdistribution hazards | default |
| C2 | New estimator `CompetingRisksForestTV`; `SurvivalForestTV` keeps a bool / {0,1} event and is bit-identical | default — **user decision** |
| C3 | Default criterion `"composite"` = Σ_j of per-cause LTRC log-rank χ²; `split_cause=k` gives the single-cause rule; bake-off decides the final default | default — **user decision** |
| C4 | Ensemble `aggregate="hazard"`: average per-cause cumulative hazards over trees, then Aalen–Johansen; `"cif"` (per-tree AJ, then average) as option; bake-off | default (D11 analogue) |
| C5 | `min_events_leaf` counts all causes; optional `min_events_leaf_cause`; per-cause leaf diagnostics | default |
| C6 | Gray-type splitting deferred (not on counting-process rows; maybe later on landmark stacks) | default |
| C7 | Metrics implemented in `rftvc.metrics`; `comprisk` / R `survival` only as test references | default |
| C8 | Cause codes are generic integer labels so that multi-state (transition codes) can layer on later | default |

## Approaches considered
- **A. Cause dimension in the engine + a new estimator.** Recommended.
  - One tree per bootstrap sample serves all causes; splits can use every cause's signal; one fit cost ≈ today's plus `O(K·J)` scoring.
  - Cost: Rust changes in data, profile, splitter, leaves and prediction; memory `4 + 8J` B per leaf event time.
- **B. J independent cause-specific forests** (fit `SurvivalForestTV` J times, treating other causes as censoring; combine hazards with Aalen–Johansen in Python).
  - No engine change; each forest is valid for its cause-specific hazard.
  - But J× fit cost and memory, and splits never share information across causes. Kept as a **baseline and test reference** (Validation 2).
- **C. Discrete-time person-period classification** (multinomial per bin). Different (discretised) estimand; row blow-up. Baseline only, via `make_person_period_data` with a cause-coded outcome.

## Data contract
- `y` fields `start`, `stop` (f8), `event` (**integer**, 0 = censored, 1..J). New helper `make_competing_risks_y(stop, event, start=None)`; `check_competing_risks_y` validates non-negative integers, at least one event, and records `causes_` (sorted observed labels, remapped internally to 1..J).
- Why a new class (C2): today `_as_bool` accepts integers {0,1} as a bool event, so "integer event" is ambiguous in `SurvivalForestTV`; its `predict` / `score` return one risk per row, which needs a cause choice under J > 1. `SurvivalForestTV` gets a clear error for integer events > 1 that points to the new class.
- `CompetingRisksForestTV` with J = 1 equals `SurvivalForestTV` (bit-identical predictions: a test).
- `ids`, contiguity, one event per id on its last row, `measured_at`, `gap_policy` — unchanged.

## Engine (Rust)
- `SurvData.event: Vec<u8>` cause codes (0..=J), `n_causes: usize`. The J = 1 path uses the same arrays, so single-event results stay identical.
- `Profile` gains a cause dimension; all-cause events stay a separate slice so existing criteria are untouched:
```rust
pub struct Profile<'a> {
    pub at_risk: &'a [f64],        // len K
    pub events: &'a [f64],         // len K, all causes
    pub cause_events: &'a [f64],   // len K*J, time-major: [k*J + j]; empty when J = 1
    pub n_causes: usize,
    pub times: &'a [f64], pub exposure: f64, pub n_units: f64,
}
```
- **Split search:** the at-risk difference array is unchanged (`bins × (K+1)`); events are accumulated into `bins × K × J` instead of `bins × K`. Scoring is `O(K·J)` per threshold. Coarsening moves the cause code with the event.
- **Composite criterion (C3):** per cause, the LTRC log-rank numerator and hypergeometric variance with "event" = cause j; the score is `Σ_j U_j² / V_j`. Unlike a signed sum of numerators (which is close to the all-cause log-rank and cancels opposing effects), this detects a covariate that raises one cause and lowers another.
```rust
impl SplitCriterion for CompositeCauseLogRank {
    fn score(&self, l: &Profile, p: &Profile, _: f64) -> f64 {
        let j_n = p.n_causes; let (mut num, mut var) = (vec![0.0; j_n], vec![0.0; j_n]);
        for k in 0..p.at_risk.len() {
            let (y, yl) = (p.at_risk[k], l.at_risk[k]);
            if y < 2.0 { continue; }
            let f = yl / y;
            for j in 0..j_n {
                let d = p.cause_events[k * j_n + j];
                if d == 0.0 { continue; }
                num[j] += l.cause_events[k * j_n + j] - d * f;
                var[j] += d * f * (1.0 - f) * (y - d) / (y - 1.0);
            }
        }
        num.iter().zip(&var).filter(|(_, v)| **v > 0.0).map(|(u, v)| u * u / v).sum()
    }
}
```
  - Bake-off candidates (S-bench slice): `"composite"` (above), the full quadratic form `Uᵀ V⁻¹ U` with the multivariate-hypergeometric cross-cause covariance, the Ishwaran et al. 2014 composite (exact form to be taken from the paper, `[K]` in research), the all-cause log-rank, and `split_cause=k`.
- **Leaves:** S9 flat layout with the per-entry value widened to J cause-specific cumulative hazards (`cumhaz[entry*J + j]`); all-cause hazard = Σ_j, not stored.
- **Leaf minimums (C5):** `min_events_leaf` counts events of any cause (default 3, as today). `min_events_leaf_cause: int | None = None` adds a per-cause floor for the cause given by `split_cause` only (a floor on every cause can make most splits infeasible for rare causes). Fit diagnostics report per-cause events per leaf. Rare-cause simulations in the bench slice decide whether a default is needed.

## Prediction
Per id and along its path, each row routes to its own leaf per tree, exactly as `predict_paths` today; per-cause increments `ΔΛ_j(t)` accumulate on the grid.
- **CIF from an origin `u`** (default: first `start`), discrete Aalen–Johansen:
  `S(t | u) = ∏_{u < v ≤ t} (1 − Σ_j ΔΛ_j(v))`, `F_k(t | u) = Σ_{u < v ≤ t} S(v− | u) ΔΛ_k(v)`.
  Clamp `1 − Σ_j ΔΛ_j` at 0 (a leaf with all its at-risk mass failing at `v`), with a count of clamps as a diagnostic.
- **Ensemble (C4):** `"hazard"` averages `Λ_j` over trees, then applies the formula; `"cif"` applies it per tree, then averages `F_k` and `S`. Both keep `Σ_k F_k + S = 1`.
- **Public methods** (same `times`, `intervals`, `ids`, `origin`, `extrapolate` keywords as today):
  - `predict_cumulative_incidence(X, times, *, cause=None, …)` → `(n, J, T)` or `(n, T)` for one cause.
  - `predict_cumulative_hazard(X, times, *, cause=None, …)` → cause-specific; `cause="all"` gives the all-cause hazard.
  - `predict_survival_function(…)` → event-free survival `S`.
  - `predict(X)` → risk score for `score_cause` (constructor param, default: the first cause label): `F_k` at the last event time (the analogue of today's mortality). `score` = Wolbers cause-specific C for `score_cause`.
- `extrapolate="none"` default (D9) unchanged; `"locf"` / scenario rows as today.

## Landmark workflow
- `make_landmark_data(…, event=…)` accepts a cause-coded event column: row `event = cause if T ≤ min(C, s+w) else 0`. Risk set and features unchanged (`U ≥ s`).
- `LandmarkCompetingRisksForest` (meta-estimator, wraps `CompetingRisksForestTV`) with `predict_risk(df_at_s, s, cause=k)` = `F_k(s+w | s, H(s))`. The clock is reset, so the AJ estimate is a direct CIF target (research §1).
- A shared private base with `LandmarkSurvivalForest` avoids duplicating the transform/CV plumbing.

## Metrics (C7)
- `brier_landmark(…, cause=k)`: CIF Brier with IPCW at `(s, w)`. Status at `s+w`: cause-k event (1), other-cause event or event-free (0), censored before `s+w` (IPCW-weighted out). **A competing event is an observed outcome**, weighted by `1/G(T−)`. `KaplanMeierCensoring` on the landmark risk set, `g_min` truncation and the exact (no-IPCW) path carry over. `integrated_brier` passes `cause`.
- `cindex_dynamic(…, cause=k)` and `concordance_index_cp(…, cause=k)`: Wolbers et al. 2014 — comparable pairs have a cause-k event first; an individual with a competing event stays a control (with IPCW in the dynamic version).
- Time-dependent AUC for competing risks (Blanche 2013): deferred.
- OOB: `oob_score_` = cause-specific C for `score_cause`; `oob_prediction_` holds `(n, J)` cause-specific risks.
- References in tests: `comprisk` metrics on right-censored data (dev dependency, optional); hand-computed small fixtures.

## Validation strategy (feeds the plan)
1. **Oracles:** a single-node tree equals `survival::survfit` Aalen–Johansen on `(start, stop]` rows with `id` (fixture exported to `tests/ref`); J = 1 equals `SurvivalForestTV` bit-for-bit; `split_cause=k` score equals the existing LTRC log-rank with other causes as censoring; the composite score equals a naive Python reference.
2. **Approach B equivalence:** with `max_features=p`, a fixed seed, `split_cause=k`, `min_events_leaf=1` and `min_events_leaf_cause=m` against the single-event forest with `min_events_leaf=m` (the leaf rules then coincide), each tree's cause-k cumulative hazard equals the single-event forest fitted on "cause k vs rest".
3. **Right-censored parity:** CIF accuracy close to `randomForestSRC` (`splitrule="logrank"`, i.e. its composite) and `comprisk` on a public dataset (e.g. `follic` / PBC with transplant as competing).
4. **Known-truth simulations:** cause-specific Cox hazards with TVCs and left truncation; opposing effects across causes; a rare cause (≤5% of events). Metrics: L2 to the true CIF; landmark CIF Brier.
5. **Bench slice:** criteria (C3), aggregation (C4), `min_events_leaf_cause` (C5); memory and time for J ∈ {2, 4} at 1M rows vs the J = 1 targets in `design.md`.

## Slice sketch (the plan stage refines this)
1. **S11** vertical core: cause-coded `y`, engine cause dimension, composite criterion, per-cause leaves, `CompetingRisksForestTV.fit` + `predict_cumulative_incidence` on a single interval per id; oracles 1.
2. **S12** paths, origins and `aggregate`; Approach-B equivalence; OOB with Wolbers C.
3. **S13** metrics (`cause=` in Brier / IBS / C) and the landmark CR meta-estimator.
4. **S14** bench: simulations, right-censored parity, criterion/aggregation/leaf-minimum bake-off; docs + case study.

## Risks
| Risk | Mitigation |
|---|---|
| Rare causes: unstable per-cause leaf increments, CIF jumps | per-cause diagnostics; `min_events_leaf_cause`; `"hazard"` aggregation smooths across trees; rare-cause sim |
| Memory `×(1+2J)/3` for leaves | report in bench; float32 cumhaz as a later option |
| Users read the CIF along internal-TVC paths as predictive | docs repeat D4: valid for external/specified paths; landmarking for dynamic prediction |
| `Σ U²/V` ignores cross-cause covariance (anti-conservative for common events) | only ranks splits (no p-values); quadratic form in bake-off |
| API sprawl (two estimators, two landmark wrappers) | shared private bases; one user-guide page |

## Out of scope
Subdistribution (Fine–Gray) targets and Gray splitting (C6); multi-state transition probabilities and recurrent events (C8 keeps the door open: cause codes → transition codes, rows grouped by from-state); competing-risks AUC; weighted criteria.
