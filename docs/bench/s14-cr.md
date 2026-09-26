# S14: competing-risks bench, bake-off and recommendation

Plan and pre-registered rule: `docs/plans/s14-plan.md` (Codex plan review applied before any run). Raw results: `docs/bench/s14-cr/*.csv`. Scripts: `bench/s14_cr_{sim,summary,parity,landmark_sim,scale}.py`, `bench/s14_cr_parity.R`. The challenger arms run at commit `66ccad8`; the challengers were removed afterwards (P5).

## Recommendation
| Decision | Result | Default |
|---|---|---|
| C3 split criterion | no challenger beats `composite`; `ishwaran` / `logrank_all` are 12–26% worse; `quadratic` ≡ `composite` on continuous times | **`composite`** (unchanged) |
| C4 aggregation | `cif` better with opposing effects (−4.0% ISE, z = −6.0), not worse elsewhere; PBC intervals include 0 | **`aggregate="cif"`** (changed from `hazard`) |
| C5 per-cause leaf floor | improves the rare target cause (z ≈ −3.7) but harms the other causes badly (z = +5 to +32) | **`min_events_leaf_cause=None`** (unchanged); use it only when the rare cause is the target |

The rule is descriptive and conservative, as in S8, with no family-wise error claim.

## Criterion finding from the source
`randomForestSRC`'s composite cause-specific log-rank (Ishwaran et al. 2014, eq. 3.2; `splitSurv.c`) is `|Σ_j w_j U_j| / sqrt(Σ_j w_j² V_j)`. With the default equal weights, `Σ_j U_j` is exactly the all-cause numerator, so the rule cannot see a covariate that shifts cause allocation without changing the all-cause hazard. rftvc's `composite` (`Σ_j U_j² / V_j`) can.

With continuous event times, two challengers coincide with other rules:
- The cross-cause covariance is non-zero only when two causes fail at the same time, so `quadratic` (`Uᵀ V⁺ U`) equals `composite`: the ISE differences are ~1e-7, and the fit is 2× slower.
- With one event per time, the all-cause variance equals `Σ_j V_j`, so `ishwaran` equals `logrank_all`.

They can differ only on tied (e.g. coarsened) times.

## 1. Simulations, known truth (30 replications per scenario)
External TVC `z`, baselines `x0` and `x1`, three noise features, delayed entry in training, closed-form path CIF (`tests/test_cr_sim_truth.py`). Metric: ISE of `F_k(t | path)` over [0, 6], mean over causes; differences are paired over replications.

Pre-run checks:
- **Scenario B:** its all-cause hazard is exactly invariant in `z`. On a 20k-id draw, the all-cause log-rank on a `z` median split was 2.06 (pre-set threshold < 2; population value 0, so this is χ²₁ noise, p ≈ 0.15; recorded, not re-drawn), and the composite was 2945.
- **Scenario C:** `a₃ = 0.007` gives cause 3 at 4.0% of events. Cause-3 training events per replicate were 14 minimum and 21.5 median, with 30/30 replicates ≥ 10.

Mean ISE (mean over causes):

| arm | A standard | B opposing | C rare |
|---|---|---|---|
| composite / hazard | 0.04204 | 0.02903 | 0.02361 |
| composite / cif | 0.04182 | **0.02787** | 0.02359 |
| quadratic / hazard | 0.04204 | 0.02903 | 0.02361 |
| ishwaran / hazard | 0.05164 | 0.03242 | 0.02659 |
| logrank_all / hazard | 0.05164 | 0.03242 | 0.02659 |
| Approach B (J survival forests + AJ) | 0.04283 | 0.03035 | 0.02314 |

**C4 (`composite` only), cif − hazard:**

| scenario | diff | SE | z | relative |
|---|---|---|---|---|
| A | −0.00021 | 0.00034 | −0.6 | −0.5% |
| B | −0.00116 | 0.00020 | −6.0 | −4.0% |
| C | −0.00002 | 0.00017 | −0.1 | −0.1% |

**C3, challenger − `composite`** (under `cif`; the conclusion is the same under `hazard`):

| scenario | quadratic | ishwaran | logrank_all | Approach B |
|---|---|---|---|---|
| A | −7e-8 (z −1.0) | +0.0111 (z 11.6) | +0.0111 (z 11.6) | +0.0010 (z 1.6) |
| B | −2e-7 (z −1.3) | +0.0038 (z 8.0) | +0.0038 (z 8.0) | +0.0025 (z 6.7) |
| C | +6e-7 (z 1.3) | +0.0038 (z 7.0) | +0.0038 (z 7.0) | −0.0005 (z −1.1) |

**C5 (scenario C):** `split_cause=3`, `min_events_leaf=1`, `min_events_leaf_cause=m` vs `m=None`, paired:

| m | cause 3 | cause 1 | cause 2 |
|---|---|---|---|
| 3 | −0.0019 (z −3.7) | +0.0334 (z 19.0) | +0.0174 (z 4.9) |
| 5 | −0.0021 (z −3.8) | +0.0488 (z 22.7) | +0.0268 (z 4.9) |
| 10 | −0.0021 (z −3.3) | +0.0749 (z 32.2) | +0.0387 (z 6.7) |

A floor on the rare cause stops the forest splitting where that cause is too sparse, which is where the other causes' structure is. The rule's "not worse on causes 1–2" condition fails for every `m`.

## 2. Right-censored parity: `survival::pbc` (transplant 1, death 2)
418 ids and 17 covariates, with missing values median-imputed **inside each fold from the training fold only** (diff review 1). There are 5 fixed folds, shared with R, which gets the same per-fold imputed rows.

The settings match rfsrc's defaults: 0.632 subsampling without replacement, leaf size 15, `mtry = 5`, 500 trees, and rfsrc `nsplit=0` (every split point).

IPCW uses a reverse KM per test fold with `g_min = 0.05`. **No weight was clipped:** 0 of 6,392 weighted subject-times (`parity_clipping.csv`). The IBS is over 365–3650 days, pooled over folds; the intervals are a subject bootstrap (B = 500) with fixed weights. The difference is vs rftvc composite/hazard.

| arm | cause | IBS | diff [95% interval] | Wolbers C | fit s (5 folds) |
|---|---|---|---|---|---|
| composite / hazard | transplant | 0.0412 | — | 0.790 | 0.15 |
| composite / cif | transplant | 0.0412 | +0.00002 [−0.0003, 0.0003] | 0.809 | 0.15 |
| Approach B | transplant | 0.0417 | +0.0006 [−0.0001, 0.0012] | 0.788 | 0.18 |
| randomForestSRC 3.9 | transplant | 0.0433 | +0.0021 [0.0012, 0.0032] | 0.727 | 5.4 (incl. R start) |
| Aalen–Johansen (no covariates) | transplant | 0.0444 | +0.0032 [0.0011, 0.0055] | 0.500 | — |
| composite / hazard | death | 0.1363 | — | 0.822 | |
| composite / cif | death | 0.1351 | −0.0013 [−0.0037, 0.0012] | 0.824 | |
| Approach B | death | 0.1373 | +0.0010 [−0.0000, 0.0020] | 0.823 | |
| randomForestSRC 3.9 | death | 0.1385 | +0.0021 [−0.0034, 0.0079] | 0.814 | |
| Aalen–Johansen (no covariates) | death | 0.1960 | +0.0596 [0.0449, 0.0744] | 0.500 | |

- **C4:** neither `cif` interval favours `hazard`, so the switch rule holds.
- **vs rfsrc (reported only):** rftvc is better on transplant (the interval excludes 0) and similar on death. Leaf-size semantics differ: rfsrc's `nodesize` is a node minimum, rftvc's `min_ids_leaf` a per-child minimum.
- **Approach B:** close to the joint forest on PBC; its intervals just include 0.

## 3. Landmark simulation (reported only; 10 replications)
Scenario A without delayed entry. Landmarks 1, 2, 3; `w = 2`; `z` (last), `x0`, `x1`. The truth is the Monte Carlo mean over the unknown next `z` (4000 draws).

| aggregate | cause | MSE to truth at s + w | IPCW Brier | oracle Brier (uncensored) | mean abs gap |
|---|---|---|---|---|---|
| cif | 1 | 0.00441 | 0.1734 | 0.1734 | 0.0051 |
| cif | 2 | 0.00296 | 0.1443 | 0.1454 | 0.0062 |
| hazard | 1 | 0.00475 | 0.1739 | 0.1739 | 0.0052 |
| hazard | 2 | 0.00301 | 0.1444 | 0.1455 | 0.0061 |

In this simulation, the cause-specific IPCW Brier tracked the oracle Brier (on the true, uncensored outcomes): the means agree to 0.001. The per-landmark gaps average 0.005–0.006. This is 10 replications and supports the metric; it does not prove unbiasedness. `cif` is slightly closer to the truth.

## 4. Scale (100 trees, 10 threads, 5 rows per id; each fit in its own process)
| rows | grid | J | fit s | × J = 1 | peak RSS GB | × J = 1 | leaf bytes × J = 1 (bound) |
|---|---|---|---|---|---|---|---|
| 100k | ntime=100 | 1 / 2 / 4 | 0.99 / 1.18 / 1.52 | 1 / 1.19 / 1.53 | 0.27 / 0.28 / 0.32 | 1 / 1.06 / 1.18 | 1 / 1.78 / 3.23 (= bound) |
| 100k | exact | 1 / 2 / 4 | 2.50 / 3.47 / 5.22 | 1 / 1.39 / 2.09 | 0.28 / 0.30 / 0.34 | 1 / 1.07 / 1.19 | 1 / 1.77 / 3.21 (= bound) |
| 1M | ntime=100 | 1 / 2 / 4 | 14.4 / 16.5 / 18.7 | 1 / 1.14 / 1.30 | 1.09 / 1.27 / 1.53 | 1 / 1.17 / 1.41 | 1 / 1.79 / 3.27 (= bound) |
| 1M | exact | 1 / 2 / 4 | 38.0 / 51.9 / 81.1 | 1 / 1.37 / 2.14 | 0.98 / 1.17 / 1.46 | 1 / 1.19 / 1.49 | 1 / 1.79 / 3.25 (= bound) |

- **Accept:**
  - fit time ≤ 1.6× (J = 2) and ≤ 2.6× (J = 4);
  - peak RSS within the same limits;
  - leaf bytes equal to `(1 + 2J)/3 · E_J/E_1` plus the leaf counts.

  All hold.
- In exact mode the entry count `E` is identical across J (812,772 at 100k). Each distinct event time lands in exactly one leaf per tree, so `E` is the in-bag event count, whatever the split rule.
- The single-event forest meets the design targets (1M exact 38 s < 180 s; `ntime=100` 14 s < 35 s; RSS 1.1 GB < 3.5 GB).
