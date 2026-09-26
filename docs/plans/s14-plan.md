# S14 slice plan: competing-risks bench, bake-off, docs, case study

Branch `feat/s14-cr-bench`. Parent: `cr-plan.md` S14 and P5, `cr-design.md` C3 / C4 / C5, "Validation strategy" 3–5. Precedent: `s8-plan.md` (a rule fixed before any run, and losers removed).
The S13 diff review returned after merge; its valid findings are fixed here in a separate commit.

## Decisions (defaults; revisit in review)

### Criteria prototyped behind `SplitCriterion` (P5)
All use `Profile.cause_events` (cause-major) and are selected by `criterion=`. They are **experimental** until the decision below.
- **`composite`** (C3 default): `Σ_j U_j² / V_j`.
- **`quadratic`**: `Uᵀ V⁻¹ U`, where `V` is the multivariate-hypergeometric covariance summed over times:
  - diagonal `d_j f(1−f)(y−d_j)/(y−1)`;
  - off-diagonal `−d_j d_k f(1−f)/(y−1)`, with `f = y_L / y` and `y ≥ 2`.
  - `V` can be singular: if every subject at risk at some time fails, the cause counts are tied together. The score therefore uses a **rank-aware pseudo-inverse**: an eigendecomposition of the `J × J` matrix, dropping eigenvalues below `1e-10 · λ_max`. Only causes with `V_jj > 0` enter.
- **`ishwaran`**: Ishwaran et al. 2014 eq. 3.2, as implemented in `randomForestSRC` (`splitSurv.c`, the cause-specific log-rank branch) with equal cause weights: `(Σ_j U_j)² / Σ_j V_j`.
  - `Σ_j U_j` is exactly the all-cause numerator, so this is the all-cause log-rank with a larger variance. It is expected to be blind to opposing effects.
- **`logrank_all`**: `LtrcLogRank` on the all-cause events (`Profile.events`).
- **`split_cause=k`**: already exists; it is scored on `F_k` for the split cause.

The J = 1 dispatch (P4a) is unchanged, so every criterion is `LtrcLogRank` for one cause.

### Simulations, known truth (`bench/s14_cr_sim.py`)
- **Data-generating process:** as in `tests/sim.py`, with external covariates, so the truth along a path is exact.
  - `z_k ~ N(0, 1)` is redrawn on each unit interval `k = 0..7`; the baselines are `x0 ~ N(0, 1)` and `x1 ~ Bernoulli(0.5)`; there are 3 noise features.
  - Cause-specific hazards are piecewise constant, `λ_j = a_j exp(β_jᵀ (z_k, x0, x1))`.
  - Censoring is `U(2, 8)`, with an administrative end at 8.
  - Training ids have delayed entry `U(0, 1)`: an id is kept only if it is event-free at entry, and its rows start at entry. Test paths start at 0.
- **Scenarios:**
  - **A standard:** cause 1 depends on `z` (0.8) and `x0` (0.5); cause 2 on `x1` (0.7) and `x0` (−0.3); `a = (0.12, 0.08)`.
  - **B opposing, with the all-cause hazard invariant in `z`** (plan review 3): `λ_1 = 0.2 · p(z) · e^{0.3 x0}` and `λ_2 = 0.2 · (1 − p(z)) · e^{0.3 x0}`, with `p(z) = logistic(1.8 z)`.
    - The total `0.2 e^{0.3 x0}` does not depend on `z`: `z` only allocates the cause, so the all-cause log-rank has no `z` signal in the population.
    - This is checked before the runs: on a 20k-id draw, the population all-cause log-rank for a median split on `z` is < 2, and the composite is > 50.
  - **C rare:** J = 3; causes 1–2 as in A, plus cause 3 on `x1` (1.0).
    - `a_3` is set, before the runs, from a 50k-id draw so that cause 3 is **3–5% of events**.
    - 800 training ids, so a replicate expects ≥ 15 cause-3 training events.
    - A replicate with < 10 realized cause-3 training events is reported but left out of the C5 comparison; the count is reported.
- **Setting:** **30** replications per scenario (seed = replication), 500 training ids (800 in C), 200 test ids, 200 trees, defaults otherwise.
  - S3/S8 used 20. The observed paired SEs there (≈ 10–15% of the mean ISE) mean 30 replications resolve differences of about ≥ 5% of ISE at 2 SE.
- **Metric:** per cause, the ISE of `F_k(t | path)` against the truth over `t ∈ [0, 6]` (61 points, trapezoid rule), averaged over test subjects. The truth is the closed-form Aalen–Johansen of the piecewise-constant hazards on the path.
  - Reported per cause and as the mean over causes (the decision uses the mean; scenario C also reports cause 3 alone).
  - Differences are paired over replications (mean ± SE).
- **Landmark simulation** (plan review 6; it validates the CR IPCW metric against the truth and is **reported, not voting**):
  - Landmarks `s ∈ {1, 2, 3}`, horizon `w = 2`. `LandmarkCompetingRisksForest` uses `z` (last value), `x0` and `x1`.
  - The true `F_k(s + w | s, H(s))` is the Monte Carlo expectation over future `z` draws (iid per interval, independent of the past; 4000 draws per test subject).
  - Reported: the IPCW cause-specific Brier at `w` on censored test data vs the oracle Brier on the uncensored test outcomes (the simulated event times are known), and the ISE of `F_k` against the truth, both aggregations.
- **Arms:** 4 criteria × `aggregate ∈ {hazard, cif}`, plus reported-only arms:
  - `split_cause = k` per cause (each scored on its `F_k`);
  - **Approach B:** J `SurvivalForestTV` fits on "k vs rest", combined by Aalen–Johansen in Python;
  - scenario C only: `split_cause=3` with `min_events_leaf_cause ∈ {None, 3, 5, 10}` and `min_events_leaf=1`.

### Right-censored parity (`bench/s14_cr_parity.py` + `bench/s14_cr_parity.R`)
- **Data:** `survival::pbc` baseline (418 ids; status 0 = censored, 1 = transplant, 2 = death), 17 covariates with median imputation.
- **Protocol:** 5-fold CV with fixed folds, written to CSV so R and Python use the same splits.
- **Arms:**
  - rftvc `composite` and the winner, hazard and cif;
  - Approach B;
  - `randomForestSRC` 3.9 (`splitrule="logrank"`, `ntree=500`, `nodesize=15`), installed in a scratch R library as in `bench/cr_rfsrc_check.R`;
  - Aalen–Johansen without covariates, as a floor.
- **Settings, matched to `randomForestSRC` defaults** (plan review 5):
  - rfsrc: `samptype="swor"`, `sampsize = 0.632 n`, `nodesize=15`, `mtry = ceil(√p) = 5`, and **`nsplit=0`** (every split point; its default of 10 random split points would not be comparable), `ntree=500`, `seed=-1`.
  - rftvc: the same subsampling (its default, 0.632 without replacement), `min_ids_leaf=15`, `max_features=5`, `min_events_leaf=1` (rfsrc has no event minimum), 500 trees, `random_state=0`, `max_bins=256`.
  - Leaf-size semantics differ (rfsrc: minimum node size; rftvc: minimum per child). This is stated in the write-up.
- **Censoring and IPCW:** `G` is a reverse KM fitted per test fold on that fold's outcomes (as in landmark CV), with `g_min = 0.05`; clipped counts are reported. The bootstrap resamples subjects of the pooled out-of-fold predictions with the per-fold weights fixed, and `G` is not refitted.
- **Metrics:**
  - cause-specific IPCW integrated Brier of `F_k` over `t ∈ (0, 3650]` (10 times), both causes;
  - Wolbers C of `F_k(3650)` (`concordance_index_cr`).
- **Uncertainty:** pooled out-of-fold predictions, with a subject bootstrap (B = 500) of paired IBS differences against rftvc `composite`.
- rfsrc is **reported, not voting** (it cannot do counting-process competing risks; parity only).

### Pre-registered decision rule (fixed before any run)
The rule is **descriptive and conservative**, as in S8. Many comparisons are made (challengers × scenarios × causes × aggregations × `m`), and no family-wise error claim is made. A change of default needs consistent evidence; otherwise the default stays.
- **Order (plan review 1, not circular):** C4 is decided first, on `composite` only. C3 is then decided under that aggregation, and the conclusion must not reverse under the other one.
- **C4 aggregation:** `"cif"` becomes the default only if **with `composite`**:
  - it beats `"hazard"` beyond 2 SE (mean-over-causes ISE) in at least one scenario;
  - it is not worse beyond 2 SE in any scenario;
  - the PBC IBS interval does not favour `"hazard"`.

  Otherwise `"hazard"` stays.
- **C3 criterion:** a challenger replaces `composite` only if all four hold. Otherwise `composite` stays.
  - (i) Under the C4 aggregation, its mean-over-causes paired ISE beats `composite` beyond 2 SE in at least one scenario, and is not worse beyond 2 SE in any.
  - (ii) Under the other aggregation, it is not worse beyond 2 SE in any scenario.
  - (iii) Its PBC parity IBS bootstrap interval (each cause) does not favour `composite`.
  - (iv) Its fit time is ≤ 2× `composite`'s.
- **C5 `min_events_leaf_cause`:** a default is set only if some `m` beats `None` on scenario C's cause-3 ISE beyond 2 SE, and is not worse beyond 2 SE on causes 1–2. Otherwise it stays `None`, and the docs give the evidence.
- **P5:** losing criteria are removed from Rust, the binding and Python after the write-up; their history stays in git. `composite` and `split_cause` stay.

### Scale (`bench/s14_cr_scale.py`)
- **Setup:** `bench/perf_fit.synth` counting-process data (5 rows per id). For J > 1, each event gets a uniformly random cause label in `1..J` (as in `bench/s11_cr_timing.py`), and the J = 1 arm is `SurvivalForestTV`. 100 trees, 10 threads, `ntime=100` and exact.
- **Sizes:** 100k and 1M rows, each fit in its own process under `/usr/bin/time -l`.
- **Measured:** fit time and peak RSS. **Leaf bytes** come from the pickled state: `4·E + 8·J·E` (`event_idx` + `cumhaz`, where `E` = total leaf entries) plus `4·J·L` leaf counts (J > 1). **Node bytes** are `forest_.nbytes` minus leaf bytes.
- **Accept (plan review 8):** these limits are pre-set from S11/S12's 100k measurements (J = 2: 1.43×, J = 4: 2.20× the J = 1 time), and they apply in both modes and at both sizes:
  - leaf bytes(J) / leaf bytes(J = 1) ≤ `(1 + 2J)/3 · E_J / E_1 + 4J·L_J / (12 E_1)`: the entry-count ratio `E_J / E_1` is reported, since J > 1 splits differently;
  - fit time ≤ 1.6× (J = 2) and ≤ 2.6× (J = 4) the J = 1 time;
  - peak RSS ≤ 1.6× / 2.6×.

### Docs and case study
- **User guide:** a `user_guide/competing_risks.rst` page covering:
  - targets (cause-specific hazards; CIF by Aalen–Johansen);
  - `causes` / labels in CV;
  - criteria (`composite`, `split_cause`, the bake-off result) and `aggregate`;
  - paths valid only for external covariates (D4), and landmarking for dynamic prediction;
  - metrics (cause-specific Brier / IBS / Wolbers C; the type-A weight note).
- **Reference pages:** `api.rst` (done in S11–S13, checked); `compatibility.rst` (CR checks).
- **Case study:** `examples/pbc2_competing.py` (landmark competing-risks CV on `pbcseq`: transplant vs death) plus a section in `case_studies/pbc2.rst`.
- **Results:** `docs/bench/s14-cr.md` with CSVs in `docs/bench/s14-cr/`, and the recommendation.

## Tests
- **New criteria** (while they exist): Rust unit tests for `quadratic` against a naive covariance build and solve, `ishwaran` against the formula, and `logrank_all` equal to `LtrcLogRank` on all-cause events. Python hypothesis tests against `cr_ref` extensions. With J = 1, every criterion equals `LtrcLogRank` bit-for-bit.
- **Sim truth:** the closed-form path CIF equals a fine-grid numerical integral (1e-6), and a single-leaf forest on a no-covariate scenario approaches the truth.
- **After P5 removal:** the tests of removed criteria go too; the remaining suite is green.
- The benchmarks are not merge gates.

## Acceptance
- A written recommendation with C3 / C4 / C5 confirmed or changed, and the non-winners removed.
- Docs build with `-W`.
- The memory bound holds, or the deviation is explained.
- The identity bench is bit-identical (`--pickle`).

## Tasks
- [x] S13 diff-review findings (separate commit)
- [x] Criteria prototypes (Rust + binding + Python) + tests
- [x] `bench/s14_cr_sim.py` (+ truth tests), run, CSV
- [x] Parity: folds, R script (rfsrc scratch lib), Python arms, run, CSV
- [x] Scale bench, run
- [x] Decision per the rule; `docs/bench/s14-cr.md`; remove losers (P5)
- [x] Docs: user guide, compatibility, case study + example
- [x] Identity bench; `cr-plan.md` status + "S14 done"; `plan.md` S11–S14 line
- [ ] Codex diff review; fixes

## Plan review (Codex, 2026-09-26)
1. C3 was judged "under the chosen aggregation" while C4 chose it from the same runs (blocker) → C4 is decided on `composite` first; C3 is then decided under it and must not reverse under the other aggregation.
2. The quadratic covariance can be singular, e.g. when the whole risk set fails (high) → an eigen-based pseudo-inverse with a tolerance. Singular fixtures in the tests.
3. Scenario B's all-cause hazard still varied with `|z|` (high) → B redesigned so the total hazard is invariant in `z`, and checked before the runs.
4. There was no multiplicity handling (high) → S8's descriptive, conservative framing is adopted, and 30 replications are justified.
5. The parity IPCW protocol and arm settings were unspecified (high) → per-fold reverse KM, `g_min`, fixed-weight bootstrap, and rfsrc / rftvc settings matched (`nsplit=0`, `mtry=5`, 0.632 subsampling).
6. The landmark CIF Brier from cr-plan was missing (medium) → a landmark simulation with a Monte Carlo truth; IPCW Brier vs the oracle Brier, reported only.
7. The rare-cause rate had no floor (medium) → a 3–5% target, 800 ids, and the exclusion rule for replicates with < 10 cause-3 events.
8. The scale acceptance was not measurable (medium) → a defined label generator, leaf bytes separated from node bytes, and numeric time and RSS limits.

## Results and deviations (2026-09-26)
- Decisions per the rule: **C3 `composite` (unchanged), C4 `aggregate="cif"` (changed), C5 `min_events_leaf_cause=None` (unchanged)**. Write-up: `docs/bench/s14-cr.md`. The losers were removed (P5); their arms rerun at `66ccad8`.
- On continuous times, `quadratic` ≡ `composite` and `ishwaran` ≡ `logrank_all` (no tied cause events). This was noted, and the pre-registered rule was not changed.
- The scenario B pre-check value was 2.06 against a threshold of "< 2" on a single draw. The population value is 0 by construction, so 2.06 is χ²₁ noise; it was recorded, not re-drawn.
- The C5 arms first recorded only cause 3. The rule needs causes 1–2 too, so they were rerun with all causes (seeds fixed; other arms identical).
- The planned "single-leaf forest approaches the truth" test was replaced by a stronger DGP test: the simulated event times match the closed-form truth (40k draws).
- The landmark simulation has 10 replications (reported only).
- The case study uses `landmark_cross_validate` with a single-leaf-tree Aalen–Johansen reference.

## Test-quality audit (2026-09-26)
- The challenger criteria were tested against independent references (a numpy pinv quadratic form with a singular fixture, an Ishwaran formula, the all-cause log-rank ref) and for J = 1 bit identity. They were removed with the criteria.
- The simulation truth is tested against numerical integration (1e-9) and empirical CIFs of the DGP.
- The default switch changed two tests that pin the hazard transform, which now set `aggregate="hazard"` explicitly. The J = 1 identity tests are unchanged.
