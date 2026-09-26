# S8 slice plan: split-criterion and aggregation bake-off

Branch `feat/s8-bakeoff`. Parent: `plan.md` S8; design.md D6 (criterion), D11 (aggregation), "Log-rank optimises separation, not calibration" risk; design-principles.md "Proper scores first".

## Decisions (defaults; revised after plan review)
1. **Prototype the criteria in Rust behind `SplitCriterion`, not in numba** (deviation from plan.md).
   - All candidates are cheap functions of per-node summaries. The existing splitter already builds these summaries per candidate split, including LTRC risk sets, the coarse grid and exact distinct-unit counts.
   - A numba prototype would need its own tree builder, so the bake-off would compare implementations, not criteria.
   - Each criterion is about 50 lines of Rust. The winner is then already ported, which removes plan.md's "its own slice if substantial".
2. **Splitter data for the new criteria** (log-rank reads nothing new and stays bit-identical; the existing fixtures and oracle tests are the guard):
   - **Durations:** `SurvData` gains `duration: Vec<f64>` = `stop − start`. In coarse mode this is computed from the **snapped** `start` / `stop`, not the originals.
   - **Exposure:** the splitter keeps `parent_exposure` and accumulates `left_exposure` as each bin enters the left child; right = parent − left. This is an `O(1)` add per row, so the `O(n + bins·K_node)` bound holds.
   - **`Profile` gains:**
     - `times: &[f64]`: node event times;
     - `exposure: f64`;
     - `n_units: f64`: distinct resampling units, taken from the existing `min_ids_leaf` counts. A unit with rows in both children counts in both, the same convention as `min_ids_leaf`.
3. **Candidates** (split score = child fit − parent fit; higher is better):
   - **C0 `logrank`**: the current LTRC log-rank (baseline, D6).
   - **C1 `grouped_lik`**: saturated grouped-time binomial log-likelihood over the node's event grid.
     - Formula: `Σ_k d log q + (y − d) log(1 − q)`, with `q = d/y` the discrete event probability (the KM decrement), not the Nelson–Aalen increment. The score is `ll_L + ll_R − ll_P` ≥ 0, with `0·log 0 = 0` (this covers `d = 0` and `d = y`).
     - Tie model: all failures at a grid time form one binomial group. This is the grouped-data convention and also covers the coarse grid.
     - Event-free times contribute 0, so node-local grids lose nothing.
   - **C2 `poisson`**: a constant hazard per node with person-time exposure (an exposure likelihood, not a risk-set statistic).
     - Formula: `ll = D log(D/E) − D`. The score is the child sum minus the parent, ≥ 0, with `0·log 0 = 0`.
     - Exposure is exact for counting-process rows routed by their covariates.
   - **C3 `km_gini`** (heuristic; **not** a Brier objective). Codex showed that a Bernoulli variance of the KM is not the IPCW Brier.
     - Node impurity: `n_units · S(τ)(1 − S(τ))`, with `S` the delayed-entry KM from the profile. The score is the impurity decrease.
     - `S(τ)` includes events at `t_k ≤ τ` (right-continuous, the project convention). `S = 1` before the node's first event.
     - `τ = criterion_horizon` is on the **reset (time-since-landmark) clock**, so C3 is run only on landmark-built data.
     - An IPCW squared-loss criterion is out of scope. It would be its own slice if `km_gini` looks promising.
4. **Public knob:** `SurvivalForestTV(split_criterion="logrank", criterion_horizon=None)`.
   - `criterion_horizon` is required iff `split_criterion="km_gini"`.
   - `LandmarkSurvivalForest` gets no duplicate params: tune it via `forest__split_criterion` / `forest__criterion_horizon` (the existing nested-param structure).
   - Fitted Rust tree state does not store the criterion, since prediction doesn't need it. Python pickling carries the estimator params as usual.
   - The params are marked experimental in the docs until item 9.
5. **Two protocols** (Codex: not every dataset fits `landmark_cross_validate`; C&L has gaps that the landmark builder doesn't support):
   - **A. Landmark protocol:** nested `landmark_cross_validate` on the reset clock, with metrics IBS over `[0, w]` and calibration at `w`.
     - **PBC2** (`pbcseq`): landmarks 1–4 y, `w = 2` y, new patients (`GroupKFold`).
     - **Panel simulation** (`tests/sim_panel.py`): the S5 set-up, `RollingOriginSplit` (future periods).
   - **B. Core-forest protocol:** `SurvivalForestTV` on counting-process rows.
     - **Simulation, known truth** (`tests/sim.py`): 20 independent replications. Metric: path ISE to the true S. C3 is excluded (no landmark clock).
     - **Cunningham & Lemke:** new-war `GroupKFold`, counting-process C. This is **supporting evidence only**, not in the decision (no landmark endpoint).
   - GBSG2 is only a sanity row.
   - Every arm is 4 criteria (3 in B) × `aggregate` ∈ {`hazard`, `survival`}, with `n_estimators=300` and `ntime=None`. `ntime=100` is checked on the winner only.
6. **Per-arm tuning:**
   - The inner loop tunes `min_ids_leaf` × `max_features` separately for each arm (landmark: `forest__…`; core: the same grid with the same inner splitter).
   - Outer CV: one 5-fold split per dataset. There are no repeats, because repeated CV understates the SE.
   - Seeds are fixed and recorded.
7. **Outputs and uncertainty (pre-specified):**
   - **Out-of-fold predictions:** `landmark_cross_validate(..., return_predictions=True)` returns them as `(fold, landmark, id, risk_at_w, survival on the IBS grid, y)` (small, tested library addition). The bench computes every metric from these.
   - **Calibration:**
     - uses `calibration_table` at `w`, pooled over landmarks, with 10 risk deciles;
     - ICI = the bin-size-weighted mean of `|KM-observed − predicted|`, where censoring is handled by the per-bin KM;
     - the slope is a weighted least-squares fit of observed on predicted over bins, with an undefined fold/bin reported as NA rather than dropped.
   - **Uncertainty:**
     - simulation: paired differences over the 20 independent replications (mean ± SE);
     - real / landmark data: a subject-cluster bootstrap (B = 500) of paired differences in pooled out-of-fold IBS / ICI. The percentile 95% interval is the unit.
   - **Multiplicity:** the rule below is **descriptive and conservative**. It makes no family-wise error claim.
8. **Pre-registered decision rule** (written before any run; the τ / `w` per dataset are fixed above):
   - **Criterion:** a challenger replaces `logrank` as the default only if all four hold:
     - (i) the paired ISE difference on `sim.py` favours it, beyond 2 SE over the replications;
     - (ii) on PBC2 and the panel, its IBS bootstrap interval does not favour log-rank and at least one excludes 0 in the challenger's favour;
     - (iii) no ICI interval favours log-rank;
     - (iv) fit time ≤ 2× log-rank's, and the conclusion is the same under both aggregations.

     Otherwise `logrank` stays the default. C&L is reported but does not vote.
   - **D11:** switch to `survival` only if its ICI interval beats `hazard` on both landmark datasets and its IBS is not worse. Otherwise `hazard` stays (it keeps conditional survival consistent across origins).
9. **After the decision:** only `logrank` and the winner (if any) stay public. Challengers that lost are removed and kept in git history; the write-up reports all arms.
10. **Files:**
    - scripts: `bench/criteria/` (`run.py` grid driver, `protocol_a.py`, `protocol_b.py`, `summarise.py`);
    - raw results: CSV in `docs/bench/s8-bakeoff/`;
    - write-up: `docs/bench/s8-bakeoff.md`, which follows the `docs/bench/` convention (deviation from `docs/scratch/bakeoff.md`).

    The full runs are not merge gates.
11. **Out of scope:**
    - an IPCW-Brier split criterion;
    - a gap-aware landmark builder;
    - R comparators (LTRCforests, randomForestRHF);
    - weighted criteria;
    - leaf-storage slimming (the S6 deferral stays);
    - O(1) log-rank updates.

## Tasks
- [x] `SurvData.duration` (snapped in coarse mode), `Profile.{times, exposure, n_units}`, splitter exposure accumulation. Log-rank is unchanged (existing tests green). New tests cover exposure in exact vs coarse mode, delayed entry, an empty-event child and a unit straddling both children.
  - Deviation: the right child's unit count cannot be derived from the parent and left (units may straddle), so `SplitCriterion::score` and `NodeScorer::score` take `n_units_right`.
  - Tests: a spy criterion checks every candidate's at-risk/events/times/exposure/units against brute force (300 random delayed-entry nodes, with straddling units and event-free children asserted to occur; a mutation of the exposure sum fails it); coarse exposure uses snapped times.
  - Log-rank predictions are bit-identical to `main` (exact, `ntime=20`, bootstrap). Fit time at 100k rows × 100 trees: +3% exact (2.66 vs 2.58 s), +4% coarse (0.94 vs 0.90 s), medians of 5.
- [x] `tests/ref/criteria_ref.py`: naive references for C1–C3, and a `_core.criterion_score` binding. Parity cases:
  - delayed entry, all-event and tied-event nodes;
  - C3 with τ before the first event, at a tie and after the last event;
  - invariants: L/R swap symmetry, zero gain for identical children, gain ≥ 0 for C1/C2, `0·log 0`.
  - Done: `GroupedLik`, `PoissonExposure`, `KmGini` in `criterion.rs`, plus `criterion(name, horizon)` (validates the horizon: required and finite > 0 for `km_gini`, rejected otherwise). Rust hand-computed tests (`tests/criteria.rs`: C1 gain = 4 ln 2, C3 at τ inside/at/after the tie). `tests/test_criteria.py` (15 tests, hypothesis parity incl. log-rank) plus a lifelines check of the delayed-entry KM reference. Three mutants (τ exclusive, wrong right exposure, no straddling) each fail 2–5 tests.
  - Note: the `km_gini` gain can be negative (straddling units count twice), so non-negativity is asserted only for C1/C2. The splitter already ignores scores ≤ 0.
- [x] `split_criterion` / `criterion_horizon` on `SurvivalForestTV` → `fit_forest_py`. Tests cover validation (`km_gini` without a horizon raises), `clone`, `get_params(deep=True)` via `forest__…`, nested-CV tuning of `forest__split_criterion`, and a fitted pickle round-trip. Re-run the compatibility matrix.
  - Done: `fit_forest(…, criterion)` in Rust. `_core.fit_forest` / `_core.best_split` take `split_criterion` / `criterion` + horizon, and Rust re-validates. Python validates in `_validate_params` with the same messages.
  - Tests (10 new in `tests/test_criteria.py`):
    - hypothesis check that `best_split` equals the brute-force argmax of the naive reference under all 4 criteria on counting-process data with units. It is non-vacuous: a split is found in 40/40 random cases (26/40 for `km_gini`);
    - param validation;
    - each criterion grows different, deterministic trees;
    - clone / pickle / `forest__` nested params;
    - nested `landmark_cross_validate` tuning `forest__split_criterion`, including `km_gini` + horizon.
  - Compatibility matrix unchanged (33 passed, 24 xfailed). Default predictions are still bit-identical to `main`. `sphinx-build -W` is clean.
- [x] `landmark_cross_validate(return_predictions=True)` + tests
  - Returns `(scores, predictions)`. Predictions hold one row per scored test row: `fold, landmark, id, time, event` (reset clock), `risk = 1 − S(w)`, and `survival` (a list over the `n_times` IBS grid). The default return is unchanged. `_score_landmark` now takes the predicted `S`, so each fold/landmark is predicted once.
  - Tests (3):
    - every fold/landmark `brier` and `integrated_brier` is reproduced from the returned rows alone;
    - under `GroupKFold`, every landmark row is predicted exactly once and each id falls in one fold;
    - the default return is unchanged, and nested CV also returns predictions.
- [x] `bench/criteria/`: protocols A and B, calibration summaries, cluster bootstrap, CSV output; smoke config runs in seconds
  - Files: `common.py` (config, metrics, bootstrap), `protocol_a.py`, `protocol_b.py`, `run.py` (`--smoke`, `--datasets`, `--out`), `summarise.py` (applies the rule mechanically → `decisions.csv`). The smoke run takes about 4 s. Raw out-of-fold predictions go to `docs/bench/s8-bakeoff/raw/` (git-ignored).
  - **Pre-registered details fixed in `common.py` before any full run:**
    - arms: 4 criteria × 2 aggregations (A); 3 × 2 (B, no `km_gini`);
    - 300 trees; grid `min_ids_leaf` ∈ {5, 15, 50} × `max_features` ∈ {sqrt, None};
    - seed 0 for every arm (paired), IBS grid of 10 points, 10 calibration deciles, B = 500;
    - A tunes by inner mean `integrated_brier`. PBC2: outer `GroupKFold(5)`, inner `GroupKFold(3)`, landmarks 1–4 y (no training landmark at 0, unlike the case study), `w = τ = 730.5` days. Panel: 2,000 units × 96 periods, landmarks every 3 periods, `w = τ = 6`; `RollingOriginSplit(5 outer / 3 inner, test_size=6, gap=6)`. The S5 12-period windows would leave the inner folds only landmarks 0–3 to train on;
    - B tunes by inner `GroupKFold(3)` counting-process C;
    - fit time = the median of 3 full-data fits at `min_ids_leaf=15`, `max_features="sqrt"`.
  - **Consequence of the rule:** `km_gini` has no `sim.py` arm, so it cannot pass check (i). It can be reported as promising but cannot become the default under this rule.
  - Row-level IPCW losses reuse each (fold, landmark) censoring fit. Their mean reproduces `landmark_cross_validate`'s scores exactly, and the bootstrap keeps those weights.
  - Tests (`tests/test_bench_criteria.py`, 10):
    - row losses reproduce the scores;
    - ICI and slope on exact and constant predictions;
    - bootstrap pairing (identical arms give 0, a constant shift is exact);
    - `align` rejects mismatched rows;
    - the decision rule on synthetic inputs (adopts, then fails each of the 4 checks in turn);
    - a smoke end-to-end run.
- [x] Full runs; `docs/bench/s8-bakeoff.md`, with the recommendation and D11 verdict against the pre-registered rule
  - Full run: 395 s (2026-09-26). **Nothing adopted:** `logrank` stays the default and `aggregate="hazard"` stays (D11).
    - `grouped_lik` / `poisson` match log-rank on landmark IBS/ICI (all intervals include 0) and are slightly, not significantly, worse on `sim.py` ISE.
    - `km_gini` is worse on the panel (ΔIBS +0.0047 [+0.0029, +0.0064]). Its forest barely splits on landmark stacks, because subjects straddle children.
    - `survival` aggregation is better in point estimate on PBC2 calibration, but the interval includes 0; it is not better on the panel or `sim.py`.
  - Log: `docs/bench/s8-bakeoff/run.log`; every rule check: `decisions.csv`.
- [x] Apply the decision: defaults, removal of losing criteria, user-guide criterion section, design.md D6/D11 updated
  - Removed:
    - `GroupedLik`, `PoissonExposure`, `KmGini`, `criterion()`;
    - `_core.criterion_score` and the criterion arguments of `_core.fit_forest` / `best_split`;
    - `split_criterion` / `criterion_horizon`;
    - `bench/criteria/` and its tests, `tests/test_criteria.py`, `tests/ref/criteria_ref.py`.
  - Kept (user decision): `Profile.{times, exposure, n_units}`, `n_units_right`, `SurvData.duration`, `fit_forest(…, criterion)` and the spy / coarse-exposure tests. `return_predictions` is also kept.
  - Predictions are still bit-identical to `main`. Docs: design.md D6/D11 and the risk table; user guide `time_grid.rst` / `prediction.rst`.
- [x] plan.md tick + "S8 done" notes

## Plan review (Codex, 2026-09-26)
1. The horizon-Brier was not a Brier objective under censoring or delayed entry (high). It is renamed `km_gini`, a heuristic, and an IPCW criterion is out of scope (items 3, 11).
2. `criterion_horizon` had no defined clock (high). It is now on the reset clock, and C3 runs on landmark data only. τ = `w` is fixed per dataset in advance (items 3, 5).
3. Not every dataset fits `landmark_cross_validate`, and C&L has gaps (high). The bake-off now has two protocols (A: landmark, B: core forest), and C&L is supporting evidence only (item 5).
4. Repeated-CV fold SEs are not independent, and multiplicity was unaddressed (high). Repeats are dropped; uncertainty comes from simulation replications plus a subject-cluster bootstrap; the rule is framed as descriptive (items 6–8).
5. `landmark_cross_validate` lacks out-of-fold predictions and calibration summaries (medium). Added `return_predictions=True`, plus pre-specified ICI, slope and NA handling (item 7).
6. C1 was mislabelled and its tie convention was implicit (medium). It is renamed `grouped_lik`, with q = d/y, grouped ties and `0·log 0 = 0`; tie and all-event fixtures are added (item 3).
7. Exposure needs durations in the splitter, snapped in coarse mode (medium). Added `SurvData.duration` and parent/left/right exposure; C2 is described as an exposure likelihood (items 2, 3).
8. C3's rules on node-local grids were unspecified (medium). Events ≤ τ are included and `S = 1` before the first event; τ edge-case tests are added (item 3).
9. The landmark pass-through duplicated nested params (low). Tuning now goes through `forest__split_criterion`, and the criterion is not stored in the Rust state (item 4).

Found sound: delayed-entry risk-set indexing for log-rank; log-rank bit-identity is feasible; C1/C2 gains are ≥ 0; nested CV already tunes nested forest params per arm.

## Diff review (Codex)
_pending_
