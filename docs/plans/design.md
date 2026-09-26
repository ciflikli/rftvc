# Design: `rftvc`, a scikit-learn-compatible survival forest with time-varying covariates

Status: **v2, approved defaults** (2026-09-25). Inputs: `research.md` (corrected), `research-review.md`, `design-principles.md` v2, `design-review.md`.
v2 folds in all 4 key and 10 minor findings from `design-review.md` (see Changelog).

## Executive summary
- `rftvc` is a general-purpose random survival forest for Python with a Rust engine.
- The engine reads **counting-process rows** `(id, start, stop, event, X)`, with left truncation and time-varying covariates handled natively. Row weights are **unit** weights; resampling and OOB are done by subject (`id`).
- **Prediction targets** are served by data-building functions and one meta-estimator, **not** by sklearn `Pipeline` transformers: row-reshaping steps change `y` and `ids`, which a `Pipeline` cannot carry.
- **Error estimation** is an explicit user choice (new subjects vs future periods), made through CV splitters.
- **v1 split criterion:** the exact LTRC log-rank on (optionally coarsened) event times. Other criteria plug in through a trait.

## Decisions
| # | Decision | Status |
|---|---|---|
| D1 | Rust core (PyO3 + maturin + rayon + rust-numpy); thin Python layer | user-confirmed |
| D2 | Covariates mostly external; goal = dynamic prediction accuracy | user-confirmed |
| D3 | General purpose; domain examples only in docs/case studies | user-confirmed |
| D4 | v1 excludes: internal-covariate path scenarios, informative visit processes, competing risks | default |
| D5 | Two public workflows: (a) the core estimator on counting-process data; (b) `make_landmark_data()` + `LandmarkSurvivalForest` meta-estimator. **No Pipeline-based stacking** | default (v2) |
| D6 | v1 criterion = LTRC log-rank, **unit weights only**; confirmed by the S8 bake-off (`docs/bench/s8-bakeoff.md`) | default (v2); S8 |
| D7 | Engine input = numpy; polars in data builders; `narwhals` for DataFrame-agnostic input | default |
| D8 | Time coarsening defined as **snapping times to the grid before counting**, so the score is an exact log-rank on the coarsened data | default (v2) |
| D9 | Unobserved future covariates are never filled silently: `extrapolate="none"` by default | default (v2) |

## Approaches considered
- **A. Own Rust counting-process forest + data builders.** Recommended.
  - Native left truncation and TVCs; id-level resampling.
  - Histogram and grid speedups; no private sklearn API.
  - Cost: most build effort, two languages.
- **B. Existing engines on landmark data.** Stack landmarks with the clock reset, then fit scikit-survival RSF or per-horizon LightGBM.
  - Fast to build, but no within-follow-up TVCs, row-level bootstrap only, and memory limits.
  - Kept as **baseline and test reference**.
- **C. Extend sklearn/sksurv Cython tree internals.** Rejected: private API that breaks across versions, row-based design, limited parallelism.

## Architecture (Approach A)
```
rftvc/
  _estimator.py      SurvivalForestTV              core estimator (counting-process data)
  landmark.py        make_landmark_data(), LandmarkData, LandmarkSurvivalForest (meta-estimator)
  person_period.py   make_person_period_data()     discrete-time view (baselines)
  _validation.py     check_counting_process(), structured-y validation
  model_selection.py RollingOriginSplit, GroupTimeSplit, landmark_cross_validate()
  metrics.py         brier_landmark, integrated_brier, cindex_dynamic, calibration_table
rust/rftvc-core/
  data.rs  grid.rs  criterion.rs  splitter.rs  tree.rs  forest.rs  predict.rs
```

### Data contract (core estimator)
- **`X`**: `(n_rows, p)` float.
- **`y`**: structured array with fields `start`, `stop` (float) and `event` (bool). Rules: `start < stop`, no NaN.
- **`ids`**: `(n_rows,)`, passed as a fit/predict argument. Separately, the CV splitter's `groups=ids` is supplied by the user or by `LandmarkData.groups`. Metadata routing is documented with `set_fit_request(ids=True)`.
- **Per-id rules**, validated by `check_counting_process`:
  - rows are **contiguous and non-overlapping**: `stop_j == start_{j+1}`. A gap raises an error, unless `gap_policy="split_id"`, which treats each segment as delayed re-entry. That option is documented as an assumption.
  - only the last row may have `event=1` (single event, v1).
- **Predictability:** X on the row `(start, stop]` must be known at `start`. If an optional `measured_at` column is supplied, the checker errors when `measured_at > start`.
- **sklearn compatibility matrix** (tested in the plan):
  - tags: `requires_y`, a custom target type
  - `n_features_in_`, `feature_names_in_`
  - `check_estimator`, with explicit, documented exclusions for checks that assume a scalar `y`.

### Time grid and coarsening (D8)
1. **Exact mode** (`ntime=None`): the grid is all unique event times.
2. **Coarse mode** (`ntime=K`): choose K grid points from event-time quantiles, then transform every row *before* any counting:
   - `start' = g(start)` and `stop' = g(stop)`, where `g(t)` is the smallest grid point `≥ t`.
   - Drop rows where `start' == stop'`. Such a row carries no at-risk time on the grid. If it carries an event, the event moves to the id's previous row. That row already ends at the same grid point, because `g(prev stop) = g(start)`. If there is no previous row (the id entered and failed within one bin), the id is dropped and counted in a fit diagnostic.
   - The statistic is then the exact LTRC log-rank on the coarsened data, including ties. Within-bin entry, event and censoring are fully defined by `g`.
3. **v1 default is `ntime=None`.** Coarse mode is opt-in until benchmarks justify a default.
4. Fixture tests put an entry, an event and a censoring inside one coarse bin and compare both modes against a delayed-entry reference (lifelines / R `survival`).

### Core algorithm: LTRC log-rank with histograms
On the (possibly coarsened) grid `t_1 < … < t_K`, for a node:
- **At risk:** `Y_k = #{r : start_r < t_k ≤ stop_r}`.
- **Events:** `d_k = #{r : event_r, stop_r = t_k}`. After coarsening, every `stop` with an event lies on the grid.

**Index convention:**
- `a_r` = first `k` with `t_k > start_r`.
- `b_r` = first `k` with `t_k > stop_r`. `K` if none.
- Row `r` is at risk for `k ∈ [a_r, b_r)`. If `a_r ≥ b_r`, it contributes nothing.

**Per candidate feature:**
1. Build a `bins × (K+1)` difference array: add `+1` at `(bin_r, a_r)` and `−1` at `(bin_r, b_r)`, then take a prefix sum over k. Events go into `(bin_r, idx(stop_r))`.
   - Time: `O(n_node + bins·K)`. Memory: `bins·K` per feature being evaluated.
2. Cumulative-sum over bins to get the left-child profiles; score each threshold in `O(K)`.
3. **Sibling subtraction** (optional, benchmarked): accumulate rows only for the smaller child. This saves at most half of the row-accumulation work. It costs retained parent histograms for the evaluated features. No general "halves the work" claim.
4. **Variance:** the hypergeometric log-rank variance, valid for the **integer unit-weight counts** used in v1. Weighted criteria are deferred until a weighted score and variance are specified.

```rust
pub trait SplitCriterion: Sync {
    fn score(&self, left: &Profile, parent: &Profile) -> f64;
}
pub struct Profile<'a> { pub at_risk: &'a [f64], pub events: &'a [f64] } // len K, integer-valued in v1
pub struct LtrcLogRank;
impl SplitCriterion for LtrcLogRank {
    fn score(&self, l: &Profile, p: &Profile) -> f64 {
        let (mut num, mut var) = (0.0, 0.0);
        for k in 0..p.at_risk.len() {
            let (y, d, yl) = (p.at_risk[k], p.events[k], l.at_risk[k]);
            if y < 2.0 || d == 0.0 { continue; }
            num += l.events[k] - d * yl / y;
            var += d * (yl / y) * (1.0 - yl / y) * (y - d) / (y - 1.0);
        }
        if var > 0.0 { num * num / var } else { 0.0 }
    }
}
```

### Leaf-size rule with ids that straddle a split
- One id's intervals can route to both children, because X changes over time.
- **Rule:** `ids(child)` = the distinct ids with ≥1 row in that child. An id **may count in both** children. A split is admissible only if `|ids(L)| ≥ min_ids_leaf`, `|ids(R)| ≥ min_ids_leaf`, and each child has at least `min_events_leaf` events.
- **Implementation:**
  1. Rank candidates by score.
  2. Check the exact distinct-id counts for the top candidates, using a per-node id→(min_bin, max_bin) scan. That makes each check `O(n_ids_node)`.
  3. Take the first admissible candidate.
- Node diagnostics report both row counts and distinct-id counts.

### Dependence knobs
| Param | Default | Meaning |
|---|---|---|
| `resample_unit` | `"id"` | the only v1 value; `"row"` and `"block"` are deferred, each needing its own OOB spec |
| `max_samples` / `bootstrap` | 0.632 of ids, without replacement / `False` | subsampling keeps OOB id-clean (D10) |
| `min_ids_leaf` | 15 | distinct ids per child (rule above); `"auto"` = max(15, √n_ids) (D12) |
| `min_events_leaf` | 3 | events per child |
| `ntime` | `None` | exact; `K` = coarse mode (D8) |
| `max_bins` / `max_features` | 255 / `"sqrt"` | |
| `oob_score` | `False` | allowed **only** with `resample_unit="id"` (error otherwise). It is documented as a *new-subject* generalisation estimate. Future-period claims require `RollingOriginSplit` or `GroupTimeSplit`; there is no automatic detection heuristic |

`row_weight` is **removed from v1**. Normalising per id or by overlap changes the estimand (review finding 4). Inverse-visit or overlap weighting can come back later as a separately specified estimand with exact per-row formulas.

### Prediction
All methods take an explicit `times` grid and use the rows supplied for each id.

1. **Per interval** `(start, stop]`: route X down each tree and add the leaf's Nelson–Aalen increments on grid points in `(start, stop]`.
2. **Per id:** the cumulative hazard is `Λ(t) = Σ` over rows. Rows are contiguous by contract, so there are no gaps.
3. **Prediction origin:** `origin=u` (default: the id's first `start`) returns the conditional survival `S(t | T > u) = exp(−(Λ(t) − Λ(u)))`.
4. **Beyond the last `stop`:** `extrapolate="none"` returns NaN.
   - `"locf"` is an opt-in *named scenario*: covariates stay at their last value.
   - Alternatively, the user appends scenario rows for a specified future path. This is valid for external covariates only.
5. **Ensemble:** `aggregate="hazard"` (the average of Λ over trees) is the default, with `"survival"` as an option. They are different ensemble quantities; the choice is validated by calibration in the bake-off slice. S8 kept `hazard`: `survival` averaging did not improve calibration on both landmark datasets, and it was worse on the known-truth simulation.

### Landmark workflow (replaces the Pipeline idea)
`make_landmark_data(df, id, start, stop, event, landmarks, horizon, history_features, step=None) -> LandmarkData(X, y, ids, groups, s)`.

For each landmark `s`, write `T` for the event time, `C` for the censoring time, and `U = min(T, C)`:
1. **Risk set:** ids with `U ≥ s`, i.e. event-free *and* uncensored at `s`. Ids that have not yet entered by `s` are excluded.
2. **Row:** `start = 0`, `stop = min(U, s+w) − s`, `event = 1{T ≤ min(C, s+w)}`.
3. **Features:** summaries of `H(s)` (information up to `s` only; a look-ahead check raises an error) plus `s`.
4. **Role of `s`:**
   - `s` is a covariate, so the forest can learn landmark-dependent effects by splitting on it (the forest analogue of a supermodel's `s`-interactions).
   - It does **not** correct for selection or censoring.
   - Validity rests on censoring being independent of `T` given `H(s)` and `s`. This is documented.
5. With the clock reset there is no delayed entry *within* the landmark dataset. Selection into it is handled by conditioning on `U ≥ s`.

`LandmarkSurvivalForest(landmarks, horizon, history_features, **forest_params)`:
- a meta-estimator whose `fit(df)` builds the landmark data internally and fits a `SurvivalForestTV`
- `predict_risk(df_at_s, s)` returns `P(T ≤ s+w | T > s, H(s))`
- it owns the joint transform of X, y, ids and groups, so it works with `landmark_cross_validate()`, which expands the splitter's groups and time blocks consistently.

It is not a sklearn `Pipeline`, and the docs say so.

### Model selection & metrics
- **`RollingOriginSplit(time_col, n_splits, test_size, gap)`:** errors if `gap < horizon` when used through `landmark_cross_validate`.
- **`GroupTimeSplit`:** groups and time blocks together.
- **Nested CV** is the documented default workflow.
- **IPCW metrics** (`brier_landmark(s, w)`, `integrated_brier`):
  - The censoring survival is estimated **within each training fold**, on the **landmark risk set** (`U ≥ s`), by KM of `C` on the time-since-`s` clock.
  - Assumption: marginal independent censoring given being at risk at `s`. An optional user `censoring_model` gets the same fold and risk-set data.
  - Positivity: weights truncated at `G ≥ 0.05` (configurable), with a diagnostic of how many weights were truncated.
  - **Exact path:** when every test id in the `(s, w)` risk set has `U ≥ s+w` or an observed event before `s+w`, no IPCW is needed and plain scores are exact.
- **`cindex_dynamic(kind="cumulative"|"incident", s, w)`, `calibration_table(s, w)`.**

## Validation strategy (feeds plan.md)
1. **Oracles:**
   - A single-node tree equals the delayed-entry Nelson–Aalen (lifelines).
   - The score equals R `survdiff` for right-censored data only. For counting-process data it equals an independent naive reference (`tests/ref/logrank_ref.py`), because `survdiff` rejects counting-process input.
   - Coarse mode equals the exact log-rank on pre-coarsened data.
   - With no TVCs or truncation, results are close to sksurv RSF.
2. **Simulations with known truth:** Cox-type TVCs, non-proportional hazards and interactions, left truncation, many vs few rows per id. Metrics: L2 distance to the true S, and landmark Brier score.
3. **Benchmarks:**
   - Datasets: PBC2 / `pbcseq`, Stanford heart transplant, a BTSCS case study, a ≥1M-row synthetic set.
   - Comparators: lifelines Cox with TVCs, `LTRCforests`, `randomForestRHF` (optional), BoXHED 2.0, Approach B.
4. **sklearn compatibility matrix tests.**
5. **Performance targets** (set in S6 from `docs/bench/s6-perf.md`; M-series, 10 cores, 100 trees, p = 10). These are regression bounds, about 1.5× the measured values:
   - `ntime=100`: 100k rows < 3 s; 1M rows < 35 s; peak RSS < 3.5 GB at 1M rows.
   - Exact grid: 100k rows (K ≈ 64k) < 12 s; 1M rows < 180 s.
   - Coarse mode keeps exact-mode test C within 0.005 on the benchmark data.
   - Stored leaves dominated peak memory; S9 cut forest memory by ~60% and peak RSS at 1M rows to ~1.5 GB (`docs/bench/s9-leaf.md`).

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| Log-rank optimises separation, not calibration | pluggable criterion; S8 bake-off judged on landmark Brier score and calibration: grouped-likelihood and Poisson criteria tied with log-rank, KM-Gini was worse (`docs/bench/s8-bakeoff.md`) |
| Exact grid slow on large K | coarse mode (D8, S6: 6× faster at 1M rows, same C) + O(1) updates [Sverdrup et al. 2025] as later optimisation |
| Top-candidate id checks reject many splits | bounded retries; diagnostics; tune `min_ids_leaf` |
| Leakage via user features | `measured_at` check; landmark look-ahead check; gap ≥ horizon check |
| sklearn routing / check_estimator friction | explicit compatibility matrix; `ids` also accepted as a DataFrame column name |
| Rust build friction | maturin abi3 wheels via cibuildwheel in CI |

## Out of scope for v1
- Competing risks and multi-state models.
- Recurrent events beyond "first event after landmark".
- Joint models for internal covariates.
- Weighted criteria.
- `"block"` resampling.
- GPU, oblique splits, honest/causal forests.

## Resolved user decisions (2026-09-25)
- D10 Resampling: **subsample 0.632 of ids without replacement** (default); `bootstrap=True` option for parity benchmarks.
- D11 Aggregation: **`aggregate="hazard"`** default (keeps conditional survival consistent across origins); `"survival"` option. Revisited in S8 and kept.
- D12 Leaf size: **`min_ids_leaf=15`** default; `"auto"` = `max(15, sqrt(n_ids))`; tune by CV.

## Changelog
- v2:
  - D8: coarsening by snapping times (review K1).
  - D5: meta-estimator instead of Pipeline (K2).
  - Exact landmark row formulas and the role of `s` (K3).
  - `row_weight` removed (K4).
  - Index convention (m1); unit-weight variance (m2); softened sibling-subtraction claim (m3).
  - Straddling-id leaf rule (m4); OOB only with `resample_unit="id"`, no heuristic (m5).
  - D9: no LOCF by default (m6); IPCW fitted per fold and landmark risk set, with truncation (m7).
  - `groups` vs `ids` (m8); compatibility matrix (m9); contiguous rows plus `origin`/`times` in prediction (m10).
