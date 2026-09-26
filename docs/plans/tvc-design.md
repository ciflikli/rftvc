# Design: statistical foundation, importance and effects for TVC forests

Status: **v2** (2026-09-26). T2 and T3 are user-confirmed (T3 adds LOCO refit to v1); the Codex design review is applied (Review log). Inputs: `tvc-questions.md`, `tvc-research.md` (Codex-corrected), `tvc-deviance.md`.
Notation:
- `M` = number of scoring windows `W_m`.
- `λ̃_m(x)` = window-average forest hazard.
- `S` = the α-mixed piecewise-exponential log score (`tvc-deviance.md` §0).
- M1–M7 = the importance methods of `tvc-research.md` §2.2.

## Executive summary
- **Foundation, stated rather than proved.**
  - A user-guide page says what the forests estimate: the hazard map `λ(t, x)` of the current covariate state, estimated with a nonparametric time curve per leaf.
  - It states the four assumptions: current state, predictability, independent censoring/entry, id-level resampling.
  - It gives the prediction functionals and when each is a probability, and it states that no consistency result exists.
- **One loss.** A new metric `rftvc.metrics.piecewise_exponential_score` (the PE score). It is proper in the qualified sense of `tvc-deviance.md`, needs no IPCW, and decomposes exactly by window, cause and subject. It is the default loss for importance.
- **Importance that respects TVC rows.** `rftvc.inspection.permutation_importance`:
  - permutation within **time strata** (M2) is the default; the naive shuffle (M1) is available only for comparison;
  - feature **groups** are permuted jointly;
  - results come per window and per cause, with subject-paired standard errors;
  - it works on held-out data or out-of-bag.
- **Landmark models:**
  - Permutation happens within each landmark's risk set, and **all history features of a raw column move together**. This equals trajectory permutation (M3) with donors from the same risk set, with no feature recomputation.
  - Conditional permutation gives the **level-vs-history diagnostic** of the current-state assumption.
- **Effects:**
  - `hazard_effect`: a time-stratified partial dependence of window hazards on a `(value × window)` grid, valid for every covariate.
  - `path_effect`: "shift `z` by δ from time `u`" contrasts in `S` / `F_k` along supplied paths, documented as prediction under a specified path (external covariates only, D4) and not as a causal effect.
- **Refit importance (M4, user-requested for v1):** `drop_column_importance` cross-fits the estimator with and without each unit and scores held-out folds. This answers "how much predictive value is lost without `z`", with cross-fitting standard errors.
- **Out of v1:** rule-release (M5), SHAP (M7), minimal depth (M6), knockoffs.

## Decisions
| # | Decision | Status |
|---|---|---|
| T1 | Foundation = user-guide page + estimator docstring notes; claims limited to `tvc-research.md` §1 (target, assumptions, functionals table, no consistency claim) | default |
| T2 | New public module `rftvc.inspection` mirroring `sklearn.inspection` (`permutation_importance`, `drop_column_importance`, `hazard_effect`, `path_effect`), functions not estimator methods | user-confirmed |
| T3 | v1 scope: PE score metric; permutation importance (M1 / M2 / groups / landmark M3 / conditional); **LOCO refit (M4)**; `hazard_effect`; `path_effect`. Deferred: rule-release (M5), minimal depth (M6), SHAP (M7) | user-confirmed (LOCO added by user) |
| T4 | Default loss = PE score: `windows=8` quantile windows of training event times on `(0, τ]` with `τ` = the last training event time, `alpha=0.01` mixture with the **training** null, zero-rate share always reported. Landmark models also accept `scoring="brier"` / `"ibs"` at horizons | default |
| T5 | Default permutation `strata="time"` (M2); `strata=None` = naive shuffle (M1), documented as extrapolation-prone; landmark models stratify by landmark `s` | default |
| T6 | Evaluation on user-supplied held-out data by default; `oob=True` rebuilds the fit-time design deterministically, checks it against a stored fingerprint, and uses a new engine call `oob_cumhaz` | default |
| T7 | Result = sklearn-style `Bunch` + TVC fields (per-window, per-cause, paired SE, zero-rate share, share of gain) | default |
| T8 | Landmark `AGGREGATIONS` gain `"slope"` and `"std"` (history summaries the research found missing) | default |
| T9 | `oob_score_` / `score` keep C for now; switching to the PE score is decided in the release pass | default (flagged) |
| T10 | Estimators store the training null at fit: `baseline_cumhaz_` (pooled Nelson–Aalen on `event_times_`; per cause for CR), so the null never comes from evaluation data | default |

## Approaches considered
- **A. Functions in `rftvc.inspection` + one new metric + one engine call.** Recommended.
  - Keeps the estimators' surface unchanged and mirrors sklearn, so users know where to look.
  - Everything except OOB runs on the public `predict_cumulative_hazard`, so the functions are model-agnostic within rftvc.
- **B. Methods on the estimators** (`forest.permutation_importance(...)`).
  - Discoverable, but the landmark and counting-process estimators need different arguments, and it duplicates code across four classes.
- **C. Drop-column refit (LOCO) as the *only* importance.** It has the best-defined estimand (Hooker et al.) and fits are fast. But its cost is p refits × folds, and it cannot answer the timing or level-vs-history questions without many refits. Shipped **alongside** permutation importance, not instead of it (T3, §3.4).

## 1. Foundation page (T1)
`docs/source/user_guide/foundation.rst`, linked from the estimator docstrings:
1. **Target.** `λ(t, x)`: the hazard at time `t` given the covariate state in force. Leaves average this hazard over the rows at risk in the cell at each `t` (composition drift caveat). The competing-risks forest does the same per cause.
2. **Assumptions:**
   - current state (history enters only through columns);
   - predictability (covariates known at `start`; `measured_at` checks this);
   - independent censoring and entry given the covariate state;
   - ids are the independent units (block resampling for long series).
3. **Prediction functionals:** the `tvc-research.md` §1.3 table (fixed profile / path / mortality / landmark), with the landmark conditions as corrected by Codex.
4. **Theory status:**
   - adjacent results: RSF consistency (Ishwaran & Kogalur 2010), splitting bias (Cui et al. 2022), LTRC trees (Fu & Simonoff 2017);
   - no consistency result exists for counting-process TVC forests;
   - **the current-state assumption is checkable** with the §4 diagnostic.
5. **Internal vs external covariates:** D4 in plain words, with the path-effect caveat of §5.

## 2. Metric: `piecewise_exponential_score` (T4)
```python
rftvc.metrics.piecewise_exponential_score(
    y, cumhaz, windows, *, null_rate=None, alpha=0.01, cause=None, ids=None, reduce="per_event",
) -> PEScore  # NamedTuple
```
- **Inputs:**
  - `y`: counting-process rows (survival or competing-risks dtype).
  - `cumhaz`: shape `(n_rows, M + 1)`, each row's fixed-profile `Λ̂` at the window edges `windows` (length `M + 1`, starting at 0). For CR, `(n_rows, J, M + 1)`.
  - `null_cumhaz`: the training null `Λ₀` at the edges, shape `(M + 1,)` (or `(J, M + 1)`): the estimator's `baseline_cumhaz_` (T10) evaluated at `windows`. **It is required.** Evaluation-fitted nulls would make `null_total` and `share_of_gain` depend on the evaluation data (Codex).
- **Windows validation:** edges must be finite and strictly increasing, starting at 0. Evaluation rows are **administratively truncated at `τ = windows[-1]`**: exposure and events after `τ` are excluded and counted in `n_truncated_events`. Nothing is silently dropped.
  - For the default windows, `τ` is the last training event time. Beyond it every leaf and the null are flat, so no rate can be scored.
  - `windows[-1]` beyond the estimator's last event time raises an error.
  - Landmark stacks require `windows[-1] <= horizon` (reset clock).
- **Computation:** cells `(e_rm, N_rm)` as in `tvc-deviance.md` §1, and `λ̃_α = (1 − α) ΔΛ̂ / |W| + α ΔΛ₀ / |W|`.
- **Returns:**
  - `total` (per event, or the sum with `reduce="sum"`);
  - `by_window (M,)` and, for CR, `by_cause (J,)` or `(J, M)`;
  - `by_id` (when `ids` is given);
  - `zero_rate_share`;
  - `null_total`: the training null's score on the same cells, for share-of-gain;
  - `n_truncated_events`.
- **Windows helper.** `event_windows(estimator, n_windows=8)`: quantiles of `event_times_`, with edges 0 and the last event time `τ`, deduplicated (fewer than `n_windows` windows if there are ties). Under coarse mode `event_times_` is the coarse grid, so the edges are grid points.
- **Warnings:** `zero_rate_share > 0.01` → `UserWarning` ("windows too fine for this forest; lower n_windows"). `alpha=0` with zero-rate event cells → `total = -inf`, never silently floored.
- **Landmark models:** the same function on stacked rows, with windows on the horizon clock `(0, horizon]`.

## 3. `permutation_importance` (T2, T3, T5–T7)
```python
rftvc.inspection.permutation_importance(
    estimator, X, y=None, *, ids=None, features=None, groups=None,
    strata="time", n_strata=10, conditional_on=None, n_bins=4,
    scoring="pe", windows=8, alpha=0.01, cause=None,
    oob=False, n_repeats=5, random_state=None, n_jobs=None,
) -> Bunch
```
**Counting-process estimators** (`SurvivalForestTV`, `CompetingRisksForestTV`):
- `X, y, ids` are held-out rows. `oob=True` needs `X, y, ids` equal to the training data and uses the forest's OOB sets (§3.3).
- **Units permuted:** `features` (names or indices; default all) or `groups` (a dict `name → [columns]`) that are permuted **jointly**. Joint permutation moves a raw column together with the lags and summaries the user derived from it, so the pair stays consistent (`tvc-research.md` §2.1 item 3).
- **Strata (`strata="time"`, M2).** Rows are binned by their `start` into `n_strata` quantile bins of the evaluation rows' starts, and values are permuted within each bin. `strata=None` is M1, with its docstring warning. `strata=array` gives user-defined strata (e.g. calendar period, region).
- **`conditional_on=[cols]`:** strata are further crossed with `n_bins` quantile bins of each conditioning column (M2+, conditional-subgroup PFI). Empty or singleton strata are left unpermuted and counted in `n_unpermuted`.
- **Score:** permutation `b` of unit `j` gives `ΔS_jb = S(intact) − S(permuted)`, recomputed per window, per cause and per id from the same predictions. One `predict_cumulative_hazard(X_perm, times=edges)` call per (unit, repeat).

**Landmark estimators** (`LandmarkSurvivalForest`, `LandmarkCompetingRisksForest`):
- `X` is the raw long frame `df` (`y`, `ids` unused). Internally it is `make_landmark_data` with the model's settings, then `model.forest_` on the stacked rows.
- **Default units** are *raw columns*: the group of all features that `history_features` built from one column. So the importance is at the variable level. `landmark` is constant within a landmark stratum, so it is not a permutable unit.
- **Strata are the landmark `s`:** each landmark's risk set is permuted separately. Moving a column's whole feature group together between ids at risk at `s` equals giving each id a donor's history up to `s` (M3 with risk-set donors), with no recomputation and no donor-coverage gaps [K, derived from the construction].
- **Level vs history (the diagnostic, `tvc-research.md` §4.3):**
  - `permutation_importance(model, df, features=["z"], conditional_on=["z_mean", "z_slope"])` permutes the level within history bins;
  - `features=["z_mean", "z_slope"], conditional_on=["z"]` permutes history within level bins.
  - The docs show the recipe: clearly positive history-given-level importance means the current-state model misses history.
- **Scoring:** the default `"pe"` works on the horizon clock. `scoring="brier"` (at `horizon`) or `"ibs"` reuses `metrics.brier_landmark` / `integrated_brier` with the IPCW settings of `landmark_cross_validate`, for users who act on horizon risk. CR models take `cause=`.

### 3.1 Result (T7)
A `Bunch` with sklearn's fields plus TVC fields:
- `importances_mean (p,)`, `importances_std (p,)` (over repeats), `importances (p, n_repeats)`: as in sklearn, in score units per event.
- `importances_window (p, M)`, `window_edges (M + 1,)`.
- `importances_cause (p, J)` (CR; with `cause=None` the total is summed over causes).
- `importances_se (p,)`: an **id-cluster bootstrap with re-permutation** (`n_bootstrap=100`; 0 disables it and gives NaN).
  - Each replicate resamples evaluation ids with replacement, redraws the within-stratum permutation on the resampled rows, and rescores.
  - Estimand: sampling variability of the importance over evaluation subjects, **conditional on the fitted forest**; fit variability is not included (LOCO §3.4 covers refits).
  - A plain paired SE over per-id differences is *not* used. Within-stratum permutation couples subjects through shared donors, so per-id differences are not independent (Codex).
  - With `oob=True` it is NaN, because OOB shares trees across ids.
- `share_of_gain (p,)`: `ΔS_j / (S_intact − S_null)`. It is an unbounded ratio, not a 0–1 share: it can be negative, or exceed 1 for correlated units. It is `NaN` with a `UserWarning` when `S_intact ≤ S_null` (the forest does not beat the training null on this data).
- `baseline_score`, `null_score`, `zero_rate_share`, `n_unpermuted`, `feature_names`.

### 3.2 Why these defaults
- Time strata keep every permuted row on the observed `(t, z)` support. This removes the extrapolation channel that `tvc-research.md` §2.1 suspects, and the S-sim of §7.1 tests it.
- `n_strata=10`: rows per stratum stay large for typical data. The docs describe `n_strata` as the conditioning knob (≈ the Debeer–Strobl threshold).

### 3.3 Engine call for OOB (T6)
- New Rust `Forest::oob_cumhaz(X, oob_offsets, oob_units, times, agg) -> (Vec<f64> n×T, Vec<u32> n_oob)`. It generalises `oob_mortality` (which is `Σ_k` of the same thing), and there is a CR twin `oob_cumhaz_causes`.
- **The OOB design must equal the fit-time one** (Codex). Coarse mode drops and rewrites rows and reindexes units; block mode splits rows and builds buffered OOB sets.
  - A private `_fit_design(X, y, ids, block_time)` is factored out of `_fit`. It returns the retained-row index, the transformed rows, the units and the OOB CSR, and it is deterministic given its inputs and the constructor parameters (no RNG).
  - `fit` stores a fingerprint `_fit_fingerprint_` (row count and a hash of `X`, `start`, `stop`, `event`, `ids`, `block_time`).
  - `permutation_importance(..., oob=True)` requires the same `X, y, ids, block_time`. It rebuilds the design, checks the fingerprint (mismatch → `ValueError`), permutes on the *transformed* rows, and scores on them. Rows dropped by coarsening are excluded and counted.
- `oob_mortality` is re-expressed through `oob_cumhaz`, with a bit-identity test in id mode, block mode and coarse mode.

### 3.4 `drop_column_importance` (LOCO, M4; T3)
```python
rftvc.inspection.drop_column_importance(
    estimator, X, y=None, *, ids=None, cv=5, features=None, groups=None,
    scoring="pe", windows=8, alpha=0.01, cause=None, n_seeds=1, n_jobs=None,
) -> Bunch
```
- **Estimand:** the drop in held-out score between the estimator fitted with all features and the estimator refitted without unit `j`. It is the predictive value of `j` for new subjects, given everything else the estimator can learn from (Hooker et al. 2021; Williamson-type VIM, survival version Wolock et al. 2025). Unlike permutation importance, correlated units can compensate for each other, so two strongly correlated variables can both have low LOCO importance. The docs pair it with grouped LOCO.
- **Cross-fitting.** `cv` is an int (`GroupKFold` on ids) or a splitter.
  - For time series, `GroupTimeSplit` / `RollingOriginSplit`, with the same administrative censoring and `gap` rules as `landmark_cross_validate`; that plumbing is reused.
  - Per fold: fit `clone(estimator)` on the training ids, and one clone per dropped unit, with the same `random_state`. Windows and `null_cumhaz` come from the full-model fit on that fold. Both models are scored on the test fold with the same windows.
- **Landmark estimators.** `X` is `df`, and units are raw columns: dropping `z` removes every `history_features` entry built on `z`. Folds follow `landmark_cross_validate`.
- **Fit noise.** Refits differ by forest randomness even for a null unit. `n_seeds > 1` averages the full and dropped fits over seeds, and a null-control column (`add_noise_control=True`, default False) gives the noise floor.
- **Uncertainty.** Each id is in exactly one test fold. Per-id score differences are pooled over folds, and `importances_se = sd(ΔS_i) · √n_ids / Σ N`. This is the cross-fit SE that Williamson/Wolock-type VIM use [S: Wolock et al.]. It is valid conditionally on the fold fits and approximately valid across refits [K]. Time splitters report per-fold values only, with SE = NaN: test folds are not exchangeable over time.
- **Result:** as in §3.1, with `importances (p, n_folds)`, `importances_window`, `importances_cause`, `importances_se`, `baseline_score`, `null_score`, `share_of_gain`, `fold_scores`.
- **Cost:** `(p + 1) × n_folds × n_seeds` fits. The docs give timings from the S8 bench sizes.

## 4. Effects (T3)
### 4.1 `hazard_effect` (all covariates, hazard scale)
```python
rftvc.inspection.hazard_effect(estimator, X, y, *, feature, values=None, windows=8, kind="average", ids=None, cause=None)
```
- **Estimand.** For each window `W_m` and grid value `v`: the exposure-weighted average of `λ̃_m(x_r with x_j = v)` over the rows at risk in `W_m` (weights `e_rm`). This is a *time-stratified* partial dependence, because each window averages over its own at-risk population.
- **Returns** `(n_values, M)` window hazards and the observed-support mask: values outside the within-window 5–95 % range of `x_j` are flagged, not dropped. `kind="individual"` returns per-row curves (ICE).
- It is valid for internal and external covariates alike: it is associational on the hazard scale, and it shows non-proportional (time-varying) effects directly.

### 4.2 `path_effect` (external covariates, prediction scale)
```python
rftvc.inspection.path_effect(estimator, X, intervals, ids, *, feature, delta, from_time, horizons, origin=None, cause=None, extrapolate="none")
```
- **Clock.** `from_time` and `horizons` are **absolute** analysis times, on the same clock as `intervals` and `origin`, as in `predict_cumulative_hazard` [R].
  - Per subject it is validated that `origin ≤ from_time ≤ h ≤ last stop` for every `h`. Otherwise a `ValueError` names the subjects, unless `extrapolate="locf"` covers `h > last stop`.
  - Paths that do not reach `from_time` are an error.
- **Computation:** the change in `P(T ≤ h | T > origin, path)` (or `F_k(h | origin)`) between the supplied path and the same path with `x_j += delta` from `from_time` onward. Rows that straddle `from_time` are split there first. It uses `predict_*(intervals=…)` [R].
- **Output:** per-subject changes `(n_subjects, n_horizons)` and their mean. For `CompetingRisksForestTV` it is `(n_subjects, J, n_horizons)` with `cause=None` (all causes, `causes_` order), or `(n_subjects, n_horizons)` for one `cause`, matching `predict_cumulative_incidence`.
- **The docstring must say:**
  - "Prediction along a specified covariate path. Valid when `x_j` is external (D4). Not a causal effect unless `x_j`'s effect on the hazard is unconfounded given the other covariates."
  - Reference: Keogh & van Geloven 2024.
- `delta` may be a callable `f(values, start) -> values`, for scenarios other than shifts.

## 5. Landmark history features (T8)
- `AGGREGATIONS += ("slope", "std")`.
  - `slope`: the OLS slope of the column on `measured_at`, or on `start` when there is no `measured_at`, over the rows known at `s`. It is NaN with fewer than 2 distinct times; the existing NaN policy of `landmark_features` applies.
  - `std`: the sample std.
- Not in v1: time-above-threshold, EWMA, time-since (all need parameters; a later spec syntax).

## 6. Competing risks
- The PE score is summed over causes and reported per cause. `cause=k` restricts importance to `λ_k` (`ΔS_k`).
- For CR landmark models, `scoring="brier", cause=k` gives importance for `F_k`.
- The user guide shows both on the S14 generator, and explains that a covariate can matter for `F_k` only through a competing cause.

## 7. Validation strategy (feeds the plan)
Every simulation predeclares, in the plan: the estimand, R = 50 replications (Monte Carlo SE of each mean ≤ 1/7 of the relevant margin, checked from a 10-replication pilot), and **margins relative to the oracle**. The oracle is the true unit's score drop, computed with the known hazard on a large (10⁵-subject) evaluation sample. Claims over windows or causes use Holm adjustment at 5 %. Pass rules are on Monte Carlo means, not on per-replication intervals, so wide SEs cannot make a test pass.
1. **Trend confounding.** `z1` has an effect; `z2` has none, trends with `t` and correlates with `z1` (ρ = 0.6).
   - M2: `|mean imp(z2)| ≤ 0.1 × oracle(z1)`, and `mean imp(z1)` is within ±25 % of the forest's own intact-vs-oracle-permuted reference.
   - LOCO: `|mean imp(z2)| ≤ 0.1 × oracle(z1)`.
   - M1 vs M2 is reported as the evidence on the extrapolation claim, without a pass rule.
2. **Level vs history.** The hazard depends on `mean(z)` over the past 2 units. Pass: the mean history-given-level importance is ≥ 0.25 × the oracle history importance. In the Markov control its mean is ≤ 0.05 × the oracle level importance.
3. **Timing.** `z` acts only on `t ∈ [2, 4]`. Pass: the Holm-adjusted per-window test (paired over replications) rejects in at least one window inside `[2, 4]` and in none outside.
4. **CR.** `z` affects cause 1 only. Pass: `mean ΔS_2 ≤ 0.1 × mean ΔS_1`, and `ΔS_1` is Holm-significant.
5. **`path_effect`** against the true Δ risk (external `z`, S3 generator): `|mean bias| ≤ 0.1 × |true Δ|` at each horizon.
5b. **Landmark specifics:** repeated landmark copies of one id (importance and SE with `step` small vs large), and heavy censoring (IPCW Brier scoring with `g_min` truncation active vs the PE score).
6. **Metric unit tests:**
   - the propriety Monte Carlo of `tvc-deviance.md` §2.3;
   - the Poisson-GLM identity (the score equals the `statsmodels` / hand-computed Poisson log-likelihood with a log-exposure offset, up to the data-only constant);
   - `alpha=0` → `-inf` on zero-rate event cells;
   - the window decomposition sums to the total.
7. **`oob_cumhaz`:** summing over grid times equals `oob_mortality` bit-for-bit, in id, block and coarse modes. The fingerprint mismatch raises. `bootstrap=True` fits are included.
7b. **LOCO:** a null-control unit has mean importance within ±2 Monte Carlo SE of 0 over 50 replications. Grouped LOCO of two copies of one variable is > 0 while each single copy is ≈ 0.
8. **Deviance follow-ups** (`tvc-deviance.md` §8): a time-varying baseline, OOB vs held-out agreement, and the zero-rate share across n and `min_events_leaf`. These tune the `windows` default.

## 8. Slice sketch (the plan stage refines this)
1. **S16 metric:** `piecewise_exponential_score`, `event_windows`, `null_window_rates`; the Rust `oob_cumhaz` (+ CR twin); the unit tests of §7.6–7.7; deviance follow-ups §7.8.
2. **S17 counting-process permutation importance:** `rftvc.inspection.permutation_importance` for `SurvivalForestTV` / `CompetingRisksForestTV` (strata, groups, conditional, windows, causes, cluster-bootstrap SE, OOB with `_fit_design`); sims §7.1 (M2 part), §7.3, §7.4.
3. **S18 LOCO:** `drop_column_importance` (cross-fitting, seeds, noise control, cross-fit SE, landmark raw-column drop); sims §7.1 (LOCO part), §7.7b.
4. **S19 landmark importance + history features:** raw-column groups, landmark strata, Brier/IBS scoring, `slope` / `std` aggregations; sims §7.2, §7.5b.
5. **S20 effects + foundation docs:** `hazard_effect`, `path_effect`, sim §7.5; the foundation page, a user-guide page on importance and effects, and a case study.

## Risks
| Risk | Mitigation |
|---|---|
| The M1 bias the research suspects may be small in practice | §7.1 reports M1 vs M2 either way; M2 stays the default because it is never worse on support grounds |
| Small strata (many conditioning bins, few rows) permute little → importance biased toward 0 | `n_unpermuted` reported; warn if > 10 % of rows are unpermuted |
| Users compare PE scores across different windows | `window_edges` in every result; docs: compare only at the same windows |
| OOB importance has no SE | documented; held-out is the default |
| LOCO cost `(p + 1) × folds × seeds` fits | groups reduce `p`; timings in docs; `n_jobs` across fits |
| LOCO fit noise masquerades as importance | `n_seeds`, a noise-control unit, the same `random_state` per fold |
| `path_effect` read causally | docstring and user-guide wording (§4.2); internal-covariate warning in the docs |
| API growth before the release pass | one new module, sklearn naming; the release pass reviews it with everything else |

## Out of scope (v1)
- Rule-release/VarPro (M5).
- Minimal depth (M6).
- TreeSHAP with vector leaves and DynSHAP-style attributions (M7).
- Knockoffs.
- Subsampling SEs for OOB importance.
- Kernel-smoothed scoring.
- Parametrised history features.

## Review log (Codex, 2026-09-26)
All 6 findings accepted:
1. **Windows unvalidated; rows beyond the last edge silently lost** (high). Fix: edge validation, administrative truncation at `τ` with `n_truncated_events`, the default `τ` = last training event, landmark windows ≤ horizon.
2. **Null fitted on evaluation data leaks into `share_of_gain`** (high). Fix: `baseline_cumhaz_` stored at fit (T10) and required by the metric; `share_of_gain` is documented as an unbounded ratio, NaN when the forest does not beat the null.
3. **OOB importance cannot line up with fit-time rows in coarse or block mode** (high). Fix: a deterministic `_fit_design`, a stored fingerprint, scoring on the transformed rows, and bit-identity tests in all modes.
4. **Paired SE is invalid under within-stratum permutation** (medium). Fix: an id-cluster bootstrap with re-permutation, and a stated conditional estimand.
5. **`path_effect` clock, origin and CR shape unspecified** (medium). Fix: absolute clock, per-subject validation, shapes matching `predict_cumulative_incidence`.
6. **Pass rules not falsifiable or powered** (medium). Fix: oracle-relative margins on Monte Carlo means, R = 50 with a pilot power check, Holm across windows and causes, and landmark-copy and heavy-censoring scenarios.
