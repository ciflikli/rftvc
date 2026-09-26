# Design: statistical foundation, importance and effects for TVC forests

Status: **v1** (2026-09-26), proposals. T2 and T3 need user confirmation; the Codex design review is pending. Inputs: `tvc-questions.md`, `tvc-research.md` (Codex-corrected), `tvc-deviance.md`.
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
- **Out of v1:** refit/LOCO importance (M4), rule-release (M5), SHAP (M7), minimal depth (M6), knockoffs.

## Decisions
| # | Decision | Status |
|---|---|---|
| T1 | Foundation = user-guide page + estimator docstring notes; claims limited to `tvc-research.md` §1 (target, assumptions, functionals table, no consistency claim) | default |
| T2 | New public module `rftvc.inspection` mirroring `sklearn.inspection` (`permutation_importance`, plus `hazard_effect`, `path_effect`), functions not estimator methods | **needs user confirmation** |
| T3 | v1 scope: PE score metric; permutation importance (M1 / M2 / groups / landmark M3 / conditional); `hazard_effect`; `path_effect`. Deferred: LOCO (M4), rule-release (M5), minimal depth (M6), SHAP (M7) | **needs user confirmation** |
| T4 | Default loss = PE score: `windows=8` quantile windows of training event times, `alpha=0.01` null mixture, zero-rate share always reported. Landmark models also accept `scoring="brier"` / `"ibs"` at horizons | default |
| T5 | Default permutation `strata="time"` (M2); `strata=None` = naive shuffle (M1), documented as extrapolation-prone; landmark models stratify by landmark `s` | default |
| T6 | Evaluation on user-supplied held-out data by default; `oob=True` uses the fitted forest's OOB sets through a new engine call `oob_cumhaz` | default |
| T7 | Result = sklearn-style `Bunch` + TVC fields (per-window, per-cause, paired SE, zero-rate share, share of gain) | default |
| T8 | Landmark `AGGREGATIONS` gain `"slope"` and `"std"` (history summaries the research found missing) | default |
| T9 | `oob_score_` / `score` keep C for now; switching to the PE score is decided in the release pass | default (flagged) |

## Approaches considered
- **A. Functions in `rftvc.inspection` + one new metric + one engine call.** Recommended.
  - Keeps the estimators' surface unchanged and mirrors sklearn, so users know where to look.
  - Everything except OOB runs on the public `predict_cumulative_hazard`, so the functions are model-agnostic within rftvc.
- **B. Methods on the estimators** (`forest.permutation_importance(...)`).
  - Discoverable, but the landmark and counting-process estimators need different arguments, and it duplicates code across four classes.
- **C. Drop-column refit (LOCO) as the default.** It has the best-defined estimand (Hooker et al.) and fits are fast. But its cost is p refits × repeats, it needs CV plumbing, and permutation within strata already stays on the data's support. Deferred to v2 as `drop_column_importance` [T3].

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
  - `null_rate`: the training-set window rates `λ₀,m` (from `null_window_rates(y_train, windows)`). If `None`, the rates come from `y` itself, with a `UserWarning` that the null is fitted on the evaluation data.
- **Computation:** cells `(e_rm, N_rm)` as in `tvc-deviance.md` §1, and `λ̃_α = (1 − α) ΔΛ̂ / |W| + α λ₀`.
- **Returns:**
  - `total` (per event, or the sum with `reduce="sum"`);
  - `by_window (M,)` and, for CR, `by_cause (J,)` or `(J, M)`;
  - `by_id` (when `ids` is given);
  - `zero_rate_share`;
  - `null_total`: the null predictor's score on the same cells, for share-of-gain.
- **Windows helper.** `event_windows(y_train, n_windows=8)`: quantiles of the training event times, with edges 0 and the maximum `stop`, deduplicated. Under coarse mode, the estimator's `coarse_grid_` can be passed as candidate edges.
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
- `importances_se (p,)`: subject-paired standard error from the `by_id` differences, averaged over repeats. It is conditional on the fitted forest, and is `NaN` with `oob=True`, because OOB shares trees across ids (use `n_repeats` and refits; subsampling SEs are deferred).
- `share_of_gain (p,)`: `ΔS_j / (S_intact − S_null)`.
- `baseline_score`, `null_score`, `zero_rate_share`, `n_unpermuted`, `feature_names`.

### 3.2 Why these defaults
- Time strata keep every permuted row on the observed `(t, z)` support. This removes the extrapolation channel that `tvc-research.md` §2.1 suspects, and the S-sim of §7.1 tests it.
- `n_strata=10`: rows per stratum stay large for typical data. The docs describe `n_strata` as the conditioning knob (≈ the Debeer–Strobl threshold).

### 3.3 Engine call for OOB (T6)
- New Rust `Forest::oob_cumhaz(X, oob_offsets, oob_units, times, agg) -> (Vec<f64> n×T, Vec<u32> n_oob)`. It generalises `oob_mortality` (which is `Σ_k` of the same thing), and there is a CR twin `oob_cumhaz_causes`.
- Python passes the OOB sets that the fit already builds (`_compute_oob`, id or block with buffer).
- `oob_mortality` is re-expressed through it, with a bit-identity test.

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
- **Computation:** the change in `1 − S(h | path)` (or `F_k`) between the supplied path and the same path with `x_j += delta` on rows whose `start ≥ from_time`, per subject and averaged. Rows that straddle `from_time` are split at `from_time` first. It uses `predict_*(intervals=…)` [R].
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
1. **Trend confounding.** `z1` has an effect; `z2` has no effect, trends with `t` and correlates with `z1`.
   - Oracle: the true score drop, from the known hazard.
   - Declared pass rule: over 20 replications, the M2 importance of `z2` has a mean paired difference to the oracle within ±2 SE. A secondary report compares M1 against M2 (the evidence for or against the extrapolation claim).
2. **Level vs history.** The hazard depends on `mean(z)` over the past 2 units. Pass: the history-given-level importance CI excludes 0; in the Markov control it covers 0.
3. **Timing.** `z` acts only on `t ∈ [2, 4]`. Pass: the window importances outside `[2, 4]` have CIs covering 0, and some window inside `[2, 4]` excludes 0.
4. **CR.** `z` affects cause 1 only. Pass: `ΔS_2` covers 0 and `ΔS_1` excludes it.
5. **`path_effect`** against the true Δ risk (external `z`, S3 generator): bias within 2 SE.
6. **Metric unit tests:**
   - the propriety Monte Carlo of `tvc-deviance.md` §2.3;
   - the Poisson-GLM identity (the score equals the `statsmodels` / hand-computed Poisson log-likelihood with a log-exposure offset, up to the data-only constant);
   - `alpha=0` → `-inf` on zero-rate event cells;
   - the window decomposition sums to the total.
7. **`oob_cumhaz`:** summing over grid times equals `oob_mortality` bit-for-bit.
8. **Deviance follow-ups** (`tvc-deviance.md` §8): a time-varying baseline, OOB vs held-out agreement, and the zero-rate share across n and `min_events_leaf`. These tune the `windows` default.

## 8. Slice sketch (the plan stage refines this)
1. **S16 metric:** `piecewise_exponential_score`, `event_windows`, `null_window_rates`; the Rust `oob_cumhaz` (+ CR twin); the unit tests of §7.6–7.7; deviance follow-ups §7.8.
2. **S17 counting-process importance:** `rftvc.inspection.permutation_importance` for `SurvivalForestTV` / `CompetingRisksForestTV` (strata, groups, conditional, windows, causes, SE, OOB); sims §7.1, §7.3 and §7.4.
3. **S18 landmark importance + history features:** raw-column groups, landmark strata, Brier/IBS scoring, `slope` / `std` aggregations; sim §7.2.
4. **S19 effects + foundation docs:** `hazard_effect`, `path_effect`, sim §7.5; the foundation page, a user-guide page on importance and effects, and a case study.

## Risks
| Risk | Mitigation |
|---|---|
| The M1 bias the research suspects may be small in practice | §7.1 reports M1 vs M2 either way; M2 stays the default because it is never worse on support grounds |
| Small strata (many conditioning bins, few rows) permute little → importance biased toward 0 | `n_unpermuted` reported; warn if > 10 % of rows are unpermuted |
| Users compare PE scores across different windows | `window_edges` in every result; docs: compare only at the same windows |
| OOB importance has no SE | documented; held-out is the default |
| `path_effect` read causally | docstring and user-guide wording (§4.2); internal-covariate warning in the docs |
| API growth before the release pass | one new module, sklearn naming; the release pass reviews it with everything else |

## Out of scope (v1)
- Refit/LOCO importance (M4; v2 candidate `drop_column_importance`).
- Rule-release/VarPro (M5).
- Minimal depth (M6).
- TreeSHAP with vector leaves and DynSHAP-style attributions (M7).
- Knockoffs.
- Subsampling SEs for OOB importance.
- Kernel-smoothed scoring.
- Parametrised history features.
