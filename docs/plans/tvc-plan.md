# Implementation Plan: TVC foundation, importance and effects (S16–S20)

Status note: v2 (2026-09-26), Codex plan review applied (Review log).
Source of truth: `tvc-design.md` v2 (T1–T10; T2/T3 user-confirmed). Background: `tvc-research.md`, `tvc-deviance.md`. Conventions, oracles and the slice workflow are those of `plan.md`:
- branch `feat/sN-*`;
- a slice plan `sN-plan.md`, with a Codex plan review before coding;
- a Codex diff review, a test-quality audit of new tests, an "SN done" note, and a user-approved merge.

Notation: `M` windows, edges `w_0 = 0 < … < w_M = τ`; `S` = α-mixed piecewise-exponential (PE) score; units = features or groups that are permuted or dropped together.

## Status
- [x] S16: PE score metric, training null (`baseline_cumhaz_`), `_fit_design` refactor, Rust `oob_cumhaz` (branch `feat/s16-pe-score`)
- [x] S17: `inspection.permutation_importance` for counting-process estimators (branch `feat/s17-perm-importance`)
- [x] S18: `inspection.drop_column_importance` (LOCO, cross-fitted) (branch `feat/s18-loco`)
- [ ] S19: Landmark importance (raw-column units, landmark strata, Brier/IBS scoring) + `slope` / `std` history features (branch `feat/s19-landmark-importance`)
- [ ] S20: `hazard_effect`, `path_effect`, foundation + importance user guide, case study, bench sims (branch `feat/s20-effects-docs`)

## Plan-level decisions (defaults; the slice plan reviews may change them)
- **Q1. The metric is NumPy, streamed by window.**
  - Cells are computed one window at a time (exposure, count, rate and contribution are 1-D per window), and events are stored as `bool` / index arrays, not float64 matrices. Peak memory is O(n) plus the `(n, M + 1)` prediction array.
  - This matters at scale: a dense float64 `(n, M)` array is 512 MB at 1M rows and M = 64, and a naive implementation needs several.
  - The forest supplies only `Λ̂` at the `M + 1` edges (`predict_cumulative_hazard(X, times=edges[1:])` with an implicit 0 at `w_0`), so no new predict path is needed.
- **Q2. One Rust row kernel.** `oob_cumhaz` reuses `ensemble_cumhaz` per row. `oob_mortality` becomes "`oob_cumhaz`'s kernel, then a sequential sum of the row buffer", the same order as today, so `oob_mortality` stays bit-identical. The test compares against `np.cumsum(H, axis=1)[:, -1]`, which is sequential, not `H.sum` (pairwise).
- **Q3. `_fit_design` is a pure refactor.**
  - It returns `FitDesign(kept, X, start, stop, event, units, n_units, oob_set, groups)`, and both `_fit` and OOB importance call it.
  - Its "no RNG" property is asserted by a test.
  - The S9 identity bench (`bench/s9_identity.py --pickle` against a fresh `main` baseline) must be bit-identical.
- **Q4. Module layout.**
  - Public `src/rftvc/inspection.py`: thin, documented functions that dispatch on the estimator family.
  - Private `_inspection/`: `_strata.py` (stratum assignment, within-stratum permutation), `_units.py` (feature/group resolution, landmark raw-column groups), `_score.py` (PE / Brier scoring glue, by-window/cause/id decomposition), `_boot.py` (id-cluster bootstrap), `_effects.py`.
  - `rftvc/__init__.py` exports `inspection`.
- **Q5. Estimator families.**
  - *Counting-process*: `SurvivalForestTV`, `CompetingRisksForestTV`, detected by `_BaseForestTV`.
  - *Landmark*: `LandmarkSurvivalForest`, `LandmarkCompetingRisksForest`, detected by `_LandmarkBase`.
  - Anything else raises `TypeError`. There is no duck typing: the functions rely on `event_times_`, `baseline_cumhaz_` and the path keywords.
- **Q6. Reproducibility.**
  - `random_state` → one `np.random.Generator`. Permutations are drawn in a fixed order (unit-major, then repeat, then stratum), and bootstrap replicates use child generators (`SeedSequence.spawn`).
  - The same seed gives identical results for any `n_jobs`: parallelism is over units, with pre-drawn permutations.
- **Q7. Simulations.** Heavy (R = 50) simulations live in `bench/tvc_*.py`, with results recorded in the slice note. `tests/test_tvc_sim_truth.py` holds reduced versions (R ≈ 10) marked `@pytest.mark.slow` (not merge gates, as `test_sim.py`), plus one fast smoke per scenario (R = 1, loose rule) in the default suite.
- **Q8. Units and names.**
  - Importance is in score units **per event** (`S / Σ N`), the same per-event normalisation in every result.
  - `feature_names` follow `feature_names_in_` when the estimator was fitted on a DataFrame, and indices otherwise.
- **Q9. `plan.md`** gets a status line "S16–S20: TVC foundation/importance — plan and status in `tvc-plan.md`".

---

## S16: PE score, training null, fit design, OOB cumhaz
**Files:**
- Rust core `forest.rs`: `oob_cumhaz(x, p, offsets, units, times, agg) -> (Vec<f64> n×T, Vec<u32>)` and `oob_cumhaz_causes` (n×J×T), sharing a row kernel with `oob_mortality` / `oob_cif`. `oob_mortality` is re-expressed through it.
- Binding `rftvc-py/src/lib.rs`: `Forest.oob_cumhaz(x, offsets, units, times, aggregate_by, n_jobs)` and `Forest.oob_cumhaz_causes(...)`, mirroring `oob_mortality`.
- Python:
  - `_estimator.py`: `_fit_design`; `_fit` uses it; `baseline_cumhaz_` (pooled Nelson–Aalen of the fitted rows on `event_times_`); `_fit_fingerprint_`; `_fit_options_` (the fit-time design options `layout`, `measured_at`, `gap_policy` and whether `block_time` was given; stored so OOB importance rebuilds with exactly these).
  - `_competing.py`: `baseline_cumhaz_` of shape `(J, K)`.
  - `metrics.py`: `piecewise_exponential_score`, `event_windows`, `PEScore`.
- Tests: `tests/test_pe_score.py`, `tests/test_oob_cumhaz.py`; additions to `tests/test_oob.py` and `tests/test_blocks.py`.
- Bench: `bench/tvc_deviance_followup.py`, the `tvc-deviance.md` §8 / design §7.8 checks (see Accept).

**Signatures:**
- `metrics.event_windows(estimator, n_windows=8) -> ndarray (M + 1,)`: quantiles of `event_times_` at `k / n_windows`, with `0` prepended and `τ = event_times_[-1]` as the last edge, deduplicated.
- `metrics.piecewise_exponential_score(y, cumhaz, windows, *, null_cumhaz, alpha=0.01, causes=None, cause=None, ids=None, reduce="per_event") -> PEScore`.
  - `causes`: the label vocabulary of the CR axis of `cumhaz` (pass `estimator.causes_`). It is required for CR inputs. Labels in `y` are mapped through it; a `y` label outside it raises. A fitted cause absent from `y` scores with `N = 0` (compensator only), and `cause=` must be in `causes`.
  - `cumhaz`: `(n, M + 1)` or `(n, J, M + 1)`, including the column at `w_0` (all 0 for fixed-profile predictions).
  - `null_cumhaz`: `(M + 1,)` or `(J, M + 1)`.
  - `PEScore(total, by_window, by_cause, by_id, zero_rate_share, null_total, n_truncated_events)`.
- **Validation:**
  - edges finite, strictly increasing and starting at 0;
  - `alpha ∈ [0, 1)`;
  - shapes consistent;
  - `cause` a label in `causes` in the CR case;
  - rows truncated at `τ` (exposure and events after `τ` dropped and counted).
- Attributes: `baseline_cumhaz_` `(K,)` / `(J, K)` on `event_times_`; `_fit_fingerprint_` (a hex digest string).

**Tests:**
- **Poisson-GLM identity:** on random cells, `S` equals the hand-computed Poisson log-likelihood with offset `log e` minus the data-only term `Σ N log e`, to 1e-12. `D = −2S − 2 Σ N log e − 2 Σ N` holds.
- **Propriety regressions** (small, seeded Monte Carlo, from `tvc-deviance.md` §2): the piecewise-constant score peaks at `c* = ΣN / Σe` over a grid of constants; the row-own-exposure v0 score prefers a constant hazard to a Weibull truth. The second test documents *why* v0 is not implemented, and asserts nothing about library code beyond the helper.
- **Decompositions:** `by_window.sum() == total·ΣN` (with `reduce="sum"`), `by_cause` sums likewise, `by_id` sums likewise; per-event normalisation.
- **Zero rates:** `alpha=0` with a zero-rate event cell → `-inf` and `zero_rate_share > 0`; `alpha > 0` stays finite; share > 1 % warns.
- **Truncation:** a row that crosses `τ` contributes its exposure up to `τ`, and an event after `τ` is counted in `n_truncated_events`, not scored.
- **Boundaries (hand-computed):**
  - an event exactly at an interior edge `w_m` belongs to window `m` (right-closed), not `m + 1`;
  - an event exactly at `τ` is scored;
  - a row with `start == τ` or `start > τ` contributes neither exposure nor an event;
  - a zero-exposure (row, window) cell contributes 0, including when its predicted rate is 0 (no `0·log 0` NaN);
  - a window with no exposure at all has `by_window = 0`.
- **CR labels:** non-consecutive labels (e.g. {2, 5}) with `causes=[2, 5]` score each cause on the right axis (checked against per-cause hand computation); an evaluation `y` lacking cause 5 scores it by compensator only; an unknown label raises.
- **Errors:** bad edges; `windows[-1]` beyond the last event time when built for an estimator (via `event_windows`); a shape mismatch.
- **`baseline_cumhaz_`:** equals a pooled Nelson–Aalen computed **directly** from the `_fit_design` rows by a NumPy reference (`d(t_k) / Y(t_k)` over all fitted rows) in id, coarse and block modes. The single-tree `max_depth=0` forest is *not* the oracle: by default it sees only a subsample of units.
- **`_fit_design`:**
  - deterministic (two calls give equal arrays);
  - no RNG (a global `np.random` state is unchanged, and `random_state` is unused);
  - `fit` results bit-identical to `main` on the S9 identity bench.
- **`oob_cumhaz`:** `np.cumsum(H, axis=1)[:, -1]` equals `oob_mortality` bit-for-bit in id mode, block mode (with buffer), coarse mode and `bootstrap=True`. The CR twin at `event_times_[-1:]` satisfies AJ consistency with `oob_cif` (1e-12). The NaN pattern equals `oob_n_trees_ == 0`.
- **Fingerprint:** a refit on the same data gives the same fingerprint. It covers `X`, `y`, `ids`, `block_time` and `_fit_options_`, so any change in any of them changes it. An OOB rebuild with `layout="stacked"` + `ntime` (each row its own chain) reproduces the fit design (regression test).
- **T9 guard:** `score` and `oob_score_` are unchanged: still concordance, equal to `main` on the identity bench data. The PE metric is separate.

**Accept:**
- tests green;
- identity bench bit-identical;
- `oob_cumhaz` at 100k rows and 8 edges costs ≤ 1.2× `oob_mortality` (reported);
- PE scoring at 1M rows and M = 64 has peak memory ≤ 2× the `(n, M + 1)` prediction array (measured with `tracemalloc`, reported).

**Deviance follow-up bench** (`bench/tvc_deviance_followup.py`, results in the S16 note):
1. S3 with a Weibull baseline (shape 0.5 and 2): the score and permutation-importance stability vs M ∈ {2, 4, 8, 16}.
2. OOB vs held-out: the PE score and the importance ranking, measured as Spearman ρ of the importances.
3. The zero-rate share of events over n ∈ {200, 1000, 5000} ids and `min_events_leaf` ∈ {1, 3, 10}.

**Decision rule for the `windows` default:** the largest M in {4, 8, 16} whose zero-rate share is ≤ 0.1 % in the worst cell of check 3 and whose importance means stay within 10 % of the M = 4 values in check 1.
- If the rule gives M ≠ 8, the default changes, and the S16 note records it.
- If OOB and held-out rankings disagree (ρ < 0.8), `oob=True` is documented as a quick screen only.

---

## S17: `permutation_importance` (counting-process estimators)
**Files:** `inspection.py`, `_inspection/_strata.py`, `_units.py`, `_score.py`, `_boot.py`; tests `tests/test_inspection_perm.py`; bench `bench/tvc_perm_sim.py` (sims §7.1 M2 part, §7.3, §7.4 of the design).

**Signature:** as in design §3:
```python
permutation_importance(estimator, X, y=None, *, ids=None, features=None, groups=None,
    strata="time", n_strata=10, conditional_on=None, n_bins=4,
    scoring="pe", windows=8, alpha=0.01, cause=None,
    oob=False, block_time=None, n_repeats=5, n_bootstrap=100, random_state=None, n_jobs=None) -> Bunch
```
- `windows` is an int (→ `event_windows`) or edges.
- `oob=True` requires `X, y, ids` (and `block_time` if the fit used it) and verifies them against `_fit_fingerprint_`.
- `strata`: `"time"` | `None` | array of labels. `strata=None` (M1) emits a `UserWarning` ("naive permutation can extrapolate off the (time, value) support; prefer strata='time'"), and its docstring carries the same warning (T5). Time strata are the `n_strata` quantile bins of the evaluation rows' `start` (bin edges from unique quantiles).
- `conditional_on`: column names/indices, crossed with `n_bins` quantile bins each. Strata with < 2 rows stay unpermuted and are counted.

**Algorithm:**
1. Resolve the units, windows and null. Predict the intact `Λ̂` at the edges and score it (`baseline_score`).
2. For each unit `j` and repeat `b`: permute the unit's columns jointly **with one shared row permutation per stratum** (so a group moves together), then predict and score. In OOB mode the prediction uses `oob_cumhaz` on the design rows.
3. Compute `ΔS` total, by window, by cause and by id.
4. Bootstrap (`n_bootstrap > 0`, held-out only): resample evaluation ids with replacement (duplicated ids are relabelled so that strata see copies as distinct rows), redraw the permutations within strata, re-predict only the permuted unit, and rescore. `importances_se` = sd over replicates of the repeat-averaged `ΔS`.
5. Return the Bunch fields of design §3.1.

**Tests:**
- **Oracle model:** a hand-built "model" (a subclass of `SurvivalForestTV` whose cumhaz depends only on column 0) gets ≈ 0 importance on column 1 and > 0 on column 0, exactly 0 on a column it ignores.
- **Strata respected:** after permutation, each value stays inside its stratum (the multiset per stratum is unchanged), with `n_strata=1` ≡ a global shuffle.
- **Groups move jointly:** rows of a 2-column group keep their pairing after permutation.
- **Reproducibility:** the same `random_state` gives the same result for `n_jobs=1` and `4`.
- **Decomposition:** `importances_window.sum(axis=1)` equals `importances_mean·ΣN` (per-event scaling handled); CR `importances_cause` sums to the total.
- **OOB:** a fingerprint mismatch raises. The OOB baseline score equals the PE score of `oob_cumhaz` computed directly.
- **Bootstrap:** the SE is finite with `n_bootstrap ≥ 2`, NaN with 0 and with `oob=True`.
  - The resampler is a private function `_resample_ids(ids, rng) -> (row_index, boot_ids)`, tested directly:
    - every sampled id copy contains **all** of that id's original rows, in order;
    - a duplicated id's copies get distinct `boot_ids`;
    - `by_id` decomposes by `boot_ids`;
    - strata assignment sees each copy's rows as distinct rows.
  - A pure-noise column's bootstrap interval covers 0 in ≥ 90 % of 20 seeded runs (a coarse sanity check, not a coverage study).
- **M1 warning:** `strata=None` warns (`pytest.warns`), and the docstring contains "extrapolat" (a doc-content assertion).
- **Conditional:** with `conditional_on` a copy of the permuted column, the importance is exactly 0 (every stratum is constant in the column).
- **Errors:** unknown feature, overlapping groups, `strata` of the wrong length, `oob=True` without training data, a landmark estimator (→ S19 path; S17 raises `NotImplementedError` until then, per the P2 convention of `cr-plan.md`).
- **Slow sims:** design §7.1 (M2 part: `|mean imp(z2)| ≤ 0.1·oracle(z1)`, `mean imp(z1) ≥ 0.5·oracle(z1)`), §7.3 (Holm windows), §7.4 (CR).

**Accept:** fast tests green; the bench results for §7.1/§7.3/§7.4 are reported in the S17 note with pass/fail against the declared rules. A failed statistical rule blocks the merge unless the user accepts a documented deviation.

---

## S18: `drop_column_importance` (LOCO)
**Files:** `inspection.py`, `_inspection/_loco.py`, the reused fold plumbing from `model_selection.py` (`_censor_at`; the splitter checks factored into a private `_cv_folds(estimator, X, y, ids, cv)` shared with `landmark_cross_validate`); tests `tests/test_inspection_loco.py`; bench `bench/tvc_loco_sim.py`.

**Signature:**
```python
drop_column_importance(estimator, X, y=None, *, ids=None, cv=5, features=None, groups=None,
    scoring="pe", windows=8, alpha=0.01, cause=None, n_seeds=1, add_noise_control=False,
    random_state=None, n_jobs=None) -> Bunch
```
- **Folds:**
  - `cv` int → `GroupKFold(cv)` on ids.
  - A splitter object is used as given, with the same checks as `landmark_cross_validate`:
    - `GroupTimeSplit` / `RollingOriginSplit` → administrative censoring at the test start and the `gap` rules;
    - any other splitter must keep ids disjoint.
- **Per fold:** fit the full model (per seed), then windows = `event_windows(full)` and null = `full.baseline_cumhaz_`. Fit one clone per dropped unit (`X` without the unit's columns) with the same seed. Score each on the test fold with the full model's windows and null.
- **Seeds:** `random_state` → per-fold, per-seed integer seeds (the same across units within a fold/seed).
- **Noise control:** `add_noise_control=True` appends a standard-normal column (drawn per row from the generator) to `X` before fitting, and reports it as unit `"_noise"`.
- **SE:**
  - Id splitters: per-id `ΔS_i` (seed-averaged), pooled over folds; `importances_se = sd(ΔS_i)·√n_ids / ΣN`.
  - Time splitters: NaN SE and per-fold values only.

**Tests:**
- **Oracle:** on data where only `x0` matters (the S3 generator), LOCO of a pure-noise column has a mean within ±3 SE of 0, and LOCO of `x0` is > 0 (seeded, small).
- **Groups:** two identical copies of `z` → single-copy LOCO ≈ 0 (|·| ≤ 0.1 × grouped), and grouped LOCO > 0.
- **Folds:** every id is scored exactly once (id splitters). Time splitters apply `_censor_at` (a training frame never contains rows after the test start); `gap < horizon` raises for landmark models, as in `landmark_cross_validate`.
- **Consistency:** windows and null come from the fold's full model (the dropped models are scored on them).
- **Seeds:** results are reproducible, and `n_seeds=2` averages two fits (checked through a spy `fit` count).
- **Cost guard:** the fit count equals `(p_units + 1) × n_folds × n_seeds` (+1 unit with the noise control).
- **DataFrame / ids column:** with a DataFrame `X` and `ids="id"` naming a column in `X`:
  - the default units exclude the ids column;
  - naming it in `features` or `groups` raises;
  - dropping an ordinary column refits on a frame whose remaining names still pass `feature_names_in_` validation (the ids column is kept in every refit frame);
  - the grouping by ids is identical across the full and dropped fits.
- **Slow sims:** design §7.1 (LOCO part), §7.7b.

**Accept:** fast tests green; sims reported against the rules; wall time for the S3 bench size is reported in the note (feeds the docs).

---

## S19: Landmark importance + history features
**Files:** `landmark.py` (`AGGREGATIONS += ("slope", "std")`; spec → raw-column map `_raw_groups(history_features)`), `inspection.py` (landmark dispatch in `permutation_importance` and `drop_column_importance`), `_inspection/_units.py`, `_score.py` (Brier/IBS scoring via `metrics.brier_landmark` / `integrated_brier` with the `landmark_cross_validate` IPCW settings); tests `tests/test_landmark.py` (aggregations), `tests/test_inspection_landmark.py`; bench `bench/tvc_landmark_sim.py` (§7.2, §7.5b).

**Details:**
- **`slope`:** the OLS slope of the column on `measured_at` (else `start`) over the rows known at `s`; NaN if < 2 distinct times. **`std`:** `ddof=1`; NaN if < 2 rows. It uses the polars expressions in `_agg_expr`.
- **Permutation for landmark models:**
  - `X = df`. Stack with the model's settings (`make_landmark_data`), and use strata = `s` (fixed; `strata=` other than `"landmark"` raises).
  - Units default to raw-column groups; `features=` names landmark features or raw columns (a raw column expands to its group). `landmark` is not permutable.
  - `conditional_on` crosses the landmark strata with quantile bins of the named features.
  - Predictions use `model.forest_.predict_cumulative_hazard` on stacked rows at horizon-clock edges.
- **Scoring:**
  - `"pe"` (windows on `(0, horizon]`, null = `forest_.baseline_cumhaz_`); Brier/IBS go **per landmark** through `_score_landmark` (per-landmark censoring fits), then aggregate as `landmark_cross_validate` does, never as a pooled stacked-data IPCW;
  - `"brier"` (at `horizon`, `risk = predict_risk`-equivalent on stacked rows);
  - `"ibs"` (over `n_times` points on `(0, horizon]`).
  - IPCW `censoring_estimator` and `g_min` are passed through with the same defaults as `landmark_cross_validate`. CR models require `cause` for Brier/IBS.
- **LOCO for landmark models:** dropping unit `z` removes every `history_features` spec on `z` (`clone(model).set_params(history_features=…)`); folds come from `_cv_folds` with the landmark rules.
- **SE:** the id-cluster bootstrap resamples **ids** (all landmark copies of an id move together).

**Tests:**
- **Aggregations:** `slope` and `std` against NumPy on hand-built histories; NaN rules; the lookahead rule (rows with `start > s` are excluded) still applies.
- **Units:** `history_features=["z", ("z", "mean"), ("z", "slope"), "x"]` → groups `{"z": [z, z_mean, z_slope], "x": [x]}`.
- **Landmark strata:** after permutation, each landmark's multiset of each feature group is unchanged, and rows move between ids at the same `s` only.
- **The M3 equivalence check** (design §3): permuting the `z` group within `s` equals recomputing features from a donor id's raw history up to `s`, row-by-row, for a fixed permutation (constructed test).
- **Level/history diagnostic:** a generator where the hazard depends on `z_mean` only → history-given-level importance > 0; a Markov control → ≈ 0 (fast, loose; strict in the slow sim §7.2).
- **Brier/IBS scoring:** the Brier/IBS scoring *is* `_score_landmark`: per landmark, with the censoring model fitted on that landmark's test outcomes, then aggregated exactly as `landmark_cross_validate` does. The test uses **several landmarks with censoring** and a single train/test split expressed as a one-fold splitter; the baseline equals the CV function's fold score to 1e-12.
- **CR:** `cause` is required for Brier; PE by cause.

**Accept:** tests green; §7.2 and §7.5b reported against the rules.

---

## S20: Effects, foundation docs, case study
**Files:** `_inspection/_effects.py`, `inspection.py` (`hazard_effect`, `path_effect`); docs `docs/source/user_guide/foundation.rst`, `docs/source/user_guide/importance.rst`, `api.rst` (the `inspection` section, the new metric), a case study `docs/source/case_studies/importance_*.py` (or a notebook, as the existing case studies do); estimator docstrings link to the foundation page; tests `tests/test_inspection_effects.py`; bench `bench/tvc_path_effect_sim.py` (§7.5).

**Signatures:**
- `hazard_effect(estimator, X, y, *, feature, values=None, windows=8, kind="average", ids=None, cause=None) -> Bunch(values, window_edges, hazard (n_values, M) | (n_values, J, M), support_mask, individual=None | (n_rows, n_values, M))`.
  - Estimand: the exposure-weighted mean over rows at risk in each window of `λ̃_m(x_r with x_j = v)`.
  - `values=None`: 20 quantiles of `x_j`.
  - `support_mask[v, m]`: `v` lies inside the 5–95 % range of `x_j` among the rows at risk in `W_m`.
- `path_effect(estimator, X, intervals, ids, *, feature, delta, from_time, horizons, origin=None, cause=None, extrapolate="none") -> Bunch(per_subject, mean, horizons)`.
  - The absolute clock and validation of design §4.2. Rows straddling `from_time` are split there (covariates copied).
  - `delta`: a float, or a callable `(values, start) -> values`.
  - Output shapes as in the design: CR `(n, J, H)` or `(n, H)`.
- Landmark estimators: `hazard_effect` on stacked rows (horizon clock); `path_effect` raises `TypeError` (paths are a counting-process concept).

**Tests:**
- **`hazard_effect`:**
  - an oracle model whose hazard is `c·exp(β v)` gives window hazards `c·exp(β v)` exactly;
  - the weights equal the exposures (a hand check on 3 rows);
  - `support_mask` is correct on constructed data;
  - `kind="individual"` averages (exposure-weighted) to `kind="average"`.
- **`path_effect`:**
  - `delta=0` → exactly 0;
  - on an oracle `SurvivalForestTV` path with a known piecewise hazard, the result equals the analytic change in risk;
  - straddling rows are split correctly (the result equals manually pre-split intervals);
  - validation errors (origin > from_time; horizon beyond the last stop without `locf`);
  - the CR shape and `cause` selection match `predict_cumulative_incidence`.
- **Docs:** `sphinx -W` builds; the case study runs in CI's docs job (as existing case studies do).
- **Slow sim:** §7.5 (`|mean bias| ≤ 0.1·|true Δ|`).

**Accept:**
- tests and docs green;
- the foundation page states only the claims of `tvc-research.md` §1 (as corrected), and the importance page says when to use M2 vs LOCO vs the landmark diagnostic, and warns against M1;
- `tests/test_docs_claims.py` (T1 guard) asserts that the foundation page:
  - contains the four assumptions;
  - contains the functionals table;
  - contains "no consistency result";
  - does **not** contain "consistent estimator" / "consistency of rftvc";
  - and that the importance page contains the M1 extrapolation warning;
- the case study shows importance, timing and the level/history diagnostic on one dataset;
- a bench summary table (all sims, pass/fail) is appended here as "S16–S20 results".

---

## Risks (plan-level)
| Risk | Mitigation |
|---|---|
| The `_fit_design` refactor changes fit results | identity bench bit-identical as the S16 gate; the refactor lands before any new behaviour in the slice |
| Bootstrap cost `p × n_bootstrap` predictions | only the permuted unit is re-predicted; `n_bootstrap=0` escape; timings in the S17 note |
| Declared statistical rules fail (e.g. M2 still biased) | report honestly; a failed rule blocks the merge unless the user accepts a documented deviation; the design may change default strata |
| LOCO wall time on large data | groups, `n_jobs` over fits, timings in docs; no default change to `n_estimators` |
| API drift before the release pass | one public module; sklearn names; the release pass reviews signatures with everything else |

## Review log (Codex plan review, 2026-09-26)
All 11 findings accepted:
1. **Deviance follow-ups missing** → S16 bench with a decision rule for the `windows` default.
2. **The M1 warning was not enforced** → a `UserWarning`, docstring and user-guide assertions.
3. **T1/T9 had no regression tests** → a docs-claims test (S20) and a `score`/`oob_score_` guard (S16).
4. **Wrong `baseline_cumhaz_` oracle** (a subsampled single tree) → a direct NumPy pooled Nelson–Aalen on the design rows.
5. **CR label mapping in the metric** → a `causes=` vocabulary argument, and tests for non-consecutive and absent labels.
6. **Brier equality to CV underspecified** → per-landmark `_score_landmark` with per-landmark censoring fits, and a multi-landmark censored test.
7. **The fingerprint missed fit options** (`layout`, `measured_at`, `gap_policy`) → `_fit_options_` stored and fingerprinted, and a stacked + coarse OOB test.
8. **Window boundary and zero-exposure cases untested** → hand-computed boundary tests.
9. **Bootstrap tests too weak** → `_resample_ids` tested directly (whole ids, distinct copy labels).
10. **The ids column in a DataFrame `X` under LOCO** → excluded from units, rejected when named, and refit frames keep it.
11. **Memory of dense intermediates** → window-streamed scoring and a 1M-row memory gate.
