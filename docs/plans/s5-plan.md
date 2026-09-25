# S5 slice plan: model selection, metrics, id-level OOB

Branch `feat/s5-model-selection`. Parent: `plan.md` S5, `design.md` "Model selection & metrics".

## Decisions (defaults; revisit in review)
1. **Metrics are per landmark, on the reset clock.** Inputs are right-censored outcomes `y` (fields `start`=0, `stop`, `event`) for the risk set at one landmark, as built by `make_landmark_data`. The landmark `s` is therefore implicit and **dropped from the metric signatures** (deviation from plan.md). They are plain right-censored metrics and work for any `start == 0` data.
2. **Censoring model:** reverse Kaplan–Meier `G(t) = P(C > t)`, events-before-censoring at ties (same as sksurv). Fitted on `y_censor`, or a user-fitted object with `predict(times, left=...) -> G` (`censoring_estimator`). Exactly one of the two. **User decision (2026-09-25, after plan review):** in `landmark_cross_validate`, `G_s` is fitted on the **test fold's risk set at `s`** (pec/riskRegression convention). Time-based splits never have training data at a test landmark, so design.md's "training fold at `s`" is impossible; pooling earlier landmarks would assume stationary censoring. `G_s` sees outcomes only, never predictions. Deviation from design.md/plan.md.
3. **IPCW Brier at `w`:** cases (`event`, `stop <= w`) weighted `1/G(stop−)`, controls (`stop >= w`, not a case) weighted `1/G(w−)`. Left limits make administrative censoring at exactly `w` (every landmark row of a survivor) a control, not a lost subject. Equals sksurv `brier_score` when no censoring time in `y_censor` equals a test `stop` or `w` **and** no test non-event has `stop == w` (sksurv controls are `stop > w`; landmark rows of survivors all have `stop == w`, which this convention counts as controls). `G` is clipped at `g_min`; `return_info=True` returns `(score, info)` with the clipped count.
4. **Exact path:** if every test subject is a case or a control (no censoring before `w`), the score is the plain mean `(1{case} − risk)²`; no censoring model is consulted.
5. **`integrated_brier(y, surv, times, ...)`:** trapezoid of the IPCW Brier over `times`, divided by the range (sksurv convention).
6. **`cindex_dynamic(y, risk, w, kind=...)`:** `"cumulative"` = cumulative/dynamic AUC at `w` (Uno 2007; sksurv `cumulative_dynamic_auc`); `"incident"` = integrated incident/dynamic concordance over `(0, w]` (Heagerty & Zheng 2005), estimated by Uno's truncated C (sksurv `concordance_index_ipcw(tau=w)`); documented as a pairwise estimand, not a single-time AUC.
7. **`calibration_table(y, risk, w, n_bins=10)`:** quantile bins of risk; per bin `n`, mean predicted risk, observed `1 − KM(w)`; returns polars. Descriptive; assumes censoring independent of the outcome within each bin (documented).
8. **Splitters, time in model units.** `RollingOriginSplit(n_splits, test_size, gap, time_col=None)`: test windows of width `test_size` stacked back from the last time, right-closed; train = times `<= min(test times) − gap`. `GroupTimeSplit` adds a `GroupKFold` partition: test = test groups ∩ test window, train = other groups ∩ train window. `split(X, y=None, groups=None)` takes times from `X[time_col]` or a 1-d `X`.
9. **`landmark_cross_validate(model, df, cv, scoring, horizon=None)`:** splits the landmark grid (and subjects for group splitters); raises if `cv.gap < horizon`. Training data are administratively censored at the earliest test landmark (belt and braces: the fit never sees later data). The model is cloned with `landmarks` = training landmarks. Test outcomes come from `make_landmark_data` at each test landmark; `G_s` is fitted on those test outcomes (decision 2); an optional unfitted `censoring_estimator` is cloned and fitted the same way. Returns a polars frame, one row per (fold, landmark).
10. **OOB (`oob_score=True`):** per row, the ensemble over trees where the row's id is out of bag (Rust, no stored bags). Errors if `gap_policy="split_id"` split an id (segments would be resampled separately). Prediction = ensemble mortality `Σ_k Λ(t_k | X_row)` over `event_times_` (sksurv convention), stored as `oob_prediction_` (NaN when no OOB tree). `oob_score_` = concordance for counting-process data: at each event time `T_i`, compare the event row with every other id's row at risk at `T_i` (`start < T_i <= stop`); pairs with the same event time or the same id are excluded; risk ties count ½. Equals R `survival::concordance` on counting-process input (fixture) and Harrell's C when `start == 0`. Documented as a *new-subject* estimate.
11. **Accept dataset:** no BTSCS data in the repo, so the nested rolling-origin example uses a simulated unit×period panel with a committed DGP (`tests/sim.py`); a real BTSCS case study is S7.

## Tasks
- [x] Rust `Forest::oob_mortality` + binding; `oob_score`/`oob_score_`/`oob_prediction_`
- [x] `concordance_cp` (counting-process C) + naive ref + R fixture
- [x] `metrics.py`: censoring KM, `brier_landmark`, `integrated_brier`, `cindex_dynamic`, `calibration_table` + sksurv parity tests
- [x] `model_selection.py`: splitters + `landmark_cross_validate` + leakage/spy tests
- [x] Simulated panel + nested rolling-origin example (`bench/s5_nested_cv.py`)
- [x] plan.md tick + "S5 done" notes

## Plan review (Codex, 2026-09-25)
1. IPCW source at test landmark (high) → user decision, item 2.
2. sksurv parity excludes test non-events at `stop == w` (medium) → item 3 wording; fixture avoids it.
3. "incident" is Uno's truncated C, not a single-time I/D AUC (medium) → documented as integrated I/D concordance (item 6).
4. Per-bin KM calibration assumes within-bin independent censoring (medium) → documented (item 7).
