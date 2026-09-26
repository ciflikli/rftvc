# S13 slice plan: cause-specific landmark metrics + landmark competing risks

Branch `feat/s13-cr-landmark`. Parent: `cr-plan.md` S13 (review findings 3–5), `cr-design.md` v2 ("Landmark workflow", "Metrics (C7)"). S12 delivered `concordance_index_cr` (whole follow-up, no IPCW). `SurvivalForestTV` / `LandmarkSurvivalForest` outputs must stay unchanged: the identity bench (`--pickle`) includes a PBC2 landmark stack.
Notation: `s` landmark, `w` horizon on the reset clock, `k` the cause of interest, `G` the censoring survival.

## Decisions (defaults; revisit in review)

### Metrics (`metrics.py`)
1. **Outcome classes at `w` with a cause.**
   - `_outcome_classes(stop, labels, w, cause=None)` returns `(case, competing, control)`:
     - case: `label == k` and `stop ≤ w`;
     - competing: another cause and `stop ≤ w`, an observed status of 0 by `w`;
     - control: `stop ≥ w` and neither.
   - Subjects censored before `w` get weight 0.
   - Weights: `1/G(stop−)` for cases and competing events, `1/G(w−)` for controls, clipped at `g_min`.
   - `cause=None` keeps today's classes exactly: `competing` is empty, and the case / control arrays are the current ones.
2. **`brier_landmark(y_test, risk, w, *, cause=None, …)`.**
   - The existing positional signature is kept, and `cause` is keyword-only.
   - With `cause=k`, `y_test` may carry cause labels (`CR_DTYPE`, or bool), `risk = F_k(w)`, and the score is `mean weight · (1{case} − risk)²`.
3. **`integrated_brier(y_test, surv, times, *, cause=None, …)`.**
   - With `cause=k`, the second argument holds **`F_k` at `times`**. It is the predicted risk, not a survival probability; the docstring says so.
   - Per-time Brier as in decision 2, trapezoid integration as today.
4. **`cindex_dynamic(…, cause=k)`.**
   - `kind="incident"`: the IPCW Wolbers C truncated at `w`. Case `i`: cause k with `T_i ≤ w`.
     - **Type A** controls are rows at risk at `T_i` with no event at `T_i`, weight `1/G(T_i−)²`.
     - **Type B** controls are competing events at `T_j ≤ T_i`, weight `1/(G(T_i−) G(T_j−))`.
   - Type A deviates from `cr-design.md`'s `1/(G(T_i−) G(T_i))`, so that with one cause the value equals today's Uno C bit-for-bit. The two agree whenever there is no censoring at `T_i`. **Design amendment** (plan review 1): `cr-design.md` "Metrics" is updated to this convention. Ties are handled as in the existing incident C, with events at `T_i` excluded as type-A controls and censoring at `T_i` still at risk. A fixture with censoring at a case time pins the weight, and it differs from the design's original form.
   - `kind="cumulative"` with `cause` raises `NotImplementedError`: competing-risks AUC (Blanche 2013) is deferred, as in the design.
   - `_competing_pairs` gains case / control weights (a float Fenwick tree); `concordance_index_cr` is unchanged.
5. **Censoring model with labels:** a `KaplanMeierCensoring` fitted on `y` with labels treats any label ≠ 0 as an event. `fit` accepts bool or labelled targets. For bool input its result is unchanged.

### Landmark (`landmark.py`)
6. **`make_landmark_data`** keeps integer labels.
   - `_lm_event` is the id's terminal label if `U ≤ s + w`, else 0. The label is read as `max` over the id's rows, which equals the last row's label because the structure check allows events only on the last row.
   - The event column is validated as bool or non-negative integers.
   - Output `y` is `SURV_DTYPE` when the column is Boolean or holds only {0, 1}, which keeps today's stacks **bit-identical**. Otherwise it is `CR_DTYPE`.
7. **`_LandmarkBase`** (private) holds `__init__`'s shared parameters, `_columns`, the stacking and block logic of `fit`, and `_features`. Subclasses provide `_default_forest()` and `_check_forest(forest, y)`.
8. **`LandmarkSurvivalForest`** is unchanged, except that it raises `ValueError` on labels > 1 and names `LandmarkCompetingRisksForest`.
9. **`LandmarkCompetingRisksForest(horizon, history_features, landmarks, step, forest=None, id, start, stop, event, measured_at, causes=None, score_cause=None)`.**
   - `forest` defaults to `CompetingRisksForestTV()`; any other class raises. `causes` and `score_cause` are set on the forest clone when given.
   - `predict_cumulative_incidence(df, s, times, cause=None)` returns `(ids, F)`, with `F` of shape `(n, J, T)`, or `(n, T)` for one cause.
   - `predict_risk(df, s, horizon=None, cause=None)` returns a polars DataFrame with the id, `landmark`, `cause` and `risk = F_k(s + w | s, H(s))`; `cause=None` means `score_cause`, else the first label.
   - `predict_survival_function(df, s, times)` returns the event-free `S`.
   - `times` must lie in `[0, horizon]`, as today.

### Model selection (`model_selection.py`)
10. **CR branch** in `landmark_cross_validate` / `_score_landmark` / `_select`, used when the model is a `LandmarkCompetingRisksForest`.
    - **One vocabulary for all folds** (plan review 2): before splitting, `causes` is the model's `causes`, or else the labels observed in the full stacked data. `k` is the model's `score_cause`, or else `causes[0]`. Every outer and nested-inner fit is a clone with `causes=` and `score_cause=` set, so no fold can switch cause or change J. The fold predicts `F_k` at `times`.
    - Scores are `brier`, `integrated_brier` and `cindex_incident` with `cause=k`. `cindex_cumulative` is rejected up front with a `ValueError`.
    - Custom scoring callables receive `cause=k` as a keyword.
    - `return_predictions` gives `cif` (list of `F_k` at `times`) and `cause` columns instead of `survival`; `risk = F_k(w)`.
    - The survival path is unchanged: same columns and values.
11. **`_censor_at`** (time splitters) keeps the label: `when(stop ≤ cutoff).then(event).otherwise(0)`, in the column's dtype. Bool columns give the same values as today.

## Tests (`tests/test_cr_metrics.py`, `tests/test_cr_landmark.py`)
- **Brier / IBS oracle:** a committed fixture `tests/fixtures/cr_brier.json` from `comprisk` 0.8.0 (`evaluation._per_time_auc_brier`, KM censoring with events first). It is generated by `tests/fixtures/make_cr_metric_fixtures.py` against a scratch install, so there is no new dev dependency. The data have continuous times (none at an evaluation time, where the conventions differ); `g_min` is tiny. Our `brier_landmark(cause=k)` and per-time `integrated_brier` values match to 1e-12.
- **Hand-worked boundary fixtures** (plan review 3; the comprisk oracle excludes times at `w`), with values written out as literals:
  - censoring exactly at `w` is a control weighted by `1/G(w−)`;
  - a competing event exactly at `w` is weighted by `1/G(w−)` with outcome 0;
  - a case exactly at `w`;
  - censoring before `w` gets weight 0;
  - an event/censoring tie before `w` checks the events-first reverse KM.
- **Dynamic C tie:** censoring at a case time `T_i` gives the `1/G(T_i−)²` value, not `1/(G(T_i−) G(T_i))`.
- **One cause:** on bool data, `cause=1` gives `brier_landmark`, `integrated_brier` and `cindex_dynamic(kind="incident")` exactly equal to `cause=None`, with and without IPCW.
- **Dynamic Wolbers C:** equals an O(n²) brute force with the same `G` (hypothesis, ties, groups). `kind="cumulative"` with `cause` raises.
- **Landmark data:**
  - A bool event column gives `SURV_DTYPE` stacks equal to today's. An int {0, 1} column gives the same stacks.
  - A cause-coded column gives labels only on terminal rows within the horizon, and a competing event beyond `s + w` is censored at `s + w`.
  - A non-terminal label raises, and so does a negative label.
- **Wrappers:**
  - `LandmarkSurvivalForest` on labels > 1 raises with a pointer.
  - `LandmarkCompetingRisksForest`: `causes=` keeps J fixed on a training subset missing a cause; `predict_risk` equals the forest's `F_k` at `w` on `landmark_features`; clone / get_params (nested `forest__…`) / pickle round trip.
- **CV:**
  - Folds lacking different causes: every fold scores the same cause, CIFs keep J columns, and there is no error.
  - `landmark_cross_validate` on the CR wrapper equals a manual loop (fold fit, `F_k`, `brier_landmark(cause=k)`, `cindex_dynamic(kind="incident", cause=k)`) for `GroupKFold` and `RollingOriginSplit`.
  - Nested selection runs.
  - Survival-model CV output is unchanged (existing tests + identity bench).
  - `_censor_at` keeps labels.
- **PBC2 fast path:** `pbcseq` with transplant (1) vs death (2) labels, a small landmark grid and a few trees; CV runs end to end, and scores are finite at most landmarks.

## Acceptance
- The metric references pass.
- The identity bench is bit-identical to `main` with `--pickle` (post-S12 baseline).
- PBC2 competing-risks landmark CV runs in the fast test path.
- The existing suite is green.

## Tasks
- [x] Baseline: identity dump on `main`
- [x] Metrics: classes, `brier_landmark` / `integrated_brier` / `cindex_dynamic` with `cause`; KM on labels; comprisk fixture script + JSON
- [x] Landmark: labels in `make_landmark_data`, `_LandmarkBase`, `LandmarkCompetingRisksForest`, exports
- [x] Model selection: CR branch, `_censor_at` labels
- [x] Tests + test-quality audit
- [x] Identity bench; docs (`api.rst`); `cr-plan.md` status + "S13 done" note
- [ ] Codex diff review; fixes

## Plan review (Codex, 2026-09-26)
1. The type-A weight `1/G(T_i−)²` contradicted the design's `1/(G(T_i−) G(T_i))` (high) → kept for exact J = 1 equality with the existing Uno C. It is recorded as a design amendment in `cr-design.md`, and a censoring-at-`T_i` fixture pins it. It is flagged for the user in the S13 summary.
2. Per-fold `causes_[0]` could silently score a different cause in a fold missing cause 1 (high) → one vocabulary and score cause are resolved before splitting and pinned in every outer and inner fit. Test with folds lacking different causes.
3. The comprisk oracle excludes times at `w` (medium) → hand-worked boundary fixtures for censoring, competing events and cases at `w`, plus the events-first KM tie.

## Test-quality audit (2026-09-26)
- Oracles are independent: the comprisk fixture (Brier, per-time IBS), hand-worked literal values (the boundary fixture 41/350, KM with events first, and the type-A tie 13/17), and a brute-force weighted Wolbers C (hypothesis).
- The CV-vs-manual test mirrors the loop on purpose. It pins the plumbing (vocabulary and cause pinning, prediction columns); the metrics are checked by the oracles.
- The time-split / nested CR test is a smoke test (weak by nature); the same plumbing is covered for survival by S5's tests.
- Mutation check: treating competing events as censored (weight 0) fails the comprisk oracle for both causes, the boundary fixture and the brute-force C.
- A real bug found while testing: `cause=0` was used as an internal "any labels" sentinel, so `brier_landmark(..., cause=0)` skipped validation → replaced by an explicit flag. Test added.
- A pre-existing S10 flake (unseeded 5-tree block-OOB test, ~1.5% "no comparable pairs") failed `main` CI after the S12 merge → seeded (separate commit).
