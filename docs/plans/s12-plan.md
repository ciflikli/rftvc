# S12 slice plan: competing-risks paths, coarsening, `aggregate="cif"`, OOB

Branch `feat/s12-cr-paths`. Parent: `cr-plan.md` S12, `cr-design.md` v2 (C4, C5, "Prediction", "Metrics: OOB / Wolbers C"). S11 note: `cause_events` is cause-major; `SurvivalForestTV` must stay bit-identical to `main` (`bench/s9_identity.py`).
Notation: `J` causes, `K` grid, `T` requested times, `u` origin.

## Decisions (defaults; revisit in review)

### One Aalen–Johansen kernel (Rust, `forest.rs`)
1. **Row and path prediction share one kernel.**
   - A "path" is rows `r0..r1`, each with its own leaf per tree. Row `r` contributes the leaf's per-cause increments `ΔΛ_j(v) = cumhaz[e, j] − cumhaz[e−1, j]` at the leaf entries with grid time `v ∈ (max(start_r, u), end_r]`. `end_r = stop_r`, except the last row under `extrapolate="locf"`, where it is `+∞`.
   - Per-row prediction (covariates fixed from time 0) is the path of one row with `start = u = −∞` and `end = +∞`. This is the S11 `predict_cif` behaviour, which the S11 tests pin.
   - Increments go into the S11 dense `K × J` buffer with a touched-index list. One sweep produces `F_j`, `S` and `Λ_j = Σ ΔΛ_j` (the conditional cumulative hazard since `u`) at the requested times.
2. **Aggregation (C4).**
   - `"hazard"`: all trees' increments in one buffer, divided by `n_trees` in the sweep (as in S11).
   - `"cif"`: one sweep per tree (divisor 1); `F`, `S` and `Λ` are then averaged over trees. `Σ F + S = 1` holds for each tree, so it holds for the average too.
   - Under `"cif"`, `Λ` is the mean conditional hazard, the same as under `"hazard"`.
3. **NaN rules as in `predict_paths`:** a time `t < u`, or `t > last stop` with `extrapolate="none"`, gives NaN in `F`, `S` and `Λ`.
4. **`Forest::oob_cif`:**
   - Per row, the tree filter of `oob_mortality` (per-row CSR unit sets, bitsets). The kernel runs as per-row prediction on the OOB trees at `times = [last event time]`.
   - Returns `(n_rows × J F_j, n_oob)`. A row with no OOB tree is NaN.
   - This follows the design: `oob_prediction_` is `F_j` at the last event time, the analogue of mortality.
5. The S11 `predict_cif` / `predict_cause_cumhaz` stay the per-row entry points. `predict_cause_cumhaz` keeps its cumhaz-at-time form, bit-identical to `SurvivalForestTV` for J = 1. The S11 test that compares the CIF with the hazard's AJ transform still holds.

### Engine options
6. **`min_events_leaf_cause` (C5).**
   - `TreeParams` and `SplitParams` gain `cause_floor: Option<(cause, m)>`. It is set only with `split_cause` (Python raises otherwise).
   - Pre-RNG gate: `can_split` also needs `n_cause_events[cause] >= 2m`.
   - Child constraint: per-bin cause-`cause` event counts, so each child has at least `m` of them.
7. **Leaf diagnostics.**
   - `Tree.leaf_cause_events: Vec<u32>`, `n_leaves × J`, row-major: the in-bag events of each cause per leaf, counting bootstrap copies. It is filled only when `TreeParams.leaf_events` is set, which `CompetingRisksForestTV` always does.
   - `SurvivalForestTV` does not set it, so its memory, `nbytes` and pickles are unchanged.
   - Flat state: an optional `leaf_cause_events` key, **written only when counts were stored** (so SF pickles are byte-identical), and validated on load to exactly `total leaves × J` entries. The format stays **v3**: S11's v3 states have no key and load with empty diagnostics. No version bump, because the key is additive and optional.
8. **Coarsening with codes.**
   - `Grid::quantile(stop, codes)` uses `code != 0`.
   - `coarsen(…, codes: &[u8])` moves the dropped row's code onto the chain's previous kept row, which is a non-terminal row with code 0. Lost events are counted as before.
   - The binding's `coarsen` takes and returns `u8`. `SurvivalForestTV` passes a bool array viewed as `u8` and gets back a bool view, so its results are unchanged (checked by the identity bench).
9. **Block resampling.** `split_at_blocks` keeps the code on the last piece (`np.where(last, event[row], 0)` in the event dtype). For bool input the result is identical.

### Python
10. **`CompetingRisksForestTV`** now supports:
    - `ntime`, `resample_unit="block"` (+ `block_length`, `oob_buffer`, `fit(block_time=)`), `oob_score`, `aggregate ∈ {"hazard", "cif"}`, and `min_events_leaf_cause` (int ≥ 1 or None; it requires `split_cause`);
    - path keywords `intervals`, `ids`, `origin`, `extrapolate` on `predict_cumulative_incidence`, `predict_cumulative_hazard` and `predict_survival_function`. Their validation and ordering are shared with `SurvivalForestTV` through a `_BaseForestTV._path_args` helper. Output rows are subjects in order of first appearance, as for SF.
    - It also gets `score` and the OOB attributes below.
    - OOB with `ntime` needs the coarsened target, which the base rebuilds with `make_survival_y` (bool only). A hook `_oob_target(stop, event, start)` is added: SF keeps `make_survival_y`, and CR builds a cause-label target (`causes_[code − 1]`, 0 stays 0), so relocated events keep their label for Wolbers C.
11. **`metrics.concordance_index_cr(y, risk, cause, ids=None)`**, the cause-specific C of Wolbers et al. on counting-process rows with time-varying risk. It extends `concordance_index_cp`:
    - A case is a row with a cause-`cause` event at `T_i`.
    - **Type A** controls are rows at risk at `T_i` (`start < T_i ≤ stop`), excluding rows with an event of *any* cause at `T_i`. This is the existing `_concordance` with `event = code != 0` and `event_mask = code == cause`.
    - **Type B** controls are rows with a **competing** event at `T_j ≤ T_i`, using that terminal row's risk. Ties `T_j = T_i` are included: such a subject can never have a cause-`cause` event, and it is not a type A control at `T_i`, so no pair is counted twice.
    - `ids` excludes same-id controls in both types. With one cause there are no type B controls, and the result equals `concordance_index_cp` exactly.
    - `score(X, y, ids)` is this C on `predict(X)` for `score_cause`.
    - Delayed entry needs no extra rule for type B: `start_j < T_j ≤ T_i`.
12. **OOB.**
    - `oob_prediction_` has shape `(n_rows, J)`, with columns in `causes_` order: `F_j` at the last event time from each row's OOB ensemble, under the fitted `aggregate`.
    - Rows dropped by coarsening are all-NaN, with `oob_n_trees_ = 0`.
    - `oob_score_` is `concordance_index_cr` on the `score_cause` column, over rows with a finite prediction (warning as for SF).
13. **Estimator diagnostics:** `leaf_cause_events_summary()` returns a small polars DataFrame, one row per cause label, with `min`, `q10`, `median` and `max` of the per-leaf counts over all leaves. It raises when the forest has no stored counts (a state loaded from S11).

## Tests (`tests/test_cr_paths.py`, `tests/test_cr_oob.py`, `tests/test_sklearn_compat.py`)
- **Paths:**
  - A one-row path starting before every event time equals per-row prediction (`F`, `S`, `Λ`).
  - A multi-row path equals a Python reference that routes each row via `apply`, reads `leaf_profile` increments on `(max(start_r, u), stop_r]`, averages them over trees and applies AJ. This covers origins (`S(u) = 1`, `F(u) = 0`), `t < u` NaN, `t > last stop` NaN, `locf` beyond it, and `aggregate="cif"` (per-tree AJ, then the mean).
  - With J = 1, the path `Λ` equals `SurvivalForestTV.predict_cumulative_hazard(intervals=…)` to 1e-12: the same forest, and the increments are differences of the same values.
- **`"cif"` aggregation:** per-row values equal the mean of per-tree AJ from `leaf_profile` (Python reference). `Σ F + S = 1` holds under both rules.
- **Approach-B equivalence:** `split_cause=k`, `min_events_leaf=1`, `min_events_leaf_cause=m` and the same seed give tree node arrays and seeds equal to `SurvivalForestTV(min_events_leaf=m)` on "cause k vs rest". Each leaf's cause-k cumhaz at the SF leaf's event times equals SF's leaf cumhaz bit-for-bit. Tested for k ∈ {1, 2} and m ∈ {1, 3}, with ids and delayed entry.
- **`min_events_leaf_cause`:** every leaf created by an admitted split has at least `m` in-bag cause-k events (`leaf_cause_events`). A tree whose root has fewer than `2m` stays a single root leaf, which may hold fewer than `m`. Without `split_cause` it raises `ValueError`.
- **Coarsening:** `ntime` with causes equals fitting on data snapped by `coarsen_ref.py` (extended to codes: the code moves with the event). J = 1 `SurvivalForestTV` coarse results are unchanged (identity bench + existing tests).
- **Blocks:** the event piece keeps its label (`split_at_blocks` on codes); a block-mode CR fit runs with block OOB. One block per id is bit-identical to id resampling (as in S10).
- **Wolbers C:**
  - A hand-worked fixture (type A and B pairs, delayed entry, a tie `T_j = T_i` across causes, a same-id exclusion, time-varying risk).
  - A brute-force O(n²) reference on random data (hypothesis).
  - With J = 1 it equals `concordance_index_cp`.
- **OOB:**
  - It matches a manual reference built from `in_bag_ids`, `apply` and `leaf_profile`, under both aggregations, for id and block units.
  - Coarsening-dropped rows are all-NaN.
  - `oob_score_ == concordance_index_cr(...)` on the finite rows.
- **Leaf diagnostics:** `leaf_cause_events(tree)` equals brute-force counting of in-bag rows (with bootstrap multiplicity) routed by `apply`. A state without the key (S11 v3) loads and the summary raises. An SF state has no `leaf_cause_events` key; a CR state has one, and one of the wrong length is rejected.
- **OOB + `ntime`:** labels `[2, 5]`, `score_cause=5`, and a relocated event: `oob_score_` equals `concordance_index_cr` on the coarsened label target.
- **sklearn:** `parametrize_with_checks` for `CompetingRisksForestTV` (id and block) with the same expected-failure mapping. The adapted tests that apply to it (fit returns self, idempotence, pickle/clone, order invariance, feature names, pipeline + routed ids + `cross_validate` via `score`) run for both classes.

## Acceptance
- References pass to 1e-12 (bit-for-bit where stated).
- `bench/s9_identity.py` stays bit-identical to `main` (post-S11 baseline `docs/scratch/s12/main.npz`), **including SF pickle bytes**: `--compare` gains `--pickle`, which counts a size difference as a failure.
- The sklearn compat matrix covers the new class.
- The existing suite is green.

## Tasks
- [ ] Rust: AJ kernel (paths, origin, extrapolate, `"cif"`), `oob_cif`, `min_events_leaf_cause`, leaf cause counts (+ flat key), coarsen / quantile with codes; Rust tests
- [ ] Binding: `predict_cif_paths`, `oob_cif`, aggregate arg on `predict_cif`, `leaf_cause_events`, `fit_forest(min_events_leaf_cause=, leaf_events=)`, `coarsen` u8
- [ ] Python: `_blocks` codes; base `_path_args`; CR options, paths, OOB, `score`, summary; `metrics.concordance_index_cr`
- [ ] Tests (above) + `coarsen_ref` codes + test-quality audit
- [ ] Identity bench; `cr-plan.md` status + "S12 done" note
- [ ] Codex diff review; fixes

## Plan review (Codex, 2026-09-26)
1. "Every leaf has ≥ m cause-k events" is false for an unsplittable root, e.g. from a bootstrap bag with few cause-k events (high) → the test covers leaves from admitted splits only; a below-floor root leaf is allowed.
2. The coarsened OOB target was rebuilt with `make_survival_y`, which rejects label 2 and loses the score cause (high) → `_oob_target` hook; test with labels [2, 5], `ntime`, `score_cause=5`.
3. An always-emitted `leaf_cause_events` key would change every SF pickle (medium) → the key is written only when counts were stored; tests for the SF, CR and S11 states.
4. The identity bench did not fail on pickle-size changes (medium) → a `--pickle` strict mode and a fresh post-S11 baseline.
