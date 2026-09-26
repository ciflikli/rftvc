# S16 slice plan: PE score, training null, fit design, OOB cumulative hazards

Branch `feat/s16-pe-score`. Parent: `tvc-plan.md` S16 (v2, Codex plan review applied); `tvc-design.md` T4, T6, T9, T10; `tvc-deviance.md`.
Notation:
- `K` = event times (`event_times_`), `J` = causes.
- `M` windows with edges `w_0 = 0 < … < w_M = τ`.
- Cell `(r, m)`: row `r` × window `W_m = (w_{m−1}, w_m]`.

## Decisions (defaults; revisit in review)
1. **`_fit_design` is a pure extraction.**
   - `_BaseForestTV._fit_design(X, y, ids, measured_at, gap_policy, layout, block_time) -> FitDesign` holds today's `_fit` lines from `split_frame` through `_block_design`, without the engine call or the OOB.
   - `FitDesign` (private NamedTuple): `X, start, stop, event, groups, n_ids, kept, block_time, fit_rows (X, start, stop, event), units, n_units, oob_set, names, ids_values`.
   - `_fit` = `_fit_design` + resolve leaf/draw params + `fit_forest` + attributes + OOB. The statement order, and therefore the RNG use (`check_random_state` after the design), is unchanged.
2. **Fit options and fingerprint.**
   - `_fit_options_` is a dict: `layout`, `gap_policy`, `has_measured_at`, `has_block_time`, and the design-relevant constructor params `ntime`, `resample_unit`, `block_length`, `oob_buffer`.
   - `_fit_fingerprint_` is the SHA-256 hex digest over:
     - the validated `X` (float64 C bytes), `start`, `stop` and `event` codes;
     - `cp.unit` (int64);
     - `measured_at` (float64) or the bytes `b"-"`, and `block_time` likewise;
     - `repr(sorted(_fit_options_.items()))`.
   - A private `_rebuild_design(X, y, ids=None, measured_at=None, block_time=None)` re-runs `_fit_design` with the *stored* options and raises `ValueError("data do not match the fitted data")` on a fingerprint mismatch. S17's `oob=True` uses it.
3. **`baseline_cumhaz_`, the training null (T10).**
   - It is the pooled Nelson–Aalen of the **fitted rows** (`fit_rows` after coarsening and block splitting) on `event_times_`: `Λ₀(t_k) = Σ_{j ≤ k} d_j / Y_j`, with `Y_j = #{rows: start < t_j ≤ stop}` and `d_j` = events at `t_j`.
   - Shape `(K,)` for survival; `(J, K)` for CR (per-cause `d`, the same `Y`).
   - `_event_counts_` (private, `(K,)` all-cause) is stored for `event_windows`.
   - In coarse mode `event_times_` is the grid, and grid points whose events were lost have `d = 0`.
4. **`event_windows(estimator, n_windows=8)`.**
   - The interior edges are inverse-CDF quantiles of the training event-time distribution *with multiplicity*: `t_(⌈k·D/n⌉)` for `k = 1..n−1`, where `D = Σ d` and `t_(i)` is the `i`-th event in time order, from `_event_counts_`.
   - Edges are `[0, interior…, τ = event_times_[-1]]`, deduplicated, so there can be fewer than `n_windows` windows under heavy ties.
   - Right-closed windows put an event at an edge in the lower window.
5. **`piecewise_exponential_score` (in `metrics.py`), streamed by window (Q1).**
   - For `m = 1..M`: `e = clip(min(stop, w_m) − max(start, w_{m−1}), 0)`, `N = event ∧ (w_{m−1} < stop ≤ w_m)`, and `ΔΛ̂ = cumhaz[:, m] − cumhaz[:, m−1]`.
   - The rate is `λ̃_α = (1 − α)·ΔΛ̂/|W_m| + α·ΔΛ₀/|W_m|`.
   - The contribution is `N·log λ̃_α − λ̃_α·e`, summed into window, cause and id accumulators.
   - `0·log 0 := 0`: cells with `N = 0` never take the log.
   - Zero rate: `λ̃ = 0` (before mixing) with `N = 1` counts toward `zero_rate_share`. With `α = 0` the contribution is `-inf`.
   - Events with `stop > τ` or `stop ≤ 0` go to `n_truncated_events`. Exposure outside `(0, τ]` is ignored.
   - `reduce="per_event"` divides every total by the number of scored events (all causes, or cause `k` when `cause=k`); `"sum"` returns raw sums. With zero scored events → `UndefinedMetricError`.
   - **Returns** `PEScore(total, by_window, by_cause, by_cause_window, by_id, id_labels, zero_rate_share, null_total, n_events, n_truncated_events)`.
     - `by_cause*` are None for survival.
     - `by_id` and `id_labels` are None without `ids`; `id_labels` are the unique ids in order of first appearance.
     - `null_total` is the same computation with `ΔΛ̂ := ΔΛ₀` (so the mixture is the identity).
   - **CR:**
     - `cumhaz (n, J, M + 1)` and `null_cumhaz (J, M + 1)`;
     - `causes` (the estimator's `causes_`) is required and maps `y` labels to axes;
     - a `y` label not in `causes` raises;
     - `cause=k` selects one axis;
     - `cause=None` sums over causes (the cause-specific likelihood factorises).
   - **Validation:**
     - `windows` is 1-D, finite and strictly increasing, with `windows[0] == 0` and `M ≥ 1`;
     - `alpha ∈ [0, 1)`;
     - `cumhaz` shape `(n, M + 1)` / `(n, J, M + 1)`, finite (NaN → error naming the rows: OOB rows with no tree must be excluded by the caller);
     - `null_cumhaz` shape;
     - `y` is survival or CR as the cumhaz shape implies.
   - `ids` is an array (not a column name) of length n.
6. **Rust `oob_cumhaz`.** A row kernel `oob_row(x_r, row_units, in_bag, times, agg, buf) -> n_trees` (the filter + `ensemble_cumhaz` of today). `oob_mortality` calls it and then sums with `buf.iter().sum()` (unchanged order). `oob_cumhaz` returns the buffers `(n × T)` plus `n_trees`, with NaN rows where `n_trees == 0`.
   - `oob_cause_cumhaz` is the OOB twin of `predict_cause_cumhaz` (tree average of `cause_cumhaz_at`; always hazard-averaged, as today) `→ (n × J × T, n_trees)`.
   - Bindings `Forest.oob_cumhaz(x, offsets, units, times, aggregate_by, n_jobs)` and `Forest.oob_cause_cumhaz(x, offsets, units, times, n_jobs)`.
7. **Public surface.**
   - `rftvc.metrics.piecewise_exponential_score`, `rftvc.metrics.event_windows`, `rftvc.metrics.PEScore`.
   - Estimator attribute `baseline_cumhaz_` (documented in both estimators' Attributes).
   - `api.rst` lists the new metrics. `_fit_options_` / `_fit_fingerprint_` / `_event_counts_` / `_rebuild_design` stay private.
8. **Old pickles** lack `baseline_cumhaz_`. `event_windows` and S17 raise `AttributeError("refit: this forest predates baseline_cumhaz_")`. Pre-release, so there is no migration.
9. **T9 guard:** `score` / `oob_score_` stay concordance.

## Tasks
- [ ] T1 `_fit_design` + `_fit_options_` + fingerprint + `_rebuild_design` (Python), `_fit` via it.
- [ ] T2 `baseline_cumhaz_` / `_event_counts_` (survival and CR).
- [ ] T3 Rust `oob_row` kernel, `oob_cumhaz`, `oob_cause_cumhaz`, `oob_mortality` through the kernel; bindings; `cargo test`, clippy.
- [ ] T4 `metrics.event_windows`, `piecewise_exponential_score`, `PEScore`.
- [ ] T5 Tests (below).
- [ ] T6 Identity: extend `bench/s9_identity.py` with CR (`CompetingRisksForestTV` cif/cumhaz/oob) and block-mode cases; baseline on a fresh `main` build, compare on this branch (arrays bit-identical; pickle sizes grow by `baseline_cumhaz_` + private attributes — reported, not gated).
- [ ] T7 Deviance follow-up bench `bench/tvc_deviance_followup.py` + the `windows` default decision.
- [ ] T8 Memory gate (1M rows, M = 64) and `oob_cumhaz` timing; docs (`api.rst`, estimator Attributes); `tvc-plan.md` tick + "S16 done" note.

## Tests
`tests/test_pe_score.py`:
- **Poisson-GLM identity:** random cells; `S` equals `Σ [N log μ − μ] − Σ N log e` with `μ = λ̃ e` computed densely (a hand reference), to 1e-12; deviance relation `D = −2S − 2ΣN log e − 2ΣN`.
- **Propriety regressions** (seeded Monte Carlo, n = 200k, one window): the score maximised over a grid of constant rates is at `ΣN/Σe` (±1 grid step). The v0 helper (in the test file only) prefers the constant hazard to the Weibull truth.
- **Boundaries** (hand-computed literals):
  - an event at interior edge `w_1` goes to window 1;
  - an event at `τ` is scored;
  - an event after `τ` is counted in `n_truncated_events` with its exposure cut at `τ`;
  - rows with `start == τ` or `start > τ` contribute nothing;
  - zero-exposure cells with zero rate give no NaN;
  - an empty window gives `by_window = 0`.
- **Decompositions** (`reduce="sum"`): `by_window.sum() == total`; `by_cause.sum() == total`; `by_cause_window.sum(1) == by_cause`; `by_id.sum() == total`; per-event = sum / `n_events`.
- **Mixture:**
  - `alpha=0` with a zero-rate event → `-inf` and `zero_rate_share > 0`;
  - `alpha=0.01` is finite;
  - a share > 1 % warns;
  - `null_total` is independent of `cumhaz`.
- **CR labels:**
  - labels {2, 5}: per-cause totals equal single-cause scoring of each axis with events recoded;
  - `cause=5` equals the axis-1 result;
  - a `y` without cause 5 scores axis 1 by compensator only;
  - label 7 raises.
- **Errors:** non-increasing edges, `windows[0] != 0`, alpha out of range, shape mismatches, NaN cumhaz, missing `causes` for CR.
- **`event_windows`:**
  - on hand-built event counts gives the inverse-CDF edges;
  - with ties, fewer windows;
  - last edge `τ`;
  - old-pickle-style estimator (attribute deleted) raises.

`tests/test_oob_cumhaz.py`:
- `np.cumsum(H, 1)[:, -1]` of `oob_cumhaz` equals `oob_prediction_` bit-for-bit for `SurvivalForestTV` in id, block (`oob_buffer` 0 and 1), coarse (`ntime`), `bootstrap=True`, and `aggregate="survival"` modes. Rows are those of `_rebuild_design`.
- The NaN pattern equals `oob_n_trees_ == 0` (a forced case with few trees).
- CR: `oob_cause_cumhaz` on the full `event_times_` grid → discrete AJ in NumPy equals `oob_cif` (`aggregate="hazard"`) at the last time to 1e-12.
- Full-bag control: a forest where no row is OOB for any tree gives all-NaN, like `oob_mortality`.

`tests/test_fit_design.py`:
- `_fit_design` twice → equal arrays; the global `np.random` state is unchanged, and it runs under `random_state=None` without drawing (spy on `check_random_state`).
- `_rebuild_design` reproduces the fit design (arrays equal) in id, block, coarse, and `layout="stacked"` + `ntime` + `block_time` modes.
- The fingerprint changes when `X`, `y`, `ids`, `measured_at`, `block_time` or a design option (`set_params(ntime=…)` after fit) changes, and `_rebuild_design` then raises.
- `baseline_cumhaz_` equals a NumPy pooled Nelson–Aalen over the design rows (id, block, coarse, stacked). CR: per-cause, and the sum over causes equals the all-cause pooled NA.
- **T9:** `score` equals `concordance_index_cp(y, predict(X))`, and `oob_score_` equals `concordance_index_cp` on the OOB rows (as today).

## Acceptance
- The full suite is green, plus `cargo test` and clippy.
- T6 identity: all arrays bit-identical to `main` (pickle-size growth reported).
- `oob_cumhaz` (100k rows, 9 edges) ≤ 1.2× `oob_mortality` time.
- PE scoring at 1M rows, M = 64: `tracemalloc` peak ≤ 2× the `(n, M + 1)` float64 prediction array.
- T7 bench run, and the `windows` default decided by the `tvc-plan.md` rule; the result is recorded in the S16 note.

## Review log
(pending: Codex plan review)
