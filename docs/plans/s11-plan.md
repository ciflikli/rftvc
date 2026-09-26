# S11 slice plan: competing-risks core

Branch `feat/s11-cr-core`. Parent: `cr-plan.md` S11 (P1–P6), `cr-design.md` v2 (C1–C8).
Scope: cause-coded `y`, a cause dimension in the engine, the composite and single-cause criteria, per-cause leaves, `CompetingRisksForestTV.fit` and per-row prediction (covariates fixed from time 0). Paths, coarsening, block resampling, OOB, `aggregate="cif"` and `score` wait for S12 and raise `NotImplementedError` (P2).
Notation: `J` causes (internal codes 1..J, 0 = censored), `K` grid times, `T` requested times.

## Decisions (defaults; revisit in review)

### Engine (Rust)
1. **`SurvData.event: Vec<u8>` codes plus `n_causes`.**
   - `SurvData::new(start, stop, event: &[bool])` stays as the J = 1 constructor (Rust tests and the debug bindings use it).
   - New `SurvData::with_causes(start, stop, codes: &[u8], n_causes)` asserts every code is `<= n_causes`.
   - `Grid::exact(stop, codes: &[u8])` keeps every time with `code != 0`, so a time at which only cause 2 has events is a grid point.
   - `Grid::quantile` and `coarsen` keep bool events in S11 (they are internal; S12 widens them to codes when the estimator first coarsens J > 1 data). This supersedes `cr-plan.md`'s "accept codes" wording.
2. **Profile.** `Profile` gains `cause_events: &[f64]` (cause-major, `[j*K + k]`; changed from time-major during implementation, see the S11 note in `cr-plan.md`) and `n_causes`. For J = 1 it aliases `events`. `LtrcLogRank` and `LogRankNode` are not changed.
3. **Splitter.**
   - `LocalRow.event: bool` becomes `cause: u8` (same 24-byte row). Event tests become `cause != 0`.
   - `NodeProfile` adds `cause_events` (K × J, empty for J = 1), `n_causes` and per-cause event counts `n_cause_events` (used by S12's `min_events_leaf_cause`).
   - In `best_split_in`, the left child's per-cause events go into one running `K × J` buffer, reset per feature and filled only when J > 1.
   - **No `bins × K × J` histogram exists:** the left child grows bin by bin over one running buffer, as today. So the extra memory is `O(K·J)` per node, and the "root histograms" risk in `cr-plan.md` does not apply (to be confirmed by the fit-time and memory check).
4. **Criteria** (`criterion.rs`):
   - `CompositeCauseLogRank`: score `Σ_j U_j² / V_j` over causes with `V_j > 0`.
     - Its `node_scorer` precomputes, per (time, cause), `mask`, `e = d_j / y` and `c = d_j (y − d_j) / (y² (y − 1))`, as `LogRankNode` does. Each candidate then costs `O(K·J)` multiply-adds.
     - `score` (the naive form) is kept for the unit test that compares both forms.
   - `SingleCause { k }` (0-based internal index): the `LogRankNode` precomputation on the parent's cause-k column. The candidate loop reads `l.cause_events[j*J + k]` in place, with the same operations in the same order as `LogRankNode`, instead of copying the column.
     - Extra all-cause event times have `mask = e = c = 0` and add exact zeros, so the score equals `LtrcLogRank` on "cause k vs rest" bit-for-bit.
   - **Dispatch (P4a):** J = 1 always uses `LtrcLogRank`, whatever `criterion` / `split_cause` say. J > 1 uses `SingleCause` when `split_cause` is set, and `CompositeCauseLogRank` otherwise.
5. **Leaves.**
   - `Tree` gains `n_causes`. `cumhaz` is entry-major: `cumhaz[e*J + j]` is cause j's Nelson–Aalen cumulative hazard at the leaf's entry `e`.
   - Leaf entries stay the node's all-cause event times. A cause without an event at an entry repeats its previous value.
   - The J = 1 branch is the current code, so leaf arrays do not change.
   - `Tree::cause_cumhaz_at(leaf, t, j)` is added; `cumhaz_at` stays (J = 1).
6. **Forest.**
   - `Forest` gains `n_causes`.
   - `predict_cause_cumhaz(x, times) → (n, J, T)` uses `"hazard"` aggregation only: the mean over trees of `cause_cumhaz_at`, accumulated in the same order as `ensemble_cumhaz`. So with J = 1 it equals `predict_cumhaz(…, Hazard)` bit-for-bit.
   - `predict_cif(x, times) → (F (n, J, T), S (n, T), n_clamped)` (P1):
     - Per row, each tree's leaf increments `ΔΛ_j = cumhaz[e, j] − cumhaz[e−1, j]` are added into a dense per-thread `K × J` buffer. A list of touched grid indices is kept, so a row costs `O(Σ leaf entries · log + T)`, not `O(K·J)`, and the buffer reset touches only those entries.
     - The sorted touched indices are divided by `n_trees` and swept with the discrete Aalen–Johansen update, `F_j += S·ΔΛ_j` and then `S *= max(0, 1 − Σ_j ΔΛ_j)`, merged with the requested times.
     - `n_clamped` counts the times at which `1 − Σ ΔΛ < 0`. Under hazard aggregation `Σ_j ΔΛ_j = mean_b d_b/y_b ≤ 1`, so a clamp can only come from rounding when a leaf's whole risk set fails at once. It is returned as a diagnostic.
7. **Flat format v3.**
   - Adds `n_causes`. The invariant is `cumhaz.len() == event_idx.len() * n_causes`, and leaf validation checks every cause column (finite, `≥ 0`, non-decreasing).
   - Saving always writes v3. A v2 state loads as J = 1 with identical predictions. `n_causes` must be in `1..=255`.
   - A J = 1 pickle grows by one scalar entry. The S9 identity bench prints pickle bytes but does not compare them.

### Binding
8. **`fit_forest`.**
   - `fit_forest(x, start, stop, event: u8[], …, n_causes=1, split_cause=None)` validates `event <= n_causes` and `split_cause` in `1..=n_causes`. J = 1 dispatches as in decision 4.
   - A `criterion` string is not added yet: `"composite"` is the only J > 1 rule until the S14 bake-off (P5), and the Python class validates the string.
   - `SurvivalForestTV` passes `event.view(np.uint8)`, a zero-copy view of the bool array.
9. **New and changed calls.**
   - `Forest.n_causes`, `Forest.predict_cause_cumhaz`, `Forest.predict_cif`.
   - `leaf_profile` returns `(times, cumhaz (n_e, J))`, always 2-d. All 8 test callers switch to `[:, 0]`: `test_oob.py:25`, `test_tvc.py:50,65`, `test_blocks.py:56,129`, `test_forest.py:66`, `test_tree_oracle.py:36,45`.
   - Debug `cause_score(start, stop, event u8, n_causes, left, split_cause=None)` gives the Rust score for the reference tests.
   - `logrank_score` and `best_split` keep bool events.

### Python
10. **`_validation.py`.**
    - `CR_DTYPE = [start f8, stop f8, event int64]`: external labels keep their value (any non-negative int64, e.g. 300); only the number of causes is capped at 255.
    - `make_competing_risks_y(stop, event, start=None)` accepts bool or non-negative integer labels (a float array with integral values is accepted, as pandas makes them).
    - `check_competing_risks_y(y, causes=None) → (start, stop, codes u8, causes_)`:
      - `causes_` is the sorted observed non-zero labels, or `np.asarray(causes)` sorted when a vocabulary is given (unique, positive integers, at most 255).
      - It raises `ValueError` for labels outside `causes`, and for no events.
      - With a vocabulary, a missing cause gives a `UserWarning` (its hazard is 0).
      - DataFrame `y` works as in `_structured`.
      - Internal codes: `code = searchsorted(causes_, label) + 1` (0 stays 0), cast to `u8` only after the vocabulary check. The engine never sees external labels.
    - `check_survival_y` / `_as_bool`: the error for integers outside {0, 1} names `CompetingRisksForestTV`.
    - `check_counting_process` uses `(event != 0) & not_last`. This is the same for bool, and it now catches label 2.
11. **`_BaseForestTV` (private) in `_estimator.py`.**
    - It holds the shared fit body, `_check_predict`, parameter validation, the resolvers and `apply`.
    - Hooks: `_check_y(y) → (start, stop, event)`, `_engine_kwargs()`, and `_check_supported()` (raises for S12 options).
    - `SurvivalForestTV` keeps its `__init__`, docstring and methods. The fit body moves unchanged; the RNG draws happen in the same order.
12. **`CompetingRisksForestTV`** (new `_competing.py`).
    - `__init__` is `SurvivalForestTV`'s parameters (same defaults, incl. `n_estimators=500`; `cr-plan.md`'s `100` was a typo) plus `causes=None, criterion="composite", split_cause=None, score_cause=None`. `aggregate` is `{"hazard", "cif"}`.
    - Supported in S11: `ntime=None`, `resample_unit="id"`, `oob_score=False`, `aggregate="hazard"`. Anything else raises `NotImplementedError` naming S12.
    - `fit(X, y, ids=None, *, measured_at=None, gap_policy="error", layout="counting_process")`. Stacked layout works: landmark stacks are S13, but nothing blocks it. Counting-process rows with delayed entry fit (P3). `block_time` is not a parameter in S11.
    - `split_cause` and `score_cause` are labels and must be in `causes_` (`ValueError`). `score_cause=None` means `causes_[0]`.
    - `predict_cumulative_incidence(X, times=None, *, cause=None, intervals=None, ids=None, origin=None, extrapolate="none")` gives `(n, J, T)`, or `(n, T)` for one cause label. Path keywords raise `NotImplementedError` (S12).
    - `predict_cumulative_hazard(X, times=None, *, cause=None)`: `(n, J, T)`; a label gives `(n, T)`; `cause="all"` gives the sum over causes.
    - `predict_survival_function(X, times=None)` is the Aalen–Johansen event-free survival `S = Π(1 − Σ_j ΔΛ_j)`. This is the product limit, **not** `exp(−Λ)` as in `SurvivalForestTV`; the docstring says so.
    - `predict(X)` returns `F_{score_cause}` at the last event time. `score` raises `NotImplementedError` (Wolbers C is S12).
    - Attributes: `causes_`, `n_causes_`, `event_times_` (unique times with any event), `n_ids_`, `n_units_`, `forest_`.

## Tests
- **Leaf oracle:** `max_depth=0`, one tree and `max_samples=1.0` use every id. The CIF equals `survival::survfit(Surv(start, stop, cause) ~ 1, id=id)` `pstate` to 1e-10, on start–stop rows with delayed entry, ties (including cross-cause ties and censoring at an event time) and a cause-2-only time.
  - Fixture: `tests/fixtures/make_aj_fixtures.py` calls Rscript, as the existing fixture scripts do (not a bare `.R` file), and writes `aj_survfit.json`.
  - The oracle also equals `cr_ref.py`'s naive AJ, and the per-cause cumulative hazard equals the naive Nelson–Aalen.
- **J = 1 bit-identity:** on {0, 1} events, `CompetingRisksForestTV` equals `SurvivalForestTV` bit-for-bit for node arrays (via the pickled state), leaf arrays, `predict_cumulative_hazard` and `apply`. It is tested with `criterion="composite"` and with `split_cause=1`, several seeds and `max_features ∈ {1, None}`, with ids and delayed entry. The identity contract is RNG use, tree structure, leaf hazards, `apply` and the hazard-aggregated cumulative hazard only; `predict`, the CIF and `S` are different quantities. So the test also asserts that the CIF equals `1 − Π(1 − ΔΛ)` computed in numpy from `SurvivalForestTV`'s hazard, and that it differs from `SurvivalForestTV.predict_risk` (`1 − exp(−Λ)`).
  - A binding test: J = 1 `leaf_profile` returns a `(n_e, 1)` array equal to the old values.
- **Single-cause score** (Rust unit test, random data): `SingleCause{k}` on J = 3 data equals `LtrcLogRank` on "cause k vs rest" data, bit-for-bit, with each node on its own event times. Also through `best_split_in`: the same best split (feature, bin, score) when the admissibility counts coincide (`min_events_leaf = 0`).
- **Composite score:**
  - The Rust `score` (naive) equals the `node_scorer` form to 1e-12 relative (Rust).
  - `cause_score` equals `cr_ref.composite_ref` (hypothesis: small random data, J ∈ {2, 3}, delayed entry, ties, empty children, `y < 2` times, a cause with no events).
  - A covariate that raises cause 1 and lowers cause 2: composite > 20 while the all-cause log-rank < 1.
- **Grid:** a time that only cause 2 has is in `event_times_` and moves the leaf's cause-2 CIF there (covered by the oracle fixture; also asserted directly).
- **Splitter:** `child_summaries_match_brute_force` extends to per-cause child counts (J = 3 data), and the J = 1 case still passes.
- **Structure:** a non-terminal event with label 1 or 2 raises; codes pass through `check_counting_process` unchanged.
- **AJ invariants:** on random fits, `Σ_k F_k + S = 1` to 1e-12, `F_k` is non-decreasing and `S` non-increasing. `n_clamped` is 0 on normal data. A single-leaf case where the whole risk set fails at the last time gives `S = 0` there, with `n_clamped ≤ 1` (whether it clamps depends on rounding).
- **Labels:**
  - Labels {2, 5} give `causes_ == [2, 5]`, and output axis 0 is label 2; labels {7, 300} (above 255) with and without `causes=[7, 300]` map to codes 1, 2.
  - `causes=[1, 2, 3]` with no label-3 events gives a zero cause-3 hazard and CIF plus a `UserWarning`.
  - Unknown `cause`, `split_cause` or `score_cause` raises `ValueError`. A label in `y` outside `causes` raises `ValueError`. Negative labels raise, and so do more than 255 causes.
- **Unsupported options:** each P2 option raises `NotImplementedError`.
- **Pickle:**
  - The v3 round trip is bit-identical (J = 2).
  - A v2 state (a v3 J = 1 state with `n_causes` removed and `format_version = 2`) loads with identical predictions.
  - Corrupt states raise `ValueError` without panicking: wrong stride, `n_causes = 0`, and a decreasing cause column.
  - The existing future-version test in `test_forest.py` moves from `format_version=3` to 4 (v3 is now valid).
- **Validation:** `make_competing_risks_y` / `check_competing_risks_y` (dtype, DataFrame input, float labels); `check_survival_y` rejects label 2 with the pointer to the new class.
- The existing suite stays green, including the S9 identity guards. `cargo test`, `cargo clippy -D warnings` and `pytest` all pass.

## Acceptance
- The oracles pass.
- `bench/s9_identity.py`: the branch equals a `main` build bit-for-bit (dumps in `docs/scratch/s11-*`, not committed).
- Fit time at 100k rows (`bench/perf_fit.synth`-like data, labels split at random into two causes): J = 2 ≤ 1.5 × J = 1. The J = 2 fit memory is reported. Both are recorded in the S11 note.

## Tasks
- [x] Baseline: `bench/s9_identity.py` dump on a `main` build
- [x] Rust: `SurvData` codes, `Grid::exact`, `Profile` / `NodeProfile` / splitter cause dimension, criteria + dispatch, leaves, `Forest::predict_cause_cumhaz` / `predict_cif`, flat v3; Rust unit tests
- [x] Binding: `fit_forest` codes / `n_causes` / `split_cause`, `predict_cause_cumhaz`, `predict_cif`, `n_causes`, `leaf_profile` 2-d, `cause_score`, v3 state
- [x] Python: validation helpers, `_BaseForestTV`, `CompetingRisksForestTV`, exports
- [x] Tests: `tests/ref/cr_ref.py`, AJ fixture, `tests/test_cr_core.py`, validation + structure tests, updated `leaf_profile` callers; test-quality audit
- [x] Identity bench + J = 2 timing; `plan.md` / `cr-plan.md` status + "S11 done" note
- [x] Codex diff review; fixes

## Plan review (Codex, 2026-09-26)
1. The J = 1 identity claimed `1 − S` equality, but `S` is a product limit in the new class and `exp(−H)` in `SurvivalForestTV` (high) → the contract is narrowed to trees, leaves, `apply` and the cumulative hazard. A relationship test was added, and the `cr-plan.md` wording was corrected.
2. Mapping external labels to internal codes was implicit, and the label dtype was questioned (high). `i8` in numpy is already int64 → written as `int64` explicitly. The `searchsorted` mapping is stated, and there are tests for labels above 255.
3. There are 8 `leaf_profile` callers, not 6 (high) → all are listed. A J = 1 shape test was added.
4. The existing future-version pickle test uses `format_version=3`, which is now valid (medium) → it moves to 4.
5. The parent plan says quantile/coarsen "accept codes" in S11 (low) → **declined**. They are internal Rust functions, and widening them now means untested code paths. The slice plan now overrides the parent wording, and S12 widens them.

## Test-quality audit (2026-09-26)
- `test_split_cause_uses_that_causes_log_rank` only asserted that forests differ, so a label → code off-by-one would pass (weak). It was replaced by a feature-attribution test: cause 3 depends on x0 and cause 8 on x1, and the root split must follow `split_cause`.
- `test_step_helper` tested an unused reference helper (slop) → both were removed.
- Mutation checks: swapping the AJ update order fails 17 tests; summing only the first cause in the composite fails the hypothesis reference test.

## Diff review (Codex, 2026-09-26)
1. A direct `_core.fit_forest` call with zero feature columns panicked in `max_features.clamp(1, 0)` (medium; also on `main`) → `ValueError`. Test added.
2. Direct binding calls with NaN / infinite times, or `start >= stop`, panicked in the grid sort or fitted rows with no at-risk time (medium; also on `main` for the bool path) → `check_times` in both binding constructors. Tests added.
3. `oob_buffer` (and a bare `resample_unit="block"` / `block_length`) passed or raised `ValueError` instead of the S12 `NotImplementedError` (medium) → the reserved options are checked before base validation. Tests added.
