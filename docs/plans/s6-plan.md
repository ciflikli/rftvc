# S6 slice plan: coarse grid mode + performance pass + benchmarks

Branch `feat/s6-coarse-grid`. Parent: `plan.md` S6, `design.md` D8 "Time grid and coarsening".

## Decisions (defaults; revisit in review)
1. **API:** `SurvivalForestTV(ntime=None)`. `None` = exact (unchanged). `ntime=K` = coarse mode (opt-in; D8.3).
2. **Event grid `Grid::quantile(K)`:** the empirical quantiles (inverse CDF, `sorted[ceil(q·n) − 1]`) of the event times (with multiplicity) at levels `j/K`, `j = 1..K`, deduplicated. There are at most `K` points, and each is an observed event time. The last point is the maximum event time.
3. **Snapping grid = event grid ∪ {earliest `start`}.** `g(t)` = the smallest snapping point `≥ t`, or `+∞` if there is none.
   - **Clarifies D8:** with event points alone, `g(0) = t_1`. Every subject entering at 0 would then be delayed to `t_1` and be out of the risk set there. Adding the origin (earliest entry, which is below every event time) keeps entry at the origin exact.
   - Other entries round up, as D8 specifies: an entry in `(t_{k−1}, t_k]` gets `start' = t_k`, so the row is at risk from `t_{k+1}` on (risk is `start' < t_j <= stop'`). A row entering and ending in the same bin collapses. (Review finding 1 corrected an earlier wording, "counts from `t_k`".)
4. **Row transform (D8.2), per observation chain.** For `layout="counting_process"` a chain is a validated group (an id, or a segment under `split_id`). For `layout="stacked"` every row is its own chain, while resampling groups stay per id:
   - `start' = g(start)`, `stop' = g(stop)`.
   - A row with `start' == stop'` (including both `+∞`, i.e. after the last event) is dropped.
   - If a dropped row carries the event, the event moves to the unit's previous **kept** row. That row ends at the same point by contiguity.
   - If the chain has no kept row, its event is lost and counted.
   - A stacked chain is one row, so a collapsed stacked event row is always dropped and counted. The id's other rows stay.
5. **One training bundle.** `coarsen()` returns the kept original row indices, `(start', stop', event')` and re-indexed resampling groups (groups with no kept row are removed).
   - Fitting uses the unchanged exact-mode machinery on that bundle.
   - `n_ids_`, `min_ids_leaf_` and `n_draw_` are resolved from the bundle.
   - `event_times_` becomes the event grid.
   - Diagnostics: `coarse_grid_`, `n_coarsen_dropped_rows_`, `n_coarsen_lost_events_`.
   - OOB: `oob_prediction_` keeps the original row order, NaN for dropped rows. `oob_score_` is computed on the kept rows with their coarsened outcomes. The `split_id` guard still uses the pre-coarsening segmentation.
6. **Prediction unchanged:** leaf hazards live on grid times, and `cumhaz_at(t)` is a step function for any `t`.
7. **Performance pass, driven by the profile:**
   - Baseline profile with macOS `sample` on a large exact-mode fit (1M rows) and a coarse-mode fit.
   - Candidate fixes:
     - Reuse split-search buffers instead of allocating `nb·(K+1)` per feature per node.
     - Skip features with a single used bin before building histograms.
     - Sibling subtraction (design: optional, benchmarked). **Not implemented.** The profile shows row accumulation is a small share of split search (scoring and the O(bins·K) prefix dominate), and subtraction saves only accumulation. It is also unspecified with node-local event grids and per-node feature sampling (review finding 5). Documented in perf.md.
   - Every change must keep all oracle tests bit-identical or within 1e-12.
8. **Benchmarks (`bench/compare.py`):**
   - Data: synthetic, one row per id, right-censored, p = 10, at n = 10k / 100k / 1M.
   - Settings, matched (review finding 4): 100 trees; `min_ids_leaf` / `min_samples_leaf` = 15; `min_events_leaf=1` (sksurv has no event minimum); `max_features="sqrt"`; `bootstrap=True, max_samples=0.632` on **both** arms; `n_jobs=10` on both.
   - Arms: rftvc exact, rftvc `ntime=100`, sksurv RSF.
   - Measures: fit time and peak RSS from `/usr/bin/time -l` in a subprocess per run.
   - Escalation rule: run the next size only if the previous one took < 10 min **and** its peak RSS × 10 < 12 GB. Otherwise record "skipped (projected T s / M GB)". Timeouts and OOM are recorded as outcomes.
   - A multi-row TVC set (≈5 rows per id) runs for rftvc only.
   - Report in `docs/bench/s6-perf.md`; profile in `docs/scratch/perf.md`.
9. **Targets** are written back into design.md from the measured numbers, e.g. "1M rows × 100 trees coarse < N s on 10 cores; peak memory < M × data size".

## Tasks
- [x] Rust: `Grid::quantile`, snapping grid, `coarsen()` + unit tests; binding `_core.coarsen`
- [x] Python: `ntime` param, fit wiring, diagnostics, docs
- [x] Tests: table-driven D8 fixtures (non-first collapsed event row, first-row collapse → dropped + counted, in-bin entry, in-bin censoring, beyond-last-event rows, stacked layout); single-node Λ on coarsened data = `nelson_aalen_ref`/lifelines on hand-built coarsened rows; coarse forest = exact forest on hand-built rows (same seed)
- [x] Baseline profile → `docs/scratch/perf.md`
- [x] Performance fixes (profile-driven), with the oracle suite green after each
- [x] `bench/compare.py` → `docs/bench/s6-perf.md`
- [x] design.md targets; plan.md tick + "S6 done" notes

## Plan review (Codex, 2026-09-25)
1. Entry wording wrong under ceil snapping (high) → corrected (item 3); D8 semantics kept.
2. No coherent post-coarsening training contract (OOB, `"auto"`, `n_draw_`) (high) → one bundle (item 5).
3. Stacked chains vs resampling groups (high) → chains separated (item 4); two diagnostics.
4. Unmatched benchmark (sampling, `n_jobs`, events per leaf, memory) (medium) → item 8.
5. Sibling subtraction unspecified with node-local grids (medium) → not implemented; profile evidence (item 7).
