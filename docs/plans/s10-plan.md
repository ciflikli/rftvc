# S10 slice plan: `"block"` resampling with its own out-of-bag spec

Branch `feat/s10-block-resampling`. Parent: `plan.md` "Deferred (post-v1)"; `design-principles.md` G2 and G4 (`resample_unit`, "Block = id × time-block, for long series with few ids"); `design-review.md` finding 5 (OOB is defined only for whole-id resampling; block OOB needs its own spec); `design.md` `resample_unit` / `oob_score` rows.

## Why
- With few ids and long series, whole-id resampling gives each tree only ~0.632 of the ids. Trees differ little, and `min_ids_leaf` stops splits early.
- Resampling id × time blocks keeps each block's rows together (short-range dependence stays inside a unit) and gives many more units.
- The engine is already unit-agnostic: `fit_forest` resamples whatever `groups` it gets, and `min_ids_leaf` / `max_samples` count those units. Most of the slice is the Python API and the OOB spec.

## Decisions (★ = user decision; 2026-09-26 the user accepted all recommendations: `block_length` on the time axis, user-defined blocks deferred, buffered OOB with default `h = 1`)
1. **API.** `SurvivalForestTV(resample_unit="block", block_length=L)`.
   - `L` is a positive finite float, required with `"block"`. Setting it with `resample_unit="id"` is an error, not silently ignored.
   - **Time-axis blocks (default):** block `k` of an id covers `(kL, (k+1)L]` on the model's time axis. For resampling, rows are **split at block boundaries** (plan review 2). This is exact for counting-process data: covariates are constant on a row, risk sets at every time are unchanged, the event stays on the last piece, and predictions are unaffected. So a block is exactly an id's person-time in that window, and no in-bag row reaches into another block's period.
   - **`block_time=` fit argument (plan review 1):** an optional per-row time used instead, with block `floor(block_time / L)` and no splitting (each row is a point on that clock).
     - It is required for `layout="stacked"`, whose `start` is 0 on every stack row. `LandmarkSurvivalForest` passes the landmark times `s` automatically.
     - It also covers the calendar-time case (analysis clock = duration). This supersedes the earlier "user-defined blocks deferred" default: it is a routed fit argument like `ids`, with `set_fit_request(block_time=True)`.
   - `gap_policy="split_id"`: blocks are keyed on the id, not the segment (the id is still the dependence unit).
2. **Unit counts.** Under `"block"`, `max_samples`, `min_ids_leaf` (and `"auto"` = `max(15, floor(sqrt(n_units)))`) count blocks. The docstring says "resampling units (ids, or id-blocks)"; the parameter keeps its name. New attributes: `n_units_` (blocks drawn from) and `oob_n_trees_` (see 4). `n_ids_` stays the number of ids.
   - Counting distinct ids per leaf instead would need a second unit array in the splitter, and with few ids it would block the splits this mode exists for.
3. **Coarsening.** Coarsen first, then split rows at block boundaries. Boundaries need not be grid points; a split at `kL` leaves every grid-time risk set unchanged. Units are re-indexed after dropped rows. With `block_time`, the labels of dropped rows go away.
4. **OOB spec (★ core decision).**
   - **Estimand:** risk for *held-out periods of training subjects*, i.e. interpolation within observed ids. It is neither new-subject nor future-period error, and the docstring and user guide say so. Future-period claims still need `RollingOriginSplit` (G2).
   - **Buffered block OOB:** OOB is reported per **original** (unsplit) row. A row of id `i` whose interval overlaps blocks `k0 … k1` uses tree `b` only if blocks `(i, k0−h) … (i, k1+h)` are all out of `b`'s bag; `h = oob_buffer`, default `1` (hv-block idea, Racine 2000). With `block_time` the row has a single block `k`. Neighbours are by block index; indices with no rows count as out.
     - Why: adjacent blocks hold the same id's rows with nearly the same covariates, i.e. near-copies. `h = 0` would leak them. Far blocks of the same id stay in the bag, so the id-level frailty still leaks. This is part of the estimand, not a bug.
     - Cost: with `U` units, `n_draw` drawn without replacement and a row's required set of `q` existing units, a tree qualifies with probability `C(U−q, n_draw) / C(U, n_draw)`. For large `U` with every neighbour block occupied, this is about `0.368^q`: 37% (q=1), 5% (q=3), 0.7% (q=5). So h=1 with 500 trees gives about 25 trees per row. `oob_n_trees_` (per row) is exposed; the existing warning covers rows with no OOB tree.
     - ★ Alternatives: (a) `h = 0` default; (b) no OOB under `"block"` (error), with CV only.
   - `oob_score_` stays `concordance_index_cp` with the **true ids**, so pairs are across ids, as in id mode.
   - Engine: `oob_mortality` takes, per row, the list of units that must be out of bag (CSR `offsets` + `units`). Id mode passes one unit per row, and its results stay bit-identical.
5. **Validation, not a new engine path:** `fit_forest` is unchanged. Block mode is `groups = block labels`.
6. **Out of scope:** `"row"` resampling (still deferred); user-defined blocks (★ above); block-aware splitters in `model_selection` (`GroupTimeSplit` already exists).

## Tests
- **Oracle 1:** when every id's rows lie in a single block (all of `(start, stop]` within one `(kL, (k+1)L]`; e.g. `start >= 0` and `L >= max stop`), the model is bit-identical to `resample_unit="id"` with the same seed (predictions, OOB with `h=0`).
- **Oracle 2:** when rows already end on block boundaries and each row is one block (`stop − start = L`, starts on multiples of `L`), no row is split and the model equals `ids=None` (each row its own unit), with the same numbering by first appearance.
- **Splitting:** pieces tile each row; the event is on the last piece; risk sets are unchanged (a single-node Nelson–Aalen on split rows equals the unsplit one); predictions from `forest_` do not depend on splitting.
- **OOB reference:** a Python reference of buffered OOB mortality (extending `tests/test_oob.py::_manual_oob_mortality`) for `h ∈ {0, 1, 2}` × both `aggregate` values, to 1e-12. A hand case with one id and 3 blocks checks exactly which trees enter.
- Block labels: contiguous per id; `split_id` segments share an id's blocks; stacked landmark data; with `ntime`, labels from the original `start`, re-indexed after drops.
- `min_ids_leaf="auto"` and `max_samples` resolve from `n_units_`. Validation errors (missing or invalid `block_length`, `block_length` with `"id"`, negative `oob_buffer`).
- sklearn compatibility: `resample_unit="block"` added to the estimator-check parameter sets.

## Evidence (bench)
`bench/s10_block.py` → `docs/bench/s10-block.md`: a simulation with few ids (e.g. 30), long follow-up, an autocorrelated TVC and an id frailty. It compares:
- OOB C for `h ∈ {0, 1, 2}`;
- a held-out-period CV with the same buffer (blocked K-fold over blocks, dropping neighbours from training), i.e. the estimand;
- new-subject CV (`GroupKFold`) and a future-period estimate (`RollingOriginSplit`) for contrast.

- **Conditional-subsampling check (plan review 3):** for a random sample of rows, refit trees whose `n_draw` units are drawn from the population minus that row's required set, and score the row. This is the distribution buffered OOB samples from.
- The blocked K-fold result, if shown, is labelled as a different estimate (it removes other folds' blocks as well).

Accept: buffered OOB matches the conditional-subsampling check within the Monte Carlo spread, and the table shows how far it is from new-subject and future-period error.

## Tasks
- [x] Rust: `oob_mortality` with per-row unit sets (CSR); binding; id mode bit-identical
- [x] Python: `block_length`, `oob_buffer`, `block_time` (+ routing, `LandmarkSurvivalForest`), row splitting at block boundaries, block labels (+ `split_id`, `ntime`), `n_units_`, `oob_n_trees_`, validation, docstrings
- [x] Tests (above) + test-quality audit (the reference computes blocks, unit numbering and bags independently; the oracles are bit-identical, not tolerance, checks)
- [x] Bench + `docs/bench/s10-block.md` (added a PBC2 landmark part: the case where the buffer matters, which led to the `block_length >= horizon` guidance and warning)
- [x] Docs: user-guide section on choosing the resampling unit and what each OOB estimates; `design.md` rows; `plan.md` S10 entry + "S10 done"
- [ ] Codex diff review; fixes

## Plan review (Codex, 2026-09-26)
1. Stacked landmark rows all have `start = 0`, so every row of an id fell into block 0 (blocker) → `block_time=` fit argument; `LandmarkSurvivalForest` passes `s`; stacked + block without it is an error.
2. Unsplit long rows let an in-bag row cover the held-out block's period, so the buffer did not hold (high) → split at block boundaries; the OOB set = every overlapped block ± h.
3. The blocked K-fold bench estimates a different training distribution (high) → a conditional-subsampling check; K-fold only as a labelled contrast.
4. Oracle 1's condition was wrong for negative starts (medium) → stated per id.
5. The OOB availability figures were approximate (medium) → exact finite-population formula, with the power form as an approximation.

## Diff review (Codex)
_pending_
