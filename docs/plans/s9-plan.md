# S9 slice plan: leaf-storage slimming

Branch `feat/s9-leaf-slim`. Parent: `plan.md` S6 deferral ("leaf storage"), `docs/scratch/perf.md` "Not done", `design.md` "Peak memory is dominated by stored leaves".

## Problem
- A fitted leaf stores `event_idx: Vec<u32>`, `d`, `y` and `cumhaz` (each `Vec<f64>`): 28 bytes per leaf event time, plus 4 `Vec` headers (96 B) and 4 heap allocations per leaf.
- Prediction reads only `event_idx` and `cumhaz` (`Tree::cumhaz_at`). `d` and `y` serve `leaf_profile` (tests only) and the pickle state. `Tree::leaf_hazard` has no callers.
- At 1M rows × 100 trees (~32k leaves per tree) the forest dominates peak RSS (2.4 GB exact, 2.3 GB `ntime=100`; `docs/bench/s6-perf.md`).

## Decisions (defaults; revisit in review)
1. **Leaves keep `event_idx` and `cumhaz` only.** `d` and `y` are dropped. Hazard increments stay available as `np.diff(cumhaz, prepend=0)`.
2. **Per-tree CSR leaf storage** replaces `Vec<Leaf>`:
   - `Tree { nodes, leaf_offsets: Vec<u32> (n_leaves + 1), event_idx: Vec<u32>, cumhaz: Vec<f64>, grid_times }`.
   - That is 12 B per leaf event time + 4 B per leaf, with no per-leaf allocations. `Leaf` and `leaf_hazard` are removed.
   - `build_tree` appends to the CSR arrays directly and shrinks them to fit at the end. The node bookkeeping is unchanged.
   - `cumhaz_at(leaf, t)` binary-searches the leaf's slice. The arithmetic is unchanged, so **predictions are bit-identical** to `main`.
3. **`leaf_profile(tree, leaf)` returns `(event_times, cumhaz)`.** This is a breaking change on `forest_` (the Rust object, not the estimator API). The tests that read `d` / `y` switch to hand-computed hazard increments.
4. **Pickle format v2:**
   - The state holds `cumhaz` instead of `d` and `y`, plus `format_version = 2`.
   - Loading validates: offsets as today; each leaf's `event_idx` strictly increasing and in range; `cumhaz` finite, `>= 0` and non-decreasing.
   - A state without `format_version` (v1, pre-S9) is **rejected** with a clear `ValueError` ("forest state from an older rftvc build (pre-S9 leaf format); refit the model"). The package is pre-release (`0.1.0.dev0`) and sklearn itself makes no cross-version pickle promise. Migrating old states (rebuilding `cumhaz` from `d`/`y`) is about 15 lines if review prefers it.
   - The flat arrays match the in-memory CSR (tree `leaf_offsets` are made global on save), so saving and loading are copies plus validation.
5. **`Forest.nbytes`** (Rust getter on `forest_`): heap bytes of nodes and leaf arrays. The benchmark and the memory test use it; it is cheap to compute from vector lengths.
6. **Forward compatibility (no code now):** competing risks would add a cause dimension to `cumhaz` (entries × causes) under the same offsets. Nothing in this layout blocks that.
7. **Out of scope:**
   - `f32` storage (not bit-identical);
   - compacting `Node` (24 B per node, about 2× fewer nodes than leaf entries);
   - changing the `ntime=None` default.

## Acceptance
- All tests pass. Predictions on `main` and the branch are **bit-identical** (`np.array_equal`) on GBSG2, the TVC simulation and the PBC2 landmark example, for both `aggregate` values, including OOB predictions. The check runs as a script against a `main` build; results go in the review log.
- Forest `nbytes` falls ≥ 50% at 1M rows. Peak RSS and pickle size are reported before and after at 1M rows (the four rftvc arms of `bench/compare.py`).
- Pickle round-trip is bit-identical; a v1 state and corrupted v2 states are rejected without panicking.

## Tasks
- [x] Baseline on `main`: `bench/compare.py` rftvc arms at 1M rows; pickle sizes; prediction dumps for the bit-identity check (`docs/scratch/s9-baseline/`, not committed)
- [x] Rust core: CSR leaves in `Tree`, `build_tree`, `cumhaz_at`; remove `Leaf` / `leaf_hazard`; `Forest::nbytes`
- [x] `FlatForest` v2 (`cumhaz`, `format_version`) + validation; Rust tests: round-trip, NaN / negative / decreasing `cumhaz`, unsorted `event_idx`, v1 rejected
- [x] Binding: `leaf_profile`, `__reduce__` / `_from_state`, `nbytes`
- [x] Python tests: update the `leaf_profile` callers; v1-state rejection; corrupt states raise `ValueError`; `nbytes` equals the layout formula (pins "no `d`/`y`")
- [x] Bit-identity check vs the baseline dumps (`bench/s9_identity.py`)
- [x] Bench after → `docs/bench/s9-leaf.md`; update `docs/scratch/perf.md`, `design.md` (memory line), `plan.md` (S9 entry + "S9 done")
- [x] Test-quality audit of new tests (each new test fails on the pre-S9 build: no `nbytes`, no `format_version`, 4-tuple `leaf_profile`); Codex diff review; fixes

## Plan review (Codex, 2026-09-26)
1. The loader did not validate `grid`: unsorted or NaN grid times break `cumhaz_at`'s binary search (high) → the grid must be finite and strictly increasing. Rust and Python tests.
2. The loader accepted impossible scalars: `n_features = 0` panics in `predict_cumhaz` (division by zero) (high) → `n_features`, `n_groups`, `n_draw >= 1`, and `n_draw <= n_groups` unless bootstrap. Tests.
3. `tests/test_oob.py::_manual_oob_mortality` also calls `leaf_profile` (medium) → updated.
4. `bench/compare.py` could not measure forest size on `main` (no `nbytes` there) (medium) → `bench/s9_leaf.py`: a separate storage run (counts, pickle bytes) from the timed RSS runs, and a logical-bytes formula for the old layout (a lower bound). `nbytes` is exact because the arrays are shrunk to fit.

## Diff review (Codex, 2026-09-26)
1. `nbytes` counted lengths but was described as heap bytes; `shrink_to_fit` is best-effort (low) → it counts `Vec` capacities; the bench text says "allocated capacity". The layout test then caught spare capacity in the shared event-time grid (after `dedup`); `Grid` now shrinks it to fit. At 1M rows this is under 5 MB, so the bench numbers stand. Predictions re-checked: bit-identical.

No panic or out-of-bounds route from a crafted state after validation, no rebasing error, no prediction change.
