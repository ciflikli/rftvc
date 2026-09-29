# Implementation Plan: Missing-Value Support

Locked design: `docs/plans/missing-values-design.md`. Two independent pieces, five
slices total. Slices 2-3 are Rust-engine-internal (not reachable from Python yet —
Python still rejects NaN until Slice 4); this is a deliberate exception to "each slice
is end-user-visible," made because the engine correctness has to exist and be
unit-tested *before* it's safe to expose, and getting it wrong silently biases every
split that uses it. Slice 1 (LOCF) is fully independent of Slices 2-5 and can land
first, in parallel, or be skipped without blocking the rest.

## Status
- [ ] Slice 1: `rftvc.impute.impute_locf` (Python-only, independent)
- [ ] Slice 2: Rust engine core — missing-aware binning + two-direction split scoring
- [ ] Slice 3: Node/tree routing + `FlatForest` format bump (v3 → v4)
- [ ] Slice 4: Python integration — allow NaN through the public API
- [ ] Slice 5: Statistical validation (known-DGP check) + docs/CHANGELOG

## Slice 1: `rftvc.impute.impute_locf`

**Files:** `src/rftvc/impute.py` (new), `src/rftvc/__init__.py` (modify: `from . import impute`,
add to `__all__`), `tests/test_impute.py` (new)

New submodule, mirroring the existing `rftvc.metrics`/`rftvc.inspection`/
`rftvc.model_selection` pattern (a family of related functions around one concern).
No new dependency — pure numpy/polars, reuses `_validation.split_frame` for
DataFrame-or-array input the same way `fit` does.

Signature:
```python
def impute_locf(X, ids, *, start=None, columns=None):
    """Fill NaN in X by carrying forward each subject's own last non-missing value.

    Rows are ordered by `start` if given, else by their position in `X` (must
    already be grouped/ordered per id, as `fit` requires). A subject's first
    row(s) for a column with no prior non-missing value are left NaN --
    nothing to carry forward from.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_rows, n_features)
    ids : array-like of shape (n_rows,) or str
        Subject of each row (or a column name of a DataFrame X).
    start : array-like of shape (n_rows,), default=None
        Sort key within each subject; default row order.
    columns : list of names or indices, default=None
        Columns to impute; default all.

    Returns
    -------
    ndarray of shape (n_rows, n_features), same dtype as X's numeric data.
    """
```

Tests:
- Hand-computed expected output on a small crafted table (2 subjects, interleaved
  rows, one column with a gap, one column with no missing).
- A subject's leading NaN (no prior value) stays NaN.
- Two subjects never see each other's values (shuffle ids, assert no leakage).
- `start`-based ordering matches expected carry-forward direction when rows aren't
  already time-sorted.
- Works with both a bare ndarray and a DataFrame `X` (pandas and polars).
- `columns` restricts which columns are touched; others pass through unchanged
  (including their own NaN, if any — this function doesn't touch excluded columns).

Acceptance criteria:
- All tests above pass.
- Output is a plain ndarray (matches the rest of the package's convention of
  numeric-array outputs from validation/transform helpers).
- Does not mutate its input.

## Slice 2: Rust engine core — missing-aware binning + two-direction split scoring

**Files:** `rust/rftvc-core/src/data.rs` (modify), `rust/rftvc-core/src/splitter.rs`
(modify), `rust/rftvc-core/tests/missing.rs` (new)

`Binned` gains missing-value tracking. Per research (Q3), `u8` bin space can be fully
used by real bins (`max_bins` up to 256), so a reserved sentinel bin value isn't safe
in general — use a parallel per-(row, feature) missing marker instead (start with
`Vec<bool>` column-major, matching `bins`'s own layout; revisit bit-packing only if a
later `bench/` timing shows it matters — this repo's own convention, per the opt-in
`ntime` coarsening, is to add complexity only once measured, not preemptively).

**Revised after `codex:rescue` plan review (see "Plan review findings" below) — this
section originally left `Binned::fit`'s NaN-panicking sort untouched and deferred it
to a Slice 4 that never actually listed `data.rs`, which would have made the whole
feature unreachable. Fixed here: `Binned::fit` itself must become NaN-safe in this
slice.**

`Binned::fit` (`data.rs:18-39`) currently computes bin edges via
`col.sort_by(|a, b| a.partial_cmp(b).expect("NaN in X"))` (`data.rs:25`), which panics
on NaN. This slice changes it to: filter NaN out before sorting/computing edges (edges
are computed from non-missing values only), and populate a new missing marker for each
(row, feature) that was NaN. `bin_of` for a missing entry is never meaningful and
should not be called; the missing marker is checked first everywhere a bin would
otherwise be read.

`best_split_in`'s single left-to-right sweep (`splitter.rs:314`) becomes two scored
sweeps per **existing** candidate threshold — missing rows folded into the left
running sums, then into the right — keeping whichever `SplitCriterion` scores higher
(the criterion itself, confirmed decoupled from raw `X` in research, needs no change).
**Plus a new candidate, independent of any real-value threshold: "missing vs.
observed"** — all missing rows in one child, all non-missing rows (regardless of their
own bin) in the other — evaluated whenever both groups are non-empty at the node. The
review caught that without this, a feature whose observed values are constant (all in
one bin, so the existing per-threshold sweep has nothing to iterate) could never be
split on missingness alone, even when missingness itself carries signal. The winning
candidate (whichever of: every real threshold's two directions, or the missing-vs-
observed split) is returned for Slice 3 to persist.

Signatures (illustrative, finalize during implementation):
```rust
// data.rs
pub struct Binned {
    pub n_rows: usize,
    pub n_features: usize,
    pub bins: Vec<u8>,
    pub missing: Vec<bool>,   // new, same column-major layout as `bins`; edges are
                              // computed excluding these rows, and `bins[i]` is
                              // meaningless (not read) wherever `missing[i]` is true
    pub edges: Vec<Vec<f64>>,
}

// splitter.rs
pub enum SplitRule {
    Threshold { bin: u8, missing_goes_right: bool },
    MissingVsObserved,   // new: no threshold; splits purely on the missing marker
}
pub struct SplitCandidate {
    // ...existing fields, `threshold: f64` etc. only meaningful for `Threshold`...
    pub rule: SplitRule,
}
```

Tests (`tests/missing.rs`, mirroring the existing `child_summaries_match_brute_force`
pattern already in `splitter.rs`):
- For several synthetic NaN patterns (single feature, multiple features, a feature
  entirely missing, missingness correlated with the outcome and not), the two-direction
  search's chosen split and score match a brute-force reference that tries both
  directions manually and picks the best, **for every existing threshold**.
- A constant-observed-value-plus-missingness case specifically exercises the new
  `MissingVsObserved` candidate (the per-threshold sweep alone has no threshold to
  offer here — this is the exact gap the plan review caught) and confirms it's chosen
  when it's genuinely the best split.
- `Binned::fit` on a column with NaN does not panic, computes edges from non-missing
  values only, and sets the missing marker exactly on the NaN rows (a direct unit test
  on `Binned::fit`, not just on the splitter).
- With zero missing rows, behavior is bit-for-bit identical to the pre-change sweep
  (regression safety — every existing `splitter.rs`/`forest.rs` Rust test must still
  pass unmodified).
- A column that is missing for every in-bag row at a node behaves sanely (no panic;
  the feature contributes no signal since there's nothing to compare against).
- **A resampling unit whose own rows mix missing and observed values for the split
  feature**: unit-counting (`min_ids_leaf`), the actual training partition, and the
  chosen split's score are all mutually consistent — checked against a manually
  constructed reference partition, for both possible missing-directions. (Plan-review
  finding: this specific case wasn't named as its own test before.)

Acceptance criteria:
- New `missing.rs` tests pass, including the `MissingVsObserved` and mixed-unit cases
  above.
- Full existing Rust suite (`cargo test -p rftvc-core`) still passes unmodified.
- `cargo clippy --all-targets -p rftvc-core -- -D warnings` and `cargo fmt --all --check`
  clean.
- **Corrected framing**: this slice makes the Rust engine itself fully NaN-safe and
  correct — the only thing still blocking end-to-end use is Python's own
  `check_array`/`allow_nan` tag (Slice 4), not any remaining Rust-side limitation.

## Slice 3: Node/tree routing + `FlatForest` format bump

**Files:** `rust/rftvc-core/src/tree.rs` (modify), `rust/rftvc-core/src/flat.rs`
(modify), `rust/rftvc-core/src/forest.rs` (modify), `rust/rftvc-core/tests/forest.rs`
(extend)

`Node::Split` gains fields carrying Slice 2's chosen `SplitRule` (both the
per-threshold missing-direction and the `MissingVsObserved` case need to be
representable), set from Slice 2's `SplitCandidate` during `build_tree`. `Tree::apply`
(`tree.rs:191`) gets an explicit `x[feature].is_nan()` branch instead of relying on
`<=`'s silent always-false-on-NaN behavior (a real, if currently unreachable, bug
found during research). Every other tree-walking prediction path in `forest.rs`
(`predict_cumhaz`, `predict_paths`, etc. — enumerate the full list during
implementation, research only confirmed `apply`'s structure directly) needs the
identical branch.

`FlatForest`'s `FORMAT_VERSION` bumps 3 → 4 (following the already-established v1→v2→v3
precedent in `flat.rs`'s own doc comment) to persist the new field(s); a v3 pickle must
still load (as a forest with no missing-direction info — every split defaulting to a
fixed direction is fine, since a v3 forest was never fit with NaN in the first place).

Tests:
- Extend the existing flat-roundtrip test: a forest with `Node::Split`s covering both
  `Threshold{missing_goes_right}` variants and `MissingVsObserved` round-trips through
  pickling correctly.
- `apply()` and at least one other prediction path route a NaN feature value per the
  node's stored rule, tested directly (construct a small tree with known rules,
  predict with NaN, assert the expected leaf) — for both `SplitRule` variants.
- **A resampling unit with both missing and observed rows for the split feature,
  carried through to a full tree fit**: the leaf cumulative hazard the unit's rows
  contribute to is consistent with the training partition Slice 2 chose for it (plan-
  review finding — Slice 2 covers this at the split-search level; this slice must
  verify it also holds end-to-end through `build_tree` into the leaf hazard actually
  stored and read back at predict time).
- A v3-format pickle (or a hand-constructed v3 state) still loads correctly (mirrors
  the existing `flat_v2_loads_as_one_cause` backward-compatibility test).

Acceptance criteria:
- All tests above pass; full existing Rust suite still passes.
- `cargo fmt`/`clippy` clean.
- Same corrected framing as Slice 2: the Rust engine is now fully NaN-correct
  end-to-end (split search through prediction); only Python's `check_array`/`allow_nan`
  (Slice 4) still blocks it from being reachable from the public API.

## Slice 4: Python integration — allow NaN through the public API

**Files:** `src/rftvc/_estimator.py` (modify), `rust/rftvc-py/src/lib.rs` (modify if
`fit_forest_py`/prediction bindings need it — check during implementation whether
anything there assumes finite `X`), `tests/test_missing.py` (new)

Flip `_BaseForestTV.__sklearn_tags__`'s `allow_nan` to `True` (`_estimator.py:107-111`);
remove the implicit NaN rejection from both `check_array` call sites
(`_estimator.py:188, 405`) — explicitly, not just by omission, so it's clear NaN is
now an intended input, not an oversight. This is on the shared `_BaseForestTV` base,
so `SurvivalForestTV` and `CompetingRisksForestTV` both get it without separate work —
state this explicitly as a test requirement, not an assumption.

Tests (`test_missing.py`):
- `fit`/`predict`/`predict_survival_function`/`predict_cumulative_hazard` all succeed
  end-to-end with NaN scattered in `X`, for both `SurvivalForestTV` and
  `CompetingRisksForestTV`.
- A column entirely NaN doesn't crash (mirrors the Rust-level test from Slice 2).
- Fitting the *same* data with **no** NaN gives bitwise-identical results to before
  this slice (regression safety — the missing-direction machinery must be a true no-op
  when nothing is missing).
- `sklearn.utils.estimator_checks.check_estimator`'s NaN-related common checks: either
  pass now, or are added to the survival-adapted expected-failures list in
  `tests/test_sklearn_compat.py` with the same reasoning pattern already used there
  (a numeric-`y` check that doesn't apply to a structured survival target).

Acceptance criteria:
- All tests above pass.
- Full default pytest tier green.
- `docs/source/compatibility.rst`'s scikit-learn-compatibility section gets a line
  about `allow_nan` (it already documents other estimator tags/behaviors there).

## Slice 5: Statistical validation + docs/CHANGELOG

**Files:** `bench/missing_mia_sim.py` (new), `tests/test_missing_sim_truth.py` (new,
default pytest tier), `docs/plans/simulation-validation-findings.md` (new row),
`CHANGELOG.md`, `README.md` (feature bullet list), `docs/source/api.rst` (if
`rftvc.impute` needs its own autosummary section, mirroring `Visualization`'s).

Reuses this repo's own established validation discipline (per
`simulation-validation-findings.md`): a closed-form-DGP check, under a declared
missingness mechanism, that a forest fit with native MIA recovers the true structure
at least as well as a complete-case-deletion baseline fit on the same data. Calibrate
the pass rule from the gate's own seeds plus >= 2 independent out-of-band seed
batches, following the pattern used in every prior slice of that inventory (Slices
5-12 there, and the S19 landmark-gate work) — do not skip the out-of-band
verification step even though it feels like ceremony; it has caught real fragility
before (Slice 10 in that inventory).

Acceptance criteria:
- New default-tier test actually runs in plain `pytest` (no `-m` flag) — verified by
  actually running it, per the standing lesson this repo has already re-learned three
  times (Slices 5-8 of the prior inventory).
- `EPSILON`/threshold calibrated from seeds distinct from the gate's own, stated
  before the gate ran conceptually (i.e. don't tune the threshold to make a
  borderline result pass).
- `simulation-validation-findings.md` gets a new row describing what this does and
  doesn't establish, matching every other row's format.
- CHANGELOG entry under `[Unreleased]`.

## Review checkpoints (per this repo's own established practice)

- One `codex:rescue` **plan** review of this file before Slice 2 starts (Rust engine
  changes are the highest-risk part; catching a design problem here is far cheaper
  than after code exists) — matches the `cr-design.md`/`tvc-design.md` precedent of a
  plan review before code on decisions this size.
- One `codex:rescue` **diff** review per slice before it's considered done, same as
  every other slice in this session so far.
