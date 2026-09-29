# Design: Missing-Value Support

## Executive summary

`rftvc` currently rejects any NaN in `X` outright (Python-level `check_array` and,
one layer down, a Rust `sort_by` panic). This is a real, current gap: `scikit-survival`
— the closest direct competitor — already handles NaN natively at the ensemble level
via MIA ("missing goes to whichever child improves the split criterion"), inherited
from sklearn 1.3's own tree splitter. `randomForestSRC` uses a different mechanism
(node-local hot-deck imputation) and explicitly rejects CART-style surrogate splits
for forests, for a reason that also applies to rftvc: surrogate search is undermined by
per-node random feature subsampling (`max_features`). Research found rftvc's
`SplitCriterion` trait is fully decoupled from raw `X`, which shrinks native MIA to a
splitter-and-routing change, not a whole-engine rewrite. Recommendation: ship two
independent, both-useful pieces — (1) a small, Python-only, TVC-aware per-subject LOCF
imputation helper as a fast, low-risk win, and (2) native MIA in the Rust splitter as
the real competitive-parity feature, landed as its own slice with its own correctness
gate. Do not build surrogate splits — both the literature and randomForestSRC's own
authors argue against them for forests specifically, for a reason rftvc already exhibits.

## Approaches considered

### A. Native MIA in the Rust splitter (recommended, as the primary piece)

At each candidate split, rows with a missing value on the feature being split are not
placed by a rule fixed in advance — the split search tries them in both children and
keeps whichever scoring is better, and that choice is persisted on the node for
prediction-time routing.

**What it touches** (per research Q3):
- `data.rs`: a way to represent "missing" for a binned column (a reserved sentinel bin
  or a parallel missing-mask), decided at bin time.
- `splitter.rs`: `best_split_in`'s single left-to-right sweep (`splitter.rs:314`)
  needs to score each threshold twice — missing-rows-left and missing-rows-right — and
  keep the better one. The existing unit-counting machinery does not need to change; it
  already works over "whatever bin is present."
- `tree.rs`: `Node::Split` needs a new field (missing-direction); `Tree::apply`
  (`tree.rs:191`) needs an explicit NaN branch instead of its current silent
  always-routes-right behavior.
- `forest.rs`: every other tree-walking prediction path (`predict_cumhaz`,
  `predict_paths`, etc.) needs the same NaN branch `apply` gets.
- `flat.rs`: a `FORMAT_VERSION` bump (v3 → v4), following the already-established
  v1→v2→v3 precedent for exactly this kind of leaf/node-format change.
- **Not touched**: `SplitCriterion`/`Profile` (`criterion.rs`) — it scores an already-
  partitioned `Profile`, agnostic to how the partition was built.
- Python side: `_BaseForestTV.__sklearn_tags__`'s `allow_nan = False` → `True`
  (`_estimator.py:107-111`), both `check_array` call sites (`_estimator.py:188, 405`),
  and whatever new `check_estimator` common tests that tag change brings into scope.

**Pros**: matches/exceeds the nearest direct competitor's actual behavior; no
assumption about *why* data is missing (works under non-ignorable missingness, per
Q2); no new fit-time preprocessing step for users; a genuine "native" story, not a
workaround.
**Cons**: real Rust engine surface area (5 files); correctness-critical (a wrong
missing-direction silently biases every split it's used in); needs its own test that
brute-force-verifies the two-direction search, mirroring the existing
`child_summaries_match_brute_force` pattern already in `splitter.rs`; likely needs a
closed-form-DGP statistical check under this repo's own validation discipline
(`docs/plans/simulation-validation-findings.md`) before being trusted, matching how
every other estimator behavior in that inventory was verified.

### B. Imputation (recommended, as a smaller complementary piece)

Fill `X` before it reaches the Rust engine at all. Two variants worth distinguishing:

- **Generic (mean/median)**: no rftvc-specific value-add — any preprocessing tool
  already does this (`sklearn.impute.SimpleImputer`). Doesn't need to be built here.
- **Per-subject LOCF**: TVC-aware — for a time-varying covariate, "this subject's own
  most recent non-missing value" is a materially better default than a population
  mean, and it reuses the *concept* already established in this codebase's mental
  model (`extrapolate="locf"` on `predict_cumulative_hazard`), even though that
  existing code is prediction-time-only and doesn't share implementation.

**Pros**: Python-only, no Rust changes, no `FlatForest` format bump, smallest surface
area of any option, ships fast, deterministic (easy to test without a new simulation).
**Cons**: statistically weaker than native handling — single imputation understates
uncertainty and can bias estimates depending on the missingness mechanism (this is
exactly the class of problem MIA's consistency results, per Q2, don't require you to
assume away). Not a "native" differentiator — every survival package can already do
this via a `Pipeline` step.

### C. Surrogate splits (not recommended — ruled out)

Both `randomForestSRC`'s own published reasoning and the surveyed literature agree
this fits poorly with random forests specifically: per-node search for a correlated
backup variable is expensive, and a forest's own per-node random feature subsampling
(rftvc's `max_features`, matching `mtry`) can easily exclude every variable that would
have made a good surrogate. rftvc already does per-node feature subsampling, so this
weakness applies directly. Not scoped further.

**Evidence caveat (added after `codex:rescue` plan review), applies to A vs. B, not
to ruling out C**: the "ruled out" verdict on surrogate splits rests on a solid,
forest-specific engineering argument (both randomForestSRC's authors and the
literature agree, and the mechanism — `max_features` undermining surrogate search —
is one rftvc genuinely has), so that part stands. But research Q2 explicitly found no
survival-specific comparison between MIA and imputation, found relative performance
between MIA and surrogate splits "depends on the covariate dependence structure," and
only read the title/abstract of the cited consistency result, not its derivation. So:
native MIA over imputation is chosen here on *fit with rftvc's architecture and
competitive parity* (it's what the nearest direct competitor does, and it's cheap
given `SplitCriterion`'s decoupling), not on a demonstrated general performance
advantage — don't oversell "MIA is provably better" to users; describe it as "the
tree-native approach with the best evidence for its own theoretical soundness under
arbitrary missingness," which is what the research actually supports.

## Recommendation

Ship both A and B, as two independently-mergeable slices, in this order:

1. **Per-subject LOCF imputation** first — small, low-risk, immediately useful, and
   directly answers "also consider imputation." Python-only helper (exact placement —
   a function in `_validation.py`, a new small module, or a documented recipe —
   is a plan-stage decision, not a design-stage one).
2. **Native MIA in the splitter** second — the real competitive-parity feature. Bigger,
   Rust-side, needs its own brute-force correctness test and probably a statistical
   validation check before being trusted, per this repo's own standing discipline.

Do not build surrogate splits.

## Representative sketch (native MIA, illustrative only — not final code)

```rust
// splitter.rs, best_split_in: today's single sweep becomes two scored sweeps
// per candidate threshold, one with the node's missing-on-this-feature rows
// folded into the left running sums, one into the right; keep whichever the
// SplitCriterion scores higher. Profile/SplitCriterion itself is untouched.
for c in 0..nb - 1 {
    let score_missing_left = criterion.score(&profile_with(missing_left_union(c)));
    let score_missing_right = criterion.score(&profile_with(missing_right_union(c)));
    // keep the better; record the winning missing-direction alongside the threshold
}
```

```rust
// tree.rs, Node::Split gains a field; Tree::apply gets an explicit NaN branch
// instead of relying on `x[feature] <= threshold` silently being false for NaN.
Node::Split { feature, threshold, missing_goes_right: bool, left, right } => {
    i = if x[feature].is_nan() {
        if missing_goes_right { right } else { left }
    } else if x[feature] <= threshold { left } else { right };
}
```

## Open questions for the plan stage (not resolved here)

- Exact `Binned` representation for "missing" (reserved sentinel bin value vs. a
  parallel bitmask) — an implementation-detail choice, not a design-level tradeoff.
- Whether the per-subject-LOCF helper ships as `rftvc._validation` code, a new small
  module, or purely as a documented recipe (mirrors the `rftvc.viz` "in-package vs.
  docs-only" question already resolved for that feature, but should be asked again
  here on its own merits).
- Whether landmark/history-feature aggregation (`landmark.py`) needs its own
  missing-value story, or whether covariate-level fixes upstream are sufficient —
  not scoped in this research pass.
