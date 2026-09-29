# Research Questions: Missing-Value Support

## Questions

1. How does `randomForestSRC` actually implement missing-data handling for
   survival/competing-risks forests -- what are `na.action`'s options, and how
   does surrogate splitting work mechanically (what is a surrogate, how is it
   selected/ranked, what happens with no usable surrogate)? Does it differ for
   counting-process/time-varying-covariate data specifically, or is TVC support
   in that package limited enough that this doesn't fully apply?
2. What do current (post-2015) references say about the relative statistical
   performance of MIA (Missing Incorporated in Attributes), surrogate splits,
   and imputation-then-fit, for tree ensembles specifically (not just any ML
   model)? Is there a consensus recommendation, and does it change for survival
   outcomes specifically vs. classification/regression?
3. In rftvc's actual Rust engine, exactly where would NaN have to be handled
   for each candidate approach: `Binned::fit` (column binning), `best_split_in`
   / `NodeProfile` (split search), the `SplitCriterion` trait (log-rank
   scoring), and `Tree::apply` (prediction-time routing)? What existing
   invariants (e.g. the `bin(x) <= b <=> x <= edges[b]` binning convention, the
   `min_ids_leaf` / `min_events_leaf` counting, competing risks' shared tree
   structure) would each approach have to preserve or touch?
4. What does "missing" mean for rftvc's specific row shape -- is it only ever
   a covariate value (`X`), or could `start`/`stop`/`event`/`measured_at` also
   be legitimately missing in practice, and does the counting-process/landmark
   dual layout change what's answerable here?
5. What existing precedent for imputation or extrapolation already exists in
   the codebase (`extrapolate="locf"` in `predict_cumulative_hazard`, any
   `_validation.py` handling of nulls from polars/pandas frames) and how
   closely does it map onto a general missing-covariate story vs. being a
   narrower, prediction-time-only mechanism?
6. What do the two most relevant existing competitors actually ship today --
   `scikit-survival`'s `RandomSurvivalForest`/`SurvivalTree` (does it accept
   NaN at all?) and any Python survival-forest package that does missing-value
   handling well? Is "missing-value support" actually a competitive
   differentiator right now, or does everyone in this space punt on it too?
7. What would each approach cost in engineering terms specific to this
   codebase -- rough surface area touched (files, whether it's Rust-only,
   Python-only, or both), and what new test/validation burden it creates
   (e.g. a new closed-form-DGP simulation needed to trust it, per this repo's
   existing validation discipline)?

## Codebase References
- `src/rftvc/_estimator.py` -- `check_array(X, dtype=np.float64, order="C")`,
  no `allow_nan`, is where NaN is currently rejected (two call sites).
- `rust/rftvc-core/src/data.rs` -- `Binned::fit`, per-feature edge computation
  and `bin(x) <= b <=> x <= edges[b]` binning convention.
- `rust/rftvc-core/src/splitter.rs` -- `best_split_in`, `NodeProfile`, the
  actual split search over binned columns.
- `rust/rftvc-core/src/criterion.rs` -- `SplitCriterion` trait, `Profile`,
  LTRC log-rank scoring.
- `rust/rftvc-core/src/tree.rs` -- `build_tree`, `Tree::apply` (row-to-leaf
  routing at prediction time).
- `src/rftvc/_estimator.py:predict_cumulative_hazard` -- existing
  `extrapolate="locf"` precedent for covariate-path handling.
- `docs/plans/simulation-validation-findings.md` -- this repo's existing
  discipline for what statistical claims need a closed-form-DGP check before
  being trusted.
