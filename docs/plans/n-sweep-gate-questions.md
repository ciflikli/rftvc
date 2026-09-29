# Research questions: n-sweep beyond Slice 1's two points (findings.md row 5 blind spot)

Closes `docs/plans/simulation-validation-findings.md` row 5's declared blind spot: "Only two
`n` values (200, 5000) checked, not a full sweep (design.md's Approach 3, deferred)." The existing
gate (`bench/pe_score_convergence_sim.py` + `tests/test_pe_score_convergence_truth.py::
test_pe_score_mean_gap_shrinks_with_n`) has two separate limitations, both in scope here:

1. It's a 2-point comparison (n=200 vs n=5000), not a sweep — a genuine convergence-rate claim
   needs at least 3 points to say anything about the *shape* of convergence, not just "bigger n is
   better than smaller n once."
2. It's `@pytest.mark.slow` — per the recurring lesson from Slices 5-9
   ([[rftvc-dev-workflow]]), a `slow`-marked gate never runs in the default CI tier at all.

`bench/pe_score_convergence_sim.py`'s `replicate`/`run` already accept an arbitrary `n_values`
tuple and already return a `(n_reps, len(n_values))` gap array — the DGP/oracle machinery needs no
changes, only the sweep width, rep count, and CI tier.

## Questions

1. What does `docs/plans/design.md`'s "Approach 3" (referenced by row 5's blind spot) actually
   propose for a full n-sweep — a specific set of `n` values, a specific statistical claim (e.g.
   monotonic decrease, a fitted power-law rate, or just "more points, same pairwise test"), or was
   it left unspecified/deferred without a concrete rule? Read `docs/plans/design.md` and
   `docs/plans/plan.md`'s Slice 1 section (including "Correction from plan review") to find out.
2. At what reduced training-set sizes and estimator counts (smaller than the design's 200/5000 at
   `n_estimators=200`) does the oracle-minus-model PE-score gap still show a real, monotonic (or
   near-monotonic) decreasing trend across >= 3 points, verified by actually running
   `bench.pe_score_convergence_sim.replicate` at candidate scales across >= 2 independent
   out-of-band seed batches — not assumed from the existing 2-point result?
3. What's the right statistical claim for a 3+ point sweep — pairwise one-sided tests between
   consecutive points (matching the existing 2-point rule's shape), a single test on the linear
   trend across `log(n)`, or something else? Does `scipy.stats` offer a standard trend test
   (e.g. Page's test for ordered alternatives, or a regression-based slope test) that's a better
   fit than repeating the existing pairwise `t.ppf`-based bound at each adjacent pair, and is it
   worth the added complexity given this codebase's established preference (Slices 6-8) for
   reusing existing statistical machinery unchanged rather than introducing new tools?
4. What replication count and runtime keeps this in the same ballpark as the other default-tier
   gates (~0.5-9s each)? The existing 2-point gate at R=10, n=(200,5000), n_estimators=200 is
   `slow`-tier presumably because n=5000 is expensive — what's the largest `n` a 3+ point sweep can
   include while still finishing fast enough for the default tier?
5. Does moving this gate to the default tier (dropping the `slow` marker) on the *existing* 2-point
   rule, without adding a sweep, partially close the row-5 blind spot on its own (the "slow gates
   never run in CI" half), leaving only the "2-point, not a sweep" half for the new sweep gate to
   close? Or should both halves be closed by one single replacement gate? Check whether the
   existing `slow`-tier test would be redundant with a new default-tier sweep gate, or worth
   keeping as a separate larger-n cross-check (mirroring `test_landmark_sim_truth.py`'s pattern of
   keeping its old `slow`-tier pilot alongside Slice 9's new default-tier gates).

## Codebase references

- `bench/pe_score_convergence_sim.py` — `replicate(seed, n_values, n_estimators)` and
  `run(n_values, n_estimators, n_reps, seed0)`, both n_values-agnostic already.
- `tests/test_pe_score_convergence_truth.py` — the existing fast closed-form check plus the
  `slow`-tier 2-point gate.
- `docs/plans/design.md` — "Approach 3" reference (row 5's blind spot text points here).
- `docs/plans/plan.md` Slice 1 section — the pass rule's original declaration and the plan-review
  correction about what the oracle can and can't claim.
- `docs/plans/simulation-validation-findings.md` row 5 — current documented state of this gap.
