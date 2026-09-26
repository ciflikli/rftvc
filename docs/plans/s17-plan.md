# S17 slice plan: `inspection.permutation_importance` (counting-process estimators)

Branch `feat/s17-perm-importance`. Parent: `tvc-plan.md` S17 (v2); `tvc-design.md` §3, §3.1–3.3, §7.1/7.3/7.4. Builds on S16: `_rebuild_design`, `forest_.oob_cumhaz` / `oob_cause_cumhaz`, `metrics.piecewise_exponential_score`, `event_windows`, `_baseline_at`.
Notation: `p` units, `R` = `n_repeats`, `B` = `n_bootstrap`, `M` windows, `J` causes, `N` = scored events.

## Decisions (defaults; revisit in review)
1. **Signature** (tvc-plan S17, plus `measured_at`):
   ```python
   permutation_importance(estimator, X, y=None, *, ids=None, features=None, groups=None,
       strata="time", n_strata=10, conditional_on=None, n_bins=4,
       scoring="pe", windows=8, alpha=0.01, cause=None,
       oob=False, measured_at=None, block_time=None,
       n_repeats=5, n_bootstrap=100, random_state=None, n_jobs=None) -> Bunch
   ```
   - `measured_at` is new: the S16 fingerprint covers it, so `oob=True` on a forest fitted with `measured_at` cannot rebuild the design without it. It and `block_time` are OOB-only; given with `oob=False` → `ValueError`.
   - `y` is required (`None` → `ValueError`; the `None` default is kept for the S19 landmark signature, where `X` is the raw frame).
   - `scoring="pe"` only; `"brier"` / `"ibs"` → `ValueError` naming S19's landmark models. Landmark estimators → `NotImplementedError` (S19); any other type → `TypeError`.
2. **Units** (`_units.py`).
   - `features`: names (need `feature_names_in_`) or integer indices; each is its own unit. Default: every feature.
   - `groups`: `dict name → [columns]`; each entry is one unit, permuted jointly. Overlapping groups, empty groups, unknown columns and duplicate columns → `ValueError`.
   - `features` and `groups` together → `ValueError` (one way to state the units).
   - A DataFrame `X` goes through `estimator._check_predict` (drops the fit-time ids column, checks names). Naming the ids column as a unit → `ValueError` ("not a feature").
   - `feature_names` in the result: the feature name (or index as str) for single units, the group key for groups.
3. **Rows and prediction.**
   - *Held-out:* rows = the evaluation rows of `X`, `y`. `ids` = array or column name (as `score`); `None` makes every row its own id (bootstrap then resamples rows, documented).
   - *OOB (`oob=True`):* `d = estimator._rebuild_design(X, y, ids, measured_at, block_time)`. Rows = `d.X` (after coarsening, before block splitting, which is what `d.oob_set` indexes). Target = `d.start/d.stop/d.event` (CR: through `_oob_target`). Row ids = the original id labels of `d.kept` rows. Rows with 0 OOB trees (from the intact call) are excluded and counted (`n_rows_excluded`, together with coarsening-dropped rows). The OOB tree count does not depend on `X`, so the exclusion set is the same for every permutation.
   - Predictions are the fixed-profile cumulative hazards **at all `M + 1` edges** (including `w_0 = 0`, so training events at negative times are handled):
     - survival: `forest_.predict_cumhaz(X, w, aggregate, n_jobs)` / `forest_.oob_cumhaz(X, *oob_set, w, aggregate, n_jobs)`;
     - CR: `forest_.predict_cause_cumhaz(X, w, n_jobs)` / `forest_.oob_cause_cumhaz(X, *oob_set, w, n_jobs)` (the engine calls directly: the public `predict_cumulative_hazard` also computes the Aalen–Johansen CIF, wasted here).
   - The core (`_score.py`) takes a `predict(X_rows) -> cumhaz` callable, so the sims' true-hazard oracle and the tests' oracle models run through the same code as the forest.
4. **Windows and null.** `windows` int → `event_windows(estimator, windows)`; array → validated edges, and `windows[-1] > event_times_[-1]` → `ValueError`. Null = `_baseline_at(estimator, w)`. Old pickles without `baseline_cumhaz_` → the S16 `AttributeError`.
5. **Strata** (`_strata.py`). A stratum label per row, then a within-stratum shuffle.
   - `"time"`: bins of the rows' `start`. `None`: one stratum, plus a `UserWarning` ("naive permutation can extrapolate off the (time, value) support; prefer strata='time'"); the docstring says the same (T5). An array: user labels, one per input row (OOB: indexed through `d.kept`); wrong length → `ValueError`.
   - **Binning rule** (for `start` and for `conditional_on` columns): a column with ≤ `n` distinct values uses the values themselves as bins. Otherwise the interior edges are `q = unique(quantile(v, k/n)), k = 1..n−1` and the bin is `searchsorted(q, v, side="left")`, so bins are right-closed and a value equal to an edge goes to the lower bin. Example: `v = [1, 2, 3, 4]`, `n = 2`, `q = [2.5]` → bins `[0, 0, 1, 1]`; with `q = [2]`, `v = 2` → bin 0. (Plain quantile edges collapse a skewed binary column into one bin.)
   - `n_strata` and `n_bins` are integers ≥ 1 (`ValueError` otherwise). User `strata` labels go through `np.unique` (any sortable dtype); NaN labels or unsortable labels → `ValueError`.
   - `conditional_on`: names or indices, crossed with the time/user strata. **A unit's own columns are removed from its conditioning set**, so conditioning is on the other listed columns (otherwise `features=all, conditional_on=[c]` would give `c` importance 0 by construction). Hence strata, and `n_unpermuted`, are per unit.
   - Rows in strata of size 1 stay in place and are counted in `n_unpermuted (p,)`. More than 10 % unpermuted rows → `UserWarning` (design risk table).
   - **Shuffle:** `order = argsort(label, stable)`, `donor = lexsort((rng.random(n), label))`; row `order[i]` receives the unit's values of row `donor[i]`. Both sort by label, so donors stay in the stratum; all of the unit's columns take the same donor (joint group permutation).
6. **Score and decompositions** (`_score.py`).
   - Each prediction is scored once with `piecewise_exponential_score(..., reduce="sum", ids=row_ids)`. `N` = the intact call's `n_events` (the same `y`, so the same for every permutation).
   - `ΔS = (S_intact − S_perm) / N` per event, and every decomposition uses **the same denominator `N`**, so `importances_window.sum(1) == importances_mean`, `importances_cause.sum(1) == importances_mean` (CR, `cause=None`) and `importances_id.sum(1) == importances_mean` exactly (to fp). This refines tvc-plan's "`·ΣN`" wording: every field is per event (Q8).
   - `cause=k`: the metric's `cause=k`; `N` = cause-k events; `importances_cause` is None.
   - `share_of_gain = importances_mean / (baseline_score − null_score)`; NaN with a `UserWarning` when the forest does not beat the null.
7. **Randomness** (refines tvc-plan Q6). `random_state` (None, int, `RandomState`, `Generator`) → one integer entropy. Each unit's stream is `SeedSequence(entropy, spawn_key=(c,))` with `c` = the unit's **smallest column index** (units never overlap, so keys are unique), not its position in the request; within a unit, repeats draw first, then each bootstrap replicate from `spawn`ed children of the unit's stream. So a unit's result does not depend on which other units are requested or in what order, and no `p × R × n` index array is pre-drawn (2 GB at 1M rows, p = 50, R = 5).
   - **`n_jobs`** goes to the engine (row-parallel prediction); units run sequentially. The engine's per-row results do not depend on the thread count, so `n_jobs=1` and `4` give identical results. Unit-level joblib parallelism on top of the engine's thread pool would oversubscribe; this deviates from Q6 ("parallelism over units") for that reason.
8. **Bootstrap SE** (`_boot.py`, held-out only).
   - `_resample_ids(ids, rng) -> (row_index, boot_ids)`: draw `n_ids` ids with replacement; copy `c` contributes **all** rows of its id in their original order, labelled `boot_ids = c` (distinct per copy).
   - Replicate `b`: rows `row_index`; strata **recomputed** on the replicate's rows by the same rules (each copy's rows are distinct rows); intact predictions reused (`cumhaz[row_index]`); `R` fresh within-stratum permutations, re-predicting only the permuted rows of this unit; `ΔS_b` = the repeat-averaged per-event drop with the replicate's own `N_b`.
   - **Estimand:** the SE of the reported statistic, the `R`-repeat average, over evaluation subjects **conditional on the fitted forest**. Because each replicate redraws its `R` permutations, the SE includes the permutation Monte Carlo noise at `R` (which is part of the reported number's variability); it is not the SE of the permutation expectation. Permutations exchange row values within strata (row-level donors, as the point estimate), while resampling is by whole id. The docstring states this.
   - `importances_se = sd(ΔS_b, ddof=1)`. NaN with `B = 0`, with `B = 1`, and with `oob=True` (design §3.1). A replicate with no scored event is redrawn from the next child (at most `B` redraws, else `UndefinedMetricError`).
   - Cost: `p × B × R` extra predictions (5000 at `p = 10`, defaults); the docstring says so and gives `n_bootstrap=0` as the fast path. Timings go in the S17 note.
9. **Result `Bunch`:**
   - `importances (p, R)`, `importances_mean`, `importances_std` (`np.std`, ddof 0, as sklearn), `importances_se`;
   - `importances_window (p, M)`, `window_edges (M + 1,)`;
   - `importances_cause (p, J)` or None;
   - `importances_id (p, n_ids)`, `id_labels` (first-appearance order, the metric's);
   - `share_of_gain (p,)`, `baseline_score`, `null_score`, `zero_rate_share` (intact), `n_unpermuted (p,)`, `n_events`, `n_truncated_events`, `n_rows_excluded` (OOB; 0 held-out), `feature_names`, `units` (the column indices per unit).
10. **Public surface.** `rftvc.inspection` (exported from `rftvc/__init__.py`), `api.rst` section. The user guide page is S20.

## Tasks
- [x] T1 `_inspection/_units.py`, `_strata.py` (+ unit tests): unit resolution, binning, stratum labels, within-stratum shuffle.
- [x] T2 `_inspection/_score.py`: the core loop over units/repeats on a `predict` callable; decompositions; `share_of_gain`.
- [x] T3 `_inspection/_boot.py`: `_resample_ids`, replicate loop, SE.
- [x] T4 `inspection.py`: public function, estimator dispatch, held-out and OOB predict callables, validation, docstring (M1 warning text, cost note, estimand of the SE).
- [ ] T5 Tests `tests/test_inspection_perm.py` (below; written, 47 pass); `/test-quality` audit (pending).
- [ ] T6 Sims `bench/tvc_perm_sim.py` (§7.1 M2 part, §7.3, §7.4; R = 50 after a 10-rep pilot for the MC SE check) + `tests/test_tvc_sim_truth.py` (slow R ≈ 10, fast smoke R = 1). Results in the S17 note.
- [ ] T7 Timings (held-out, OOB, bootstrap), `api.rst`, `__init__`, `tvc-plan.md` tick + "S17 done" note.

## Tests (`tests/test_inspection_perm.py`)
- **Oracle model.** A fitted `SurvivalForestTV` whose `forest_` is replaced by a stub whose `predict_cumhaz` depends only on column 0 (a known hazard): importance of column 1 is **exactly 0** (predictions unchanged), column 0 > 0; the same through a CR stub (`predict_cause_cumhaz`) with column 0 acting on cause 1 only: `importances_cause[0, 1] == 0`.
- **Strata respected:** a spy `predict` records the permuted `X`; per stratum the multiset of the unit's values is unchanged, and rows move only within their stratum; `n_strata=1` gives a permutation of the whole column (multiset equality) that differs from the identity.
- **Groups move jointly:** in the recorded `X`, a 2-column group keeps its row pairing; a column outside the unit is untouched.
- **Binning:** a skewed binary column gives 2 bins; a continuous column gives `n` bins of near-equal size; ties at edges go to the lower bin.
- **Singletons:** `conditional_on` with many bins produces singleton strata; `n_unpermuted` equals the hand count, and > 10 % warns.
- **Conditional:** `conditional_on` = an exact copy of the permuted column (a column the forest ignores does not matter) → importance exactly 0; the own-columns rule: `features=[0], conditional_on=[0]` behaves as unconditional (same result as without it, same seed).
- **Reproducibility:** the same `random_state` gives identical Bunches for `n_jobs=1` and `4`; a different seed differs; `features=[1, 0]` and `features=[0]` give unit 0 the same values as `features=[0, 1]` (streams keyed by column, not position).
- **Decomposition:** `importances_window.sum(1)`, `importances_id.sum(1)`, CR `importances_cause.sum(1)` each equal `importances_mean` (1e-12); `importances.mean(1) == importances_mean`; `cause=k` equals the hand-computed per-cause-k drop from two `piecewise_exponential_score` calls.
- **Literal known-hazard case:** 4 rows, 2 time strata of 2 rows, a stub hazard `exp(x0)` per unit time, windows `[0, 1, 2]`, `n_repeats=1`: the importance equals the hand-computed literal for the swap pattern the seed produces (both patterns' literals are in the test; the recorded `X` selects which).
- **Hand check of one permutation:** with `n_repeats=1`, the result equals `(S(intact) − S(permuted)) / N` computed from the recorded permuted `X` via `piecewise_exponential_score` directly.
- **OOB:**
  - baseline score equals the PE score of `oob_cumhaz` on the rebuilt design computed directly (id, block, coarse modes); in block mode also against an independent reference: the rows are the unsplit coarsened training rows and their OOB prediction reproduces `oob_prediction_` (`cumsum` at `event_times_`), as the S16 test does;
  - a fingerprint mismatch (perturbed `y`) raises; `oob=True` without `measured_at` on a forest fitted with it raises; `measured_at` / `block_time` with `oob=False` raise;
  - `n_rows_excluded` equals coarsening-dropped rows plus rows with `oob_n_trees_ == 0`;
  - `importances_se` is NaN.
- **Bootstrap:**
  - `_resample_ids` directly: every copy contains all of its id's rows, in order; duplicated ids get distinct `boot_ids`; `row_index` length = sum of the drawn ids' row counts;
  - strata see copies as distinct rows (a recorded replicate `X`: two copies of one id can exchange values between each other);
  - SE finite with `B ≥ 2`, NaN with 0 and 1;
  - a pure-noise column's `mean ± 1.96·se` covers 0 in ≥ 18 of 20 seeded runs (coarse sanity check).
- **M1 warning:** `strata=None` warns; `"extrapolat"` is in the docstring.
- **Share of gain:** equals `importances_mean / (baseline − null)`; a model that does not beat the null (stub = the null) gives NaN and warns.
- **Errors:** unknown feature name/index, feature names without `feature_names_in_`, overlapping / empty groups, `features` + `groups`, ids column as a unit, strata of the wrong length, `y=None`, `scoring="brier"`, bad windows (beyond `τ`), `n_repeats < 1`, `n_bootstrap < 0`, a landmark estimator (`NotImplementedError`), a non-rftvc estimator (`TypeError`), an unfitted estimator.
- **DataFrame:** a DataFrame `X` with an ids column named by `ids="id"` gives the same result as arrays; features by name.

## Simulations (T6; rules from design §7, predeclared)
Oracle: the true unit's score drop with the **known hazard** as the `predict` callable, on a 10⁵-subject evaluation sample, under the same strata (`"time"`, `n_strata=10`) and windows. Forest: 1000 training ids, 1000 evaluation ids, 200 trees, defaults otherwise. R = 50; a 10-rep pilot checks MC SE ≤ margin / 7.
1. **§7.1 trend confounding (M2 part).** Unit intervals `k = 0..7`; `z1_k ~ N(0,1)`; `z2_k = 0.6 z1_k + 0.8 e_k + 0.5 (k − 3.5)` (trend; the within-interval correlation is 0.6); hazard `0.1 · 1.2^k · exp(0.8 z1_k)` on `(k, k+1]` (a time-varying baseline, so `z2` tracks time); plus two baseline noise columns; `X = [z1, z2, n1, n2]`.
   - Pass (design §7.1, as declared): `|mean imp(z2)| ≤ 0.1 · oracle(z1)` and `mean imp(z1) ≥ 0.5 · oracle(z1)`.
   - **Estimand caveat (Codex):** time-stratified PFI measures the *fitted forest's* reliance on `z2`, and a forest may legitimately use `z2` as a partial proxy for `z1` (ρ = 0.6), so the `z2` rule can fail for a correct forest. The rule stays as declared by the design; reported alongside it, without a rule: `z2` with `conditional_on=["z1"]` (the "no effect given `z1`" estimand) and M1 (`strata=None`) means for `z1`, `z2`. If the declared rule fails and the conditional arm is ≈ 0, that is the evidence for a user-approved deviation.
2. **§7.3 timing.** S3-type generator; `z` acts only on `t ∈ (2, 4]` (`0.15 · exp(1.0 z · 1{2 < t ≤ 4} + 0.4 x0)`); `windows = [0, 1, 2, 3, 4, 5, 6]` (fixed, so windows align across replications).
   - Windows (right-closed): 1 = `(0, 1]`, 2 = `(1, 2]`, 3 = `(2, 3]`, 4 = `(3, 4]`, 5 = `(4, 5]`, 6 = `(5, 6]`; inside = 3, 4.
   - Oracle: `O = oracle importance of z summed over windows 3–4`; margin `δ = 0.1 · O`.
   - Pass (a): one-sided t-tests (> 0) of `importances_window[z, m]` over replications, Holm across the 6 windows at 5 %, reject in at least one inside window.
   - Pass (b), equivalence outside (not mere non-rejection): for each outside window, the one-sided upper 95 % bound (Bonferroni over the 4 outside windows) of the mean is `≤ δ`.
   - The pilot checks MC SE of each window mean `≤ δ / 7`.
3. **§7.4 CR.** The S14 scenario A generator (`z` acts on cause 1 only).
   - Oracle `O_1` = the true-hazard cause-1 importance of `z`; margin `δ = 0.1 · O_1`.
   - Pass: `ΔS_1(z)` Holm-significant (one-sided t-tests over replications, 2 causes), and the one-sided upper 95 % bound of `mean ΔS_2(z)` is `≤ δ`. The design's `mean ΔS_2 ≤ 0.1 · mean ΔS_1` is also reported. The pilot checks MC SE `≤ δ / 7`.
- A failed rule blocks the merge unless the user accepts a documented deviation.

## Acceptance
- Full suite green (fast), slow sims run once and reported.
- The three sim rules pass (or a user-approved deviation).
- Timings reported: held-out with `n_bootstrap=0` and 100, and OOB, on 1000 ids × ~5 rows, p = 4, 200 trees.

## Review log
Codex plan review (2026-09-26), 7 findings: 5 accepted, 1 partly, 1 rejected.
1. **Rejected — "block-mode OOB rows misaligned with `oob_set`"**. `_blocks.oob_sets` builds one CSR set per *unsplit* row (`id_index = groups`), and `fit` already computes `oob_prediction_` on `d.X` with `d.oob_set`; the S16 test `test_stacked_coarse_block_rebuild_matches_the_fit` pins this. A block-mode reference check is added anyway (OOB tests).
2. **Partly accepted — the §7.1 `z2` rule can fail for a correct forest** (proxy use of a correlated `z2`). The declared design rule is kept; the estimand caveat and a reported `conditional_on=["z1"]` arm are added as evidence for a possible deviation.
3. **Accepted — per-unit streams depended on request order** (`spawn` is positional) → `spawn_key` from the unit's smallest column index.
4. **Accepted — timing/CR rules not oracle-relative; "none outside" was non-rejection** → oracle margins `δ = 0.1·O`, an equivalence (upper-bound) test outside, pilot MC SE against `δ`.
5. **Accepted — binning edge ownership and parameter validation** → `side="left"` with worked examples; `n_strata` / `n_bins` / label validation.
6. **Accepted — bootstrap estimand unstated** → the SE of the `R`-repeat statistic conditional on the forest, including permutation noise; documented.
7. **Accepted — no literal known-hazard test** → a 4-row literal case; the block-mode OOB reference check.
