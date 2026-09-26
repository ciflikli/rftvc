# Implementation Plan: competing risks (S11–S14)

Source of truth: `cr-design.md` v2 (C1–C8; C2/C3 user-confirmed). Background: `cr-research.md`, `design.md` v2. Conventions, oracles and the slice workflow are those of `plan.md` (branch `feat/sN-*`, slice plan `sN-plan.md` with a Codex plan review before coding, Codex diff review, "SN done" note, user-approved merge).
Notation: `J` causes (labels remapped to 1..J; 0 = censored), `K` grid (event) times, `T` requested times.

## Status
- [x] S11: Core — cause-coded `y`, engine cause dimension, composite criterion, per-cause leaves, `CompetingRisksForestTV` fit + per-row CIF (branch `feat/s11-cr-core`)
- [x] S12: Paths, coarsening, `aggregate="cif"`, OOB + Wolbers C, Approach-B equivalence (branch `feat/s12-cr-paths`)
- [x] S13: Cause-specific metrics + landmark competing-risks workflow (branch `feat/s13-cr-landmark`)
- [x] S14: Bench (simulations, parity, bake-off) + docs + case study (branch `feat/s14-cr-bench`)

## Plan-level decisions (defaults; the slice plan reviews may change them)
- **P1. Aalen–Johansen runs in Rust.** The CIF needs the increments at every event time between the origin and `t`, not only at requested times. Per row (or path), increments are accumulated into a dense per-thread `K × J` buffer over the leaf entries, then one sweep writes `S` and `F_k` at the requested times. `O(K·J)` per row, no `n × J × K` array crosses into Python.
- **P2. Unsupported options raise until their slice lands.** S11's `CompetingRisksForestTV` raises `NotImplementedError` for `intervals`/`ids` at predict (paths), `ntime`, `resample_unit="block"`, `oob_score=True`, `aggregate="cif"` and `score`. Nothing silently falls back.
- **P3. Counting-process fitting is in S11.** The engine handles `(start, stop]` rows and left truncation unchanged, and the leaf oracle (`survfit` with `id`) is on start–stop rows, so S11 fits TVC data; only path *prediction* waits for S12.
- **P4a. J = 1 dispatch.** With J = 1 the estimator always uses the existing `LtrcLogRank` scorer, leaf accumulation and RNG use, whatever `criterion` says (`"composite"` is mathematically the same statistic but a different floating-point expression, so it is not relied on for bit identity).
- **P4. Single-cause route.** `split_cause=k` wraps the existing `LogRankNode` on the cause-k column (copied contiguous per node/candidate). Extra all-cause event times add exact zeros (`mask = e = c = 0`), so scores equal the single-event forest's bit-for-bit.
- **P5. Bake-off challengers are prototyped behind `SplitCriterion` in S14 and removed unless adopted** (as in S8; history kept in the commit).
- **P6. `plan.md`:** S11–S14 get status lines pointing here; "Competing risks" leaves "Deferred" (multi-state and recurrent events stay).

---

## S11: Core (vertical: fit + per-row CIF)
**Files:**
- Rust core: `data.rs` (`SurvData.event: Vec<u8>`, `n_causes`), `grid.rs` (exact grid from `code != 0`, so every cause's event times are grid points; quantile grid / coarsening stay bool (J = 1) until S12), `criterion.rs` (`Profile.cause_events`, `n_causes`; `CompositeCauseLogRank`; `SingleCause { k }`), `splitter.rs` (events accumulated `bins × K × J`; per-cause counts in `NodeProfile`), `tree.rs` (per-cause leaf cumhaz, entry-major, `n_causes` in `Tree`), `forest.rs` (`n_causes` in `Forest`; `predict_cif` per row, `aggregate="hazard"`), `flat.rs` (format v3, stride invariant, v2 states load as J = 1).
- Binding `rftvc-py/src/lib.rs`: `fit_forest(event: u8[], n_causes, criterion=…, split_cause=None)`, `Forest.predict_cif`, `leaf_profile` returns `(times, cumhaz[n_e, J])`; existing bool callers convert (`event.astype(u8)`).
- Python: `_validation.py` (`make_competing_risks_y`, `check_competing_risks_y`; structural checks in `check_counting_process` use `event != 0`, not `&`, so a non-terminal label 2 is caught (`2 & True == 0` today); `check_survival_y` errors on integer events > 1 with a pointer to the new class), `_estimator.py` (shared private `_BaseForestTV`; `SurvivalForestTV` keeps its surface), new `_competing.py` (`CompetingRisksForestTV`), `__init__.py`.
- Tests: `tests/test_cr_core.py`, `tests/ref/cr_ref.py` (naive composite score + naive per-cause Nelson–Aalen / AJ with delayed entry), `tests/fixtures/make_aj_fixtures.R` → `aj_survfit.json`; Rust unit tests in `criterion.rs` / `splitter.rs`.

**Signatures:**
- `make_competing_risks_y(stop, event, start=None)`; `check_competing_risks_y(y, causes=None) -> (start, stop, codes u8, causes_)`.
- `CompetingRisksForestTV(n_estimators=100, *, causes=None, criterion="composite", split_cause=None, score_cause=None, aggregate="hazard", min_events_leaf=3, …same as SurvivalForestTV…)`.
- `.fit(X, y, ids=None)`; `.predict_cumulative_incidence(X, times, *, cause=None)` → `(n, J, T)` or `(n, T)`; `.predict_cumulative_hazard(X, times, *, cause=None | "all")`; `.predict_survival_function(X, times)` (event-free); `.predict(X)` → `F_{score_cause}` at the last event time; attributes `causes_`, `n_causes_`.

**Tests:**
- **Leaf oracle:** a single-node tree (`max_depth=0`) equals `survival::survfit(Surv(start, stop, cause) ~ 1, id=id)` `pstate` on start–stop rows with delayed entry and ties, to 1e-10; also equals `cr_ref.py`.
- **J = 1 bit-identity:** `CompetingRisksForestTV` (default `criterion="composite"` and `split_cause=1`) on {0,1} events equals `SurvivalForestTV` (node arrays, leaf arrays, hazard-aggregated cumhaz, `apply`), several seeds and `max_features`; its CIF is `1 − Π(1 − ΔΛ)` of the same hazard, not `1 − exp(−Λ)` (`s11-plan.md` review 1).
- **Single-cause score:** `SingleCause{k}` equals `LtrcLogRank` on "cause k vs rest" bit-for-bit (Rust unit test) and `logrank_ref.py`.
- **Composite score:** equals `cr_ref.py` (hypothesis, small random data incl. empty children and `y < 2`); a covariate that raises cause 1 and lowers cause 2 scores > 0 where the all-cause log-rank ≈ 0.
- **Grid:** an event time that only cause 2 has is a grid point and enters the leaf AJ.
- **Structure:** non-terminal events with labels 1 and 2 raise; codes pass through unchanged.
- **Splitter:** per-cause child counts equal brute force (extends `child_summaries_match_brute_force`).
- **AJ invariants:** `Σ_k F_k + S = 1`, `F_k` non-decreasing, clamp counted when `1 − Σ ΔΛ < 0`.
- **Labels:** non-contiguous labels (e.g. {2, 5}) map to `causes_`; `causes=` vocabulary with a missing cause gives zero hazard + `UserWarning`; unknown `cause` → `ValueError`; labels outside `causes` → `ValueError`.
- **Pickle:** v3 round trip; a v2 (single-event) state loads as J = 1 with identical predictions; corrupt stride rejected without panic.
- Existing suite green (S9 bit-identity guards unchanged).

**Accept:** oracles pass; `SurvivalForestTV` bit-identical to `main` on the S9 identity bench (`bench/s9_identity.py`); J = 2 fit time at 100k rows ≤ 1.5× J = 1 (reported in the S11 note).

---

## S12: Paths, coarsening, ensemble options, OOB
**Files:** `forest.rs` (`predict_cif_paths` with origin/`extrapolate`, per-tree AJ for `"cif"`, `oob_cif`), `grid.rs`/`coarsen` (cause code moves with the event), `tree.rs` (`min_events_leaf_cause` in the pre-RNG `can_split` gate and the child constraint; per-leaf per-cause event counts `leaf_cause_events[leaf*J + j]` (u32) stored in `Tree` and the flat state, C5 diagnostics), binding; `_competing.py`, `_blocks.py` (codes through `split_at_blocks`), `metrics.py` (`concordance_index_cr`); tests `tests/test_cr_paths.py`, `tests/test_cr_oob.py`, `tests/test_sklearn_compat.py`.

**Signatures:**
- `predict_cumulative_incidence(X, times, *, cause=None, intervals=None, ids=None, origin=None, extrapolate="none")` (same path keywords as `SurvivalForestTV`); same for hazard / survival.
- `aggregate="hazard" | "cif"`; `min_events_leaf_cause: int | None` (requires `split_cause`).
- `forest_.leaf_cause_events(tree)` → `(n_leaves, J)`; estimator `leaf_cause_events_summary()` (per-cause min / quantiles over leaves).
- `concordance_index_cr(y, risk, cause, ids=None)`; `.score(X, y, ids=None)`; `oob_prediction_ (n, J)`, `oob_score_`, `oob_n_trees_`.

**Tests:**
- **Paths:** a one-row path equals per-row prediction; the path AJ equals a Python reference that routes each row and sums increments on `(start_r, stop_r]`; origin `u` resets `S(u) = 1`, `F(u) = 0`; `extrapolate="locf"` as in S3.
- **`"cif"` aggregation:** equals the mean of per-tree AJ (Python reference); both aggregations keep `Σ F + S = 1`.
- **Approach-B equivalence:** `split_cause=k`, `min_events_leaf=1`, `min_events_leaf_cause=m`, fixed seed, exact grid ⇒ tree seeds and node arrays equal `SurvivalForestTV(min_events_leaf=m)` on "cause k vs rest", and each tree's cause-k cumhaz equals it at the cause-k event times (leaf arrays differ only by zero-increment entries).
- **Coarsening:** `ntime` with causes equals coarsening then fitting on the snapped cause-coded data (`coarsen_ref.py` extended); J = 1 unchanged.
- **Block resampling:** codes survive row splitting (event piece keeps its label); block OOB runs.
- **Wolbers C:** hand-worked fixture (type A / B pairs, delayed entry, ties) and a brute-force O(n²) reference; J = 1 equals `concordance_index_cp`.
- **OOB:** matches a manual OOB reference (extends `test_oob.py::_manual_oob_mortality`) for both aggregations; coarsening-dropped rows are all-NaN.
- **Leaf diagnostics:** `leaf_cause_events` equals brute-force in-bag terminal-leaf assignment counts (in-bag rows via `in_bag_ids` + `apply`).
- **sklearn:** `check_estimator` parameter sets for the new class with the same expected-failure mapping.

**Accept:** all references pass to 1e-12; sklearn compat matrix regenerated with the new class.

---

## S13: Metrics + landmark competing risks
**Files:** `metrics.py` (`cause=` in `brier_landmark`, `integrated_brier`, `cindex_dynamic`), `landmark.py` (integer labels in `make_landmark_data`; `_LandmarkBase`; `LandmarkCompetingRisksForest`; `LandmarkSurvivalForest` errors on labels > 1), `model_selection.py` (a CR branch in `_score_landmark`, `landmark_cross_validate` and nested `_select`: request `predict_cumulative_incidence(…, cause=k)`, score with the cause-specific Brier / IBS / Wolbers C, and return `cif` columns plus `cause` metadata instead of a `survival` column), `__init__.py`; tests `tests/test_cr_metrics.py`, `tests/test_cr_landmark.py`.

**Signatures:**
- `brier_landmark(y_test, risk, w, *, cause=None, y_censor=None, censoring_estimator=None, g_min=0.05, return_info=False)`: the existing positional signature is kept and `cause` is keyword-only; `cause=None` keeps today's behaviour; `integrated_brier` likewise; with `cause=k`, status at `s+w` is cause k (1), competing or event-free (0), censored before `s+w` (IPCW-out); a competing event is weighted `1/G(T−)`.
- `cindex_dynamic(…, cause=k)`: Wolbers IPCW version (type A `1/(G(T_i−)G(T_i))`, type B `1/(G(T_i−)G(T_j−))`).
- `make_landmark_data(…)`: `_lm_event` = terminal cause label if `T ≤ min(C, s+w)`, else 0; the terminal label is the id's last row's label (not `any()`).
- `LandmarkCompetingRisksForest(…, causes=None, score_cause=None).predict_risk(df, s, horizon=None, cause=None)` = `F_k(s+w | s, H(s))`; `predict_cumulative_incidence(df, s, times, cause=None)`.

**Tests:**
- CIF Brier / IBS equal a hand-computed fixture and, on right-censored data without delayed entry, `comprisk` (optional dev dependency, `importorskip`); `cause=None` results unchanged (existing tests).
- With a single cause, every `cause=1` metric equals its survival version.
- Dynamic Wolbers C equals a brute-force reference with the same `G`.
- Landmark: a bool event column gives today's stacks bit-identically; a cause-coded column gives labels only on terminal rows within the horizon; `LandmarkSurvivalForest` raises on labels > 1; `causes=` keeps J fixed across CV folds; clone / nested params / pickle.
- `landmark_cross_validate` on a `LandmarkCompetingRisksForest` scores `F_k` (not `1 − S`): equals a manual per-fold CIF Brier / C; survival-model output unchanged.
- Landmark input with a non-terminal cause label raises (structural check from S11).

**Accept:** metric references pass; PBC2 (`pbcseq`, transplant vs death) landmark CV runs end-to-end in the tests' fast path (small subset).

---

## S14: Bench, docs, case study
**Files:** `bench/s14_cr_sim.py`, `bench/s14_cr_parity.py` (+ R script, rfsrc in a scratch lib per `bench/cr_rfsrc_check.R`), `bench/s14_cr_bakeoff.py`; `docs/bench/s14-cr.md` (+ CSVs); `docs/source/user_guide/competing_risks.rst`, `api.rst`, `compatibility.rst`; `examples/pbc2_competing.py` + `docs/source/case_studies/pbc2.rst` section.

**Scope:**
- **Simulations (known truth):** cause-specific Cox hazards with TVCs and left truncation; opposing effects across causes; a rare cause (≤ 5% of events). Metric: L2 to the true CIF, landmark CIF Brier.
- **Parity (right-censored):** CIF accuracy vs `randomForestSRC` (`splitrule="logrank"`) and Approach B on a public dataset (PBC baseline, transplant competing, or `follic`).
- **Bake-off, rule fixed in `s14-plan.md` before runs (as S8):** criteria `composite` / full quadratic form / Ishwaran et al. 2014 composite / all-cause log-rank / `split_cause=k`; aggregation `hazard` vs `cif`; `min_events_leaf_cause` default for rare causes.
- **Scale:** memory and fit time for J ∈ {2, 4} at 1M rows vs the J = 1 targets in `design.md`.
- **Docs:** one user-guide page (targets, CIF along external paths only — D4, landmarking for dynamic prediction, cause labels / `causes=` in CV); PBC2 competing-risks case study.

**Accept:** a written recommendation with defaults confirmed or changed (C3, C4, C5) and non-winners removed (P5); docs build with `-W`; memory within `(1+2J)/3 ×` the J = 1 leaf memory plus the reported split-search overhead.

---

## Risks specific to the plan
| Risk | Mitigation |
|---|---|
| S11 is large (engine + binding + new class) | P2 keeps scope to per-row prediction; the slice plan may split the Rust part into its own PR if the diff exceeds review size |
| Root split histograms grow to `bins × K × J` f64 | measured in S11's accept check; fallback: sparse per-bin event lists (events are one per row) |
| Bit-identity drift in `SurvivalForestTV` from shared refactors | S9 identity bench + existing node-array tests run on every slice |
| `survfit` / `comprisk` fixtures need R or optional packages | fixtures generated once and committed (JSON); optional deps `importorskip` |

## Plan review (Codex, 2026-09-26)
1. S11 changed `SurvData.event` to codes but left grid construction to S12, while `SurvData::new` builds the exact grid from bool events (blocker) → exact grid from `code != 0` in S11; test a cause-2-only event time.
2. J = 1 bit identity was claimed for the default `"composite"`, a different floating-point expression than `LogRankNode` (high) → P4a: J = 1 always dispatches to `LtrcLogRank`; tested with both spellings.
3. The planned `brier_landmark` signature broke the public one used positionally in CV (high) → existing signature kept, `cause` keyword-only.
4. "Pass `cause`" to CV still scored `1 − S` (high) → explicit CR branch in `_score_landmark`, `landmark_cross_validate`, `_select`; CIF output columns.
5. The one-event-per-id check uses `event & not_last`, which misses label 2 (high) → `event != 0` in S11, tests for labels 1 and 2 incl. landmark input.
6. C5 per-cause leaf diagnostics were dropped (medium) → per-leaf per-cause counts stored in S12, exposed and tested against brute force.

**S11 done (2026-09-26). Deviations / notes** (slice plan and review logs: `s11-plan.md`):
- `CompetingRisksForestTV` fits counting-process rows with cause labels. It predicts per-row CIF, cause-specific / all-cause hazard and event-free survival. `SurvivalForestTV` is bit-identical to `main` (`bench/s9_identity.py`); its pickle grows by 13 B (the `n_causes` key, format v3). v2 states load as J = 1.
- `cause_events` is **cause-major** (`[j·K + k]`), not the time-major layout in `cr-design.md`. The first version (time-major, two allocations per candidate) fitted J = 2 in 2.9 × the J = 1 time. With a contiguous column per cause, the composite runs `LtrcLogRank`'s dense candidate loop per cause, skipping causes with no event in the node. `SingleCause` is that loop on one column, so it stays bit-identical to `LtrcLogRank`.
- No `bins × K × J` histogram exists (one running `K × J` left-child buffer), so the root-histogram risk does not apply.
- Fit at 100k rows, 100 trees (`bench/s11_cr_timing.py`): J = 1 is 1.0 ×, J = 2 1.32 × and J = 4 2.11 × the single-event time. The forest's leaf memory is 1.40 × at J = 2 and 2.21 × at J = 4.
- `n_estimators` defaults to 500 (the plan's 100 was a typo). `leaf_profile` returns `(times, cumhaz (n_e, J))`. Event-free `S` is the Aalen–Johansen product limit, not `exp(−Λ)`.
- `Grid::quantile` / `coarsen` stay bool until S12 (plan-review finding 5 was declined).

**S12 done (2026-09-26). Deviations / notes** (slice plan, plan/diff review logs and test audit: `s12-plan.md`):
- One Rust Aalen–Johansen kernel serves per-row prediction, paths (origin, `extrapolate`) and OOB. A row is a one-row path covering all times, and S11 results are unchanged. `aggregate="cif"` runs the kernel per tree and averages `F`, `S` and the hazard.
- `min_events_leaf_cause` (with `split_cause`) is in the pre-RNG gate and the child constraint. Approach-B equivalence holds bit-for-bit (seeds, node arrays, cause-k leaf hazards).
- Per-leaf per-cause in-bag counts are stored only for `CompetingRisksForestTV`. Their pickle key is optional, so `SurvivalForestTV` pickles are byte-identical (the identity bench now checks pickle bytes with `--pickle`). The format stays v3.
- `concordance_index_cr` (Wolbers) on counting-process rows: type A as in `concordance_index_cp`, type B = competing events at `T_j ≤ T_i`, ties included. It backs `score` and `oob_score_`. `oob_prediction_` is `(n, J)`.
- Coarsening and block splitting carry cause codes. The coarsened OOB target keeps labels (`_oob_target` hook).
- Block OOB is checked by equivalence (one block per id = id resampling), not by a separate manual reference.
- Fit time at 100k rows: J = 2 is 1.43× the single-event time and J = 4 2.20×. The J = 2 forest is 1.47× the size, including the leaf counts (`bench/s11_cr_timing.py`).

**S13 done (2026-09-26). Deviations / notes** (slice plan, review logs and test audit: `s13-plan.md`):
- `brier_landmark`, `integrated_brier` and `cindex_dynamic(kind="incident")` take a keyword `cause`. With `cause=k`, competing events by `w` are observed outcomes weighted `1/G(T−)`, and `integrated_brier` takes `F_k`. Competing-risks AUC (`kind="cumulative"`) raises, as designed.
- **Design amendment:** the dynamic Wolbers type-A weight is `1/G(T_i−)²` (Uno), not `1/(G(T_i−) G(T_i))`, so `cause=1` on one-cause data equals the survival C exactly. The two differ only when censoring ties a case time; a test pins the chosen form.
- `make_landmark_data` keeps terminal cause labels: `SURV_DTYPE` for bool / {0, 1} columns (bit-identical stacks), `CR_DTYPE` otherwise.
- `_LandmarkBase` is shared. `LandmarkCompetingRisksForest` adds `causes`, `score_cause`, `predict_risk(cause=)` and `predict_cumulative_incidence`. `LandmarkSurvivalForest` rejects labels > 1.
- `landmark_cross_validate` has a CR branch: one vocabulary and scored cause are pinned before splitting and in every inner fit, and `cif` / `cause` prediction columns replace `survival`. `_censor_at` keeps labels.
- The Brier oracle is a committed comprisk 0.8.0 fixture generated from a scratch install; no new dependency.

**S14 done (2026-09-26). Deviations / notes** (slice plan, review logs and results: `s14-plan.md`, `docs/bench/s14-cr.md`):
- Bake-off: `composite` stays (C3); **`aggregate="cif"` becomes the default** (C4); `min_events_leaf_cause` stays `None` (C5: a rare-cause floor harms the other causes).
- The challengers `quadratic`, `ishwaran` and `logrank_all` were removed (P5; rerun at `66ccad8`).
- randomForestSRC's composite (Ishwaran eq. 3.2) was verified from its C source: with equal weights it sums numerators into the all-cause numerator, and it was 12–26% worse in simulation.
- PBC parity (per-fold imputation, no IPCW clipping): rftvc is better than rfsrc on transplant and similar on death, and ~35× faster including R start-up.
- Scale: J = 2 is ≤ 1.4× and J = 4 ≤ 2.2× the single-event fit time, peak RSS ≤ 1.5×, and leaf bytes equal the `(1 + 2J)/3` bound.
- Docs: a competing-risks user-guide page, compatibility, and a PBC2 transplant-vs-death case study (`examples/pbc2_competing.py`).
- Out of scope, as designed: Fine–Gray / Gray splitting (C6), competing-risks AUC, multi-state.
