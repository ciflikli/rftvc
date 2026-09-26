# S15 slice plan: competing-risks cumulative/dynamic AUC

Branch `feat/s15-cr-auc`. Parent: `cr-design.md` "Metrics" (CR AUC deferred), `s13-plan.md` decision 4, user request 2026-09-26. The S13 metric machinery (`_classes`, `_ipcw` with `cause`) is reused.
Notation: `w` horizon (landmark reset clock), `k` cause of interest, `G` censoring survival (reverse KM, events first), `r` the predicted risk `F_k(w)`.

## Decisions (defaults; revisit in review)
1. **Estimand:** the cumulative/dynamic AUC of Blanche, Dartigues and Jacqmin-Gadda (2013), **definition 2**: `P(r_i > r_j | case i, control j)`, ties ½.
   - Cases: a cause-`k` event by `w`.
   - Controls: every subject whose status at `w` is observed and is not a case, i.e. event-free through `w` **or** with a competing event by `w`.
   - Definition 2 matches the Brier classification of S13 (a competing event is an observed "no") and is the usual default (`timeROC`, `comprisk`).
   - Definition 1 (controls event-free only) is not offered; it answers a different question.
2. **Weights:** the IPCW weights of S13's `_ipcw` with `cause=k`: `1/G(T_i−)` for cases, `1/G(w−)` for event-free controls, `1/G(T_j−)` for competing controls, all clipped at `g_min`.
   - `AUC = Σ_i Σ_j w_i w_j (1{r_i > r_j} + ½ 1{r_i = r_j}) / (Σ_i w_i · Σ_j w_j)`.
   - It is computed by sorting the controls by risk with cumulative weights, `O(n log n)`.
   - **Tie convention (plan review 1):** rftvc's, not comprisk's. Event-free controls are `stop ≥ w` (a subject censored exactly at `w` is a control), weighted `1/G(w−)`, as in rftvc's Brier and survival cumulative AUC. comprisk uses `stop > t` and `G(t)`, so the two agree only when no observed time equals `w`. The oracle claim is limited to that case, and a boundary test pins our convention.
   - A dedicated helper `_ipcw_classes` returns `(case, competing, control, weights, info)`. `_ipcw` delegates to it unchanged, so the competing mask is not lost (plan review 2).
3. **API:** `cindex_dynamic(y_test, risk, w, kind="cumulative", cause=k, …)` stops raising `NotImplementedError`.
   - **Without competing controls, the existing cumulative-AUC code runs verbatim** (plan review 3), so `cause=1` on one-cause data equals `cause=None` exactly.
   - No cases, or no controls, raises `UndefinedMetricError`, as today.
4. **CV:** `landmark_cross_validate` stops rejecting `"cindex_cumulative"` for `LandmarkCompetingRisksForest` and passes `cause=k`.
5. **Docs (plan review 4, explicit):**
   - the user-guide competing-risks page ("Competing-risks AUC is not implemented" → the definition);
   - the `cindex_dynamic` docstring;
   - `cr-design.md`: both the Metrics line "Time-dependent AUC for competing risks … deferred" and the "Out of scope" entry are replaced;
   - `s13-plan.md`: the "raises" statements are marked superseded by S15, keeping the history;
   - the `cr-plan.md` note.

## Tests (`tests/test_cr_metrics.py`, `tests/test_cr_landmark.py`)
- **Oracle:** the committed comprisk 0.8.0 fixture is extended with `_per_time_auc_brier`'s AUC (same data: continuous times, none at an evaluation time, so `G(w−) = G(w)`). Our value matches to 1e-12 for both causes at every evaluation time.
- **Brute force (hypothesis):** an O(n²) weighted pair sum with the same reverse KM, with ties in risk and in times (including competing events and censoring exactly at `w`), `g_min` clipping, and exact vs IPCW paths.
- **Hand-worked literals:**
  - The S13 boundary fixture (41/350) gets a written-out AUC value covering a competing control at `w`, a censored control at `w` and clipping-free weights.
  - The same data under comprisk's convention (`stop > w`, `G(w)`) gives a different literal, so the tie convention is pinned.
  - A case whose only controls are competing events gives a defined AUC.
- **One cause:** `cause=1` on bool data equals `cause=None` (with and without IPCW).
- **CV:** `cindex_cumulative` runs for the CR wrapper and equals a manual per-fold `cindex_dynamic(kind="cumulative", cause=k)`.
- Mutation check: controls restricted to event-free (definition 1) must fail the oracle and the brute force.

## Acceptance
- The oracle and brute force pass, and the existing suite is green.
- The identity bench is bit-identical (`--pickle`).
- Docs build with `-W`.

## Tasks
- [ ] Fixture: extend `make_cr_metric_fixtures.py` with the AUC and regenerate
- [ ] `cindex_dynamic` cumulative with `cause`; CV branch; docs
- [ ] Tests + test-quality audit (mutation)
- [ ] Identity bench; `cr-plan.md` note
- [ ] Codex diff review; fixes

## Plan review (Codex, 2026-09-26)
1. comprisk's control convention (`stop > t`, `G(t)`) differs from ours at `t = w`, so the oracle could not check our boundary (blocker) → our convention is stated explicitly, the oracle claim is limited to non-boundary data, and a literal boundary test pins it.
2. `_ipcw` drops the competing mask that the AUC needs (high) → a `_ipcw_classes` helper, plus a literal test with only competing controls.
3. "Bit-for-bit" with one cause was not guaranteed by the weighted formula (medium) → a fast path through the existing code when there are no competing controls.
4. The parent docs would contradict S15 (low) → the explicit doc edits are listed.
