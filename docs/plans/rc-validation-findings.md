# Real-data validation findings: Rossi + EBMT4

Parent: `docs/plans/rc-validation-plan.md` (Decision 7's deliverable). Produced by `examples/rc_validation.py`, run 2026-09-27. Raw CSVs in `docs/plans/rc-validation/`.

## Scope reminder

Per the plan's Decision 8/scope correction: this is an API/reshape/plausibility exercise on real data, not a clinical or criminological validation against an established published effect. `permutation_importance`/`drop_column_importance` report **relevance magnitude only** — never read as a signed effect anywhere below. Direction comes only from `hazard_effect` contrasts and the matching `CoxTimeVaryingFitter` coefficient's sign.

## Rossi (single-event TVC)

`SurvivalForestTV(n_estimators=500, min_ids_leaf=10)`, 432 subjects, 1405 counting-process rows. Fit in 0.05s; OOB concordance 0.687.

**Relevance ranking** (`oob=True` permutation and 5-fold drop-column importance; `docs/plans/rc-validation/rossi_relevance.csv`):

| feature | perm_importance | loco_importance |
|---|---|---|
| employed | 0.2334 | 0.1096 |
| age | 0.1049 | 0.0613 |
| wexp | 0.0307 | 0.0190 |
| fin | 0.0238 | 0.0127 |
| educ | -0.0033 | 0.0150 |
| race | -0.0025 | 0.0006 |
| paro | -0.0012 | -0.0057 |
| mar | -0.0012 | -0.0021 |
| prio | -0.0395 | -0.0291 |

`employed` and `age` dominate by both measures; `prio`'s *negative* relevance under both measures is a real, correctly-reported possibility (not an error) — a negative score drop means permuting/dropping the unit made the model's held-out score *better*, which the docstrings already document as possible for a weak or noisy covariate (relevance is a magnitude comparison across units here, not a claim that every unit helps).

**Direction** (`hazard_effect` two-point contrasts vs `CoxTimeVaryingFitter`'s coefficient sign; `docs/plans/rc-validation/rossi_directions.csv`):

| feature | contrast | forest direction | Cox coef | Cox direction | agree |
|---|---|---|---|---|---|
| fin | 0 -> 1 | lower hazard | -0.336 | lower | **yes** |
| age | 20 -> 30 | lower hazard | -0.048 | lower | **yes** |
| prio | 0 -> 10 | higher hazard | +0.078 | higher | **yes** |
| employed | 0 -> 1 | lower hazard | -1.336 | lower | **yes** |

All four agree with Cox, and all four match the well-replicated textbook result (Rossi, Berk & Lenihan 1980; financial aid, employment and age lower the hazard of rearrest, prior convictions raise it) that the plan's Decision 4 named as the expected direction. This is a real, if informal, sanity check that `hazard_effect` on real data recovers the same qualitative story a standard Cox fit does, on a genuine time-varying covariate (`employed`).

## EBMT4 (competing risks + TVC)

`CompetingRisksForestTV(n_estimators=500, min_ids_leaf=10, causes=[1,2])`, 2279 patients, 3413 counting-process rows. Fit in 0.05s.

**Relevance ranking** (per cause; `docs/plans/rc-validation/ebmt4_relevance_cause{1,2}.csv`; regenerated after the determinism fix below — see the note at the end of this subsection):

| feature | cause 1 perm | cause 1 loco | cause 2 perm | cause 2 loco |
|---|---|---|---|---|
| year | 0.0229 | -0.0316 | 0.0762 | 0.0289 |
| agecl | 0.0157 | -0.0252 | 0.1067 | 0.0376 |
| proph | 0.0099 | 0.0182 | 0.0162 | 0.0254 |
| ae | 0.0000 | 0.0120 | 0.0000 | 0.0870 |
| match | -0.0220 | -0.0344 | 0.0339 | 0.0018 |

`ae`'s permutation importance is ~0 for both causes but its LOCO importance is clearly positive (0.0120 for cause 1, 0.0870 for cause 2) — the two measures can legitimately disagree (LOCO answers "predictive value for new subjects given everything else", permutation answers "reliance holding time fixed"; the docs already warn correlated/weak-signal units can show this split). `agecl` is the most relevant baseline covariate for cause 2 (death without relapse) by both measures, plausibly since age at transplant is a well-known driver of transplant-related mortality.

**Direction** (`docs/plans/rc-validation/ebmt4_directions.csv`):

| cause | feature | n_levels | forest direction | Cox direction | agree | gated |
|---|---|---|---|---|---|---|
| 1 | ae | 2 | lower | lower | **yes** | yes |
| 1 | proph | 2 | higher | higher | **yes** | yes |
| 1 | match | 2 | higher | higher | **yes** | yes |
| 1 | year | 3 | lower | lower | yes | **excluded** |
| 1 | agecl | 3 | higher | lower | no | **excluded** |
| 2 | ae | 2 | higher | higher | **yes** | yes |
| 2 | proph | 2 | higher | higher | **yes** | yes |
| 2 | match | 2 | higher | higher | **yes** | yes |
| 2 | year | 3 | lower | lower | yes | excluded |
| 2 | agecl | 3 | higher | higher | yes | excluded |

**Investigated, not papered over:** `agecl` disagrees at cause 1 (`year` happens to agree here, but is still excluded — see below). Both `year` and `agecl` are 3-level *nominal* categories (`year`: `1985-1989`/`1990-1994`/`1995-1998`; `agecl`: `<=20`/`20-40`/`>40`), ordinally coded for the fit. `CoxTimeVaryingFitter` fits **one linear coefficient** across all 3 levels; the forest's `hazard_effect` makes no such linearity assumption and can legitimately show a different sign at the two extreme codes even on the identical fitted data, because a 3-level nominal variable's true relationship with the hazard has no reason to be monotonic in an arbitrary code order. This is not a forest-vs-Cox modeling conflict and not a library bug — it is a limitation of comparing a forced-linear coefficient against a two-point contrast for a variable that shouldn't have been given a single linear "direction" in the first place; `year`'s *agree=yes* here is not evidence against that (a >2-level nominal variable's agreement/disagreement under an arbitrary code order isn't a meaningful signal either way, which is exactly why both rows are excluded regardless of which way they happen to land). Every genuinely **binary** covariate checked — `ae`, `proph`, `match` (both causes) and all four Rossi covariates — agrees between forest and Cox. The internal-consistency gate is scoped to binary covariates for this reason; `year`/`agecl` are still reported, with this explanation, not hidden.

**Note on these numbers (added in a later release pass):** `examples/rc_validation.py`'s original ordinal encoding (bare `pl.col(...).cast(pl.Categorical).to_physical()`) turned out to assign category codes by first appearance in a way that is **not stable across separate process runs** on identical data — rerunning this exact script against the same fixture gave a third and fourth distinct set of code numbers each time (an API-friction finding recorded below). The script now uses an explicit `pl.Enum` level order (verified deterministic across repeated runs), and the two tables above were regenerated with it. Only the numbers changed; every qualitative conclusion in this subsection (which covariates are gated, that all gated/binary covariates agree, and that `year`/`agecl` are correctly excluded rather than papered over) was true before this fix and remains true after it.

**Cause-specific structure check:** `ae`'s `hazard_effect` contrast (`ae=1` vs `ae=0`) is clearly distinguishable between the two causes — cause 1's per-window contrast is small and slightly negative (peaking at -0.0005 around window 3), cause 2's is consistently positive and larger (peaking at +0.0012 around window 2). This is evidence the forest is picking up genuinely cause-specific structure from the same shared trees, not evidence about `ae`'s real clinical meaning (the plan's Decision 1 already removed any GvHD-specific claim: `ae` is a generic, simplified adverse-event flag per `mstate`'s own documentation).

## Decision 4 verdict

**All internal-consistency checks that apply (binary covariates, both datasets) pass.** No stop-and-report condition was triggered on a covariate where the check is well-defined. The one disagreement found was investigated, explained as a scoping/encoding issue rather than a modeling conflict, and the script was corrected to scope the gate accordingly (`examples/rc_validation.py`, `n_levels <= 2`).

`ae`'s two-cause `hazard_effect` curves are distinguishable, as the plan's weaker (no-clinical-claim) check asked for.

## API friction / release-pass follow-ups

- **`polars.DataFrame.to_pandas()` needs `pyarrow`, which is not installed in this venv and isn't an rftvc dependency.** Had to hand-roll a pyarrow-free conversion (`pd.DataFrame({c: d[c].to_numpy() for c in d.columns})`). Not an rftvc bug, but worth a one-line note in a future real-data-facing docs page (or the `docs` dependency group) if polars fixtures become more common. **Resolved:** the pyarrow-free conversion is now noted in `docs/source/compatibility.rst`, next to the existing DataFrame/narwhals bullet.
- **`hazard_effect`'s result `Bunch` field `"values"` shadows `dict.values`** (already known from S20; `result["values"]`, not `result.values`). Confirmed as a real trap again here — worth surfacing more prominently (e.g. a `UserWarning` on `.values` attribute access, if sklearn's `Bunch` ever supports that, or at minimum keeping the docstring note highly visible) since a user copying a pattern from elsewhere will hit it immediately on real data. **Resolved (superseded 2026-09-27, PR #37):** the `_HazardEffectResult` subclass and its warn-and-return-grid `.values` property (described above) turned out to have its own real bug — the property fully replaced the `values` slot, so `result.values()` (the ordinary way to call the inherited `dict.values` method) raised `TypeError: 'numpy.ndarray' object is not callable`, breaking any generic Mapping-consuming code. Reverted: `hazard_effect` now returns a plain `Bunch`, matching how `sklearn.inspection.partial_dependence` handles the identical `"values"`-key collision (no workaround; `result.values` is the bound method, `result["values"]` is the documented way to get the array). `result["values"]` itself is unchanged.
- **No documented guidance for comparing a `hazard_effect` contrast against a linear-model coefficient for a >2-level nominal covariate.** This validation exercise tripped over exactly this (the `year`/`agecl` "disagreement"), and it cost real investigation time to distinguish "real modeling conflict" from "meaningless comparison for this covariate type". A short note in `importance.rst` (or a new `hazard_effect` docstring caveat) — "a `hazard_effect` two-point contrast is only comparable to a linear model's single coefficient when the covariate is binary or genuinely continuous; for a >2-level nominal covariate, encode with care and don't expect a single 'sign' to be meaningful" — would have saved this step and would help future users avoid mis-reading a real disagreement into an encoding artifact (or the reverse). **Resolved:** this caveat is now in both `hazard_effect`'s docstring and `importance.rst`, in essentially the words drafted here.
- **`permutation_importance(..., oob=True)` always gives `importances_se = NaN`** (documented, expected — OOB shares trees across ids). This surfaced immediately and as expected; a first-time reader of the relevance tables who skips the docstring could plausibly mistake all-`NaN` standard errors for a bug. No action needed beyond what's already documented; noting it here because it is the first thing a new real-data user will see in the output.
- **`UserWarning: the model does not beat the training null on these data; share_of_gain is NaN`** fired routinely across several of the cause-specific EBMT4 fits (weaker per-cause signal on real, moderate-sized subsets). Expected and benign, but on real data (unlike S16-S20's synthetic sims, tuned to have a clear signal) this warning will likely be the norm rather than the exception for at least one cause/covariate combination in a typical real analysis. Worth a one-line mention in the release-pass docs pass that this is common and not itself diagnostic of a problem. **Resolved:** noted in `importance.rst`'s new "Two warnings you'll likely see on real data" section.
- **`UserWarning: ... % of events fall in windows where the predicted rate is 0; the windows are too fine for this model`** appeared during EBMT4's cause-specific `drop_column_importance` (the rarer-event cause). The default `windows=8` may be too fine for a cause with fewer events in a real, moderate-sized competing-risks dataset; users fitting `CompetingRisksForestTV` on real data with an imbalanced cause split should expect to need a smaller `windows` for the rarer cause. Not a bug — the warning did its job — but worth a concrete real-data example in the docs since S16-S20's synthetic generators were tuned to avoid triggering it. **Resolved:** noted alongside the above in the same new `importance.rst` section.
- **Runtime/memory:** both fits completed in well under a second (500 trees, 1405 and 3413 rows respectively); `drop_column_importance` at `cv=5` (5-10 refits per cause) also completed quickly. No runtime or memory concern at these realistic-but-modest real-world sizes.
- **Categorical encoding is entirely the user's responsibility** (as designed — S20's release-pass note already flags API consistency around this as a release-pass topic, not fixed here). This validation exercise needed five different encoding decisions (Rossi's `race`/`mar`/`fin`/`wexp`/`paro`, EBMT4's `year`/`agecl`/`proph`/`match`) with no library guidance beyond "encode it yourself"; a real-data case study (see below) would be a natural place to demonstrate a recommended pattern.
- **`examples/rc_validation.py`'s `pl.col(...).cast(pl.Categorical).to_physical()` encoding of EBMT4's `year`/`agecl`/`proph`/`match` is not stable across separate process runs on identical data** (discovered incidentally while re-validating the fixtures after the Decision 1 download-on-demand switch below; confirmed by rerunning the *original* committed CSV data twice more, getting a third and fourth distinct set of category-code numbers each time). This is a script issue, not an rftvc library issue, and not caused by the data-source change -- the underlying per-patient `year`/`agecl`/`proph`/`match` values are confirmed bit-identical across all runs and sources (checked directly), and the qualitative conclusions (`agree`, `gated`, and every `forest_direction`/`cox_direction`) are stable across every run observed. Only the raw numeric codes (and the exact `cox_coef` floats, which shift with them) churn. Likely cause: bare `.cast(pl.Categorical)` without `pl.enable_string_cache()` or an explicit level order, so code assignment depends on incidental global categorical-cache state built up by whatever else ran earlier in the process (forest fits, CV folds). Worth a fix in a later release pass -- e.g. `.cast(pl.Enum(sorted(...)))` with an explicit, stable level order -- if this script's exact numbers are ever depended on again. **Resolved:** `examples/ebmt4_case_study.py` (the promoted case study, see Recommendation below) uses exactly this `pl.Enum` fix; `examples/rc_validation.py` itself was fixed the same way in a further release-pass commit, and the two EBMT4 tables above were regenerated with it (the "Note on these numbers" at the end of the EBMT4 Direction subsection).

## Recommendation (not a decision)

Both fixtures ran cleanly end-to-end and recovered a real, previously-known qualitative result (Rossi) and produced an interpretable, internally-consistent competing-risks fit (EBMT4) on genuine data neither S16-S20 nor the existing case studies had exercised. Promoting one or both to a real `docs/source/case_studies/*.rst` page (per the plan's Decision 6, left as a follow-up) seems worthwhile for the release pass, particularly Rossi's clean agreement with the textbook result — but that is a documentation decision for the user, not made here.

**Resolved:** the user asked for both to be promoted. `docs/source/case_studies/rossi.rst` and `ebmt4.rst` (new `examples/rossi_case_study.py`/`ebmt4_case_study.py`, following `btscs.rst`'s structure) report the same headline numbers as above. One improvement made along the way: the new EBMT4 case-study script encodes `year`/`agecl`/`proph`/`match` with an explicit `pl.Enum` level order (the R factor levels themselves) rather than bare `.cast(pl.Categorical)`, fixing the cross-run nondeterminism noted in the API-friction list above (verified deterministic across repeated runs) — `examples/rc_validation.py` itself is unchanged (out of scope; it is the plan's own T4 deliverable, not the case study).
