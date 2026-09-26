# TVC statistical foundation and importance: research questions (seed)

Status: **seed; research done in `tvc-research.md`** (2026-09-26). Next: a research doc (`tvc-research.md`), then a Codex review, a design (`tvc-design.md`) and a plan, as in the competing-risks track (`cr-questions.md` → `cr-research.md` → `cr-design.md` → `cr-plan.md`). The release-readiness pass comes after this track (user decision), so the release covers the final API.

## User goal
Outside users need a statistical foundation for rftvc's trees:
- how time-varying covariates influence the predictions;
- which measures help them understand the covariate ↔ outcome relationship;
- whether standard RF variable importance is enough, or whether a TVC-specific measure is needed.

The library must stay general purpose (see memory `rftvc-general-purpose`).

## Questions for the research stage
1. **Foundation:** what exactly does a counting-process forest estimate?
   - Each row routes by its covariates in force on `(start, stop]`, and leaves estimate the hazard given the current covariate value.
   - What does this assume? A Markov-type dependence on the current value only; history enters only through features (landmarking).
   - How does it relate to time-dependent Cox models (a hazard ratio for the current value) and to RSF theory (consistency results, the LTRC log-rank)?
   - What is valid for external vs internal covariates? D4 limits prediction along paths to external covariates.
2. **Permutation importance on counting-process rows:** why a naive row-level shuffle is wrong.
   - It breaks each subject's trajectory, and it scrambles the covariate–time association, because TVCs often trend with time.
   - Candidates: subject-level trajectory permutation; conditional permutation within time strata (Strobl-style conditional importance); a knockoff or refit (LOCO) alternative.
   - Which question does each answer?
3. **Scoring:** importance as a drop in a proper score (integrated IPCW Brier, S13 metrics), not only in C. OOB vs held-out; id vs block units (S10).
4. **Level vs history vs timing:**
   - separate the importance of the current value, of history summaries (slope, max, time above a threshold; the landmark features), and of *when* a covariate matters;
   - time-resolved importance: leaves store whole hazard curves, so contributions can be attributed per time window.
5. **Effect curves:**
   - Path-intervention effects: "shift z by +δ from time u onwards → Δ S or Δ F_k at horizon w". This is a TVC-native analogue of partial dependence / ALE, valid for external covariates and computable with `predict_*(intervals=…)`.
   - Also individual-conditional-expectation-style curves along paths.
6. **Competing risks:** cause-specific importance and effects on `F_k` vs on the cause-specific hazards.
7. **Prior art to survey:** RSF VIMP and minimal depth (Ishwaran); conditional permutation importance (Strobl 2008; Hooker, Mentch & Zhou 2021 on permute-and-predict); LOCO; SHAP for survival (SurvSHAP(t)); time-dependent variable importance in dynamic prediction (landmarking literature); partial dependence / ALE for survival; `randomForestSRC` / `LTRCforests` / `pec` / `survex` implementations.
8. **Validation:** simulations with a known truth: which covariates matter, when, and through level or history. The S3 / S14 generators can be extended.

## Initial hypotheses (to be tested, not decisions)
- A single importance number is not enough. Users likely need (a) level importance, (b) history importance via landmark features, (c) effect curves by path intervention, and (d) optionally a time-resolved view.
- Naive permutation importance is biased for TVCs that correlate with time; a trajectory or time-conditional permutation is needed.
