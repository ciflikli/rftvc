# Research Questions: competing risks in rftvc

Goal (context only, not a solution): extend the general-purpose counting-process
survival forest (`SurvivalForestTV`, `LandmarkSurvivalForest`) to competing risks:
K ≥ 2 mutually exclusive event types, censoring, left truncation and time-varying
covariates. Multi-state models and recurrent events are adjacent (noted, not the target).

## Questions
1. **Estimands.** What targets are well defined with time-varying (external vs internal) covariates and left truncation: cause-specific hazards, cumulative incidence functions (CIF) conditional on history, subdistribution hazards? Which one can a counting-process forest estimate directly, and which only via landmarking?
2. **Prior art.** Which tree/forest methods handle competing risks (split rules, leaf estimators, aggregation)? Which of them accept counting-process `(start, stop]` rows or left truncation? Is there a Python implementation, and what does it lack?
3. **Split criteria.** How do cause-specific log-rank, composite log-rank and Gray's test extend to left-truncated counting-process rows? What do they optimise, and what do they cost relative to the current LTRC log-rank?
4. **Leaf estimators and aggregation.** Aalen–Johansen with delayed entry: validity, small-risk-set instability. Along a covariate path: how is a CIF formed from per-row leaves? Ensemble rule: average CIFs, or average cause-specific hazards then combine (the analogue of D11)?
5. **Evaluation.** CIF Brier/IBS with IPCW, time-dependent AUC and concordance for competing risks, and landmark (dynamic) versions. What changes in the censoring model?
6. **Engine and API impact.** Event coding, per-cause node profiles, criterion trait, leaf storage (S9 layout), `min_events_leaf`, OOB score, block resampling, prediction API, sklearn target format.
7. **Scope boundary.** What would multi-state (illness–death) and recurrent events need beyond competing risks, so that competing-risks choices do not block them?
