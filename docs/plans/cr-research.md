# Research Findings: competing risks for rftvc

Status: research stage (2026-09-26); Codex research review corrections applied (see "Review log" at the end). No design decisions yet. Answers `cr-questions.md`.
Tags: [S] = backed by a source fetched or searched this session; [S: title] = the source was found but only its title/abstract was read; [K] = background knowledge, verify before relying on it; [R] = from this repository's code or docs.

---

## 1. Estimands with time-varying covariates and left truncation
- **Cause-specific hazards** `λ_k(t | X(t))`, k = 1..K. These are well defined for both external and internal TVCs, and they are the natural target of a counting-process fit: an event of type k is an event for `λ_k`, and every other event and censoring ends follow-up [S: Cortese & Andersen 2010].
- **Cumulative incidence** `F_k(t | ·) = P(T ≤ t, cause = k | ·)`.
  - With only external covariates (or a specified future path), `F_k` along the path is the Aalen–Johansen product integral of all K cause-specific hazards. In the discrete (Nelson–Aalen increment) form a forest produces: `S(u) = ∏_{v ≤ u} (1 − Σ_j ΔΛ_j(v))` and `F_k(t) = Σ_{u ≤ t} S(u−) ΔΛ_k(u)`. The form `S = exp(−Σ_j Λ_j)` holds only for continuous hazards [K].
  - **Internal TVCs:** cause-specific hazards conditional on the observed history remain estimable. A predictive CIF generally needs a model for the future internal-covariate process (e.g. a joint model); the fitted conditional hazards alone do not identify it [S: Cortese & Andersen 2010, as summarised; K for the general statement]. This is the same boundary as v1's D4 for survival.
  - The dynamic answer is **landmarking**: condition on `T > s` and the history `H(s)`, and predict `F_k(s + w | s, H(s))` [S: Nicolaie, van Houwelingen, de Witte & Putter 2013; dynamic pseudo-observations, Biometrics 2013].
- **Subdistribution hazard (Fine–Gray).** Its risk set keeps subjects after a competing event, so the covariate value after that event is undefined for internal TVCs. Time-dependent covariates in the Fine–Gray model are "problematic" and need care [S: title — Latouche, Porcher & Chevret 2005/2008 (Biostatistics 9:765); Austin et al. review of TVCs in Fine–Gray].
- **Consequence for rftvc:**
  - The counting-process estimator (`SurvivalForestTV`) should target **cause-specific hazards**. CIF is then a derived output, valid along external/specified paths, like `predict_cumulative_hazard(intervals=…)` today [R].
  - The landmark workflow (`LandmarkSurvivalForest`) is where **direct CIF targets** are coherent: covariates are fixed at `s`, and the outcome is on the reset clock [R].

## 2. Prior art
| Method | Split rule(s) | Leaf estimate | Aggregation | TVC / left truncation | Language |
|---|---|---|---|---|---|
| RSF for competing risks (Ishwaran, Gerds, Kogalur et al. 2014), `randomForestSRC` | generalised log-rank (cause-specific hazards); Gray-type `logrankCR` (CIF, default); composite weighting across causes via `cause` | CIF, cause-specific CHF, CPC | tree estimates averaged | documents competing-risks forests **and** a separate start–stop time-dependent-covariate mode (subject ids); whether multi-cause status works in the start–stop mode is **unverified** [S: vignette; `utilities_survival.R` per review] | R/C [S] |
| `comprisk` (Yang, Zhao & Zhao, arXiv 2607.09431, 2026) | competing-risks log-rank, composite and cause-specific; histogram splits on uint8 bins | Aalen–Johansen CIF, Nelson–Aalen | "standard" (not detailed) | no TVC or delayed-entry support identified in the paper reviewed; the released input contract is unverified | Python + numba; sklearn-style; ~10–22× faster than randomForestSRC, 1M rows in ~1 min [S] |
| DynForest (Devaux et al. 2023/2025) | Fine–Gray test for competing risks | CIF | averaged | longitudinal markers via mixed models fitted within nodes (not `(start, stop]` rows) | R [S] |
| Trees/ensembles for CIF (pseudo-value based) | regression on pseudo-observations | CIF | averaged | right-censored [S: title — IJB 2022, doi 10.1515/ijb-2021-0014] | R |
| Static vs dynamic RF for EHR with competing risks (arXiv 2404.16127) | randomForestSRC | CIF | — | landmark-style dynamic models [S: title] | R |
| `randomForestRHF` 2.1.0 (2026-09-17) | hazard likelihood, counting-process TVCs | hazard | — | yes, but **no competing risks mentioned** [S] | R/C |
| `LTRCforests` (Yao et al. 2022) | LTRC log-rank / Poisson | survival | — | yes (start–stop pseudo-subjects); no competing-risks feature in the documented API [S: per review] | R |
| Discrete-time / person-period competing-risks trees and forests (e.g. multinomial hazard per time bin; Moradian et al.-style dynamic forests for the binary case) | multinomial / classification splits on person-period rows | per-bin cause-specific hazards → CIF by recursion | averaged | delayed entry = absence of rows before entry; TVCs per period | R (ad hoc) [K; binary-case prior art in `research.md`] |
| scikit-survival 0.28 | — | nonparametric CIF only (`cumulative_incidence_competing_risks`) | — | — | Python [S] |

**Gap (provisional):** no **Python** forest with competing risks and counting-process TVCs plus left truncation was identified. `comprisk` is the closest Python tool (fast, sklearn-style, with competing-risks metrics), and it is the obvious baseline and parity reference for the right-censored special case. In R, whether `randomForestSRC` combines its competing-risks and start–stop modes must be **checked experimentally** before the gap is claimed there. Discrete-time person-period forests are a practical baseline with a different (discretised) estimand.

## 3. Split criteria
- **Cause-specific LTRC log-rank for cause k:** the current exact LTRC log-rank with "event" meaning an event of type k [R]. The risk sets are unchanged, because risk is `start < t <= stop` whatever the event type. The per-node cost is the same as today once per-cause event counts exist.
- **Composite (all causes):** a weighted combination of the K cause-specific statistics, as in the `randomForestSRC` "generalised log-rank" with the `cause` weights [S]. The exact form (sum of squared standardised statistics vs sum of numerators over summed variances) must be taken from Ishwaran et al. 2014 [K: verify].
- **Gray's test (CIF equality):** a weighted log-rank on the subdistribution risk set. With right censoring it needs censoring weights; under **left truncation and right censoring** it needs truncation–censoring weights [S: title — "Cause-specific cumulative incidence estimation and the Fine and Gray model under both left truncation and right censoring", PubMed 20377575]. With counting-process TVC rows the subdistribution risk set is ill-defined for internal covariates (section 1).
  - So Gray-type splitting fits the **landmark stacks** (time-fixed features at `s`, clock reset), not the counting-process estimator.
  - An LTRC-valid Gray-type statistic needs a specified truncation/censoring-weight estimator. A design must say whether it is estimated once per parent node or globally per training fold, and must prevent test-fold leakage [K].
- **Criterion trait [R]:** `SplitCriterion::score` receives node summaries (`Profile`: at-risk counts, events, times, exposure, `n_units`). Competing risks needs per-cause event counts in `Profile` and in the split-search difference arrays: a cause dimension, i.e. K+1 counts per event time instead of 2.

## 4. Leaf estimators and aggregation
- **Aalen–Johansen with delayed entry** is the canonical CIF estimator under random left truncation [S: Stegherr et al. 2020]. With delayed entry, early small risk sets cause large jumps that propagate over the whole CIF [S: PubMed 37850535]. This is the survival problem v1 already handles with `min_ids_leaf` / `min_events_leaf` [R], but per cause it is worse, because events are split K ways.
- **Leaf content:** per leaf, the event times, the per-cause Nelson–Aalen increments `d_k / Y`, and optionally the all-cause survival. The S9 flat layout can take a cause dimension (entries × K) under the same offsets [R: `s9-plan.md` decision 6].
- **Path prediction with TVCs [K, derived]:** along a path, each row routes to its own leaf, and the cause-specific hazards accrue piecewise, as in `predict_paths` today [R]. The CIF is the product integral over the concatenated path, `F_k(t | path) = Σ_{u ≤ t} S(u− | path) ΔΛ_k(u | x(u))`, conditional on the origin as for the conditional cumulative hazard.
- **Ensemble rule (analogue of D11):**
  - (a) average cause-specific cumulative hazards, then Aalen–Johansen;
  - (b) Aalen–Johansen per tree, then average the CIFs (`randomForestSRC` averages tree CIFs [S]).
  - Both give CIFs in [0, 1] whose sum is ≤ 1. They differ like `aggregate="hazard"` vs `"survival"` [R], and the choice is an empirical question (S8-style) [K].

## 5. Evaluation
- **CIF Brier / IBS with IPCW** at `(s, w)` [K: Gerds & Schumacher; `riskRegression`]. A competing event is an observed outcome, not censoring. The IPCW rules of `design.md` / `plan.md` S5 carry over:
  - `G_s` is fitted on the landmark risk set, on the reset clock;
  - a KM `G_s` assumes marginal independent censoring there, and history-dependent censoring needs a conditional model (opt-in);
  - the positivity / `g_min` truncation policy stays [R].
- **Time-dependent AUC for competing risks** with IPCW, with two definitions of controls (Blanche, Dartigues & Jacqmin-Gadda 2013) [S].
- **Concordance for competing risks** (Wolbers, Blanche, Koller, Witteman & Gerds 2014), related to the AUC definitions [S].
- **Dynamic / landmark versions:** evaluated per landmark, as for survival [S: Nicolaie et al. 2013].
- `comprisk` ships IPCW CIF Brier/IBS, cause-specific C with confidence intervals, and IPCW AUC, which is a possible cross-check for the metrics [S].
- **rftvc impact [R]:** `rftvc.metrics` (`brier_landmark`, `integrated_brier`, `cindex_dynamic`, `concordance_index_cp`) needs cause-aware versions. `KaplanMeierCensoring` (G_s on the landmark risk set) carries over unchanged.

## 6. Engine and API impact: changes required from the current single-event engine
Everything below is **new**. Today `SurvData.event` is `Vec<bool>`, validation accepts bool or `{0, 1}`, `Profile.events` is one vector, flat leaves hold one `cumhaz` per event time, `predict_paths` returns one cumulative hazard, and `coarsen` moves a Boolean event flag [R].
- **Target:** `y` with an integer `event` (0 = censored, 1..K) instead of bool. `make_survival_y` / `check_survival_y` must accept both; a bool `event` stays the K = 1 case, bit-identical [R].
- **Engine [R]:**
  - `SurvData` event → cause codes;
  - `node_profile` / `Profile` with per-cause counts;
  - the split-search difference array with a cause dimension;
  - leaf storage with per-cause cumulative hazards (S9 flat layout × K);
  - `predict_cumhaz` / `predict_paths` per cause;
  - coarsening (`coarsen`) moves an event together with its cause.
- **Leaf minimums:** an all-cause `min_events_leaf` can leave a leaf with no events of a rare cause; a per-cause minimum can make splitting infeasible.
  - A candidate default: split feasibility uses `min_events_leaf` over all causes, and leaf diagnostics always report per-cause counts.
  - An optional `min_events_leaf_cause` would apply only when fitting for a selected cause.
  - This needs rare-cause simulations [K: design question].
- **OOB score:** needs a cause-specific concordance (Wolbers) and a chosen cause [S/R].
- **Block resampling (S10) and landmarking** carry over unchanged. Units and risk sets do not depend on event type [R].
- **Prediction API:** `predict_cumulative_incidence(X, times, cause=…, intervals=…)` and `predict_cause_specific_hazard(...)`, plus the existing survival (all-cause) outputs [K: design question].
- **Memory:** leaf storage grows from 12 B to 4 + 8K B per leaf event time [R: S9 numbers].

## 7. Scope boundary: multi-state and recurrent events
- **Illness–death and general multi-state:** competing risks out of each state, i.e. transition-specific hazards [K]. In counting-process form, each transition's rows carry a from-state, and each state's exits form a competing-risks problem. Clock choice (forward vs reset) and non-Markov issues are the hard part [S: title — arXiv 1304.2293 on non-Markov illness–death].
- If competing risks is built as "K cause-specific hazards over counting-process rows", multi-state is a later layer: grouping rows by from-state, plus transition-probability prediction. None of the competing-risks choices above block it [K].
- **Recurrent events** need an event count per id (the current contract allows one event per id, on its last row [R]), and a target such as a mean cumulative function or gap times. There is prior forest work [S: title — BMC Med Res Methodol 2025, RSF for recurrent events]. Separate slice.

## 8. Open questions for the design stage
1. **Criterion** for the counting-process estimator: a composite cause-specific log-rank (default?) vs a single-cause log-rank (`cause=k`). Is Gray-type splitting offered only on landmark stacks?
2. **Ensemble rule** (hazard-then-Aalen–Johansen vs per-tree CIF average): decide empirically, as in S8?
3. `min_events_leaf` semantics under K causes.
4. **API shape:** a new estimator (`CompetingRisksForestTV`) vs `SurvivalForestTV` handling integer events. sklearn `score` / `predict` conventions for K outputs.
5. **Validation references:** `randomForestSRC` (right-censored, via rpy2) and `comprisk` for the special case; `survival::survfit` multi-state (Aalen–Johansen with `(start, stop]`) for single-node leaf oracles [K: verify that survfit accepts counting-process competing-risks data].
6. **Metrics:** implement Wolbers C, Blanche AUC and the CIF Brier score in `rftvc.metrics`, or depend on an external package?

---

## Sources
- randomForestSRC competing risks vignette: https://www.randomforestsrc.org/articles/competing.html ; start–stop utilities: https://rdrr.io/cran/randomForestSRC/src/R/utilities_survival.R
- Ishwaran et al. 2014, RSF for competing risks: https://ncbi.nlm.nih.gov/pmc/articles/PMC4173102
- comprisk: https://arxiv.org/abs/2607.09431 ; https://arxiv.org/html/2607.09431 ; https://pypi.org/project/comprisk/
- scikit-survival competing risks: https://scikit-survival.readthedocs.io/en/stable/user_guide/competing-risks.html
- DynForest: https://journal.r-project.org/articles/RJ-2025-002/ ; https://arxiv.org/pdf/2302.02670
- CIF trees/ensembles: https://doi.org/10.1515/ijb-2021-0014
- Static vs dynamic RF with competing risks (EHR): https://arxiv.org/pdf/2404.16127
- randomForestRHF: https://cran.r-project.org/web/packages/randomForestRHF/index.html
- Cortese & Andersen 2010: https://onlinelibrary.wiley.com/doi/10.1002/bimj.200900076 ; https://pubmed.ncbi.nlm.nih.gov/20029852/
- TVCs in the subdistribution model: https://academic.oup.com/biostatistics/article/9/4/765/259139 ; review: https://www.researchgate.net/publication/336884956
- Nicolaie et al. 2013, landmarking in competing risks: https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.5665 ; dynamic pseudo-observations: https://academic.oup.com/biometrics/article/69/4/1043/7492355
- Aalen–Johansen with left truncation: https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8421 ; small risk sets: https://pubmed.ncbi.nlm.nih.gov/37850535/
- Fine–Gray / CIF under LTRC: https://pubmed.ncbi.nlm.nih.gov/20377575/
- Blanche et al. 2013 AUC with competing risks: https://onlinelibrary.wiley.com/doi/10.1002/sim.5958
- Wolbers et al. 2014 concordance with competing risks: https://pubmed.ncbi.nlm.nih.gov/24493091/
- Non-Markov illness–death: https://arxiv.org/pdf/1304.2293
- RSF for recurrent events: https://link.springer.com/article/10.1186/s12874-025-02678-z

## Review log (Codex, 2026-09-26)
1. The internal-TVC claim was too absolute, and the path formula used `exp(−ΣΛ)` (high) → a predictive CIF needs a model for the future internal-covariate process; the discrete Aalen–Johansen product is now given.
2. The `randomForestSRC` row ("none native") was overstated: it has a separate start–stop mode (high) → combined support is unverified, and the gap is provisional and Python-specific until checked.
3. "The censoring model is unchanged" was incomplete (high) → the landmark IPCW rules, assumptions and truncation policy from S5 are restated.
4. The Gray weights claim was over-prescriptive and over-tagged (medium) → `[K]`, with the design choices stated.
5. Discrete-time person-period competing-risks forests were missing (medium) → prior-art row and baseline.
6. Leaf-minimum rule left open (medium) → a candidate default plus rare-cause simulations.
7. Engine list could be read as existing behaviour (medium) → prefixed as required changes, with the current state stated.
8. The `comprisk` absence claim was untagged (low) → reworded as "not identified in the material reviewed".
Not verifiable in the review: the exact LTRC Gray weighting, and a `survival::survfit` start–stop multi-state oracle (both stay `[K]`).
