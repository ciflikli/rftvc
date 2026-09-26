# Research Findings: statistical foundation and importance for TVC forests

Status: research stage (2026-09-26); Codex research review applied (see "Review log"). No design decisions yet. Answers `tvc-questions.md`. Next: Codex research review, then `tvc-design.md`.
Tags: [S] = backed by a source fetched or searched this session; [S: title] = found, only title/abstract/snippet read; [K] = background knowledge or own derivation, verify before relying on it; [R] = from this repository's code or docs.

---

## 1. Foundation: what a counting-process forest estimates

### 1.1 The estimand
- **Data model.** Each row carries covariates `x_r` in force on `(start_r, stop_r]`. The trees split on `x` only; time is never a split variable [R: `fit_forest` gets `X` only]. The split statistic is the LTRC log-rank, which compares the two children **at each event time among the rows at risk then** [R: `criterion.rs`]. Splitting is therefore time-matched: a covariate that trends with time but does not affect the hazard gives no split signal in expectation [K, derived].
- **Leaf estimate.** Nelson–Aalen over the leaf's rows: `Λ_A(t) = Σ_{t_k ≤ t} d_A(t_k) / Y_A(t_k)`, with `Y_A(t_k)` the rows in cell `A` with `start < t_k ≤ stop` [R: `tree.rs`]. The leaf stores jumps only at its own in-bag event times (S9 layout), so **its cumulative hazard is flat between those times**: the predicted increment over any window without a leaf event is 0, including where the leaf has no rows at risk. `cumhaz_at` carries the last value forward [R: `tree.rs`].
- **Target.** Assume the intensity model `P(dN(t) = 1 | F_{t−}) = Y(t) λ(t, X(t)) dt` with a predictable covariate state `X(t)`. Then a leaf estimates the at-risk-averaged hazard `λ_A(t) = E[λ(t, X(t)) | X(t) ∈ A, at risk at t]`. **Heuristically**, cells that shrink tend to the **hazard map `(t, x) ↦ λ(t, x)`**. That needs positivity (rows at risk near every relevant `(t, x)`), local cells with enough at-risk observations, independent subjects, and the censoring/entry conditions of §1.2. The implementation guarantees none of these, and no consistency result exists (§1.4) [K, derived].
  - `randomForestRHF` states the same target explicitly: "Markovian in the predictable state but not necessarily in the full history", with a no-lookahead rule [S: Ishwaran, Hsich, Kogalur & Lee 2026, arXiv 2608.21597].
  - rftvc reaches the target with a partition in `x` and a nonparametric time curve per leaf. RHF uses a time-constant working model during splitting, then estimates on a time grid [S].
- **Composition drift.** `λ_A(t)` depends on which rows of `A` are at risk at `t`. That set changes through selection (frailty) and through covariate movement. In coarse leaves (large `min_ids_leaf`), part of the leaf's hazard *shape* therefore reflects the population mix rather than time itself. This is an interpretation caveat, not a bias in the target [K].

### 1.2 Assumptions, stated for users
1. **Current-state dependence.** The target is always well defined as "the hazard given the row's covariates". It equals the full-history hazard only if the row captures all relevant history: current values plus any lags or summaries the user adds as columns [K]. §4.3 turns this into a testable diagnostic.
2. **Predictability.** The row's `x` must be known at `start` (no lookahead). The landmark module enforces this for history features: a row is known at `s` only if `start ≤ s` [R: `_history_features`]. For raw counting-process rows it is the user's responsibility; `measured_at` exists as a check [R].
3. **Independent censoring and entry given the covariate history** (the standard Andersen–Gill conditions) [K].
4. **Within-subject rows.** A subject's rows are dependent, but the counting-process likelihood factorises over time (martingale increments), so the pooled log-rank and Nelson–Aalen remain valid across one subject's rows. Uncertainty and OOB need id-level (or block-level) resampling, which rftvc already does [K; R: `resample_unit`].

### 1.3 Prediction functionals and their validity
| Call | Quantity | Valid as a survival probability when |
|---|---|---|
| `predict_*` without `intervals` | `Λ(t \| x held fixed from 0)` | `x` is time-fixed, or as a named "fixed profile" scenario [R] |
| `predict_*(intervals=…)` | `Λ(t \| path) − Λ(origin \| path)` = `∫ λ(u, x(u)) du` | the covariates are **external** and the path is observed or specified (Kalbfleisch–Prentice) [K; R: D4] |
| `predict()` / `oob_prediction_` / `oob_score_` | mortality `Σ_k Λ(t_k \| x)` over the fixed profile | a ranking of covariate *states*, not of subjects [R] |
| landmark `predict_risk(df, s, w)` | `P(T ≤ s + w \| T > s, H(s))` | for internal or external covariates, under censoring independent of the event given `H(s)` and `s` [R: `landmark.py`], adequate support, and a fitted model that transports to the prediction population |

- For **internal** covariates, the hazard map is still estimable and interpretable, but a path-based survival curve is not a probability for any real subject. This is D4 again [R].
- **Finding for the release pass:** `oob_score_` ranks each TVC row by its fixed-profile mortality. Under non-proportional leaf hazards, that ranking can disagree with the ranking by hazard at the row's own time. A time-local C (hazard in a window around the event time, compared within the risk set) matches the estimand better (§3) [K, derived].

### 1.4 Relation to other models and theory
- **Time-dependent Cox** `λ0(t) exp(β'x(t))` is the special case with multiplicative, time-constant effects. The forest target allows `x × t` interactions (non-PH), because leaf hazards are arbitrary time functions [K].
- **Tree ancestry.** LTRC survival trees for TVC data via pseudo-subjects (Fu & Simonoff 2017, Biostatistics 18:352) [S]; LTRC forests (Yao, Frydman, Larocque & Simonoff 2022, SMMR) [S]; Bacchetti & Segal 1995 [K].
- **Consistency.**
  - RSF consistency: Ishwaran & Kogalur 2010, right-censored data with time-fixed covariates [K].
  - Splitting bias under censoring and a concentration bound for non-i.i.d. within-node samples: Cui, Zhu, Zhou & Kosorok 2022 (Stat. Sinica 32:1245) [S].
  - **No consistency result was found for counting-process TVC forests.** The RHF paper is empirical on this point [S, per summary]. The foundation doc should state the target and assumptions (§1.1–1.2), cite the adjacent theory, and not claim consistency [K].
- **Competing risks.** Everything above holds per cause for the cause-specific hazards `λ_k(t, x)` [R: `cr-research.md` §1].

## 2. Permutation importance on counting-process rows

### 2.1 Why a naive row-level shuffle misleads
Standard RSF VIMP is the increase in OOB error after permuting (or noising up) `x_j` [S: Ishwaran; DynForest]. On counting-process rows, three distinct problems arise:
1. **Off-support (time, value) pairs: the main one.** A shuffle gives a row at time `t` a value typical of time `t'`. If `z` trends with time, the row lands in a leaf with no events (often no rows at risk) near `t`. That leaf's cumulative hazard is flat there (§1.1 [R]), so the row gets no predicted hazard over that stretch: a prediction from a region with no local event support. The measured "importance" can then be extrapolation damage, not reliance on `z`. This is the Hooker, Mentch & Zhou (2021, Stat. Comput. 31:82) permute-and-predict failure, possibly made worse by event-free time regions in leaves [S; mechanism K, derived]. **Whether it materially inflates VIMP is unproven**: the §8.1 simulation must show it.
2. **Dependence on other covariates** (the classic PFI bias). Correlated `x_{−j}` inflate marginal importance [S: Hooker et al.; Strobl et al. 2008 K; Molnar et al. 2023].
3. **Trajectory breakage and derived features.** A row-level shuffle turns each subject's path into noise. For a current-state model and a row-additive loss, the expected importance depends only on each row's donor-value distribution, not on the joint structure across rows. Two schemes that give every row the same donor distribution have the same expectation and differ only in variance [K, derived]. Schemes with different donor distributions (M2 vs M3 below) are different estimands. But if the user added derived columns (a lag, a running mean), permuting one column breaks the consistency between them. "Importance of the variable `z`" must then permute the raw series and **recompute every derived feature** [K].

Prior art on this exact point: DynForest permutes "at the individual level when p is time-fixed and at the observation level when p is time-dependent", with no correction for time trends [S: Devaux et al., arXiv 2208.05801, §2.5].

### 2.2 Candidate measures and the question each answers
| # | Method | Question answered | Cost | Notes |
|---|---|---|---|---|
| M1 | Marginal row permutation (Breiman/Ishwaran VIMP) | reliance on `z`, contaminated by extrapolation | predict only | baseline, known biased; keep for comparison only |
| M2 | **Time-conditional permutation**: permute `z` among rows at risk in the same time window (risk-set strata) | reliance on `z` beyond what time explains; stays on the `(t, z)` support | predict only | Strobl-style conditional PI with `t` as the conditioning variable [S: Debeer & Strobl 2020, `permimp`]; conditional-subgroup PFI (Molnar, König, Bischl & Casalicchio 2023) [S]; window width = the conditioning knob (≈ Debeer–Strobl threshold) [K] |
| M2+ | M2 also conditional on `x_{−j}` (subgroups from the forest's own splits or a transformation tree) | partial importance given other covariates | predict only | `permimp`-style [S]; subgroup size limits power |
| M3 | **Subject-level trajectory permutation**: give subject A subject B's `z(t)` looked up at A's times; recompute derived features | importance of the variable as a whole process, including its history features | predict only | needs B's `z` defined over A's follow-up. External covariates usually have that, internal ones do not: report undefined donor coverage. A **different estimand from M2** by default: donor values come from other subjects' paths at A's times, not from A's risk-set stratum. The two coincide only under synchronised external paths with exchangeable donors [K] |
| M4 | **LOCO / condition-and-refit**: drop `z` (or its group), refit, compare held-out loss | predictive value of `z` for the population (model-agnostic) | one refit per variable/group | Hooker et al.'s "gold standard" [S]. Survival VIM with cross-fitting and doubly robust inference: Wolock, Gilbert, Simon & Carone 2025 (Biometrika 112, `survML`) [S], right-censored, time-fixed. rftvc fits are fast (Rust), so this is affordable [R] |
| M5 | Rule-release / VarPro (`importance.rhf`) | time-localised local effect of releasing `z` from each leaf rule, on log integrated OOB hazard | no refit, no permutation | windows on the time grid; start–stop records that overlap the window [S]. Avoids extrapolation by construction [K] |
| M6 | Minimal depth / split counts | how early and often the forest splits on `z` | free | cheap; blurred when `mtry` < p (DynForest recommends max `mtry`) [S]; biased toward many-valued features [K] |
| M7 | Local attributions: SurvSHAP(t) [S: Krzyziński et al. 2023, KBS 262]; TreeSHAP with vector leaf values [K]; DynSHAP / TimeSHAP (time × feature players) [S: arXiv 2609.13042; Bento et al. 2021] | per-prediction contribution of each feature over the time curve | TreeSHAP exact and fast; others sample | TreeSHAP is linear in leaf values, so it applies to each leaf's whole `Λ` vector (hazard aggregation) or `S` vector (survival aggregation) at once [K, verify]. Path predictions route one path through several leaves, which needs time × feature players (DynSHAP) |

**Knockoffs** for time series are immature and are left out [K].

## 3. Scoring: what loss to measure a drop in
- **Counting-process deviance (no IPCW needed).** Worked out in `tvc-deviance.md`, which supersedes the bullets below where they differ. The row-additive log-likelihood `Σ_r [∫ log λ̂ dN_r − ∫_{start_r}^{stop_r} λ̂(u, x_r) du]` is the natural proper loss for this data; RHF minimises exactly this risk [S].
  - With Nelson–Aalen leaves the pointwise version fails: an OOB event time is usually not a jump of the OOB trees, so `log λ̂ = −∞` [K, derived from §1.1].
  - Workable form: a **binned Poisson deviance**. Per (row, window `W_m`), compare the observed events `N` with the expected `E = Λ̂(min(stop, w_m) | x) − Λ̂(max(start, w_{m−1}) | x)`, using `D = 2 Σ [N log(N/E) − (N − E)]` [K, proposal; needs a zero-`E` floor].
  - It is **time-resolvable by construction**: sum over windows = overall, per window = timing (§4.2).
- **Landmark IPCW Brier / IBS** at `(s, w)`, with the S13 metrics and `landmark_cross_validate` [R]. This is proper for the quantity users act on, `P(T ≤ s + w | T > s, H(s))`, and gives importance per landmark `s` and horizon `w`.
  - For `SurvivalForestTV` it needs a path functional: the observed future path (external covariates) or LOCF. The importance is then *of that prediction functional*, which must be stated [K].
- **Concordance.** Rank-based and calibration-blind. The present `oob_score_` uses fixed-profile mortality (§1.3). If C is kept, prefer a time-local version [K].
- **Which data.**
  - OOB per tree (Breiman/Ishwaran; S10 block OOB sets with buffer) is cheap and needs no refit [R].
  - Held-out data (`GroupKFold` on ids, or `GroupTimeSplit` for time series) is needed for M4 and for model-agnostic use [R].
  - Conditional permutation (M2) must stratify within the evaluation set.
- **Uncertainty.** VIMP standard errors by subsampling / delete-d jackknife (Ishwaran & Lu 2019, Stat. Med. 38:558) [S]; cross-fitting inference for M4 (Wolock et al.) [S]. Resampling unit = id or id-block, consistent with fitting [R].

## 4. Level vs history vs timing

### 4.1 Where history can enter
- **`SurvivalForestTV` sees only row columns.** History enters only through user-built columns; the library does no feature engineering on counting-process rows [R].
- **The landmark API builds history features from raw columns** with `last`, `first`, `mean`, `min`, `max`, `sum`, `count` [R: `landmark.py` `AGGREGATIONS`]. Common TVC summaries are missing: slope, time since last change or peak, time above a threshold, exponentially weighted mean, lagged value. These are gaps, and each is a candidate for the design [R/K].
- Because the landmark spec maps each feature to its raw column, the landmark workflow can already do **variable-level grouped importance**: permute the raw series across subjects within a landmark and recompute all its features (M3) [R/K]. This is the gVIMP idea (Gregorutti et al., used by DynForest) [S].

### 4.2 Timing: two different time axes
- **Analysis time `t`** (counting-process estimator): restrict the loss to window `W_m` (binned deviance, §3), or use rule-release per window (M5, as `importance.rhf` does) [S]. This answers "during which periods does `z` drive the hazard?"
- **Landmark `s` and horizon `w`** (landmark workflow): importance per `s` shows how a covariate's predictive value changes during follow-up [S: title/snippet, dynamic-prediction literature]. SurvSHAP(t) localises over the prediction horizon [S].
- Leaves store whole hazard curves, so the per-window attributions for M2 come from one prediction pass per permutation [R/K].

### 4.3 Level vs history, and a diagnostic of the current-state assumption
- **Level importance:** permute the current value **conditional on its history features** (M2+ with history features as conditioners).
- **History importance:** permute the history features jointly, **conditional on the current value**.
- A clearly positive history-given-level importance says that the hazard depends on more than the current state. That is exactly the §1.2 assumption, so the importance tool doubles as a **model-adequacy diagnostic** ("add these history features") [K, derived].

## 5. Effect curves
- **Fixed-profile PD / ALE on `Λ(t | x)`** as in `survex` (`model_profile`, types partial and ALE, with a time dimension) [S].
  - Marginal PD has the same off-support problem as M1.
  - Time-stratified PD (average over the rows at risk in the window) or ALE avoids it [K; S: Molnar et al. subgroups].
- **Hazard-scale effect surface on `(z, t)`.**
  - The estimand is the hazard map, so the most direct TVC-native effect is the windowed hazard contrast `ΔΛ_W(z + δ) − ΔΛ_W(z)`, averaged over the rows at risk in `W`.
  - Plotted over `t` it shows time-varying effects (non-PH) and is valid for internal and external covariates alike, since it is associational on the hazard scale [K, derived].
- **Path-intervention effects** (TVC-native PD).
  - Definition: `Δ(s, w) = mean_i [F̂(s + w | path_i with z → z + δ on (u, ∞)) − F̂(s + w | path_i)]`, computable today with `predict_*(intervals=…)` [R]. ICE-style per-subject curves come from the same computation.
  - It is a valid **prediction under a specified covariate path** only for external covariates (D4) [R].
  - A **causal** reading needs more: no unmeasured confounding of the `z`–hazard link, and a covariate process not affected by the intervention. The prediction-under-interventions literature (Keogh & van Geloven 2024, Epidemiology) shows how the evaluation differs [S]. Docs must say "what the model predicts along this path", not "the effect of changing `z`" [K].
- **Competing risks:** the same curves on `F_k` via Aalen–Johansen along the path [R: `predict_cumulative_incidence(intervals=…)`].

## 6. Competing risks
- **Cause-specific importance.** Binned deviance on cause-`k` events uses `Λ̂_k`, and gives importance for `λ_k` [K]. `randomForestSRC` reports cause-specific VIMP for competing-risks forests [K: verify].
- **Importance for `F_k`** comes from the CR IPCW Brier at landmarks (S13 metrics) [R].
  - A covariate can matter for `F_k` purely through a competing cause, by changing `λ_j`.
  - So `F_k` importance and `λ_k` importance answer different questions, and both should be reported, as in the CIF-vs-cause-specific-hazard distinction of `cr-research.md` §1 [R/K].
- DynForest lists competing risks among its settings [S: keywords]; whether its VIMP is cause-specific was not checked [K: verify].

## 7. Prior art summary
| Tool / paper | TVC input | Importance | Time-resolved | Effects | Language |
|---|---|---|---|---|---|
| `randomForestSRC` (Ishwaran) | CR yes; start–stop only with `splitrule="random"` for CR [R: `cr-design.md`] | VIMP (permute/noise), minimal depth, subsampling CIs [S] | no | partial plots [K] | R/C |
| `randomForestRHF` (2026) | counting process, `Surv(id, start, stop, event)` [S] | **time-localised VarPro rule-release** on log integrated OOB hazard (`importance.rhf`) [S] | **yes, windows on the time grid** | — | R/C |
| DynForest (Devaux et al.) | longitudinal markers summarised by mixed models inside nodes [S] | VIMP (observation-level permutation for TVCs), grouped VIMP, minimal depth, on OOB IBS [S] | via landmark | — | R |
| `LTRCforests` | start–stop pseudo-subjects [S] | none found in the documented API [S] | — | — | R |
| `survex` | fixed covariates | PFI on Brier / time-dependent loss [S] | loss over time [S] | PD, ALE, SurvSHAP(t), SurvLIME [S] | R |
| `survshap` | fixed | SurvSHAP(t) [S] | yes (horizon) | — | Python |
| `survML` (Wolock et al.) | fixed, right-censored | LOCO-type VIM with inference [S] | per landmark time [S: title] | — | R |
| `permimp` | fixed | conditional permutation, threshold knob [S] | — | — | R |
| TimeSHAP / DynSHAP | sequences | Shapley over time × feature [S] | yes | — | Python (TimeSHAP) |

**Gap (provisional):** no **Python** tool offers importance or effects that respect counting-process TVC rows: time-conditional permutation, variable-level importance with derived-feature recomputation, time-resolved loss, or path-intervention effects. Among R tools, RHF's time-localised rule-release is the closest.

## 8. Validation: simulations with known truth
Extend `tests/sim.py` (external `z` redrawn per unit interval, Cox-type hazard with a threshold) [R] and the S14 CR generators [R] with these families:
1. **Trend confounding.** `z_1` has an effect; `z_2` trends with `t` and is correlated with `z_1` but has no effect. Expect M1 to inflate `z_2` through extrapolation and M2/M2+ to give ≈ 0. Compare against an oracle "true loss increase" from the known hazard [K].
2. **History vs level.** The hazard depends on a lag `z(t − 1)` or on cumulative exposure. Show (a) the current-state forest's loss vs the oracle, (b) history-given-level importance > 0 when the lag feature is supplied, and ≈ 0 in a Markov control [K].
3. **Timing.** `z` acts only for `t ∈ [2, 4]`. Windowed importance should localise; landmark importance should peak for `s` near that window [K].
4. **Correlated groups.** A raw variable plus derived features. Grouped (M3) vs per-feature importance [K].
5. **Competing risks.** `z` affects cause 1 only. Expect `λ_1` importance > 0 and `λ_2` importance ≈ 0, while `F_2` importance is > 0 through competition [K].
6. **Effects.** Path-intervention `Δ(s, w)` against the true `Δ` from the known hazard (external `z`). Coverage of subsampling intervals [K].

Pass rules are declared in advance, as in S3/S14 [R].

## 9. Hypotheses revisited (from `tvc-questions.md`)
- *"A single importance number is not enough."* **Supported.** The literature and §2–4 give at least three distinct questions: overall reliance (M2 or M4), variable vs feature level (M3 / grouped), and timing (windowed loss or rule-release), plus effect curves.
- *"Naive permutation is biased for TVCs that correlate with time."* **Plausible, with a candidate mechanism:** off-support rows fall into event-free regions of leaf cumulative hazards and get no predicted hazard there (§2.1). This is unproven until the §8.1 simulation shows it.
- **New:** history-given-level importance is a diagnostic of the forest's core assumption (§4.3).
- **New:** a binned piecewise-exponential (Poisson) score is a candidate IPCW-free, time-resolvable loss for counting-process rows. It is proper only in a qualified sense (`tvc-deviance.md`).

## 10. Open questions for the design stage
1. **Scope of v1.** Which of M2 / M3 / M4 ship first? Is M5 (rule-release) worth porting, or does windowed M2 cover the timing question?
2. **Default loss.** Binned deviance (native to the counting-process estimator), landmark IBS (native to the landmark models), or both, per estimator?
3. **API shape.** An `rftvc.inspection` module mirroring `sklearn.inspection` (`permutation_importance`, `partial_dependence`), plus TVC-specific arguments (`strata="time"`, `groups=`, `windows=`)? Or methods on the estimators? Keep it general purpose (memory `rftvc-general-purpose`).
4. **History features.** Add slope, time-since, time-above-threshold, EWMA and lag to `landmark.AGGREGATIONS`, and offer a helper that builds the same features on counting-process rows?
5. **`oob_score_` / `score`.** Switch TVC rows to a time-local C or to deviance? This is a release-pass decision, flagged by §1.3.
6. **TreeSHAP with vector leaves.** In scope, or deferred?

## Review log
- **2026-09-26, Codex research review (4 findings, all accepted):**
  1. "Hazard exactly 0" misstated: leaves carry the cumulative hazard forward, so only window increments are 0. Reworded §1.1, §2.1 and §9, and the mechanism is marked unproven.
  2. Floored deviance is not strictly proper: moved to `tvc-deviance.md` with a qualified propriety statement.
  3. "Trajectory breakage changes only variance" and "M3 = M2 in expectation" were too broad: the condition (same per-row donor distribution) is now stated, and M2 and M3 are different estimands.
  4. The hazard-map limit is heuristic, with its assumptions listed. Landmark validity is not "always": it needs conditional independent censoring, support and transportability.
