# Research Findings: Survival Random Forests with Time-Varying Covariates

Status: research stage done (2026-09-25); Codex review corrections applied (see research-review.md). No design decisions yet. Answers `questions.md`.
Tags: [S] = backed by a source fetched this session; [K] = background knowledge, verify before relying on it.

---

## i) State of the art: random forests for survival

### Time-fixed covariates (the mature part)
| Model | Split rule | Leaf estimator | Implementations |
|---|---|---|---|
| RSF (Ishwaran 2008) | log-rank / log-rank score / C-index | KM + Nelson–Aalen (NA) | randomForestSRC (R), scikit-survival `RandomSurvivalForest` (Py) [S] |
| Extra survival trees | random thresholds + log-rank | KM / NA | scikit-survival `ExtraSurvivalTrees` [K] |
| Conditional inference forest | permutation test on log-rank scores (less bias toward many-level variables) | KM | partykit `cforest` (R) [K] |
| Oblique RSF (Jaeger 2019; aorsf 2023) | linear combos of features via Cox Newton–Raphson in each node | KM / NA | aorsf (R); often top C-index in benchmarks, and hundreds of times faster than earlier oblique RSF [S] |
| grf survival forest | log-rank, now with constant-time split updates (Sverdrup, Yang, LeBlanc 2025, arXiv 2510.03665) | KM | grf (R) [S] |
| ranger survival forest | log-rank / C-index / maxstat | KM / NA | ranger (C++/R); fast baseline; no native counting-process TVCs [S: Wright & Ziegler 2017, arXiv 1508.04409] |
| Gradient-boosted survival | Cox / AFT loss | – | xgboost, lightgbm, scikit-survival GBSA [K] |

This table covers **time-fixed** forests only and is not exhaustive. Competitor families not yet reviewed: survival BART, and deep dynamic-survival models (e.g. Dynamic-DeepHit, DeepHazard) [gap: to review if they become baselines].

Takeaways:
- Benchmarks: RSF, ORSF and boosting beat Cox when effects are non-linear or interact. With linear additive effects, Cox does as well or better [S: Sci Rep 2025 simulation].
- Efficiency: a naive log-rank split search costs O(M) per candidate split, where M is the number of distinct event times in the node. The grf trick makes each update O(1) [S]. scikit-survival's RSF is known to use a lot of memory on large n and many unique times, which is why it has `low_memory` [K].

### Time-varying covariates: what exists now (key finding)
| Method | Year | TVC mechanism | Language |
|---|---|---|---|
| Bou-Hamad et al., discrete-time trees/forests | 2011 | person-period rows, binary hazard | R (ad hoc) [S: title] |
| Fu & Simonoff, LTRC **trees** (`LTRCtrees`: `LTRCART`/`LTRCIT`; the forests are in `LTRCforests`) | 2017 | counting-process rows treated as independent left-truncated (LTRC) "pseudo-subjects" | R [S] |
| RF-SLAM (Wongvibulsin, Wu, Zeger) | 2020 (accepted 2019) | counting-process information units (CPIUs), Poisson log-likelihood split | R (`rfSLAM`) [S] |
| Pickett et al., RSF landmarking | 2021 | landmark snapshots → ordinary RSF per landmark | R [S] |
| Moradian, Yao et al., discrete-time dynamic forests | 2021 | person-period data, pooled "super person-period" fit | R [S] |
| **Yao, Frydman, Larocque, Simonoff: CIF-TV / RRF-TV (`LTRCforests`)** | 2022 | pseudo-subjects; LTRC log-rank scores or Poisson deviance with NA exposure | R (CRAN) [S] |
| DynForest (Devaux et al.) | 2023/2025 | fits a mixed model for each longitudinal marker inside every node, then splits on the random effects | R [S] |
| Laurent & Vo Van, LTRC survival forest | 2024 | LTRC CART → forest, with a "simple API" | IPOL: source and online demo exist; **implementation language and TVC input format unverified**, so this row does not count for or against the Python-gap claim [S] |
| **BoXHED 2.0 (Pakbin, Wang, Mortazavi, Lee)** | JSS 2025 | boosted nonparametric hazard λ(t, X(t)), counting-process data, C++/GPU | **Python (PyPI `boxhed`)** [S] |
| **Random Hazard Forests (Ishwaran, Kogalur et al.) `randomForestRHF` 2.1.0** | CRAN, 17 Sep 2026 | `Surv(id,start,stop,event)`; splits on a nonparametric hazard likelihood for predictable covariate processes; step-function hazard in each leaf | R/C, OpenMP [S] |

**Gap confirmed:** no maintained Python package offers a *random-forest* survival model with native counting-process TVCs.
- BoXHED 2.0 is the closest Python tool, but it is boosting, not a forest.
- RHF is the Ishwaran group's own answer to "RSF can't do TVCs", but it is R only.

## ii) How TVCs are handled, and the challenges

### Four data representations
1. **Counting process / pseudo-subjects (LTRC)**: each (start, stop] interval with constant X becomes one row, treated as a left-truncated observation. Used by Fu & Simonoff, LTRCforests, RF-SLAM and RHF. Needs risk sets that respect left truncation, both in the split statistic and in the leaf KM/NA [S].
2. **Person-period / discrete time / survival stacking**: bin time and turn survival into a binary hazard classification, with time as a feature. Any sklearn classifier works. The results depend on how time is binned, and the data can get very large [S].
3. **Landmarking**: at landmark time s, take everyone still at risk, summarise the history up to s as features, and fit an ordinary RSF for the horizon. This makes the prediction target explicit, but you need one model per landmark (or a stacked super-model), and data are thrown away [S].
4. **Longitudinal summaries inside nodes** (DynForest): mixed models deal with measurement error and irregular visit times, but they are expensive and you have to specify the mixed model up front [S].

### Practical challenges
- **Within-subject dependence**: pseudo-rows from one subject are not independent. Per a tool summary of arXiv 2006.00567, Yao et al. found that bootstrapping by subject or by row gave similar accuracy (not re-verified in the text; the review could not locate it). Bootstrapping by row lets a subject's rows land both in-bag and out-of-bag, which makes OOB assessment of how well the model generalises to new subjects optimistic [K]. For the new-subject estimand, resample, cross-validate and compute OOB by `id`.
- **Routing a subject across leaves**: a subject can fall in different leaves at different times. Prediction then chains conditional survival: S(t) = Π over intervals of S_leaf_j(t_{j+1}) / S_leaf_j(t_j) [S: Yao eq.]. A cumulative-hazard sum is the equivalent in hazard form, and RHF works this way natively [S].
- **You need the future covariate path**: a survival curve beyond the last observed covariate time requires either assuming X stays constant (LOCF) or supplying a scenario path [S].
- **Node sizes**: a tool summary attributes to Yao et al. the rule nodesize = max(default, √n_pseudo), plus tuning mtry by OOB integrated Brier score (IBS). **Unverified**: the review could not locate this in the text. Treat it as a package-default heuristic, not a statistical result.
- **Computation**: counting-process expansion makes N_rows much larger than n_subjects. Split search also has to handle left-truncated risk sets, so the usual sorted-by-time cumulative trick needs entry/exit event sweeps [K]. A time grid (RHF's `ntime`) caps the cost [S].

## iii) Statistical implications

1. **Three separate targets. Don't conflate them.**
   - (a) **Hazard / path-scenario estimation.** A forest that uses only the current X(t) estimates a model-dependent conditional hazard λ(t | X(t)). This is not necessarily the full-history hazard λ(t | H(t)).
     - For **external** covariates, S(t | X path) under a specified path is a genuine survival probability.
     - For **internal** covariates, a survival curve along an arbitrary future path cannot be identified or predicted without modelling that path. This is why Yao et al. restrict their paper to estimation [S].
   - (b) **Landmark dynamic prediction** P(T > s+w | T > s, H(s)). This is well defined for internal *and* external covariates and needs no future covariate path [S: Snell et al.; van Houwelingen].
   - (c) **Joint longitudinal–survival prediction.** This models the covariate path explicitly (joint models, DynForest).
2. **Predictability.** A valid hazard model needs X(t) to be known just before t (predictable). A covariate measured at or after the event time leaks the outcome, which is look-ahead bias. RHF makes this an explicit requirement [S].
3. **Split statistics with pseudo-subjects.**
   - A log-rank score with LTRC risk sets remains a usable predictive split candidate. Counting-process likelihoods factor into conditional interval contributions, so conditional on the past they do not need rows to be independent [S: RF-SLAM].
   - **Unverified:** whether a particular split test's p-value or permutation calibration is valid under within-subject dependence. It depends on the score, the null, the permutation unit, and the censoring/visit process. It has to be derived or checked by simulation. Row-level resampling or permutation is the wrong default for generalising to new subjects; use subject-level units.
   - Subjects with many visits carry more weight in splits, which gives implicit weighting by how often a subject was measured [K].
4. **Informative visit times.** If sicker patients are measured more often, the visit process carries information about the outcome. Pseudo-subject methods and LOCF ignore this. Joint models and DynForest partly address it [K].
5. **Discretisation (person-period).**
   - As the bins get finer, this approaches the continuous-time hazard.
   - Coarse bins bias the hazard and blur event ordering.
   - A probabilistic classifier trained on correctly built person-period rows estimates the interval event probability, which is the discrete-time hazard. Calibration has to be checked (and recalibrated if needed) at the chosen interval and horizon. Poisson fitting is an alternative likelihood, not a calibration guarantee [S: RF-SLAM].
6. **Landmarking.** Each landmark model is valid on its own. Predictions from different landmarks do not have to agree with each other, and the method is not efficient because it discards data [S+K].
7. **Leaf estimator under left truncation.** A KM with delayed entry is unstable when the risk set is small early on, which is another reason for larger nodesize [S: PMC11345615]. Averaging cumulative hazards and then exponentiating, versus averaging tree survival curves, target different ensemble quantities; neither is uniformly better. Choose explicitly and validate by calibration.
8. **Evaluation.**
   - Match the evaluation to the prediction origin s and horizon w.
     - For landmark models, use landmark risk sets (T ≥ s) and score P(T > s+w | T > s, H(s)).
     - For fixed-origin predictions, standard cumulative/dynamic or incident/dynamic AUC and C-index work. Always name the definition used.
   - IPCW requires independent censoring plus a censoring-survival estimate.
     - Condition that model on history only if censoring is independent *only given* history. Otherwise a marginal model is enough.
     - IPCW can be unstable or misspecified [S: Snell et al.; JMLR 24 19-1030].
   - Report calibration at (s, w) as well as discrimination. Use subject-level outer splits for the new-subject estimand, and time-based splits for future-period deployment (see design-principles.md).

## Open questions for the design stage
- ~~Target / covariate type~~ → answered: mostly external covariates, dynamic prediction (target b).
- Pseudo-subject log-rank (LTRC-RSF) vs hazard-likelihood (RHF-style) vs Poisson (RF-SLAM) split criterion?
- ~~Engine~~ → answered: Rust core (see design.md).
- Which baselines to benchmark against: LTRCforests, randomForestRHF (via rpy2), BoXHED 2.0, landmark-RSF with sksurv.

## Sources
- Yao et al. 2022, Ensemble methods… TVC: https://arxiv.org/html/2006.00567 ; https://journals.sagepub.com/doi/abs/10.1177/09622802221111549
- LTRCforests: https://rdrr.io/cran/LTRCforests/man/LTRCforests-package.html ; https://github.com/weichiyao/TimeVaryingData_LTRCforests
- Fu & Simonoff 2017: https://academic.oup.com/biostatistics/article/18/2/352/2739324
- RF-SLAM: https://link.springer.com/article/10.1186/s12874-019-0863-0
- Discrete-time dynamic forests: https://ar5iv.arxiv.org/html/2103.01355
- Survival stacking: https://arxiv.org/pdf/2107.13480
- DynForest: https://arxiv.org/pdf/2302.02670 ; https://journal.r-project.org/articles/RJ-2025-002/
- RSF landmarking (Pickett 2021): https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-021-01375-x
- Random Hazard Forests: https://cran.r-project.org/web/packages/randomForestRHF/index.html ; https://rdrr.io/cran/randomForestRHF/man/rhf.html
- BoXHED 2.0: https://arxiv.org/abs/2103.12591 ; https://github.com/BoXHED/BoXHED2.0
- aorsf: https://arxiv.org/pdf/2208.01129 ; ORSF vs RSF sim: https://www.nature.com/articles/s41598-025-27747-7
- Efficient log-rank updates: https://arxiv.org/abs/2510.03665
- LTRC survival forest (IPOL 2024): http://www.ipol.im/pub/art/2024/466/
- scikit-survival RSF: https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html
- Bou-Hamad 2011: https://www.researchgate.net/publication/254133473
- Snell et al., dynamic prediction: https://academic.oup.com/jrsssa/article/184/1/3/7056431
- Landmark conditional survival target: https://pmc.ncbi.nlm.nih.gov/articles/PMC5957493/
- Left-truncation small risk sets: https://pmc.ncbi.nlm.nih.gov/articles/PMC11345615/
- IPCW Brier limitations: https://www.jmlr.org/beta/papers/v24/19-1030.html
- ranger: https://arxiv.org/abs/1508.04409
- LTRCtrees CRAN: https://CRAN.R-project.org/package=LTRCtrees
