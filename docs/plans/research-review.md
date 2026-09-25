# Adversarial review of `research.md`

Reviewed 2026-09-25 against `questions.md` and the linked primary/package documentation.  Line numbers refer to `docs/plans/research.md`.  This is a review, not a proposed edit.

## Key findings

### 1. The internal-TVC conclusion is too categorical and would rule out the intended dynamic-prediction use case

**Confidence: high. Priority: high. Location: lines 59–61.**

The document says pseudo-subject forests estimate only “a *conditional hazard given the current covariate value*,” then says that, for internal covariates, “the curve is not a survival probability you can predict with.”  This conflates two distinct targets:

- A survival probability conditional on a *specified future internal-marker path* is generally not operational for an individual, because that future path is unknown and the marker is only observed while alive.
- A dynamic prediction at landmark `s`, such as `P(T > s+w | T > s, observed history through s)`, is nevertheless a well-defined survival probability and is precisely the target of landmarking and joint models.  It does not require observing the future marker path at prediction time.

Yao et al. do restrict *their* forest paper to population-level estimation rather than individual prediction because of the internal-covariate problem; that does **not** imply that internal covariates make all survival prediction invalid.  Their paper explicitly distinguishes the two purposes, and landmarking literature defines the individual conditional target above.  The design consequence is material: the plan should distinguish (a) hazard/path-scenario estimation, (b) landmark dynamic prediction conditional on observed history, and (c) joint longitudinal-survival prediction, rather than excluding (b) outright.

Concrete correction: replace the quoted conclusion with: “For internal covariates, an unconditional survival curve under an arbitrary future path is not identified/predictable without modelling that path.  Dynamic conditional survival given the observed history and being event-free at a landmark is still meaningful; landmarking or a joint model is needed when that is the target.”  Also state that a forest using only the current value estimates a model-dependent conditional hazard, not necessarily a full-history conditional hazard.

Sources: [Yao et al., 2022, pp. 2–3 (their estimation—not individual-prediction—scope)](https://www.weichiyao.com/papers/EnsembleDynamicEstimationsTimeVaryingCovariates.pdf); [Snell et al., dynamic-prediction definition](https://academic.oup.com/jrsssa/article/184/1/3/7056431); [landmarking conditional survival target](https://pmc.ncbi.nlm.nih.gov/articles/PMC5957493/).

### 2. The statement that classifier forests cannot produce calibrated discrete hazards is false

**Confidence: high. Priority: high. Location: lines 68–71, especially line 71.**

The claim is: “Classification forests give probabilities that are *not* calibrated hazards unless the forest is fit as a regression / Poisson forest.”  In person-period/discrete-time survival data, the binary target for an at-risk interval is the discrete hazard, `P(event in interval | alive at interval start, covariates)`.  A probabilistic classifier trained for that target can estimate that hazard; whether its probabilities are calibrated is an empirical/property-of-the-learning-and-calibration question, not something made impossible by classification.  A Poisson/piecewise-exponential forest is an alternative likelihood/continuous-time approximation, not a prerequisite for calibrated hazards.  Conversely, regression or Poisson fitting does not automatically confer calibration.

Concrete correction: say that binary classifiers estimate *interval event probabilities (discrete hazards)* when rows/risk sets and labels are constructed correctly; assess calibration at the chosen interval/horizon (and recalibrate if needed).  Clarify that coarser bins target interval risk rather than an instantaneous hazard, while a sufficiently fine, correctly specified piecewise model can approximate a continuous-time hazard.  Do not present Poisson fitting as the calibration criterion.

Source: RF-SLAM’s primary description explains why the conditional event indicators can use a binomial or Poisson likelihood after conditioning on the past—these are alternative representations, not a claim that the binomial/classification representation is invalid: [Wongvibulsin, Wu & Zeger, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6937754/).  For dynamic survival calibration/Brier assessment, see [Snell et al.](https://academic.oup.com/jrsssa/article/184/1/3/7056431).

### 3. The claimed invalidity of LTRC log-rank/permutation calibration is asserted without support and is too broad

**Confidence: high. Priority: high. Location: lines 63–66.**

Lines 64–65 say an LTRC log-rank statistic is valid only as a heuristic and that its “p-value / permutation-test calibration (CIF) is not valid, because the effective sample size is `n_subjects`, not `n_rows`.”  Dependence/repeated intervals absolutely matters for resampling, standard errors, and inference, but that explanation is not a general proof that a counting-process score or every conditional-inference permutation calibration is invalid.  Standard counting-process survival likelihoods deliberately factor conditional interval contributions; RF-SLAM explicitly notes that its Poisson/binomial interval likelihood does not require directly assuming CPIUs are independent after conditioning on the past.  A correct result depends on the exact score, null, permutation unit, and censoring/visit process.

The cited Yao paper describes pseudo-subjects as treated as independent LTRC observations, but the document supplies no source establishing the categorical p-value claim.  The claim is especially risky for implementation because it may lead the project to reject an otherwise usable split statistic for the wrong reason.

Concrete correction: mark the inferential claim as **unverified** unless a clustered counting-process/permutation reference is supplied.  Say: “Row-level pseudo-observation resampling/permutation is not an appropriate default for new-subject generalization or cluster-level inference; resample, cross-validate, and permute at subject level.  The finite-sample/null calibration of a particular LTRC split test must be derived or empirically validated under the intended within-subject process.”  Keep a log-rank criterion as a predictive split candidate rather than calling it mathematically invalid.

Sources: [Fu & Simonoff’s LTRC-tree description/package](https://pages.stern.nyu.edu/~jsimonof/survivaltree/); [RF-SLAM’s conditional-interval likelihood explanation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6937754/); [Yao et al. on pseudo-subject reformatting](https://www.weichiyao.com/papers/EnsembleDynamicEstimationsTimeVaryingCovariates.pdf).

### 4. The evaluation rules overstate what IPCW and time-dependent discrimination “must” do

**Confidence: high. Priority: high. Location: lines 74–77.**

The document says a time-dependent C-index/AUC “must be computed per landmark or with dynamic risk sets,” and that IBS “needs IPCW, with the censoring model conditional on history.”  The useful underlying point is right—evaluation must match the prediction time, horizon, risk set, and information available at prediction—but the absolute wording is not.

For a landmark model, landmark-specific risk sets and an IPCW Brier score conditional on being uncensored through the landmark are appropriate.  For a model producing predictions at a fixed baseline or a fixed prediction time, standard time-dependent concordance/AUC definitions can be used without fitting a separate model/metric at every landmark.  IPCW requires an appropriate independent-censoring assumption and an estimate of the censoring survival probability.  Conditioning that model on history is needed when censoring is only independent **conditional on** that history; a marginal censoring model is sufficient under unconditional independent censoring.  IPCW can also be unstable/misspecified when censoring relates to covariates, so it should not be presented as a mechanical requirement.

Concrete correction: specify an evaluation protocol by prediction origin `s` and horizon `w`: subject-level outer splits; construct test risk sets with `T >= s`; score `P(T > s+w | T > s, H(s))`; use an IPCW estimator whose censoring model matches the assumed conditional-independence set; report calibration at `s,w` as well as discrimination.  Name the chosen C-index/AUC definition (e.g., cumulative/dynamic or incident/dynamic), rather than “time-dependent C-index” generically.

Sources: [landmark IPCW Brier-score construction](https://academic.oup.com/jrsssa/article/184/1/3/7056431); [conditions for landmark consistency/prediction](https://link.springer.com/article/10.1007/s12561-016-9157-9); [limitations of IPCW Brier scores when covariates inform censoring](https://www.jmlr.org/beta/papers/v24/19-1030.html).

## Minor findings

- **Confidence: high. Priority: medium. Location: line 29.**  “RF-SLAM … 2019” is a misleading publication year.  The paper was received/accepted in 2019 but appeared in *BMC Medical Research Methodology* in 2020 (the PMC record gives collection date 2020).  Cite it as Wongvibulsin, Wu & Zeger (2020), while optionally noting the 2019 online/accepted history.  Source: [article record](https://pmc.ncbi.nlm.nih.gov/articles/PMC6937754/).

- **Confidence: high. Priority: medium. Location: lines 28, 45, 51 and 77.**  The table/description calls `LTRCtrees` pseudo-subject handling, but the package is a **tree** package, not a forest; the maintained CRAN package now lists Fu, Simonoff, and maintainer Wenbo Jing and provides `LTRCART`/`LTRCIT`.  `LTRCforests` is the forest implementation.  More importantly, the paper/package APIs distinguish the sampling unit via `id`; use subject-level bootstrap/OOB/CV for the intended “new subject” estimand.  Row bootstrap does create in-bag/OOB subject overlap and optimistic subject-generalization assessment, but “Yao et al. found [row and subject bootstrap] similar accuracy” and the asserted OOB-importance result need a direct source or should be marked unverified.  Sources: [LTRCtrees CRAN metadata](https://CRAN.R-project.org/package=LTRCtrees); [`LTRCforests::ltrcrrf` documentation](https://search.r-project.org/CRAN/refmans/LTRCforests/help/ltrcrrf.html); [RF-SLAM’s person-bootstrap description](https://d-nb.info/1208068555/34).

- **Confidence: high. Priority: medium. Location: lines 34 and 38–40.**  “IPOL (probably Python, not confirmed)” is not adequate evidence either for or against the headline Python-package gap.  The Laurent–Vo Van article confirms an LTRC survival forest, downloadable source, and an online demo, but the cited material does not establish a maintained Python/scikit-learn implementation or native TVC API.  Resolve its implementation language and input contract before using it in the gap claim; do not describe it as Python in the table.  The present review found no maintained Python **random forest** with native `id,start,stop,event` support that contradicts the headline; BoXHED2.0 is instead a maintained Python *gradient-boosted* counting-process hazard estimator.  Sources: [Laurent & Vo Van, 2024](https://doi.org/10.5201/ipol.2024.466); [BoXHED2.0 JSS paper](https://www.jstatsoft.org/article/view/v113i03/4712); [PyPI package](https://pypi.org/project/boxhed/).

- **Confidence: high. Priority: medium. Location: lines 10–18 and questions 9–16.**  The state-of-the-art table does not answer several methods explicitly requested in `questions.md`: notably `ranger`, and newer competitor families such as survival BART/deep dynamic-survival methods.  Add `ranger` at minimum: it is a maintained C++/R survival forest implementation (and is relevant as a performance baseline), but it is not evidence of native counting-process TVC support.  The table should separate ordinary right-censored survival forests from TVC-capable forests rather than imply its short list is exhaustive.  Source: [Wright & Ziegler’s ranger paper](https://arxiv.org/abs/1508.04409).

- **Confidence: medium. Priority: medium. Location: line 73.**  It is reasonable to warn that delayed-entry KM estimates can become unstable with small early risk sets; that is supported.  The follow-on assertion that “NA / hazard-scale averaging across trees is better behaved than averaging survival curves” is unsupported and lacks a stated estimand.  Averaging cumulative hazards then exponentiating and averaging tree survival functions are different ensemble functionals; neither is uniformly “better behaved.”  State the chosen aggregation and justify it theoretically or by calibration experiments.  Source for the narrow, supportable claim: [review of small-risk-set instability under left truncation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11345615/).

- **Confidence: medium. Priority: low. Location: line 54.**  The numerical recommendation `nodesize = max(default, sqrt(n_pseudo))`, the “wide margin” result, and tuning by OOB IBS are attributed to Yao et al. but not locatable in the supplied source text during this review.  Treat these as implementation-specific/default-package guidance, not a general statistical recommendation, unless the exact simulation/table or package documentation is cited.

## Verified facts worth retaining

The core gap is presently defensible with qualification: `randomForestRHF` 2.1.0 is a maintained R/C/OpenMP package published on CRAN on 2026-09-17 and accepts `Surv(id, start, stop, event)`; it is a hazard-likelihood forest, not a log-rank adaptation.  BoXHED2.0 is a maintained Python counting-process **boosted-tree** hazard estimator, not a random forest.  RF-SLAM uses CPIUs and a Poisson log-likelihood split.  These distinctions are important and correctly motivate a scikit-learn-compatible forest gap.

Sources: [randomForestRHF CRAN page](https://CRAN.R-project.org/package=randomForestRHF); [RHF getting-started documentation](https://www.randomforestrhf.org/articles/getstarted.html); [BoXHED2.0](https://www.jstatsoft.org/article/view/v113i03/4712); [RF-SLAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC6937754/).
