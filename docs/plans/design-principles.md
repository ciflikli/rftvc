# Design Principles: General-Purpose Survival Forest with Time-Varying Covariates

Status: draft v2 (2026-09-25). Input to `design.md`. Builds on `research.md` + `research-review.md`.
v2 change: the library is general purpose. Country-month conflict data is **one use case**, not the design target.
Tags: [S] = sourced; [K] = background knowledge / reasoning, validate empirically.

User decisions so far: mostly external covariates; goal = dynamic prediction accuracy.

---

## G0. What the library must not assume
- A data frequency (discrete months vs continuous time).
- Whether "out of sample" means **new subjects**, **future time**, or both.
- One outcome structure: single event, recurrent events, or alternating states.
- One domain: clinical, conflict, churn, reliability.

## G1. One engine, several data views

**Core engine:** a counting-process survival forest over rows `(id, start, stop, event, X)`.
- Left truncation is handled natively, in both the split statistic and the leaf estimator.
- Every tree-building quantity that is a "count" (resampling, min leaf, weights) uses **`id` groups, not rows**.

**Data views:** transformers that produce engine input. The engine never changes.

| View | What the transformer produces | Serves |
|---|---|---|
| Counting process (identity) | rows as given | hazard given current X; path-conditional survival (external covariates) |
| Landmark stack | rows at each landmark `s`: entry = `s`, covariates = `H(s)`, administrative censoring at `s+w` | dynamic prediction `P(T > s+w \| T > s, H(s))` with no future covariates [S: van Houwelingen; Tanner et al. 2021] |
| Discrete / person-period | binned rows | discrete-time data such as BTSCS; benchmarking against classifiers |

Key point: a landmark row *is* a left-truncated row with entry `s`. So the stacked landmark super-model needs no second algorithm, only a transformer [K].

## G2. Error estimation principle (general)

**The error estimator must reproduce the deployment distribution shift.** The library exposes this as a choice; it does not hard-code an answer.

| Deployment question | Splitter (sklearn-style CV object) |
|---|---|
| New subjects | `GroupKFold` by `id` (default) |
| Future periods, same subjects | `RollingOriginSplit(gap≥horizon)`: expanding window with an embargo [S: Cerqueira et al. 2020; Bergmeir et al. 2018] |
| Both | group × time blocked split |

Rules that hold in every setting:
1. **Nested CV:** tuning and reporting use separate loops [S: Neunhoeffer & Sternberg 2019].
2. **Never split rows of one `id` across train/test at random.** Row-level OOB error and importance are optimistic when rows are dependent.
   - OOB is computed per `id`, and only when resampling is by group.
   - It is disabled or warned against when the user declares temporal deployment [K].
3. **Embargo ≥ horizon** whenever a label's look-ahead window can overlap the test period (landmark or horizon targets).

## G3. Metrics (general)
- **Proper scores first:** time-dependent Brier score and integrated Brier score (IPCW), plus log loss where applicable.
- **Discrimination:** a named time-dependent C-index or AUC(t) definition (cumulative/dynamic).
- **Landmark evaluation:** scored at `(s, w)` pairs.
- **IPCW censoring model:** marginal by default; covariate-conditional when the user opts in [S: research-review finding 4].
- **Shortcut:** if test windows have complete follow-up, no IPCW is needed and plain scores are exact.
- Calibration curves at chosen horizons.

## G4. Training under dependence (general knobs, defaults conservative)
- `resample_unit`: `"id"` (default) | `"row"` | `"block"`.
  - Block = id × time-block, for long series with few ids.
- `min_leaf`: counted in distinct ids (or episodes), not rows [K].
- `landmark_step` / overlap weights: control the ~w-fold redundancy of stacked landmarks [K].
- `split_criterion`: pluggable. Candidates:
  - LTRC log-rank (Fu & Simonoff; Yao et al.)
  - hazard likelihood (RHF-style)
  - Poisson (RF-SLAM)
  - a horizon-Brier criterion

  Chosen empirically [S: research.md].
- Leaf estimator: Nelson–Aalen / KM with delayed entry. Aggregation (cumulative hazard vs survival averaging) is an explicit option, validated by calibration [S: research-review minor].

## G5. Explicitly out of scope for v1 (documented, not silently wrong)
- Internal covariates with path-conditional survival. The landmark view is the supported route [S: research-review finding 1].
- Informative visit / measurement processes (joint models, DynForest territory).
- Competing risks and multi-state models. Possible v2: the counting-process format extends naturally.

## G6. Validation suite (general, not one dataset)
- Simulations with known truth:
  - Cox-type time-varying covariates
  - non-proportional hazards / interactions
  - left truncation
  - many rows per id vs few
- Public datasets across domains:
  - clinical TVC data (e.g. PBC2 / `pbcseq`, Stanford heart transplant)
  - a BTSCS example (e.g. UCDP country-month)
  - a large-n dataset for scaling
- Baselines:
  - Cox with TVCs (lifelines)
  - `LTRCforests` and `randomForestRHF` (R, via rpy2, optional)
  - BoXHED 2.0
  - scikit-survival RSF on landmark data

## Appendix: case study — BTSCS / conflict (was v1 of this file)
An illustration of how the general knobs map to one domain, not a design target.
- Deployment = future months, same countries → `RollingOriginSplit(gap≥H)`, nested; final untouched window [S: ViEWS / Hegre et al. 2021].
- Landmark view with `w = H`; the last `H` months of training enter as censored rows.
- `resample_unit="block"`; `min_leaf` in episodes.
- Onset and termination: disjoint risk sets → two fits, or one fit with a transition indicator. Clock = time since the last transition [K: Beck, Katz & Tucker 1998]. Caution on overlapping landmarks with recurrent events [S: Musoro et al. 2018].
- Do not down-sample rare events for the forest (distorts probabilities; see the Muchlinski debate).

## Open decisions for design.md
- [ ] Split criterion default (bake-off in slice 1 or later).
- [x] Engine language — **confirmed by user 2026-09-25**:
  - Rust core: PyO3 + maturin + rayon + rust-numpy.
  - Thin Python sklearn wrapper.
  - numba only for prototyping split criteria.
  - Polars for data-view transformers.
  - Engine input = numpy arrays.

  Algorithmic speed levers matter more than language:
  - pre-binned features (histogram)
  - time grid (`ntime`)
  - O(1) log-rank updates [S: Sverdrup et al. 2025]
  - index-based subject sampling
  - leaf counts stored on the grid rather than full curves
- [ ] Prediction API: `predict_survival_function(X_history, landmark=s)` vs a transformer-first workflow.
- [ ] v1 scope confirmation (G5).

## Sources
- Neunhoeffer & Sternberg 2019: https://www.cambridge.org/core/journals/political-analysis/article/how-crossvalidation-can-go-wrong-and-what-to-do-about-it/CA8C4B470E27C99892AB978CE0A3AE29
- Cerqueira, Torgo & Mozetič 2020: https://arxiv.org/pdf/1905.11744
- Bergmeir, Hyndman & Koo 2018: https://robjhyndman.com/publications/cv-time-series/
- Tanner et al. 2021 (landmarking + ML): https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssa.12611
- Landmark supermodel: https://github.com/thehanlab/dynamicLM/blob/main/tutorials/theory-landmark-supermodel.md
- Musoro et al. 2018: https://doi.org/10.1177/0962280216643563
- ViEWS2020: https://journals.sagepub.com/doi/10.1177/0022343320962157
- Muchlinski debate: https://www.cambridge.org/core/journals/political-analysis/article/comparing-random-forest-with-logistic-regression-for-predicting-classimbalanced-civil-war-onset-data-a-comment/B62CC1DA390C58435004D4C5D56DBF71
