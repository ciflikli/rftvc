# Design Review: rftvc

## Key Findings

### 1. Quantile-grid event binning does not preserve the stated LTRC log-rank risk sets

**Confidence:** high  
**Priority:** high  
**Section:** `### Core algorithm: LTRC log-rank on a time grid with histograms` (lines 84–95); `### Dependence handling (knobs; defaults conservative)` (line 126)  
**Problem:** The formula defines risk at each grid point as `start < t_k <= stop`, but bins all events with `stop ∈ (t_{k-1}, t_k]`. When `ntime=100` combines multiple event times, a person who fails or is censored earlier in that bin is absent from `Y_k` even though their event is included in `d_k`; delayed entrants inside the bin have the converse timing problem. Thus the displayed log-rank score is exact on an all-unique-event-time grid, but is not the ordinary LTRC log-rank statistic on the advertised quantile grid (and can produce incoherent event/risk totals).  
**Fix:** For v1, make exact LTRC log-rank require the unique event-time grid, or derive and implement a grouped-time LTRC score with explicitly defined within-bin entry, event, and censoring handling. Add fixtures where an event/censor/entry occur within one coarse bin and compare both modes to a trusted delayed-entry calculation.

### 2. The proposed `LandmarkStacker → estimator` sklearn Pipeline cannot fit

**Confidence:** high  
**Priority:** high  
**Section:** `### Data views (polars transformers)` (lines 142–151); `### Prediction` (line 140); `## Architecture (Approach A)` (lines 61–64)  
**Problem:** `LandmarkStacker` emits a different number of rows and a new structured target (`start=0`, reset `stop`, and event), but a standard sklearn `Pipeline` transformer returns only transformed `X`; it does not return replacement `y` (or replacement `ids`). The pipeline will pass the original `y` to the estimator, whose length then disagrees with stacked `X`. Metadata routing routes named metadata; it does not make a transformer able to replace `y` as pipeline output.  
**Fix:** Do not present this as a standard sklearn Pipeline. Provide an explicit dataset-building API returning `(X_stacked, y_stacked, ids_stacked[, metadata])`, then fit the estimator; or implement a dedicated meta-estimator/CV wrapper that owns the joint transformation of X, y, ids, and splitter inputs. Test this public workflow, including GridSearchCV/CV, rather than relying on Pipeline compatibility.

### 3. Clock reset is conditionally valid, but the landmark outcome/censoring construction and role of `s` are underspecified

**Confidence:** high  
**Priority:** high  
**Section:** `### Data views (polars transformers)` (lines 143–149); `### Model selection & metrics` (lines 153–160)  
**Problem:** Resetting the clock removes delayed entry only after restricting to people event-free *and uncensored* at landmark `s`; it does not by itself make the original selection process disappear. Step 3 says `stop=min(T−s,w)` without defining whether `T` is event time or observed event/censoring time, or how censoring before `s+w` is encoded. Adding `s` lets a flexible supermodel condition on calendar/landmark time, but is not by itself a correction for landmark selection, censoring, or a guarantee that the effect of history/horizon varies appropriately with `s`. This conflicts with the more precise target in `research.md`, `## iii) Statistical implications`, lines 62–67 and 80–86.

**Fix:** Specify the landmark row as: include only `min(T_event, C) >= s`; set `stop=min(T_event, C, s+w)-s`; set `event=1{T_event <= min(C,s+w)}`; and state the conditional-independence assumptions. Define whether `s` is required, optional, or used with a stated supermodel/interaction strategy, and evaluate on the same `(s,w)` risk sets.

### 4. `row_weight="per_id"` is not an id-counting correction as defined and changes the survival estimand

**Confidence:** high  
**Priority:** high  
**Section:** `### Dependence handling (knobs; defaults conservative)` (lines 118–129); `## Risks & mitigations` (lines 173–181)  
**Problem:** With non-overlapping counting-process rows, an id already contributes at most one row to a risk set at a given time. Normalising all of an id's interval weights to sum to one makes a subject with many intervals contribute less at every time and gives its terminal event a fractional, row-count-dependent weight. It therefore is not simply “every count ... in subject units” as claimed in `## Executive summary` (lines 7–9); it is a different, undocumented weighted risk-set/event estimand. The overlapping-window option is likewise not defined for boundary overlap, varying horizons, or its interaction with `per_id`.

**Fix:** Retain unit subject weight in ordinary counting-process risk sets. If inverse-visit or overlap weighting is intended, expose it as an explicitly defined estimand with exact per-row formulas, mutually compatible/precedence rules, and weighted split/leaf validation; do not describe it as a neutral dependence correction.

## Minor Findings

### 1. Difference-array indices need strict-boundary definitions

**Confidence:** high  
**Priority:** medium  
**Section:** `### Core algorithm: LTRC log-rank on a time grid with histograms` (lines 84–91)  
**Problem:** “Add `+w` at `entry_idx` and `−w` at `exit_idx`” is ambiguous and, if those are the grid indices of `start` and `stop`, is off by one for `start < t_k <= stop`.  
**Fix:** Define `entry_idx` as the first grid index with `t_k > start` and the subtraction index as the first with `t_k > stop` (or give an equivalent inclusive convention), including no-contribution cases and boundary tests.

### 2. The displayed variance formula is only justified for unweighted integer risk/event counts

**Confidence:** high  
**Priority:** medium  
**Section:** `### Core algorithm: LTRC log-rank on a time grid with histograms` (lines 97–115); `### Dependence handling (knobs; defaults conservative)` (line 125)  
**Problem:** The finite-population log-rank variance in line 111 assumes the ordinary hypergeometric event allocation with integer `y`, `d`, and child risk count. The documented fractional `per_id`/`overlap` weights are fed into those quantities, but no weighted-score variance or intended heuristic status is specified.  
**Fix:** Either restrict v1's LTRC log-rank score to unit weights, or specify a valid weighted score/variance and its constraints. If it is solely a predictive heuristic under weights, say so and test its numerical behavior separately from the unweighted equivalence oracle.

### 3. The sibling-subtraction performance statement overclaims and omits its memory prerequisite

**Confidence:** high  
**Priority:** medium  
**Section:** `### Core algorithm: LTRC log-rank on a time grid with histograms` (lines 89–95)  
**Problem:** Building only the smaller child's histogram reduces row accumulation by at most half; `bins × K` initialization/prefix work remains, and subtraction requires retaining the parent histogram/profile for every evaluated feature. “This halves the work” is therefore not a general complexity result.  
**Fix:** State separate time and memory costs, including selected features and retained parent histograms; describe the saving as a bound on child row-accumulation work, then benchmark it.

### 4. Id-counted leaf size needs a precise rule when an id's rows route to both children

**Confidence:** high  
**Priority:** medium  
**Section:** `### Dependence handling (knobs; defaults conservative)` (lines 121–125)  
**Problem:** A time-varying id can have different interval covariates and hence appear in both children. The statement “leave ≥ this many ids on each side” does not say whether that id counts in both, is assigned once, or makes the split inadmissible. Those choices change admissible splits and the claimed dependence protection.  
**Fix:** Define the distinct-id set for each child and explicitly allow/disallow overlap; implement the corresponding exact distinct-id scan rather than inferring it from row histograms. Report both row and distinct-id counts in node diagnostics.

### 5. OOB validity is not determined by whether ids “look like” repeated units

**Confidence:** high  
**Priority:** high  
**Section:** `### Dependence handling (knobs; defaults conservative)` (lines 121–129); `## Risks & mitigations` (lines 173–181)  
**Problem:** All counting-process ids may recur over time, so that heuristic cannot identify a future-period deployment target. Id-level OOB is appropriate for new-id generalisation only when whole ids are the resampling units; it is not cleanly defined under the offered `"row"` or `"block"` modes, yet line 129 presents it as id-level OOB generally. This accords with `design-principles.md`, `## G2`, lines 43–48, which distinguishes new-subject OOB from temporal deployment.

**Fix:** Require an explicit deployment/estimand setting. Enable id-OOB only for whole-id resampling, disable it (or provide separately specified block-OOB) for row/block resampling, and require RollingOriginSplit/GroupTimeSplit for future-time claims.

### 6. LOCF is an unlabelled future-path assumption, not merely an extrapolation convenience

**Confidence:** high  
**Priority:** medium  
**Section:** `### Prediction` (lines 131–140)  
**Problem:** Default `extrapolate="locf"` asserts that future external covariates remain at their last value. Externality makes a survival probability conditional on a *specified* path meaningful; it does not establish that LOCF is the right future path. The data contract also permits gaps between non-overlapping rows without saying whether LOCF fills them. `research.md`, `## ii) How TVCs are handled`, line 56 makes this path assumption explicit.

**Fix:** Require callers to select/provide a scenario path for horizons outside (and gaps within) observed rows, with LOCF as an opt-in named scenario; otherwise return missing values. Document the prediction origin and conditioning interval.

### 7. IPCW defaults need their assumptions and landmark conditioning encoded in the API

**Confidence:** high  
**Priority:** medium  
**Section:** `### Model selection & metrics` (lines 153–160)  
**Problem:** Marginal KM IPCW is valid only under the relevant marginal independent-censoring assumption. For landmark scores it must also be based on the landmark risk set/conditional censoring survival; a generic `censoring_model=` does not state how it is trained per CV fold, what history it can use, or how zero/small weights are handled. `research.md`, `## iii) Statistical implications`, lines 80–86 identifies these qualifications.

**Fix:** Define the censoring estimand and fitting data for each metric, fit it within each training/evaluation fold, document the required independence assumption, and specify positivity handling (for example truncation plus a diagnostic). Keep the complete-follow-up path, but define “complete” per `(s,w)`.

### 8. Metadata routing does not make `ids` automatically usable by GroupKFold or CV search

**Confidence:** high  
**Priority:** medium  
**Section:** `### Data contract` (lines 76–81); `### Model selection & metrics` (lines 153–156); `## Risks & mitigations` (lines 173–181)  
**Problem:** Fit metadata named `ids` reaches an estimator only if it requests it. `GroupKFold` consumes `groups`, supplied to the splitter/CV machinery, not an estimator's fit-only `ids` metadata. After stacking, groups must also be expanded to the transformed rows. The statement “so `GroupKFold` works unchanged” is therefore ambiguous and is false for the proposed Pipeline path.  
**Fix:** Specify supported sklearn versions and `set_fit_request`/routing behavior; document a CV call that supplies splitter `groups` and estimator `ids` separately, and make the dataset/meta-estimator workflow expand both consistently.

### 9. Structured survival `y` requires an explicit sklearn-validation and estimator-check policy

**Confidence:** medium  
**Priority:** medium  
**Section:** `### Data contract` (lines 76–81); `## Architecture (Approach A)` (lines 61–64)  
**Problem:** A one-dimensional structured array with `start`, `stop`, `event` is a legitimate library contract, but it is outside the ordinary scalar/classification target assumptions exercised by many sklearn validation utilities and estimator checks. The document does not specify estimator tags, field/dtype coercion, missing-value handling, or which `check_estimator` checks are expected to pass/are deliberately unsupported.  
**Fix:** Add a compatibility matrix and test plan: custom structured-y validator; documented supported dtypes/field names; estimator tags and `n_features_in_`; explicit check_estimator exclusions with reasons, or adapters for checks that should pass.

### 10. Prediction does not define gaps, delayed prediction origin, or how leaf hazards are restricted to a requested horizon

**Confidence:** medium  
**Priority:** medium  
**Section:** `### Data contract` (lines 76–81); `### Prediction` (lines 131–140)  
**Problem:** Rows need only be non-overlapping, so gaps are allowed, but “sum over its rows” does not define the hazard during a gap. Nor does the API show a prediction origin/time grid argument, despite a cumulative hazard being indexed by absolute time in the core design.  
**Fix:** Require contiguous coverage over every requested prediction interval or define a gap policy/scenario; make origin, requested time grid/horizon, and conditioning survival explicit in each prediction method.

## Explicit Answer: Does the sklearn Pipeline API work with LandmarkStacker?

No. As designed, it does not work with a standard sklearn `Pipeline`: LandmarkStacker changes the number of rows and must create a new `y` and `ids`, while Pipeline transformers only replace `X` and forward the original `y`. Metadata routing cannot repair that output-contract mismatch. It can work only through an explicit pre-fit dataset transformation or a custom meta-estimator/CV workflow that jointly transforms X, y, ids, and groups.
