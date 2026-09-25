# Adversarial review of `plan.md` against approved `design.md` v2

## Key findings

1. **Finding:** S3's proposed log-rank oracle cannot run: `survival::survdiff()` rejects counting-process response data, so `survdiff(Surv(start, stop, event))` cannot validate the LTRC score. `Surv()` does create counting-process objects, but the `survdiff` implementation stops with “survdiff not defined for counting process data.” This leaves the principal D6/LTRC oracle unimplemented for the slice that introduces delayed entry and TVCs.
   **Reference:** `plan.md` S3, Tests (“The log-rank score equals `survdiff(Surv(start, stop, event))` fixtures”); `design.md` “Core algorithm: LTRC log-rank with histograms” and “Validation strategy,” Oracle 2.  
   **Confidence:** high. **Priority:** high.  
   **Concrete fix:** Replace this oracle with a committed, independently implemented counting-process reference that calculates the stated risk sets, events, numerator, and hypergeometric variance (including ties), and fixtures with checked expected scores. If retaining an R cross-check, use a counting-process-capable procedure only after pinning its version, tie method, and equality to the stated statistic; do not name `survdiff`.

2. **Finding:** The S5 IPCW API cannot meet the approved training-fold requirement. `brier_landmark(y_true_lm, risk, s, w, ...)` has no argument for the training-fold landmark-risk-set outcomes (or a prefit censoring estimator). Yet the design requires KM for censoring to be fit *within the training fold*, on the `U >= s` risk set and the time-since-`s` clock; the planned spy test likewise asserts a training-fold fit. The proposed scikit-survival parity test is therefore not obtainably specified either: `brier_score` requires separate `survival_train` and `survival_test` structured right-censored inputs.
   **Reference:** `plan.md` S5, Signatures and Tests; `design.md` “Model selection & metrics,” IPCW metrics bullets.  
   **Confidence:** high. **Priority:** high.  
   **Concrete fix:** Change the metric/cross-validation contract to pass `y_censor_train_lm` (or an explicitly prefit censoring estimator) separately from test landmark outcomes, and specify that both are restricted to `U >= s` and represented on the reset clock. Add a parity fixture that supplies the same train/test arrays and survival probabilities to this API and to `sksurv.metrics.brier_score`, with `G` safely above `g_min`; separately test truncation.

3. **Finding:** The Nelson--Aalen oracle is underspecified and can assert the wrong estimand at tied event times. `lifelines.NelsonAalenFitter` defaults `nelson_aalen_smoothing=True`, whereas the design’s leaf profile is the discrete increment `d_k/Y_k`. The plan neither disables smoothing nor states the timeline/step convention, despite testing ties in S1 and relying on the same oracle for delayed entry in S3. A 1e-10 comparison is consequently not a reliable acceptance criterion as written.
   **Reference:** `plan.md` S1, Tests; `plan.md` S3, Tests; `design.md` “Core algorithm: LTRC log-rank with histograms” (ties and discrete counts) and “Validation strategy,” Oracle 1.  
   **Confidence:** high. **Priority:** high.  
   **Concrete fix:** Pin the oracle call to `NelsonAalenFitter(nelson_aalen_smoothing=False)` with an explicit timeline and comparison convention, and add tied-event and delayed-entry fixtures. State whether the implementation returns cumulative hazard immediately after each event time and compare that same convention.

4. **Finding:** S6 does not fully test D8’s specified coarsening semantics. Equivalence to an unspecified Python “pre-snapped” input verifies neither the required handling of a same-bin event (move it to the prior interval) nor the first-interval event/drop rule against an independent expected result. The plan mentions an entry/event/censoring-in-bin fixture and a dropped-id diagnostic, but it does not assert the reassigned event or compare the constructed coarse data / resulting Nelson--Aalen estimate with the delayed-entry reference required by the approved design.
   **Reference:** `plan.md` S6, Tests; `design.md` “Time grid and coarsening (D8),” step 2 and fixture-test requirement.  
   **Confidence:** high. **Priority:** high.  
   **Concrete fix:** Add table-driven fixtures that assert the exact coarsened `(id, start', stop', event)` rows and diagnostic count for: a non-first collapsed event row, a first-row collapsed event, and in-bin entry/censoring. Then compare the resulting single-node cumulative hazard/risk sets with an independently constructed delayed-entry reference, not only another implementation of the snapping transform.

## Minor

1. **Finding:** D9 prediction coverage is incomplete: S3 tests an explicit conditional-survival identity and `extrapolate="none"` beyond the final stop, but not the required default origin (the id’s first `start`), the opt-in `"locf"` behavior, or scenario rows as the alternative future-covariate path.  
   **Reference:** `plan.md` S3, Signatures and Tests; `design.md` “Prediction,” items 3--4 (D9).  
   **Confidence:** high. **Priority:** medium.  
   **Concrete fix:** Add tests for `origin=None == first start`, explicit `origin`, default NaN, and a named `locf` scenario; document and test appended future scenario rows separately.

2. **Finding:** D12 is present in the S2 signature but lacks an acceptance test for its approved defaults/formula. In particular, no test verifies `min_ids_leaf=15`, `"auto" = max(15, sqrt(n_ids))`, or that leaf admissibility uses that selected value with straddling ids.  
   **Reference:** `plan.md` S2, Signatures; `plan.md` S3, straddling-id test; `design.md` “Dependence knobs” and D12.  
   **Confidence:** high. **Priority:** medium.  
   **Concrete fix:** Add boundary fixtures for `n_ids` below and above 225, and assert the selected `min_ids_leaf` and rejection/admission decision with an id straddling both children.

3. **Finding:** The S2 GBSG2 comparison is not reproducibly specified. C-index and IBS can be computed with the stated scikit-survival dev dependency rather than waiting for S5, so it is not necessarily a code dependency; however, “matched parameters” and a ±0.02 five-fold result omit fixed splits/seeds, preprocessing, evaluation-time grid, IBS implementation, and a mapping from `min_ids_leaf` to scikit-survival's row-based leaf settings. The result is measurable but likely brittle and not a stable merge gate.
   **Reference:** `plan.md` S2, Tests and Accept; `plan.md` S5, metrics introduced later; `design.md` “Validation strategy,” Oracle 3.  
   **Confidence:** high. **Priority:** medium.  
   **Concrete fix:** Keep the comparison in S2 using external metric functions, but commit fixed folds, preprocessing, seeds, estimator parameters, prediction times, and a tolerance rationale; make it a benchmark/regression report unless the two algorithms are deliberately made comparable at every relevant setting.

4. **Finding:** The S3 simulation acceptance criterion is underspecified and probabilistic: “integrated L2 error … below” gives no data-generating parameters, integration grid, forest/baseline tuning, seed policy, aggregation across the 20 replications, or uncertainty threshold. It is therefore not a deterministic CI acceptance test.
   **Reference:** `plan.md` S3, Simulation and Accept; `design.md` “Validation strategy,” simulations with known truth.  
   **Confidence:** high. **Priority:** medium.  
   **Concrete fix:** Commit the DGP and seeds, define the time integral and aggregation statistic (for example mean paired error with a predeclared confidence bound), and separate a slow statistical regression test from the normal unit-test gate.

5. **Finding:** The plan names `resample_unit="id"|"row"` only in the approved design, while S2 exposes id-only bags and S5 introduces the OOB restriction without a visible `resample_unit` parameter or tests for the row setting. This is a cross-slice API ambiguity rather than a necessary implementation dependency.
   **Reference:** `plan.md` S2, Signatures/Tests; `plan.md` S5, Signatures; `design.md` “Dependence knobs,” `resample_unit` and `oob_score`.  
   **Confidence:** medium. **Priority:** medium.  
   **Concrete fix:** Introduce and validate `resample_unit` in S2 (even if S2 implements only `"id"` initially), or explicitly defer `"row"` and remove it from the v1 public contract; in S5 test that row resampling rejects OOB.

