PBC2: dynamic prediction from repeated biomarkers
=================================================

Data: ``pbcseq`` from R's ``survival`` package. It holds 312 patients with
primary biliary cirrhosis and 1,885 clinic visits. Bilirubin, albumin and
prothrombin time are measured at each visit. Visits become counting-process
rows: a visit's measurements apply until the next visit. Death is the event;
transplant and end of follow-up are censoring.

**Question.** At *s* = 1, 2, 3 and 4 years after enrolment, which patients who
are still alive will die within the next 2 years?

**Models.** All models are fitted and evaluated in 5-fold cross-validation over
**patients** (``GroupKFold``, i.e. new patients):

- **Counting-process forest**: ``SurvivalForestTV`` on the visit rows (age,
  sex, treatment, edema, ascites, log bilirubin, albumin, prothrombin). At *s*
  it predicts along each patient's observed path up to *s*. After *s* the
  covariates are carried forward (``extrapolate="locf"``), since future visits
  are unknown at *s*.
- **Landmark super-model**: ``LandmarkSurvivalForest`` on landmarks
  0, 1, …, 4 years. Its features are the latest values, plus the maximum and
  first log bilirubin, and *s*.
- **Kaplan–Meier**: training patients' outcomes at *s*, no covariates.

**Metrics.** IPCW Brier score and cumulative/dynamic AUC at *w* = 2 years.
The censoring weights come from the test risk set at each landmark.

.. csv-table:: Mean over folds, by model
   :file: generated/pbc2_overall.csv
   :header-rows: 1

.. csv-table:: By landmark (counts are summed over folds)
   :file: generated/pbc2_landmarks.csv
   :header-rows: 1

.. csv-table:: Landmark super-model calibration (all test landmarks pooled; observed = 1 − Kaplan–Meier within bin)
   :file: generated/pbc2_calibration.csv
   :header-rows: 1

**Reading.**

- Both forests beat the no-covariate reference by a wide margin at every
  landmark.
- The landmark super-model has a slightly lower Brier score and a higher AUC
  than the counting-process forest. It is fitted directly for the target
  ``P(T <= s + w | T > s, H(s))`` and can use history summaries (such as the
  maximum bilirubin so far). The counting-process forest relies on the LOCF
  scenario after *s*.
- The counting-process forest answers a different question as well: survival
  along any specified covariate path.

Reproduce with ``python -m examples.pbc2_landmark`` from the repository root.
