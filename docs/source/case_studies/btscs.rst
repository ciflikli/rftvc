BTSCS: how long do wars last?
=============================

**Data.** Cunningham and Lemke (2013) combine civil and interstate wars
(1946–2008) in one war-year panel. This binary time-series cross-section (BTSCS)
has covariates that change each year: troop numbers, the troop ratio,
population and democracy. The same data underlie the conflict-duration models
in Ciflikli (2018), *Learning conflict duration: insights from predictive
modelling* (PhD thesis, LSE).

The replication archive is downloaded from the first author's website on first
use and checked against a pinned SHA-256. It states no licence, so rftvc does
not redistribute it (``examples/data/cunningham_lemke.py``).

**From war-years to counting-process rows.** Preprocessing follows the authors'
``stset clenddate, id(CLID) origin(clstartdate) failure(clend==1)`` with
``stcox``:

- each war-year covers ``(clstartdate, clenddate]``;
- time is measured in days since the war's onset. We read the onset as the
  war's earliest ``clstartdate``, because ``clstartdate`` varies within a war
  in the archive;
- in 7 wars a later year's row repeats the episode start and overlaps earlier
  rows, so it is made to start the day after the previous row ends;
- war-years with a missing covariate are dropped (listwise, as ``stcox`` does).
  The resulting gaps are not at risk (``gap_policy="split_id"``). Each war stays
  one resampling unit.

.. csv-table:: Preprocessing
   :header-rows: 1

   "wars", "war-years", "terminations", "rows with a missing covariate", "rows kept", "wars kept", "terminations kept", "chains (segments)"
   382, 1957, 341, 371, 1586, 280, 236, 334

**The paper's model.** Cox proportional hazards with the time-varying
covariates (lifelines, all kept rows). A positive coefficient means a shorter
war.

.. csv-table::
   :file: generated/btscs_cox.csv
   :header-rows: 1

**New-war cross-validation.** The splits are 5-fold ``GroupKFold`` by war. The
metric is the counting-process concordance: at each war end in the test fold,
that war-year is compared with the war-years of the other test wars still
ongoing. Models: the forest (500 trees, ``min_ids_leaf=10``) and the Cox model,
both on the same covariates.

.. csv-table::
   :file: generated/btscs_cv.csv
   :header-rows: 1

On the same fit, the forest's id-level out-of-bag concordance is 0.626.

**Reading.**

- With these seven covariates, the Cox model discriminates somewhat better than
  the forest (C = 0.654 vs 0.636). The effects look close to log-linear, and
  280 wars is a small sample for a forest.
- The forest's value here is in the modelling it allows without specifying the
  functional form:
  - survival along a war's actual covariate path;
  - conditional predictions from any point in a war (``origin``);
  - honest new-war and future-period evaluation.

**Path predictions.** The table shows P(war still ongoing at *t*) along each
war's observed covariate path, from onset, for the longest civil and
interstate wars observed from onset without a gap.

.. csv-table::
   :file: generated/btscs_paths.csv
   :header-rows: 1

Reproduce with ``python -m examples.btscs_war_duration`` from the repository
root (downloads about 1 MB on first run).

References: D. E. Cunningham and D. Lemke (2013), "Combining Civil and
Interstate Wars", *International Organization* 67(3): 609–628. G. Ciflikli
(2018), *Learning conflict duration: insights from predictive modelling*, PhD
thesis, London School of Economics.
