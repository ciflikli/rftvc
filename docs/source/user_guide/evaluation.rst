Choosing the error estimate
===========================

Which error you estimate depends on where the model will be used. rftvc makes
the choice explicit.

.. list-table::
   :header-rows: 1

   * - Target
     - Estimate with
   * - **New subjects** (same period)
     - ``oob_score=True`` (id-level out-of-bag concordance), or
       ``GroupKFold`` with ``groups=ids``
   * - **Future periods** (same population)
     - ``RollingOriginSplit(test_size, gap)``: train on earlier times, test on
       later ones
   * - **New subjects in future periods**
     - ``GroupTimeSplit``: group folds crossed with rolling time windows
   * - **Held-out periods of training subjects** (interpolation)
     - ``resample_unit="block"`` with ``oob_score=True`` (buffered block
       out-of-bag concordance; see below)

Out-of-bag and group splits keep all of a subject's rows on one side. Rolling
splits use calendar or landmark time. With landmark models, use ``gap >= horizon``
so that training outcomes cannot overlap the test period;
``landmark_cross_validate`` enforces this and also censors the training data at
the first test landmark.

.. code-block:: python

   from rftvc.model_selection import RollingOriginSplit, landmark_cross_validate

   results = landmark_cross_validate(
       model, df, RollingOriginSplit(3, test_size=12, gap=6),
       scoring=["brier", "cindex_cumulative"],
       param_grid={"forest__min_ids_leaf": [5, 15, 50]},            # nested CV
       inner_cv=RollingOriginSplit(2, test_size=12, gap=6))

Block resampling and its out-of-bag estimate
--------------------------------------------

With few subjects followed for a long time, whole-id resampling gives each tree
only about 63% of a handful of ids. ``resample_unit="block", block_length=L``
resamples an id's person-time in windows ``(kL, (k+1)L]`` instead. Rows are
split at window boundaries, which leaves every risk set, and so every split
score and leaf hazard, unchanged. ``min_ids_leaf`` and ``max_samples`` then
count blocks. Pass ``block_time=`` to ``fit`` to form blocks on another clock
(for example calendar time when the model's clock is a duration).
``LandmarkSurvivalForest`` passes the landmark times automatically.

The out-of-bag estimate then answers a different question. A row is scored
only by trees whose bag leaves out its blocks and ``oob_buffer`` neighbouring
blocks of the same subject on each side (default 1). This stops near-copies in
adjacent periods from leaking. The rest of the subject's history stays in the
bag, so the estimate is for **held-out periods of subjects the model was trained
on**: neither new-subject nor forecast error. Each extra buffer block cuts the
qualifying trees by roughly a further factor of 0.37 under the default
subsampling; check ``oob_n_trees_``.

With **landmark stacks**, landmarks less than a horizon apart share an outcome
window, so a buffer that reaches less than a horizon still leaks. Use
``block_length >= horizon`` with ``oob_buffer=1``;
``LandmarkSurvivalForest`` warns otherwise. On PBC2 (horizon 2 years) this gave
an out-of-bag C of 0.836 against 0.837 from ``GroupKFold`` by patient, while
``oob_buffer=0`` with half-year blocks gave 0.891 (``docs/bench/s10-block.md``).

.. code-block:: python

   model = SurvivalForestTV(resample_unit="block", block_length=12,
                            oob_score=True, n_estimators=1000).fit(X, y, ids)
   model.oob_score_, np.median(model.oob_n_trees_)

Metrics (``rftvc.metrics``)
---------------------------

All landmark metrics are computed per landmark, on the reset clock, at horizon
``w``.

- ``brier_landmark``: IPCW Brier score.

  - Cases have an event by ``w``; controls are event-free through ``w``
    (including administrative censoring at ``w``).
  - The censoring survival ``G`` is taken as a left limit and clipped at
    ``g_min``, and ``return_info=True`` reports how many weights were clipped.
  - Under complete follow-up no weights are needed (the exact path).

- ``integrated_brier``: the Brier score integrated over a time grid.
- ``cindex_dynamic``: ``kind="cumulative"`` is the cumulative/dynamic AUC at
  ``w``; ``kind="incident"`` is Uno's C truncated at ``w``.
- ``calibration_table``: observed (Kaplan–Meier) vs predicted risk by risk
  quantile.
- ``concordance_index_cp``: concordance for counting-process rows with
  time-varying risk scores. It equals R ``survival::concordance`` for
  ``(start, stop]`` data.

In ``landmark_cross_validate`` the censoring model ``G_s`` is a reverse
Kaplan–Meier fitted on each test landmark's risk set. It uses outcomes only,
never predictions.
