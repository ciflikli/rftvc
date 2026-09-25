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
