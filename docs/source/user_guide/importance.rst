Importance and effects
=======================

``rftvc.inspection`` answers four distinct questions about a fitted forest.
See :doc:`foundation` first for what the forest estimates.

.. list-table::
   :header-rows: 1

   * - Question
     - Use
   * - Does the model rely on this covariate, holding time fixed?
     - :func:`~rftvc.inspection.permutation_importance` (default: time-strata
       permutation, "M2")
   * - What is this covariate's predictive value for new subjects?
     - :func:`~rftvc.inspection.drop_column_importance` (cross-fitted, LOCO)
   * - Does the current-state assumption hold for a landmark model?
     - :func:`~rftvc.inspection.permutation_importance`'s level-vs-history
       diagnostic (below)
   * - What does the hazard/prediction look like as this covariate varies?
     - :func:`~rftvc.inspection.hazard_effect` / :func:`~rftvc.inspection.path_effect`

``permutation_importance``
---------------------------

The importance of a feature (or a jointly-permuted ``groups=`` unit) is the
drop in the piecewise-exponential score when its values are permuted among
the evaluation rows.

**The naive shuffle (M1) is biased for time-varying covariates and is not the
default.** Permuting a covariate across *all* rows regardless of time can
give a row at time ``t`` a value typical of some other time ``t'``. If the
covariate trends with time, that row lands off the observed ``(time, value)``
support -- in a region with no local event support -- and the resulting
"importance" measures extrapolation damage, not reliance on the covariate.
``strata="time"`` (the default) permutes only within time strata, keeping
every permuted row on the observed support; pass ``strata=None`` for the
naive M1 shuffle, for comparison only.

.. code-block:: python

   from rftvc import inspection

   result = inspection.permutation_importance(forest, X_test, y_test, ids=ids_test)
   result.importances_mean, result.importances_se       # id-cluster bootstrap SE
   result.importances_window                             # per scoring window

Use ``groups={"z": ["z", "z_lag1", "z_mean"]}`` to permute a raw covariate
with its derived features jointly, so their consistency is preserved.
``conditional_on=[...]`` crosses the strata with bins of other columns, for
conditional (subgroup) importance.

``drop_column_importance`` (LOCO)
-----------------------------------

Refits the estimator with and without each unit, cross-fitted over folds, and
scores the drop in held-out loss. This is the predictive value of a covariate
for *new* subjects, given everything else the model can learn from --
**unlike permutation importance, two strongly correlated covariates can
compensate for each other and both show low LOCO importance.** Report both
measures when covariates are correlated.

.. code-block:: python

   result = inspection.drop_column_importance(forest, X, y, ids=ids, cv=5)

The level-vs-history diagnostic (landmark models)
----------------------------------------------------

For a fitted ``LandmarkSurvivalForest`` / ``LandmarkCompetingRisksForest``,
:func:`~rftvc.inspection.permutation_importance` conditions on the current
value or on the history features to isolate which one drives the hazard --
and so tests the :doc:`foundation` page's current-state assumption directly:

.. code-block:: python

   # history-given-level: is there useful information in the trend beyond
   # today's value? A clearly positive importance says the current-state
   # model is missing information the history features would supply.
   inspection.permutation_importance(
       model, df, features=["z_mean", "z_slope"], conditional_on=["z"])

   # level-given-history: does the instantaneous value still matter once the
   # history summaries are held fixed?
   inspection.permutation_importance(
       model, df, features=["z"], conditional_on=["z_mean", "z_slope"])

Competing risks: cause-specific hazard vs cumulative incidence
--------------------------------------------------------------

A covariate can matter for a cause's cumulative incidence ``F_k`` purely
through a *competing* cause -- slowing cause 2 mechanically raises ``F_1``,
even if the covariate has no effect on ``lambda_1`` at all. The two questions
need different importance measures:

- **cause-specific hazard** ``lambda_k``: ``permutation_importance``/
  ``drop_column_importance`` with ``cause=k`` (the piecewise-exponential
  score, per cause);
- **cumulative incidence** ``F_k``: for a landmark model, ``scoring="brier"``
  or ``"ibs"`` with ``cause=k`` (there is no counting-process analogue --
  ``F_k`` is not itself a hazard).

The :doc:`../case_studies/importance` case study shows both computed on one
dataset and reports where they disagree.

``hazard_effect``
-------------------

A time-stratified partial dependence: the exposure-weighted average window
hazard as one covariate varies over a grid, holding every other covariate at
its observed value. Valid for internal and external covariates alike (it is
associational on the hazard scale) and shows time-varying (non-proportional)
effects directly.

.. code-block:: python

   effect = inspection.hazard_effect(forest, X_test, y_test, feature="z")
   effect["values"], effect.hazard, effect.support_mask

``effect["values"]`` (not ``effect.values`` -- see the function's docstring;
this one field shadows ``dict.values``).

``path_effect``
------------------

A prediction contrast along a covariate path: "shift ``z`` by ``delta`` from
``from_time`` onward", read off ``S`` or ``F_k`` at given horizons. This is a
**prediction under a specified path**, valid only for external covariates
(:doc:`foundation`'s internal/external distinction) -- it is **not a causal
effect** unless the covariate's effect on the hazard is unconfounded given
everything else in the model (Keogh & van Geloven, 2024).

.. code-block:: python

   effect = inspection.path_effect(
       forest, X_path, intervals, ids, feature="z", delta=1.0,
       from_time=3.0, horizons=[4.0, 5.0, 6.0])
   effect.mean

Not defined for landmark models (a path is a counting-process concept); use
``hazard_effect`` there instead.
