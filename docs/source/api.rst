API reference
=============

.. currentmodule:: rftvc

Estimators
----------

.. autosummary::
   :toctree: generated/api
   :nosignatures:

   SurvivalForestTV
   CompetingRisksForestTV
   LandmarkSurvivalForest
   LandmarkCompetingRisksForest

Data
----

.. autosummary::
   :toctree: generated/api
   :nosignatures:

   make_survival_y
   check_survival_y
   make_competing_risks_y
   check_competing_risks_y
   check_counting_process
   make_landmark_data
   landmark_features
   LandmarkData

.. data:: CR_DTYPE

   Structured ``numpy.dtype`` of a competing-risks target: fields ``start`` (f8),
   ``stop`` (f8), ``event`` (i8, a cause label; 0 = censored). See
   :func:`make_competing_risks_y`.

.. data:: SURV_DTYPE

   Structured ``numpy.dtype`` of a survival target: fields ``start`` (f8),
   ``stop`` (f8), ``event`` (bool). See :func:`make_survival_y`.

Model selection
---------------

.. currentmodule:: rftvc.model_selection

.. autosummary::
   :toctree: generated/api
   :nosignatures:

   RollingOriginSplit
   GroupTimeSplit
   landmark_cross_validate

Inspection
----------

.. currentmodule:: rftvc.inspection

.. autosummary::
   :toctree: generated/api
   :nosignatures:

   permutation_importance
   drop_column_importance
   hazard_effect
   path_effect

Metrics
-------

.. currentmodule:: rftvc.metrics

.. autosummary::
   :toctree: generated/api
   :nosignatures:

   brier_landmark
   integrated_brier
   cindex_dynamic
   calibration_table
   concordance_index_cp
   concordance_index_cr
   piecewise_exponential_score
   event_windows
   PEScore
   KaplanMeierCensoring
   UndefinedMetricError

Visualization
-------------

Optional (``pip install rftvc[viz]``); every function renders an existing rftvc
return value. The five Altair charts need the extra; :func:`~rftvc.viz.plot_tree`
does not (it returns a plain SVG string).

.. currentmodule:: rftvc.viz

.. autosummary::
   :toctree: generated/api
   :nosignatures:

   plot_survival_curve
   plot_cumulative_incidence
   plot_importance
   plot_hazard_effect
   plot_calibration
   plot_tree
