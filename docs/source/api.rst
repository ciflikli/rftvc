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
