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
   KaplanMeierCensoring
   UndefinedMetricError
