scikit-learn compatibility
==========================

``SurvivalForestTV`` is a scikit-learn estimator.

- ``get_params`` / ``set_params`` / ``clone`` and pickling work.
- ``n_features_in_`` is set on fit, and ``feature_names_in_`` too when ``X`` is
  a DataFrame (pandas, polars, pyarrow, via narwhals).
- ``predict`` returns a risk score (the ensemble mortality, higher is riskier)
  and ``score`` returns a concordance index, so ``Pipeline``,
  ``cross_validate`` and ``GridSearchCV`` work.
- ``ids`` is fit metadata: with metadata routing enabled,
  ``set_fit_request(ids=True)`` (and ``set_score_request(ids=True)``) route it
  through meta-estimators, sliced to each fold.

The target ``y`` is a structured array (or a DataFrame) with ``start``,
``stop`` and ``event``, not a numeric vector. ``check_estimator`` checks that
build a numeric ``y`` therefore cannot fit a survival model. They are declared
as expected failures, and each is covered by a survival-adapted test with the
same intent. The table below is generated from that list in the test suite.
All other ``check_estimator`` checks pass.

.. csv-table:: Expected failures of ``check_estimator`` and their survival-adapted tests
   :file: generated/compat.csv
   :header-rows: 1

``LandmarkSurvivalForest`` fits a DataFrame and needs no ``y``, so it is outside
``check_estimator``'s scope. ``clone``, nested ``get_params`` (``forest__...``)
and pickling are tested directly.
