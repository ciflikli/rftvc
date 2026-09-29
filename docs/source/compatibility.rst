scikit-learn compatibility
==========================

``SurvivalForestTV`` and ``CompetingRisksForestTV`` are scikit-learn estimators.

- ``get_params`` / ``set_params`` / ``clone`` and pickling work.
- Both forest estimators accept NaN feature values during fit and prediction
  (``allow_nan=True``); positive and negative infinity remain invalid.
- ``n_features_in_`` is set on fit, and ``feature_names_in_`` too when ``X`` is
  a DataFrame (pandas, polars, pyarrow, via narwhals).

  - This accepts a polars ``X`` directly; it does not require converting to
    pandas first. If your own script or fixture-building code does need a
    polars-to-pandas conversion (e.g. to hand data to another library that
    only takes pandas), note that ``polars.DataFrame.to_pandas()`` itself
    requires ``pyarrow``, which is not an rftvc dependency. A pyarrow-free
    conversion: ``pd.DataFrame({c: d[c].to_numpy() for c in d.columns})``.
- ``predict`` returns a risk score (higher is riskier), and ``score`` returns a
  concordance index, so ``Pipeline``, ``cross_validate`` and ``GridSearchCV``
  work.

  - For ``SurvivalForestTV`` these are the ensemble mortality and the
    counting-process C.
  - For ``CompetingRisksForestTV`` they are ``F_k`` of ``score_cause`` and
    Wolbers' cause-specific C.

- ``ids`` is fit metadata: with metadata routing enabled,
  ``set_fit_request(ids=True)`` (and ``set_score_request(ids=True)``) route it
  through meta-estimators, sliced to each fold.

The target ``y`` is a structured array (or a DataFrame) with ``start``,
``stop`` and ``event``, not a numeric vector. ``check_estimator`` checks that
build a numeric ``y`` therefore cannot fit a survival model. They are declared
as expected failures, and each is covered by a survival-adapted test with the
same intent. The table below is generated from that list in the test suite.
Both classes run the same matrix (with id and block resampling), and the
adapted tests run for both. All other ``check_estimator`` checks pass.

.. csv-table:: Expected failures of ``check_estimator`` and their survival-adapted tests
   :file: generated/compat.csv
   :header-rows: 1

``LandmarkSurvivalForest`` and ``LandmarkCompetingRisksForest`` fit a DataFrame
and need no ``y``, so they are outside ``check_estimator``'s scope. ``clone``, nested ``get_params`` (``forest__...``)
and pickling are tested directly.

Support matrix
---------------

- Python: >= 3.10 (``requires-python = ">=3.10"``, no declared ceiling); CI tests
  3.10 and 3.13.
- scikit-learn: >= 1.6.
- Platforms: Linux (x86_64, aarch64), macOS (aarch64, x86_64), Windows (x64) —
  prebuilt abi3 wheels for each, built and smoke-tested in ``wheels.yml``.
- polars >= 1.0, numpy >= 1.24, narwhals >= 1.30.
