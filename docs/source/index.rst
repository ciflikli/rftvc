rftvc
=====

Random survival forests for **time-varying covariates**, with a Rust engine and a
scikit-learn compatible Python API.

.. image:: ../../assets/readme/tvc-survival.png
   :alt: Predicted survival for two changing covariate paths with the same baseline
   :width: 48%

.. image:: ../../assets/readme/importance.png
   :alt: Cause 1 hazard permutation importance with standard-error bars
   :width: 48%

- Counting-process data ``(id, start, stop, event, X)`` with delayed entry
  (left truncation) and covariates that change over follow-up.
- Whole-subject resampling, id-level out-of-bag estimates, and leaf sizes
  counted in subjects, not rows.
- Landmark data building and a landmark super-model for dynamic prediction.
- Time-aware cross-validation and IPCW landmark metrics.
- An exact log-rank split criterion, and an opt-in coarse time grid for large
  data.

.. code-block:: python

   from rftvc import SurvivalForestTV, make_survival_y

   y = make_survival_y(stop, event, start=start)
   forest = SurvivalForestTV(n_estimators=500, random_state=0).fit(X, y, ids=ids)
   risk = forest.predict_risk(X_path, horizon, intervals=y_path, ids=ids_path)

.. toctree::
   :maxdepth: 2

   user_guide/index
   case_studies/index
   compatibility
   api
