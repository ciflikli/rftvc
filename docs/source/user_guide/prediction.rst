Prediction along covariate paths
================================

With fixed covariates, each row of ``X`` is one subject whose covariates hold
from time 0:

.. code-block:: python

   forest.predict_survival_function(X, times)       # (n_subjects, n_times)
   forest.predict_risk(X, horizon)                  # P(T <= horizon)
   forest.predict(X)                                # risk score (ensemble mortality)

With time-varying covariates, pass the rows of a **path** and their intervals:

.. code-block:: python

   forest.predict_survival_function(
       X_rows, times, intervals=y_rows, ids=subject, origin=None, extrapolate="none")

- Output has one row per subject, in order of first appearance.
- Each row's covariates apply on its ``(start, stop]``, and the ensemble
  cumulative hazard accumulates along the path.
- ``origin=u`` conditions on survival to ``u``: ``S(t | T > u)``. The default is
  the subject's first ``start``, which makes delayed entry exact. Times before
  the origin give NaN.
- Beyond the last ``stop`` the covariates are unknown, and the result is NaN
  unless you choose a scenario:

  - ``extrapolate="locf"``: the last covariates stay in force (a named assumption);
  - or append rows for the future path you want to evaluate. This is valid for
    external covariates.

Aggregation
-----------

``aggregate="hazard"`` (default) averages the trees' cumulative hazards, so that
``S = exp(-mean Λ_b)``. ``aggregate="survival"`` averages the trees'
survival curves instead. The hazard rule keeps conditional predictions
consistent across origins. In the aggregation benchmark (``docs/bench/s8-bakeoff.md``
in the repository), survival averaging calibrated slightly better on PBC2, but the
difference was within the bootstrap interval. It was no better on a simulated
panel and worse on a known-truth simulation, so ``"hazard"`` stays the default.
