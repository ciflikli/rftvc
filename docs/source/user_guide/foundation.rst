Statistical foundation
=======================

What the forest estimates, what it assumes, and what is (and is not) known
about it in theory.

The target
----------

Each row carries the covariates in force on ``(start, stop]``. The trees
split on the covariates only -- time is never a split variable -- and the
split statistic (the LTRC log-rank) compares two candidate children at each
event time among the rows at risk then. A leaf estimates the
**at-risk-averaged hazard**, ``lambda_A(t) = E[lambda(t, X(t)) | X(t) in A, at
risk at t]``, by a Nelson-Aalen curve over the leaf's own rows. As leaves
shrink, they heuristically target the **hazard map** ``(t, x) -> lambda(t,
x)``: the hazard at time ``t`` given the covariate state in force. The
competing-risks forest does the same per cause, giving cause-specific hazard
maps ``lambda_1(t, x), ..., lambda_J(t, x)``.

A leaf's cumulative hazard is flat between its own in-bag event times (it
carries the last value forward), so a leaf's *shape* over time partly
reflects which rows are at risk there -- a composition-drift effect, not a
bias in the target.

Assumptions
-----------

1. **Current-state dependence.** The target is always well defined as "the
   hazard given the row's covariates". It equals the *full-history* hazard
   only if the row captures all relevant history -- current values plus any
   lags or summaries added as columns. :doc:`importance` shows a diagnostic
   for this.
2. **Predictability.** A row's covariates must be known at its ``start`` (no
   lookahead); ``measured_at`` checks this for counting-process rows, and the
   landmark module enforces it for history features built up to a landmark.
3. **Independent censoring and entry** given the covariate history (the
   standard Andersen-Gill conditions).
4. **Ids are the independent resampling unit.** A subject's own rows are
   dependent, but the counting-process likelihood factorises over time
   (martingale increments), so uncertainty and out-of-bag estimation need
   id-level (or id-block) resampling, which the forest already does.

Prediction functionals
-----------------------

.. list-table::
   :header-rows: 1

   * - Call
     - Quantity
     - Valid as a survival probability when
   * - ``predict_*`` without ``intervals``
     - ``Lambda(t | x held fixed from 0)``
     - ``x`` is time-fixed, or as a named "fixed profile" scenario
   * - ``predict_*(intervals=...)``
     - ``Lambda(t | path) - Lambda(origin | path)``
     - the covariates are **external** and the path is observed or specified
   * - ``predict()`` / ``oob_prediction_`` / ``oob_score_``
     - mortality ``sum_k Lambda(t_k | x)`` over the fixed profile
     - a ranking of covariate *states*, not of subjects
   * - landmark ``predict_risk(df, s, w)``
     - ``P(T <= s + w | T > s, H(s))``
     - internal or external covariates, censoring independent given
       ``H(s)`` and ``s``, adequate support, and a model that transports to
       the prediction population

For **internal** covariates, the hazard map is still estimable and
interpretable, but a path-based survival curve is not a probability for any
real subject: the future path depends on outcomes the intervention would
itself change. This internal/external distinction governs which prediction
calls, and which of :doc:`importance`'s effect functions, apply.

Theory status
-------------

Adjacent results exist for related models: RSF consistency for time-fixed
covariates (Ishwaran & Kogalur, 2010), a splitting-bias analysis under
censoring (Cui, Zhu, Zhou & Kosorok, 2022), and LTRC survival trees built on
pseudo-subjects (Fu & Simonoff, 2017). **No consistency result was found for
counting-process TVC forests.** The current-state assumption above is
checkable, not provable, with the level-vs-history diagnostic of
:doc:`importance`.

Internal vs external covariates
--------------------------------

An **external** covariate's future path does not depend on the subject's
outcome (e.g. an environmental exposure); an **internal** one does (e.g. a
biomarker that worsens as disease progresses). Fixed-profile predictions and
the hazard-scale effect of :func:`rftvc.inspection.hazard_effect` are valid
for both. Path-based predictions (``predict_*(intervals=...)``) and
:func:`rftvc.inspection.path_effect` are a *prediction under a specified
path*, valid only when that path is externally determined -- for an internal
covariate, supplying a future path pre-empts the very outcome being
predicted.
