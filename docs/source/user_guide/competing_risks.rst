Competing risks
===============

With several event types (causes), only the first event is observed, so each
cause "competes" with the others. ``CompetingRisksForestTV`` handles this on the
same counting-process rows as ``SurvivalForestTV``: time-varying covariates,
delayed entry, id and block resampling, and coarsening all carry over.

Data
----

``event`` holds a **cause label**: 0 is censored, and any positive integer is
an event of that cause. Labels need not be contiguous. Only an id's last row
may carry an event, whatever its cause.

.. code-block:: python

   from rftvc import CompetingRisksForestTV, make_competing_risks_y

   y = make_competing_risks_y(stop, event_labels, start=start)   # 0 = censored
   model = CompetingRisksForestTV(causes=[1, 2]).fit(X, y, ids)
   model.causes_                                                # output axis order

Outputs are indexed by label, in ``causes_`` order. Pass ``causes=`` (the full
label vocabulary) whenever a training set might lack a cause, as in
cross-validation folds: the cause axis then keeps its shape, and a missing cause
gets a zero hazard (with a warning).

Targets
-------

Each leaf stores the delayed-entry Nelson–Aalen **cause-specific cumulative
hazard** of every cause. Predictions combine them with the discrete
Aalen–Johansen estimator:

- ``predict_cumulative_incidence``: the cumulative incidence
  ``F_k(t) = P(T <= t, cause k)``, shape ``(n, J, T)``, or ``(n, T)`` for one
  ``cause``;
- ``predict_survival_function``: the event-free survival ``S = 1 - sum_k F_k``.
  This is the product limit, **not** ``exp(-Λ)`` as in ``SurvivalForestTV``;
- ``predict_cumulative_hazard``: cause-specific hazards; ``cause="all"`` gives
  the all-cause hazard;
- ``predict``: ``F_k`` of ``score_cause`` at the last event time, the risk score
  behind ``score`` and ``oob_score_``.

Subdistribution (Fine–Gray) hazards are not modelled. On counting-process rows
the cause-specific hazards are the natural target, and the cumulative incidence
follows from them.

Covariate paths
---------------

As for survival, ``intervals`` / ``ids`` / ``origin`` / ``extrapolate`` predict
along a covariate path, conditional on being event-free at ``origin``. This is
valid **only for external (or specified) covariates**. For internal covariates
such as biomarkers, the path is not known in advance. For dynamic prediction
from such a history, use the landmark super-model:

.. code-block:: python

   from rftvc import LandmarkCompetingRisksForest

   lm = LandmarkCompetingRisksForest(horizon=730.5, landmarks=[365.25, 730.5, 1095.75],
                                     history_features=["log_bili", "albumin"], causes=[1, 2])
   lm.fit(df)                              # event column: 0 / 1 / 2
   lm.predict_risk(df_now, s=730.5, cause=2)   # F_2(s + w | event-free at s, history)

The clock is reset at each landmark, so the leaf Aalen–Johansen estimate is a
direct estimate of ``F_k(s + w | s, H(s))``.

Split rule and ensemble
-----------------------

- ``criterion="composite"`` (default) sums the per-cause LTRC log-rank
  chi-square statistics. It detects a covariate that raises one cause and lowers
  another. The all-cause log-rank cannot, because the signed per-cause
  numerators add up to the all-cause numerator exactly. So can the
  equal-weight composite of Ishwaran et al. (2014), as implemented in
  ``randomForestSRC``.
- ``split_cause=k`` splits on cause *k* alone, with other causes as censoring:
  a cause-specific forest, useful when one cause is the target.
  ``min_events_leaf_cause`` adds a floor on that cause's events per child.
- ``aggregate="cif"`` (default) runs the Aalen–Johansen estimator in each tree
  and averages the cumulative incidences. ``"hazard"`` averages the hazards first.

These defaults come from the S14 bake-off (``docs/bench/s14-cr.md``):

- ``composite`` beat the all-cause and Ishwaran rules by 12–26% in integrated
  squared error to the true cumulative incidence;
- the full quadratic form with cross-cause covariance was indistinguishable
  from it;
- ``"cif"`` aggregation helped when causes have opposing effects and was not
  worse elsewhere;
- a per-cause floor improved a rare target cause, but badly hurt the other
  causes. So set ``min_events_leaf_cause`` only when the rare cause *is* the
  target.

Evaluation
----------

- ``oob_score_`` and ``score`` use ``metrics.concordance_index_cr``, the
  cause-specific C of Wolbers et al. Each ``score_cause`` event is compared
  with the subjects still at risk, and with those who already had a competing
  event: they can no longer have this cause.
- Landmark metrics take ``cause=k``:

  - ``brier_landmark`` and ``integrated_brier`` (pass ``F_k``, not a survival
    probability): a competing event by the horizon is an observed "no", weighted
    by ``1/G(T-)``;
  - ``cindex_dynamic(kind="incident")``: the IPCW Wolbers C. Pairs with subjects
    still at risk are weighted ``1/G(T_i-)^2``, as in the survival version;
    pairs with competing events ``1/(G(T_i-) G(T_j-))``.

  Competing-risks AUC is not implemented.
- ``landmark_cross_validate`` accepts a ``LandmarkCompetingRisksForest``. It fixes
  one cause vocabulary and one scored cause before splitting, so every fold
  scores the same cause.
