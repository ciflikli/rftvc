Importance and effects: cause-specific hazard vs cumulative incidence
=======================================================================

Data: the S14 competing-risks generator, scenario A (``bench/s14_cr_sim.py``).
``z`` (redrawn on each unit interval) drives cause 1's hazard; ``x0`` and
``x1`` drive cause 2's; three columns are pure noise. This shows, on one
simulated dataset, the :doc:`../user_guide/importance` point that
cause-specific-hazard importance and cumulative-incidence importance answer
different questions.

**Cause-specific-hazard importance** (``CompetingRisksForestTV``, the
piecewise-exponential score, permutation and LOCO):

.. csv-table:: Mean importance per event, by cause
   :file: generated/tvc_importance_cause_hazard.csv
   :header-rows: 1

``z`` dominates cause 1's importance by both measures, and is near zero for
cause 2, matching the generator (``z`` only enters cause 1's hazard).

**When does ``z`` matter?** The per-window decomposition of ``z``'s
permutation importance for each cause:

.. csv-table:: importances_window for feature "z"
   :file: generated/tvc_importance_windowed_z.csv
   :header-rows: 1

**The hazard effect of ``z`` on cause 1** (``hazard_effect``, exposure-weighted
window hazard on a grid of ``z``, other covariates held at their observed
values):

.. csv-table:: cause-1 window hazard as z varies
   :file: generated/tvc_importance_hazard_effect.csv
   :header-rows: 1

**Cumulative-incidence importance** (``LandmarkCompetingRisksForest``, IPCW
Brier score, ``scoring="brier"``): ``z``'s importance for its own cause,
``F_1``, is clearly positive. The general point from
:doc:`../user_guide/importance` -- that a covariate can matter for a
*competing* cause's ``F_k`` purely by changing how long subjects stay at risk
for it -- is a real mechanism, but it is not always a *large* one: here
``z``'s measured importance for ``F_2`` is indistinguishable from zero at
this sample size. The two importance measures still answer genuinely
different questions (one is on the cause-specific hazard, the other on the
absolute risk of an event of that cause), even when, as here, they happen to
agree that ``z`` does not matter much for cause 2.

.. csv-table:: cause-specific F_k importance of z (Brier)
   :file: generated/tvc_importance_cif_brier.csv
   :header-rows: 1

**The level-vs-history diagnostic**, on the same landmark fit
(``history_features=["z", ("z", "mean"), "x0", "x1"]``): the generator's
hazard depends on the instantaneous ``z``, not its history, so history-given-
level importance should be small and level-given-history should be the larger
of the two.

.. csv-table:: level-vs-history diagnostic for z
   :file: generated/tvc_importance_level_history.csv
   :header-rows: 1

**Timings** (800 training ids, 400 held-out, 300 trees):

.. csv-table::
   :file: generated/tvc_importance_timing.csv
   :header-rows: 1

Reproduce with ``python -m examples.tvc_importance_case_study`` from the
repository root.
