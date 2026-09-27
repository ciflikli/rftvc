EBMT4: competing risks with a time-varying covariate
=======================================================

**Data.** 2,279 patients transplanted at European Society for Blood and
Marrow Transplantation (EBMT) centres, 1985-1998 (``mstate::ebmt4``). Two
competing causes of treatment failure: relapse, and death without relapse.
``ae`` is a generic adverse-event flag that occurs, for some patients,
strictly before the terminal event -- a genuine time-varying covariate.
``mstate``'s own documentation states the data are *simplified for
illustration, not clinical conclusions*: this page makes no clinical claim
about ``ae`` and draws no graft-versus-host-disease inference from it.
``tests/fixtures/ebmt4.py`` reshapes the wide table into counting-process
rows, splitting a patient's row at ``ae`` when it was observed.

The data are downloaded on demand from the package author's own upstream
repository and checksum-verified on first use, not committed with rftvc --
``mstate`` is GPL >= 2 with no separate data-specific licence (see
``docs/plans/rc-validation-findings.md`` for the fuller discussion).

.. csv-table:: Patients, rows and events
   :file: generated/ebmt4_summary.csv
   :header-rows: 1

**Models.** ``CompetingRisksForestTV`` (500 trees, ``min_ids_leaf=10``,
``causes=[1, 2]``) against two cause-specific ``lifelines.CoxTimeVaryingFitter``
fits (one per cause, censoring the other) -- the standard cause-specific-hazard
approach, comparable to what the forest's per-cause hazards target. Baseline
covariates (``year``, ``agecl``, ``proph``, ``match``) are ordinal-coded for
both fits with an explicit, fixed level order (the R factor levels
themselves, already ordinal for ``year``/``agecl``).

**Relevance, per cause** (``permutation_importance``/``drop_column_importance``,
``oob=True`` -- a score-drop magnitude, not a signed effect):

.. csv-table::
   :file: generated/ebmt4_relevance.csv
   :header-rows: 1

``agecl`` is the strongest baseline covariate for cause 2 (death without
relapse) by both measures -- plausible, since age at transplant is a
well-known driver of transplant-related mortality. ``ae``'s permutation
importance is ~0 for both causes but its LOCO importance is clearly
positive: the two measures can legitimately disagree (LOCO answers
"predictive value for a new patient given everything else"; permutation
answers "reliance holding time fixed"; :doc:`../user_guide/importance`).

**Direction** (``hazard_effect`` two-point contrasts vs. each cause-specific
Cox coefficient's sign):

.. csv-table::
   :file: generated/ebmt4_directions.csv
   :header-rows: 1

``year`` and ``agecl`` are 3-level *nominal* categories: Cox fits one linear
coefficient across all three levels, while the forest's ``hazard_effect``
makes no such linearity assumption, so a sign read at the two extreme codes
is not a same-contrast comparison for either (:doc:`../user_guide/importance`'s
caveat) -- ``gated`` marks the rows excluded from the agreement check for
exactly this reason, not hidden, just not counted. Every genuinely **binary**
covariate agrees between the forest and Cox.

**Is the forest picking up cause-specific structure at all?** ``ae``'s
``hazard_effect`` contrast (``ae=1`` minus ``ae=0``), per scoring window, for
each cause:

.. csv-table::
   :file: generated/ebmt4_ae_contrast.csv
   :header-rows: 1

**Reading.**

- Every binary covariate checked agrees in direction between the forest and
  Cox, on both causes. The one *disagreement* found (``agecl`` at cause 1) is
  exactly the kind the caveat above predicts for a >2-level nominal
  covariate, not a forest-vs-Cox modelling conflict.
- ``ae``'s per-window contrast is small and slightly negative for cause 1
  (relapse), and consistently positive and larger for cause 2 (death without
  relapse) -- clear evidence the forest is picking up genuinely
  cause-specific structure from the same shared trees, not evidence about
  ``ae``'s real clinical meaning (there is none claimed here).

Reproduce with ``python -m examples.ebmt4_case_study`` from the repository
root (downloads a small file on first run).

References: M. Fiocco, H. Putter and H. C. van Houwelingen (2008),
"Reduced-rank proportional hazards regression and simulation-based
prediction for multi-state models", *Statistics in Medicine* 27: 4340-4358.
L. C. de Wreede, M. Fiocco and H. Putter (2011), "mstate: An R Package for
the Analysis of Competing Risks and Multi-State Models", *Journal of
Statistical Software* 38(7).
