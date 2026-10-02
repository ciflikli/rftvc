Rossi: does financial aid lower the hazard of rearrest?
=========================================================

**Data.** Rossi, Berk & Lenihan (1980) followed 432 men released from
Maryland prisons for one year, recording weekly employment status as they
were followed. Employment is a genuine time-varying covariate: whether a
man is employed changes from week to week, and only the weeks up to his last
observed one are known. ``carData::Rossi`` ships it wide (``emp1..emp52``);
``tests/fixtures/rossi.py`` reshapes it to counting-process rows, one per run
of unchanged employment status, with ``arrest`` as the event on the last row.

The data are downloaded on demand from ``carData``'s own upstream and
checksum-verified on first use, not committed with rftvc -- ``carData`` is
GPL >= 2 with no separate data-specific licence.

.. csv-table:: Subjects, rows and events
   :file: generated/rossi_summary.csv
   :header-rows: 1

**Models.** ``SurvivalForestTV`` (500 trees, ``min_ids_leaf=10``) and
``lifelines.CoxTimeVaryingFitter``, both on the same counting-process rows
and covariates (financial aid, age, race, work experience, marital status,
parole status, prior convictions, education, and the time-varying employment
status).

.. csv-table:: Cox coefficients (all covariates)
   :file: generated/rossi_cox.csv
   :header-rows: 1

**Relevance** (``permutation_importance``, ``oob=True`` -- a score-drop
magnitude, not a signed effect; see :doc:`../user_guide/importance`):

.. csv-table::
   :file: generated/rossi_relevance.csv
   :header-rows: 1

Employment status and age dominate; prior convictions' negative score under
both permutation and LOCO importance is possible for a covariate with a weak
signal relative to its noise. A negative score is not an error.

**Direction** (``hazard_effect`` two-point contrasts vs. the Cox
coefficient's sign; only meaningful for a binary or continuous covariate,
:doc:`../user_guide/importance`):

.. csv-table::
   :file: generated/rossi_directions.csv
   :header-rows: 1

**Reading.**

- All four checked directions agree between the forest and Cox, and all four
  match the well-replicated textbook result: financial aid, employment and
  age lower the hazard of rearrest; prior convictions raise it.
- This is a real, if informal, sanity check that ``hazard_effect`` on real
  data recovers the same qualitative story a standard Cox fit does, on a
  genuine time-varying covariate (employment status) -- not a claim that the
  forest improves on Cox here: no Cox concordance is computed on this page to
  compare against, only the forest's own out-of-bag concordance (0.687).

Reproduce with ``python -m examples.rossi_case_study`` from the repository
root (downloads a small file on first run).

References: P. H. Rossi, R. A. Berk and K. J. Lenihan (1980), *Money, Work,
and Crime: Some Experimental Evidence*, Academic Press.
