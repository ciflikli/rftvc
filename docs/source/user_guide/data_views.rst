Three views of the data
=======================

``SurvivalForestTV`` reads one table shape: **counting-process rows**. The
three common data views below all map onto it.

One row per subject (right-censored)
------------------------------------

Covariates are measured once. Each subject has one row with ``start = 0``,
``stop`` = its event or censoring time, and ``event``. This is the classic
random survival forest setting.

.. code-block:: python

   from rftvc import SurvivalForestTV, make_survival_y

   y = make_survival_y(time, event)            # start defaults to 0
   forest = SurvivalForestTV().fit(X, y)

Counting-process rows (time-varying covariates, delayed entry)
--------------------------------------------------------------

Each subject has several rows ``(start, stop]``. The covariates on a row are
those in force over that interval and must be **known at its start** (pass
``measured_at`` to have this checked). Rules per subject, enforced by
``check_counting_process``:

- rows are contiguous and do not overlap (``stop_j == start_{j+1}``);
- only the last row may carry the event;
- a first ``start > 0`` is delayed entry (left truncation): the subject joins
  the risk set only after it.

Gaps are an error by default. ``gap_policy="split_id"`` treats a gap as time
not at risk, with re-entry after it. The segments are then separate chains, but
the subject stays **one resampling unit**, so bootstrap/subsampling, leaf sizes
and out-of-bag estimates remain per subject.

.. code-block:: python

   y = make_survival_y(stop, event, start=start)
   forest = SurvivalForestTV().fit(X, y, ids=subject_id)
   # or, with a DataFrame X holding the id column:
   forest = SurvivalForestTV().fit(df[features + ["subject"]], y, ids="subject")

Trees split rows. One subject's rows can therefore fall on both sides of a
split, and it then counts towards ``min_ids_leaf`` in both children.

Categorical covariates
----------------------

Both counting-process forest estimators accept string-valued columns in ``X``
and DataFrame categorical dtypes (including numeric category labels). At fit, each
observed level becomes a 0/1 indicator. The fitted vocabulary is reused by
``predict``, ``apply``, and covariate-path prediction; an unseen level raises
``ValueError``. Missing category values become NaN in all of that column's
indicators, allowing the tree to learn their route. Numeric columns remain
continuous; mark numeric category codes with a categorical dtype or convert
them to strings if they represent unordered levels.

The encoding is one-hot, so a column with ``K`` observed levels contributes
``K`` split features. ``max_features`` samples these indicators separately,
and ``export_tree`` reports their names as ``column=level``. This supports
single-level-versus-rest splits; it does not search all multi-level partitions
as ranger's ``respect.unordered.factors="partition"`` mode does. For columns
with many rare levels, group levels before fitting to control width.

.. code-block:: python

   X = df[["age", "treatment"]]  # treatment contains labels such as "A" and "B"
   forest = SurvivalForestTV().fit(X, y)
   risk = forest.predict_risk(X.iloc[:5], horizon=365)

Stacked landmark data
---------------------

For dynamic prediction, ``P(T <= s + w | T > s, history up to s)``, the
landmark view builds one row per (subject, landmark ``s``):

- the clock is reset to ``s``;
- the features summarise the history up to ``s`` (``history_features``: last
  value, or ``(column, agg)`` with ``first``, ``mean``, ``min``, ``max``,
  ``sum``, ``count``);
- the outcome is administratively censored at ``s + w``.

The builder only aggregates rows with ``start <= s``, so a feature cannot look
ahead.

.. code-block:: python

   from rftvc import LandmarkSurvivalForest, make_landmark_data

   data = make_landmark_data(df, horizon=730, step=365,
                             history_features=["bili", ("bili", "max")])
   model = LandmarkSurvivalForest(horizon=730, step=365,
                                  history_features=["bili", ("bili", "max")]).fit(df)
   model.predict_risk(df_now, s=1095)

A subject at risk at several landmarks appears several times in the stack.
Resampling keeps all of a subject's landmark rows together. The estimand is the
landmark super-model's, in which subjects at risk at many landmarks carry more
weight.
