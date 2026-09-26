The time grid: exact and coarse
===============================

Splits maximise the **log-rank statistic for left-truncated, right-censored
data**. It is computed over a time grid. The S8 bake-off compared it with a
grouped-time likelihood, a Poisson (person-time) likelihood and a Kaplan–Meier
impurity, under nested cross-validation. None improved Brier score or
calibration (``docs/bench/s8-bakeoff.md`` in the repository), so log-rank is the
only criterion.

``ntime=None`` (default, exact)
    The grid is every distinct event time. Split search costs about
    ``O(n_node + bins × K_node)`` per candidate feature, where ``K_node`` is the
    number of event times in the node.

``ntime=K`` (coarse)
    The grid is ``K`` quantiles of the event times, plus the earliest entry as
    the origin. Every ``start`` and ``stop`` is rounded **up** to the next grid
    point *before* counting, so the statistic is the exact log-rank on the
    coarsened rows. Consequences:

    - an event or censoring inside a bin counts at the bin's end;
    - an entry inside a bin counts from the following grid point;
    - a row that starts and ends inside one bin is dropped, and its event moves
      to the subject's previous row (``n_coarsen_lost_events_`` counts events
      with no such row).

    The fitted grid is ``coarse_grid_``.

On 100k to 1M rows, ``ntime=100`` fitted 4–6× faster than exact mode
with the same test concordance (see ``docs/bench/s6-perf.md`` in the
repository). Exact mode remains the default until wider benchmarks justify a
change.
