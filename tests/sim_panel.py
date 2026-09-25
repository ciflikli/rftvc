"""Simulated unit × period panel with a binary onset (a BTSCS-shaped example).

Data-generating process (committed; ``seed`` selects the replication):
- Units ``i = 0..n_units-1`` enter at period ``e_i ~ U{0..n_periods // 2}``.
- Baseline covariate ``z_i ~ N(0, 1)``; external covariate ``x_{i,t}``, AR(1):
  ``x_t = 0.7 x_{t-1} + N(0, 1)``, known at the start of period ``t``.
- Onset hazard on ``(t, t+1]``: ``0.012 * exp(0.8 x_t + 0.5 z_i + 0.6 * 1{x_t > 1})``.
- Units leave at their first onset (event), by dropout (probability 0.005 per
  period, censored) or at the end of the panel (``n_periods``).
- Rows: one per observed period, ``(id, start=t, stop=t+1, event, x, z)``.

Time is calendar time, so a rolling origin over landmarks tests future periods.
"""

import numpy as np
import polars as pl


def simulate_panel(n_units=300, n_periods=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        entry = int(rng.integers(0, n_periods // 2 + 1))
        z = rng.normal()
        x = rng.normal()
        for t in range(entry, n_periods):
            if t > entry:
                x = 0.7 * x + rng.normal()
            h = 0.012 * np.exp(0.8 * x + 0.5 * z + 0.6 * (x > 1))
            event = rng.random() < 1 - np.exp(-h)
            rows.append((i, float(t), float(t + 1), bool(event), x, z))
            if event or rng.random() < 0.005:
                break
    return pl.DataFrame(rows, schema=["id", "start", "stop", "event", "x", "z"], orient="row")
