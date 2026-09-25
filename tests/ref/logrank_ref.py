"""Deliberately naive LTRC log-rank reference (independent of the Rust code).

Risk set at event time t: rows with start < t <= stop. Events at t: rows with
event and stop == t. Hypergeometric variance with ties. O(n * K).
"""

import numpy as np


def logrank_ref(start, stop, event, left):
    start, stop = np.asarray(start, float), np.asarray(stop, float)
    event, left = np.asarray(event, bool), np.asarray(left, bool)
    num = var = 0.0
    for t in np.unique(stop[event]):
        at_risk = (start < t) & (t <= stop)
        dead = event & (stop == t)
        y, d = at_risk.sum(), dead.sum()
        yl, dl = (at_risk & left).sum(), (dead & left).sum()
        if y < 2 or d == 0:
            continue
        num += dl - d * yl / y
        var += d * (yl / y) * (1 - yl / y) * (y - d) / (y - 1)
    return num * num / var if var > 0 else 0.0


def nelson_aalen_ref(start, stop, event):
    """Delayed-entry Nelson–Aalen: (event_times, cumulative hazard just after each)."""
    start, stop, event = np.asarray(start, float), np.asarray(stop, float), np.asarray(event, bool)
    times = np.unique(stop[event])
    inc = [(event & (stop == t)).sum() / ((start < t) & (t <= stop)).sum() for t in times]
    return times, np.cumsum(inc)
