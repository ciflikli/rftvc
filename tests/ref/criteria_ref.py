"""Deliberately naive references for the S8 candidate split criteria (independent of the Rust code).

Risk set at event time t: rows with start < t <= stop. Events at t: rows with
event and stop == t. Children are ``left`` and ``~left``; each score is
child fit minus parent fit, computed from rows, not from profiles.
"""

import numpy as np


def _arrays(start, stop, event, left):
    return (np.asarray(start, float), np.asarray(stop, float),
            np.asarray(event, bool), np.asarray(left, bool))


def _xlogx(x):
    return x * np.log(x) if x > 0 else 0.0


def _grouped_ll(start, stop, event, mask, times):
    ll = 0.0
    for t in times:
        y = (mask & (start < t) & (t <= stop)).sum()
        d = (mask & event & (stop == t)).sum()
        # d ln(d/y) + (y - d) ln((y - d)/y), 0 ln 0 = 0
        ll += _xlogx(d) + _xlogx(y - d) - _xlogx(y)
    return ll


def grouped_lik_ref(start, stop, event, left):
    start, stop, event, left = _arrays(start, stop, event, left)
    times = np.unique(stop[event])
    everyone = np.ones_like(left)
    return (_grouped_ll(start, stop, event, left, times) + _grouped_ll(start, stop, event, ~left, times)
            - _grouped_ll(start, stop, event, everyone, times))


def _poisson_ll(start, stop, event, mask):
    d, e = event[mask].sum(), (stop - start)[mask].sum()
    return (d * np.log(d / e) if d > 0 else 0.0) - d


def poisson_ref(start, stop, event, left):
    start, stop, event, left = _arrays(start, stop, event, left)
    everyone = np.ones_like(left)
    return (_poisson_ll(start, stop, event, left) + _poisson_ll(start, stop, event, ~left)
            - _poisson_ll(start, stop, event, everyone))


def km_ref(start, stop, event, horizon):
    """Delayed-entry Kaplan–Meier at ``horizon`` (events at ``t <= horizon``)."""
    start, stop, event = np.asarray(start, float), np.asarray(stop, float), np.asarray(event, bool)
    s = 1.0
    for t in np.unique(stop[event]):
        if t > horizon:
            break
        y = ((start < t) & (t <= stop)).sum()
        s *= 1.0 - (event & (stop == t)).sum() / y
    return s


def km_gini_ref(start, stop, event, left, horizon, units=None):
    start, stop, event, left = _arrays(start, stop, event, left)
    units = np.arange(len(start)) if units is None else np.asarray(units)

    def impurity(mask):
        if not mask.any():
            return 0.0
        s = km_ref(start[mask], stop[mask], event[mask], horizon)
        return len(np.unique(units[mask])) * s * (1 - s)

    return impurity(np.ones_like(left)) - impurity(left) - impurity(~left)
