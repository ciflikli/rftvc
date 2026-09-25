"""Deliberately naive D8 coarsening reference (independent of the Rust code).

grid: sorted event-time quantiles; points = [earliest start] + grid.
g(t) = smallest point >= t, clamped to the last point. Per chain (an id's rows
sorted by start, or each row alone when stacked): drop rows with
g(start) == g(stop); a dropped event moves to the chain's previous kept row,
otherwise it is lost.
"""

import math

import numpy as np


def quantile_grid(stop, event, k):
    t = sorted(float(s) for s, e in zip(stop, event) if e)
    n = len(t)
    pts = [t[math.ceil(j * n / k) - 1] for j in range(1, k + 1)]
    return sorted(set(pts))


def coarsen_ref(start, stop, event, ids, k, stacked=False):
    """Returns (kept original indices, start', stop', event', grid, lost)."""
    grid = quantile_grid(stop, event, k)
    points = [min(start)] + [p for p in grid if p > min(start)]

    def g(t):
        for p in points:
            if p >= t:
                return p
        return points[-1]

    if stacked:
        chains = [[i] for i in range(len(start))]
    else:
        chains = {}
        for i, key in enumerate(ids):
            chains.setdefault(key, []).append(i)
        chains = [sorted(rows, key=lambda i: start[i]) for rows in chains.values()]
    kept, s_out, t_out, e_out, lost = [], [], [], [], 0
    for rows in chains:
        mine = []
        for i in rows:
            s, t = g(start[i]), g(stop[i])
            if s < t:
                kept.append(i)
                s_out.append(s)
                t_out.append(t)
                e_out.append(bool(event[i]))
                mine.append(len(e_out) - 1)
            elif event[i]:
                if mine:
                    e_out[mine[-1]] = True
                else:
                    lost += 1
    return np.array(kept), np.array(s_out), np.array(t_out), np.array(e_out, bool), np.array(grid), lost
