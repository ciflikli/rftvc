"""``hazard_effect`` / ``path_effect`` core."""

import numpy as np

from ..metrics import _window_exposure_1


def default_values(x, n=20):
    """20 quantiles of ``x``, deduplicated (fewer than 20 under heavy ties)."""
    return np.unique(np.quantile(x, np.linspace(0.0, 1.0, n)))


def hazard_grid(Xe, start, stop, predict, w, feature, values, kind, competing, cause_idx,
                category_group=None, category_levels=None):
    """Exposure-weighted window hazard of ``feature`` at ``values``, per window.

    ``predict(Xr, rows)`` returns the fixed-profile cumulative hazard of ``Xr``
    at the edges ``w``: ``(n, M+1)`` (survival or one selected cause) or
    ``(n, J, M+1)`` (competing risks, ``cause_idx=None``).

    Returns ``(hazard, support_mask, individual)``:
    - ``hazard``: ``(n_values, M)`` or CR ``(n_values, J, M)``.
    - ``support_mask``: ``(n_values, M)`` (numeric ``v`` inside the 5-95%
      range, or categorical ``v`` observed among rows at risk in that window;
      ``False`` when no row is at risk).
    - ``individual``: ``None`` unless ``kind="individual"``, else the
      per-row rate, ``(n_rows, n_values, M)`` or CR ``(n_rows, n_values, J, M)``,
      ``NaN`` where the row has no exposure in that window.
    """
    n = Xe.shape[0]
    M = w.size - 1
    n_values = len(values)
    categorical = category_group is not None
    x = None if categorical else Xe[:, feature]
    # observed-support bounds: one window's exposure array alive at a time (O(n), not O(n*M) --
    # the array itself is not retained; recomputed per (value, window) below for the same reason).
    support_lo = np.full(M, np.nan)
    support_hi = np.full(M, np.nan)
    for m in range(M):
        e = _window_exposure_1(start, stop, w[m], w[m + 1])
        at_risk = e > 0
        if at_risk.any() and not categorical:
            support_lo[m], support_hi[m] = np.quantile(x[at_risk], [0.05, 0.95])

    hazard = None
    individual = None
    support_mask = np.zeros((n_values, M), dtype=bool)
    for i, v in enumerate(values):
        Xr = Xe.copy()
        if categorical:
            level = category_levels.index(v)
            Xr[:, category_group] = 0.0
            Xr[:, category_group[level]] = 1.0
        else:
            Xr[:, feature] = v
        H = predict(Xr, np.arange(n))
        if competing and cause_idx is None:
            J = H.shape[1]
            if hazard is None:
                hazard = np.zeros((n_values, J, M))
                if kind == "individual":
                    individual = np.full((n, n_values, J, M), np.nan)
        else:
            if competing:
                H = H[:, cause_idx, :]
            if hazard is None:
                hazard = np.zeros((n_values, M))
                if kind == "individual":
                    individual = np.full((n, n_values, M), np.nan)
        for m in range(M):
            e = _window_exposure_1(start, stop, w[m], w[m + 1])
            width = w[m + 1] - w[m]
            at_risk = e > 0
            if competing and cause_idx is None:
                d = H[:, :, m + 1] - H[:, :, m]  # (n, J)
                rate = d / width
                if kind == "individual":
                    individual[at_risk, i, :, m] = rate[at_risk]
                if at_risk.any():
                    hazard[i, :, m] = (e[at_risk, None] * rate[at_risk]).sum(axis=0) / e[at_risk].sum()
                else:
                    hazard[i, :, m] = np.nan
            else:
                d = H[:, m + 1] - H[:, m]  # (n,)
                rate = d / width
                if kind == "individual":
                    individual[at_risk, i, m] = rate[at_risk]
                if at_risk.any():
                    hazard[i, m] = (e[at_risk] * rate[at_risk]).sum() / e[at_risk].sum()
                else:
                    hazard[i, m] = np.nan
            if categorical:
                support_mask[i, m] = bool(at_risk.any()) and bool((Xe[at_risk, category_group[level]] == 1).any())
            else:
                support_mask[i, m] = bool(at_risk.any()) and (support_lo[m] <= v <= support_hi[m])
    return hazard, support_mask, individual


def split_at(X, start, stop, offsets, from_time):
    """Split, per subject (delimited by ``offsets``), the row straddling ``from_time``.

    A row with ``start < from_time < stop`` becomes two rows ``(start, from_time)``
    and ``(from_time, stop)``, both carrying the original row's covariates.
    Returns ``(X, start, stop, offsets)`` with the same subject order and, for a
    split subject, one extra row. Subjects with no straddling row are unchanged.
    """
    out_X, out_start, out_stop, out_offsets = [], [], [], [0]
    for i in range(offsets.size - 1):
        lo, hi = offsets[i], offsets[i + 1]
        for r in range(lo, hi):
            s, e = start[r], stop[r]
            if s < from_time < e:
                out_X.append(X[r])
                out_start.append(s)
                out_stop.append(from_time)
                out_X.append(X[r])
                out_start.append(from_time)
                out_stop.append(e)
            else:
                out_X.append(X[r])
                out_start.append(s)
                out_stop.append(e)
        out_offsets.append(len(out_X))
    return (
        np.asarray(out_X),
        np.asarray(out_start, dtype=float),
        np.asarray(out_stop, dtype=float),
        np.asarray(out_offsets, dtype=np.int64),
    )


def shift_from(X, start, feature, from_time, delta):
    """``X`` with ``feature`` shifted by ``delta`` on rows with ``start >= from_time``."""
    Xs = X.copy()
    mask = start >= from_time
    if callable(delta):
        Xs[mask, feature] = delta(Xs[mask, feature], start[mask])
    else:
        Xs[mask, feature] = Xs[mask, feature] + delta
    return Xs
