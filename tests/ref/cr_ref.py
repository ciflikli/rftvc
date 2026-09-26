"""Deliberately naive competing-risks references (independent of the Rust code).

Risk set at time t: rows with start < t <= stop. Cause-j events at t: rows with
code j and stop == t. Codes are 1..J (0 = censored).
"""

import numpy as np


def composite_ref(start, stop, codes, left, n_causes):
    """Composite LTRC log-rank ``sum_j U_j^2 / V_j`` (causes with ``V_j > 0``)."""
    start, stop = np.asarray(start, float), np.asarray(stop, float)
    codes, left = np.asarray(codes), np.asarray(left, bool)
    total = 0.0
    for j in range(1, n_causes + 1):
        num = var = 0.0
        for t in np.unique(stop[codes == j]):
            at_risk = (start < t) & (t <= stop)
            dead = (codes == j) & (stop == t)
            y, d = at_risk.sum(), dead.sum()
            yl, dl = (at_risk & left).sum(), (dead & left).sum()
            if y < 2:
                continue
            num += dl - d * yl / y
            var += d * (yl / y) * (1 - yl / y) * (y - d) / (y - 1)
        if var > 0:
            total += num * num / var
    return total


def aalen_johansen_ref(start, stop, codes, n_causes):
    """Delayed-entry per-cause Nelson–Aalen and Aalen–Johansen at every event time.

    Returns ``(times, cumhaz (K, J), cif (K, J), surv (K,))``, each value just after its time.
    """
    start, stop, codes = np.asarray(start, float), np.asarray(stop, float), np.asarray(codes)
    times = np.unique(stop[codes != 0])
    cumhaz = np.zeros((times.size, n_causes))
    cif = np.zeros((times.size, n_causes))
    surv = np.zeros(times.size)
    h, f, s = np.zeros(n_causes), np.zeros(n_causes), 1.0
    for i, t in enumerate(times):
        y = ((start < t) & (t <= stop)).sum()
        dh = np.array([((codes == j) & (stop == t)).sum() / y for j in range(1, n_causes + 1)])
        h = h + dh
        f = f + s * dh
        s = s * max(0.0, 1.0 - dh.sum())
        cumhaz[i], cif[i], surv[i] = h, f, s
    return times, cumhaz, cif, surv



def cause_moments_ref(start, stop, codes, left, n_causes):
    """Numerators ``U_j`` and the multivariate hypergeometric covariance ``V`` over times with ``y >= 2``."""
    start, stop = np.asarray(start, float), np.asarray(stop, float)
    codes, left = np.asarray(codes), np.asarray(left, bool)
    U, V = np.zeros(n_causes), np.zeros((n_causes, n_causes))
    for t in np.unique(stop[codes != 0]):
        at_risk = (start < t) & (t <= stop)
        y, yl = at_risk.sum(), (at_risk & left).sum()
        if y < 2:
            continue
        f = yl / y
        d = np.array([((codes == j) & (stop == t)).sum() for j in range(1, n_causes + 1)], float)
        dl = np.array([((codes == j) & (stop == t) & left).sum() for j in range(1, n_causes + 1)], float)
        U += dl - d * f
        V += f * (1 - f) / (y - 1) * (np.diag(d * y) - np.outer(d, d))
    return U, V


def quadratic_ref(start, stop, codes, left, n_causes):
    U, V = cause_moments_ref(start, stop, codes, left, n_causes)
    if not V.any():
        return 0.0
    return float(U @ np.linalg.pinv(V, rcond=1e-10, hermitian=True) @ U)


def ishwaran_ref(start, stop, codes, left, n_causes):
    U, V = cause_moments_ref(start, stop, codes, left, n_causes)
    var = np.trace(V)
    return float(U.sum() ** 2 / var) if var > 0 else 0.0
