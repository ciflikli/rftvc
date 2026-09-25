"""S3 simulation with a known truth: external time-varying covariate, Cox-type hazard.

Data-generating process (committed; seeds are the replication index):
- Subjects have a baseline covariate ``x0 ~ N(0, 1)`` and an external covariate
  ``z_k ~ N(0, 1)`` redrawn on each unit interval ``(k, k+1]``, k = 0..7.
- Hazard on ``(k, k+1]``: ``0.15 * exp(0.8 z_k + 0.8 * 1{z_k > 1} + 0.4 x0)``
  (log-linear plus a threshold non-linearity).
- Censoring ``C ~ Uniform(2, 8)``, administrative end at 8.
- Training rows: one row per observed interval, ``X = [x0, z_k]``.
- Fixed-covariate baseline: one row per subject, ``X = [x0, z_0]``.

Metric: for test subjects, the integrated squared error of the predicted
``S(t | covariate path)`` against the true ``exp(-Λ(t))`` over ``t ∈ [0, 6]``
(trapezoid rule, 61 points), averaged over subjects. The true path is known
because ``z`` is external. The time-varying forest predicts along each test
subject's full path through t = 6; the baseline predicts from ``[x0, z_0]``.

Pass rule (declared in advance): over 20 replications, the upper 95% bound of
the mean paired difference (TVC − baseline) is < 0.
"""

import numpy as np
from scipy import stats

from rftvc import SurvivalForestTV, make_survival_y

K, END, HORIZON = 8, 8.0, 6.0
GRID = np.linspace(0.0, HORIZON, 61)


def hazard(x0, z):
    return 0.15 * np.exp(0.8 * z + 0.8 * (z > 1) + 0.4 * x0[:, None])


def simulate(n, rng):
    x0 = rng.normal(size=n)
    z = rng.normal(size=(n, K))
    lam = hazard(x0, z)
    # Event time from piecewise-constant hazards on unit intervals.
    e_draw = rng.exponential(size=n)
    cum = np.cumsum(lam, axis=1)
    k_evt = (cum < e_draw[:, None]).sum(axis=1)  # interval containing the event (K = none)
    prev = np.where(k_evt > 0, cum[np.arange(n), np.maximum(k_evt - 1, 0)], 0.0)
    within = (e_draw - prev) / lam[np.arange(n), np.minimum(k_evt, K - 1)]
    T = np.where(k_evt < K, k_evt + within, np.inf)
    C = np.minimum(rng.uniform(2, END, size=n), END)
    return x0, z, np.minimum(T, C), T <= C


def rows(x0, z, U, event):
    X, start, stop, ev, ids = [], [], [], [], []
    for i in range(len(x0)):
        k = 0
        while k < U[i]:
            X.append([x0[i], z[i, k]])
            start.append(float(k))
            stop.append(min(k + 1.0, U[i]))
            ev.append(bool(event[i] and k + 1.0 >= U[i]))
            ids.append(i)
            k += 1
    return np.array(X), make_survival_y(np.array(stop), np.array(ev), start=np.array(start)), np.array(ids)


def true_survival(x0, z):
    lam = hazard(x0, z)
    cum = np.concatenate([np.zeros((len(x0), 1)), np.cumsum(lam, axis=1)], axis=1)
    k = np.minimum(np.floor(GRID).astype(int), K - 1)
    H = cum[:, k] + lam[:, k] * (GRID - k)
    return np.exp(-H)


def ise(S_hat, S_true):
    return np.trapezoid((S_hat - S_true) ** 2, GRID, axis=1).mean()


def replicate(seed, n_train=400, n_test=200, n_estimators=200):
    rng = np.random.default_rng(seed)
    x0, z, U, event = simulate(n_train, rng)
    X, y, ids = rows(x0, z, U, event)
    tvc = SurvivalForestTV(n_estimators=n_estimators, random_state=seed).fit(X, y, ids=ids)
    base = SurvivalForestTV(n_estimators=n_estimators, random_state=seed).fit(
        np.column_stack([x0, z[:, 0]]), make_survival_y(U, event)
    )
    tx0, tz = rng.normal(size=n_test), rng.normal(size=(n_test, K))
    S_true = true_survival(tx0, tz)
    n_int = int(np.ceil(HORIZON))
    Xp = np.column_stack([np.repeat(tx0, n_int), tz[:, :n_int].ravel()])
    starts = np.tile(np.arange(n_int, dtype=float), n_test)
    iv = make_survival_y(starts + 1.0, np.zeros(len(starts), bool), start=starts)
    S_tvc = tvc.predict_survival_function(Xp, GRID, intervals=iv, ids=np.repeat(np.arange(n_test), n_int))
    S_base = base.predict_survival_function(np.column_stack([tx0, tz[:, 0]]), GRID)
    return ise(S_tvc, S_true), ise(S_base, S_true)


def run(n_reps=20, **kw):
    res = np.array([replicate(seed, **kw) for seed in range(n_reps)])
    diff = res[:, 0] - res[:, 1]
    upper = diff.mean() + stats.t.ppf(0.975, n_reps - 1) * diff.std(ddof=1) / np.sqrt(n_reps)
    return res, diff, upper
