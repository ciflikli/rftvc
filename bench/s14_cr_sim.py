"""S14 competing-risks simulations with a known truth (s14-plan.md; not a merge gate).

    python -m bench.s14_cr_sim check          # pre-run scenario checks (plan review 3, 7)

The challenger criteria ("quadratic", "ishwaran", "logrank_all") were removed
after the bake-off (P5); rerun their arms at commit 66ccad8.
    python -m bench.s14_cr_sim run [n_reps]   # writes docs/bench/s14-cr/sim.csv

Data-generating process: external covariates, so the truth along a path is exact.
- Baselines ``x0 ~ N(0, 1)``, ``x1 ~ Bernoulli(0.5)``, three noise features;
  ``z_k ~ N(0, 1)`` redrawn on each unit interval ``(k, k+1]``, k = 0..7.
- Cause-specific hazards are constant on each interval (scenario functions below).
- Censoring ``U(2, 8)``, administrative end 8. Training ids have delayed entry
  ``U(0, 1)`` and are kept only if event-free and uncensored at entry; test
  paths start at 0 and run through t = 6.
- Metric: per cause, the integrated squared error of ``F_k(t | path)`` over
  ``t ∈ [0, 6]`` (61 points, trapezoid rule), averaged over test subjects.
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from rftvc import CompetingRisksForestTV, SurvivalForestTV, _core, make_competing_risks_y, make_survival_y

K, END, HORIZON = 8, 8.0, 6.0
GRID = np.linspace(0.0, HORIZON, 61)
OUT = Path(__file__).resolve().parents[1] / "docs" / "bench" / "s14-cr"
# Cause-3 baseline for scenario C, set by `check` (3-5% of events on a 50k draw).
A3 = 0.007


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def hazards(scenario, x0, x1, z):
    """Cause-specific hazards, shape ``(n, K, J)``, on each unit interval."""
    x0, x1 = x0[:, None], x1[:, None]
    if scenario == "A":
        lam = [0.12 * np.exp(0.8 * z + 0.5 * x0), 0.08 * np.exp(0.7 * x1 - 0.3 * x0) + 0 * z]
    elif scenario == "B":  # all-cause hazard 0.2 e^{0.3 x0} whatever z; z only allocates the cause
        total = 0.2 * np.exp(0.3 * x0) + 0 * z
        p = _logistic(1.8 * z)
        lam = [total * p, total * (1 - p)]
    elif scenario == "C":
        lam = [
            0.12 * np.exp(0.8 * z + 0.5 * x0),
            0.08 * np.exp(0.7 * x1 - 0.3 * x0) + 0 * z,
            A3 * np.exp(1.0 * x1) + 0 * z,
        ]
    else:
        raise ValueError(scenario)
    return np.stack(lam, axis=-1)


def covariates(n, rng):
    return rng.normal(size=n), rng.integers(0, 2, n).astype(float), rng.normal(size=(n, 3)), rng.normal(size=(n, K))


def event_times(lam, rng):
    """First event time and cause from piecewise-constant hazards on unit intervals."""
    n = lam.shape[0]
    total = lam.sum(axis=2)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(total, axis=1)], axis=1)  # at integer times
    e = rng.exponential(size=n)
    k = np.minimum((cum[:, 1:] < e[:, None]).sum(axis=1), K - 1)
    t = np.where(cum[:, -1] < e, np.inf, k + (e - cum[np.arange(n), k]) / total[np.arange(n), k])
    p = lam[np.arange(n), k] / total[np.arange(n), k][:, None]
    u = rng.random(n)[:, None]
    cause = 1 + (u > np.cumsum(p, axis=1)).sum(axis=1)
    return t, np.minimum(cause, lam.shape[2])


def training_rows(scenario, n_ids, rng):
    """Counting-process rows of ``n_ids`` kept (delayed-entry) training ids."""
    Xs, starts, stops, labels, ids = [], [], [], [], []
    kept = 0
    while kept < n_ids:
        m = 2 * (n_ids - kept) + 50
        x0, x1, noise, z = covariates(m, rng)
        t, cause = event_times(hazards(scenario, x0, x1, z), rng)
        c = np.minimum(rng.uniform(2, 8, m), END)
        entry = rng.uniform(0, 1, m)
        keep = np.flatnonzero((t > entry) & (c > entry))[: n_ids - kept]
        for i in keep:
            u = min(t[i], c[i])
            label = int(cause[i]) if t[i] <= c[i] else 0
            a = entry[i]
            while a < u:
                b = min(np.floor(a) + 1.0, u)
                k = int(np.floor(a))
                Xs.append([z[i, k], x0[i], x1[i], *noise[i]])
                starts.append(a)
                stops.append(b)
                labels.append(label if b == u else 0)
                ids.append(kept)
                a = b
            kept += 1
    y = make_competing_risks_y(np.array(stops), np.array(labels), start=np.array(starts))
    return np.array(Xs), y, np.array(ids)


def test_paths(scenario, n, rng):
    """Test subjects' paths on (0, 6] and the true ``F`` (n, J, len(GRID))."""
    x0, x1, noise, z = covariates(n, rng)
    lam = hazards(scenario, x0, x1, z)
    nk = int(HORIZON)
    X = np.array([[z[i, k], x0[i], x1[i], *noise[i]] for i in range(n) for k in range(nk)])
    iv = np.zeros(n * nk, dtype=[("start", float), ("stop", float)])
    iv["start"] = np.tile(np.arange(nk, dtype=float), n)
    iv["stop"] = iv["start"] + 1
    ids = np.repeat(np.arange(n), nk)
    return X, iv, ids, true_cif(lam, GRID)


def true_cif(lam, t):
    """Closed-form Aalen–Johansen of piecewise-constant hazards: ``F`` of shape (n, J, len(t))."""
    n, _, J = lam.shape
    total = lam.sum(axis=2)
    S_k = np.concatenate([np.ones((n, 1)), np.exp(-np.cumsum(total, axis=1))], axis=1)  # S at integers
    F_k = np.zeros((n, K + 1, J))
    for k in range(K):
        F_k[:, k + 1] = F_k[:, k] + lam[:, k] / total[:, k, None] * (S_k[:, k] * (1 - np.exp(-total[:, k])))[:, None]
    k = np.minimum(np.floor(t).astype(int), K - 1)
    dt = t - k
    h = total[:, k]  # (n, len(t))
    frac = lam[:, k, :] / h[:, :, None]
    F = F_k[:, k, :] + frac * (S_k[:, k] * (1 - np.exp(-h * dt)))[:, :, None]
    return np.transpose(F, (0, 2, 1))


def ise(F_hat, F_true):
    """Mean over subjects of the trapezoid integral of the squared error, per cause: shape (J,)."""
    return np.trapezoid(np.square(F_hat - F_true), GRID, axis=2).mean(axis=0)


def approach_b(X, y, ids, Xt, iv, ids_t, J, **kw):
    """J cause-specific survival forests, combined by Aalen–Johansen along the test paths."""
    start, stop, labels = y["start"], y["stop"], y["event"]
    grid = np.unique(stop[labels != 0])
    H = []
    for k in range(1, J + 1):
        sf = SurvivalForestTV(**kw).fit(X, make_survival_y(stop, labels == k, start=start), ids)
        H.append(sf.predict_cumulative_hazard(Xt, grid, intervals=iv, ids=ids_t))
    dL = np.diff(np.stack(H, axis=1), axis=2, prepend=0.0)  # (n, J, V)
    S_after = np.cumprod(1 - dL.sum(axis=1), axis=1)
    S_before = np.concatenate([np.ones((dL.shape[0], 1)), S_after[:, :-1]], axis=1)
    F = np.cumsum(S_before[:, None] * dL, axis=2)
    idx = np.searchsorted(grid, GRID, side="right")
    return np.concatenate([np.zeros(F.shape[:2] + (1,)), F], axis=2)[:, :, idx]


def arms(scenario, J):
    out = [(f"{c}/{a}", dict(criterion=c, aggregate=a), None)
           for c in ("composite", "quadratic", "ishwaran", "logrank_all") for a in ("hazard", "cif")]
    out += [(f"split_cause={k}", dict(split_cause=k), k) for k in range(1, J + 1)]
    if scenario == "C":
        # All causes are recorded: C5 also checks causes 1-2 of these forests.
        out += [(f"split_cause=3/m={m}", dict(split_cause=3, min_events_leaf=1, min_events_leaf_cause=m), None)
                for m in (None, 3, 5, 10)]
    return out


def replicate(scenario, seed, n_trees=200):
    rng = np.random.default_rng(seed)
    J = 3 if scenario == "C" else 2
    n_train = 800 if scenario == "C" else 500
    X, y, ids = training_rows(scenario, n_train, rng)
    Xt, iv, ids_t, F_true = test_paths(scenario, 200, rng)
    n_events = [int((y["event"] == k).sum()) for k in range(1, J + 1)]
    rows = []
    base = dict(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    for name, params, only in arms(scenario, J):
        t0 = time.perf_counter()
        m = CompetingRisksForestTV(causes=list(range(1, J + 1)), **base, **params).fit(X, y, ids)
        fit_s = time.perf_counter() - t0
        F = m.predict_cumulative_incidence(Xt, GRID, intervals=iv, ids=ids_t)
        err = ise(F, F_true)
        for k in range(1, J + 1):
            if only is None or k == only:
                rows.append(dict(scenario=scenario, rep=seed, arm=name, cause=k, ise=float(err[k - 1]),
                                 fit_s=fit_s, n_events=n_events[k - 1]))
    t0 = time.perf_counter()
    F = approach_b(X, y, ids, Xt, iv, ids_t, J, **base)
    fit_s = time.perf_counter() - t0
    err = ise(F, F_true)
    rows += [dict(scenario=scenario, rep=seed, arm="approach_b", cause=k, ise=float(err[k - 1]), fit_s=fit_s,
                  n_events=n_events[k - 1]) for k in range(1, J + 1)]
    return rows


def check():
    """Pre-run checks: scenario B's all-cause invariance and scenario C's cause-3 share."""
    rng = np.random.default_rng(12345)
    X, y, _ = training_rows("B", 20_000, rng)
    left = X[:, 0] > np.median(X[:, 0])
    codes = y["event"].astype(np.uint8)
    start, stop = np.ascontiguousarray(y["start"]), np.ascontiguousarray(y["stop"])
    lr = _core.logrank_score(start, stop, codes != 0, left)
    comp = _core.cause_score(start, stop, codes, left, 2)
    print(f"B: all-cause log-rank on z median split = {lr:.2f} (want < 2), composite = {comp:.1f} (want > 50)")
    rng = np.random.default_rng(54321)
    x0, x1, _, z = covariates(50_000, rng)
    t, cause = event_times(hazards("C", x0, x1, z), rng)
    c = np.minimum(rng.uniform(2, 8, t.size), END)
    ev = t <= c
    share = (cause[ev] == 3).mean()
    print(f"C: A3 = {A3}: cause-3 share of events = {share:.3%} (want 3-5%), events/id = {ev.mean():.2f}")


def run(n_reps=30):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in ("A", "B", "C"):
        for seed in range(n_reps):
            t0 = time.perf_counter()
            rows += replicate(scenario, seed)
            print(f"{scenario} rep {seed}: {time.perf_counter() - t0:.1f}s", flush=True)
        pl.DataFrame(rows).write_csv(OUT / "sim.csv")
    print("wrote", OUT / "sim.csv")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        run(*(int(a) for a in sys.argv[2:]))
