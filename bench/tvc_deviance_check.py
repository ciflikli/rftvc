"""Numerical checks behind the piecewise-exponential score analysis (research stage, not a test).

1. Propriety (Monte Carlo, one window, Weibull truth): the row-own-exposure score
   ``N log E_hat - E_hat`` is improper; the piecewise-constant score is maximised
   at the exposure-weighted window rate c*.
2. Zero-rate event cells and score vs window count M on the S3 simulation.
3. Permutation-importance stability across M.

Run: .venv/bin/python bench/tvc_deviance_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from sim import hazard, rows, simulate  # noqa: E402

from rftvc import SurvivalForestTV  # noqa: E402


def cells(start, stop, event, edges):
    """Per (row, window): exposure ``e`` and event count ``N``; windows ``(edges[m-1], edges[m]]``."""
    lo, hi = edges[:-1], edges[1:]
    e = np.clip(np.minimum(stop[:, None], hi) - np.maximum(start[:, None], lo), 0, None)
    N = (event[:, None] & (stop[:, None] > lo) & (stop[:, None] <= hi)).astype(float)
    return e, N


def score(rate, e, N):
    """Piecewise-exponential log score per event (higher is better); -inf if an event cell has rate 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.where(N > 0, N * np.log(rate), 0.0) - rate * e).sum() / N.sum()


def window_rates(forest, X, edges):
    H = np.c_[np.zeros(len(X)), forest.predict_cumulative_hazard(X, times=edges[1:])]
    return np.diff(H, axis=1) / np.diff(edges)


def propriety():
    rng = np.random.default_rng(0)
    n, k, a = 2_000_000, 1.0, 2.5
    T = (rng.exponential(size=n) / k) ** (1 / a)  # Λ(t) = k t^a on the window (0, 1]
    C = rng.uniform(0, 2, size=n)
    stop = np.minimum(np.minimum(T, C), 1.0)
    N = ((T <= C) & (T <= 1)).astype(float)
    print("1a. row-own-exposure score N log Ê − Ê (higher is better), Λ̂ = k t^b:")
    for b in (1.0, 1.5, 2.0, 2.5, 3.0):
        E = k * stop**b
        print(f"    b={b}: {np.mean(N * np.log(E) - E):.5f}{'  <- truth' if b == a else ''}")
    c_star = N.sum() / stop.sum()
    print("1b. piecewise-constant score N log c − c e:")
    for c in (0.5 * c_star, c_star, 1.0 * k, 1.5 * c_star):
        print(f"    c={c:.4f}: {np.mean(N * np.log(c) - c * stop):.6f}")
    print(f"    c* = {c_star:.4f} (exposure-weighted); uniform window average of the truth = {k:.4f}")


def zero_cells(n_reps=5):
    res = []
    for seed in range(n_reps):
        rng = np.random.default_rng(seed)
        X, y, ids = rows(*simulate(400, rng))
        Xt, yt, _ = rows(*simulate(400, rng))
        ev_times = y["stop"][y["event"]]
        for n_trees in (70, 200):  # 70 ≈ the OOB tree count of a 200-tree forest
            f = SurvivalForestTV(n_estimators=n_trees, random_state=seed).fit(X, y, ids=ids)
            for M in (4, 8, 16, 32, 64):
                edges = np.r_[0.0, np.quantile(ev_times, np.arange(1, M) / M), 8.0]
                rate = window_rates(f, Xt, edges)
                e, N = cells(yt["start"], yt["stop"], yt["event"], edges)
                pooled = N.sum(0) / e.sum(0)  # test-set null rate, for scale only
                out = dict(
                    trees=n_trees,
                    M=M,
                    zero_ev=((rate == 0) & (N > 0)).sum() / N.sum(),
                    truth=score(np.broadcast_to(hazard(Xt[:, 0], Xt[:, 1:2]), rate.shape), e, N),
                    null=score(np.broadcast_to(pooled, rate.shape), e, N),
                )
                for alpha in (0.0, 0.001, 0.01, 0.1):
                    out[f"mix{alpha}"] = score((1 - alpha) * rate + alpha * pooled, e, N)
                res.append(out)
    print("2. zero-rate event share and score per event, by trees and M:")
    print(pd.DataFrame(res).groupby(["trees", "M"]).mean().round(4).to_string())


def importance(n_reps=5, alpha=0.01):
    res = []
    for seed in range(n_reps):
        rng = np.random.default_rng(100 + seed)
        X, y, ids = rows(*simulate(400, rng))
        Xt, yt, _ = rows(*simulate(400, rng))
        X, Xt = np.c_[X, rng.normal(size=len(X))], np.c_[Xt, rng.normal(size=len(Xt))]  # null column
        f = SurvivalForestTV(n_estimators=200, random_state=seed).fit(X, y, ids=ids)
        ev_times = y["stop"][y["event"]]
        for M in (2, 4, 8, 16):
            edges = np.r_[0.0, np.quantile(ev_times, np.arange(1, M) / M), 8.0]
            e, N = cells(yt["start"], yt["stop"], yt["event"], edges)
            pooled = N.sum(0) / e.sum(0)

            def S(Xe):
                return score((1 - alpha) * window_rates(f, Xe, edges) + alpha * pooled, e, N)

            base = S(Xt)
            for j, name in enumerate(["x0", "z", "noise"]):
                d = []
                for _ in range(3):
                    Xp = Xt.copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    d.append(base - S(Xp))
                res.append(dict(M=M, var=name, imp=np.mean(d)))
    print(f"3. permutation importance (score drop per event, alpha={alpha}), mean and sd over reps:")
    print(pd.DataFrame(res).pivot_table(index="M", columns="var", values="imp", aggfunc=["mean", "std"]).round(4))


if __name__ == "__main__":
    propriety()
    zero_cells()
    importance()
