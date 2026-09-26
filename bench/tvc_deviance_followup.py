"""S16 T7: deviance follow-up checks (tvc-deviance.md §8) and the ``windows`` default.

1. Weibull baseline (shape 0.5 and 2) in an S3-type generator: held-out PE score
   and permutation importance vs M in {2, 4, 8, 16}.
2. OOB vs held-out: PE score and importance ranking (Spearman) at M = 8.
3. Zero-rate share of events vs n ids {200, 1000, 5000} and min_events_leaf
   {1, 3, 10}, at M in {4, 8, 16}.

Decision rule (tvc-plan.md S16): the largest M in {4, 8, 16} whose zero-rate
share is <= 0.1% in the worst cell of check 3 and whose importance means stay
within 10% of the M = 4 values in check 1 (for units with M = 4 importance
>= 0.05 per event; smaller ones are noise-level).

Run: .venv/bin/python -m bench.tvc_deviance_followup  (writes docs/scratch/tvc_deviance_followup.txt)
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from rftvc import SurvivalForestTV, make_survival_y
from rftvc.metrics import _baseline_at, event_windows, piecewise_exponential_score

K, END = 8, 8.0
NAMES = ["x0", "x1", "z", "n1", "n2"]
BETA = {"x0": 0.4, "x1": 0.2}


def simulate(n, rng, shape=1.0):
    """S3-type: z redrawn per unit interval; hazard 0.15 a t^(a-1) exp(lp_k)."""
    x = rng.normal(size=(n, 4))  # x0, x1, n1, n2
    z = rng.normal(size=(n, K))
    lp = 0.8 * z + 0.8 * (z > 1) + BETA["x0"] * x[:, [0]] + BETA["x1"] * x[:, [1]]
    H0 = 0.15 * np.arange(K + 1.0) ** shape
    inc = np.exp(lp) * np.diff(H0)
    cum = np.cumsum(inc, axis=1)
    e = rng.exponential(size=n)
    k = (cum < e[:, None]).sum(axis=1)
    prev = np.where(k > 0, cum[np.arange(n), np.maximum(k - 1, 0)], 0.0)
    kk = np.minimum(k, K - 1)
    base = H0[kk] + (e - prev) / np.exp(lp[np.arange(n), kk])
    T = np.where(k < K, (base / 0.15) ** (1 / shape), np.inf)
    C = rng.uniform(2, END, size=n)
    U, ev = np.minimum(T, C), T <= C
    Xr, start, stop, evr, ids = [], [], [], [], []
    for i in range(n):
        j = 0
        while j < U[i]:
            Xr.append([x[i, 0], x[i, 1], z[i, j], x[i, 2], x[i, 3]])
            start.append(float(j))
            stop.append(min(j + 1.0, U[i]))
            evr.append(bool(ev[i] and j + 1.0 >= U[i]))
            ids.append(i)
            j += 1
    return np.array(Xr), make_survival_y(np.array(stop), np.array(evr), start=np.array(start)), np.array(ids)


def edges_for(m, M):
    return event_windows(m, M)


def score(m, X, y, w, cumhaz=None):
    if cumhaz is None:
        cumhaz = np.c_[np.zeros(len(X)), m.predict_cumulative_hazard(X, times=w[1:])]
    return piecewise_exponential_score(y, cumhaz, w, null_cumhaz=_baseline_at(m, w), alpha=0.01)


def importances(m, X, y, w, rng, n_rep=3):
    base = score(m, X, y, w).total
    out = {}
    for j, name in enumerate(NAMES):
        d = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d.append(base - score(m, Xp, y, w).total)
        out[name] = np.mean(d)
    return base, out


def check1(n_reps=5):
    rows = []
    for shape in (0.5, 2.0):
        for rep in range(n_reps):
            rng = np.random.default_rng(1000 * rep + int(shape * 10))
            X, y, ids = simulate(600, rng, shape)
            Xt, yt, _ = simulate(600, rng, shape)
            m = SurvivalForestTV(n_estimators=200, random_state=rep).fit(X, y, ids)
            for M in (2, 4, 8, 16):
                w = edges_for(m, M)
                base, imp = importances(m, Xt, yt, w, rng)
                rows.append(dict(shape=shape, rep=rep, M=M, score=base, **imp))
    return pd.DataFrame(rows)


def oob_scores(m, X, y, ids, w, rng, n_rep=3):
    d = m._rebuild_design(X, y, ids)

    def S(Xd):
        H, n = m.forest_.oob_cumhaz(Xd, *d.oob_set, w, m.aggregate, 4)
        ok = n > 0
        H[:, 0] = 0.0  # w_0 = 0: no hazard before time 0
        return piecewise_exponential_score(y[ok], H[ok], w, null_cumhaz=_baseline_at(m, w), alpha=0.01).total

    base = S(d.X)
    imp = {}
    for j, name in enumerate(NAMES):
        vals = []
        for _ in range(n_rep):
            Xp = d.X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            vals.append(base - S(Xp))
        imp[name] = np.mean(vals)
    return base, imp


def check2(n_reps=5, M=8):
    rows = []
    for rep in range(n_reps):
        rng = np.random.default_rng(500 + rep)
        X, y, ids = simulate(800, rng)
        Xt, yt, _ = simulate(800, rng)
        m = SurvivalForestTV(n_estimators=200, random_state=rep).fit(X, y, ids)
        w = edges_for(m, M)
        hb, himp = importances(m, Xt, yt, w, rng)
        ob, oimp = oob_scores(m, X, y, ids, np.ascontiguousarray(w), rng)
        rho = stats.spearmanr([himp[k] for k in NAMES], [oimp[k] for k in NAMES]).statistic
        rows.append(dict(rep=rep, heldout=hb, oob=ob, spearman=rho,
                         **{f"h_{k}": himp[k] for k in NAMES}, **{f"o_{k}": oimp[k] for k in NAMES}))
    return pd.DataFrame(rows)


def check3(n_reps=3):
    rows = []
    for n in (200, 1000, 5000):
        for mel in (1, 3, 10):
            for rep in range(n_reps):
                rng = np.random.default_rng(n + mel + 97 * rep)
                X, y, ids = simulate(n, rng)
                Xt, yt, _ = simulate(max(n, 500), rng)
                m = SurvivalForestTV(n_estimators=100, min_events_leaf=mel, random_state=rep).fit(X, y, ids)
                for M in (4, 8, 16):
                    w = edges_for(m, M)
                    r = score(m, Xt, yt, w)
                    rows.append(dict(n=n, min_events_leaf=mel, rep=rep, M=M, windows=len(w) - 1,
                                     zero_rate_share=r.zero_rate_share))
    return pd.DataFrame(rows)


def main(out="docs/scratch/tvc_deviance_followup.txt"):
    import warnings

    warnings.simplefilter("ignore", UserWarning)
    lines = []
    c1 = check1()
    g1 = c1.groupby(["shape", "M"])[["score"] + NAMES].mean()
    lines += ["1. Weibull baseline: held-out PE score and permutation importance (per event), mean over reps",
              g1.round(4).to_string(), ""]
    c2 = check2()
    lines += ["2. OOB vs held-out (M = 8)", c2.round(4).to_string(),
              f"mean Spearman = {c2.spearman.mean():.3f}", ""]
    c3 = check3()
    g3 = c3.groupby(["n", "min_events_leaf", "M"]).zero_rate_share.max().unstack("M")
    lines += ["3. zero-rate share of events (max over reps)", g3.to_string(), ""]
    # decision rule
    ok_zero = {M: g3[M].max() <= 0.001 for M in (4, 8, 16)}
    ok_imp = {}
    for M in (4, 8, 16):
        good = True
        for shape in (0.5, 2.0):
            ref = g1.loc[(shape, 4), NAMES]
            cur = g1.loc[(shape, M), NAMES]
            sig = ref >= 0.05
            good &= bool((np.abs(cur[sig] - ref[sig]) <= 0.1 * ref[sig]).all())
        ok_imp[M] = good
    chosen = max([M for M in (4, 8, 16) if ok_zero[M] and ok_imp[M]], default=None)
    lines += [f"zero-rate rule: {ok_zero}", f"importance rule: {ok_imp}", f"DECISION: default windows = {chosen}"]
    text = "\n".join(lines)
    print(text)
    with open(out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
