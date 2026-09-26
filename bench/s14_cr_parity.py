"""S14 right-censored parity on survival::pbc (transplant = 1, death = 2), s14-plan.md.

    RL=<R lib with randomForestSRC> python -m bench.s14_cr_parity

5 fixed folds shared with R. Arms: rftvc composite (hazard, cif), Approach B,
randomForestSRC (splitrule="logrank", reported only), and covariate-free
Aalen–Johansen. Metrics: cause-specific IPCW integrated Brier of F_k over
365..3650 days (G: reverse KM per test fold, g_min = 0.05), pooled over folds
with per-fold weights fixed; subject bootstrap (B = 500) of paired differences
vs rftvc composite/hazard; Wolbers C of F_k(3650) per fold.
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from bench.s14_cr_sim import OUT
from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y, make_survival_y
from rftvc.metrics import KaplanMeierCensoring, _ipcw, concordance_index_cr

TIMES = np.arange(365.0, 3651.0, 365.0)
SCRATCH = Path(__file__).resolve().parents[1] / "docs" / "scratch" / "s14"
R_EXPORT = """library(survival); d <- pbc; d$sex <- as.integer(d$sex == "f")
for (v in names(d)) if (any(is.na(d[[v]]))) d[[v]][is.na(d[[v]])] <- median(d[[v]], na.rm = TRUE)
write.csv(d, "{path}", row.names = FALSE)"""


def load(seed=2026):
    path = SCRATCH / "pbc.csv"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    subprocess.run(["Rscript", "-e", R_EXPORT.format(path=path)], check=True)
    d = pl.read_csv(path)
    folds = np.random.default_rng(seed).permutation(d.height) % 5
    return d.with_columns(pl.Series("fold", folds))


def feats(d):
    return [c for c in d.columns if c not in ("id", "time", "status", "fold")]


def rftvc_arm(train, test, cols, aggregate):
    kw = dict(n_estimators=500, min_ids_leaf=15, min_events_leaf=1, max_features=5, random_state=0, n_jobs=-1)
    y = make_competing_risks_y(train["time"].to_numpy().astype(float), train["status"].to_numpy())
    m = CompetingRisksForestTV(aggregate=aggregate, causes=[1, 2], **kw).fit(train.select(cols).to_numpy(), y)
    return m.predict_cumulative_incidence(test.select(cols).to_numpy(), TIMES)


def approach_b(train, test, cols):
    kw = dict(n_estimators=500, min_ids_leaf=15, min_events_leaf=1, max_features=5, random_state=0, n_jobs=-1)
    t, s = train["time"].to_numpy().astype(float), train["status"].to_numpy()
    grid = np.unique(t[s > 0])
    H = [SurvivalForestTV(**kw).fit(train.select(cols).to_numpy(), make_survival_y(t, s == k))
         .predict_cumulative_hazard(test.select(cols).to_numpy(), grid) for k in (1, 2)]
    dL = np.diff(np.stack(H, axis=1), axis=2, prepend=0.0)
    S_after = np.cumprod(1 - dL.sum(axis=1), axis=1)
    S_before = np.concatenate([np.ones((dL.shape[0], 1)), S_after[:, :-1]], axis=1)
    F = np.cumsum(S_before[:, None] * dL, axis=2)
    idx = np.searchsorted(grid, TIMES, side="right")
    return np.concatenate([np.zeros(F.shape[:2] + (1,)), F], axis=2)[:, :, idx]


def aj_null(train, test):
    t, s = train["time"].to_numpy().astype(float), train["status"].to_numpy()
    grid = np.unique(t[s > 0])
    y = np.array([(t >= g).sum() for g in grid], float)
    dL = np.stack([np.array([((t == g) & (s == k)).sum() for g in grid]) / y for k in (1, 2)])
    S_after = np.cumprod(1 - dL.sum(axis=0))
    S_before = np.r_[1.0, S_after[:-1]]
    F = np.cumsum(S_before * dL, axis=1)
    idx = np.searchsorted(grid, TIMES, side="right")
    F = np.concatenate([np.zeros((2, 1)), F], axis=1)[:, idx]
    return np.broadcast_to(F, (test.height, 2, TIMES.size))


def contributions(d, preds):
    """Per subject, cause and time: IPCW weight * (case - F)^2 with per-fold reverse KM weights."""
    n = d.height
    out = {arm: np.zeros((n, 2, TIMES.size)) for arm in preds}
    for f in range(5):
        rows = np.flatnonzero(d["fold"].to_numpy() == f)
        stop = d["time"].to_numpy().astype(float)[rows]
        lab = d["status"].to_numpy()[rows]
        cens = KaplanMeierCensoring().fit(make_competing_risks_y(stop, lab))
        for k in (1, 2):
            for j, t in enumerate(TIMES):
                case, _, w, _ = _ipcw(stop, lab, t, None, cens, 0.05, cause=k)
                for arm, F in preds.items():
                    out[arm][rows, k - 1, j] = w * np.square(case - F[rows, k - 1, j])
    return out


def ibs(contrib, idx=None):
    c = contrib if idx is None else contrib[idx]
    per_t = c.mean(axis=0)  # (2, T)
    return np.trapezoid(per_t, TIMES, axis=1) / (TIMES[-1] - TIMES[0])


def main(n_boot=500):
    d = load()
    cols = feats(d)
    OUT.mkdir(parents=True, exist_ok=True)
    preds = {a: np.zeros((d.height, 2, TIMES.size)) for a in
             ("composite/hazard", "composite/cif", "approach_b", "aj_null", "rfsrc")}
    fit_s = {a: 0.0 for a in preds}
    folds = d["fold"].to_numpy()
    for f in range(5):
        tr, te = d.filter(pl.col("fold") != f), d.filter(pl.col("fold") == f)
        rows = np.flatnonzero(folds == f)
        for arm, fn in (("composite/hazard", lambda: rftvc_arm(tr, te, cols, "hazard")),
                        ("composite/cif", lambda: rftvc_arm(tr, te, cols, "cif")),
                        ("approach_b", lambda: approach_b(tr, te, cols)),
                        ("aj_null", lambda: aj_null(tr, te))):
            t0 = time.perf_counter()
            preds[arm][rows] = fn()
            fit_s[arm] += time.perf_counter() - t0
    data_csv, r_out = SCRATCH / "pbc_folds.csv", SCRATCH / "rfsrc_pred.csv"
    d.write_csv(data_csv)
    t0 = time.perf_counter()
    subprocess.run(["Rscript", str(Path(__file__).with_suffix(".R")), str(data_csv), str(r_out)], check=True,
                   env={**os.environ, "RL": os.environ["RL"]})
    fit_s["rfsrc"] = time.perf_counter() - t0
    r = pl.read_csv(r_out)
    pos = {int(i): p for p, i in enumerate(d["id"].to_list())}
    for row in r.iter_rows(named=True):
        preds["rfsrc"][pos[row["id"]], row["cause"] - 1] = [row[f"t{int(t)}"] for t in TIMES]
    contrib = contributions(d, preds)
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, d.height, d.height) for _ in range(n_boot)]
    rows = []
    base = "composite/hazard"
    y = make_competing_risks_y(d["time"].to_numpy().astype(float), d["status"].to_numpy())
    for arm in preds:
        point = ibs(contrib[arm])
        diffs = np.array([ibs(contrib[arm], b) - ibs(contrib[base], b) for b in boot])
        for k in (1, 2):
            cs = [concordance_index_cr(y[folds == f], preds[arm][folds == f, k - 1, -1], cause=k) for f in range(5)]
            lo, hi = np.percentile(diffs[:, k - 1], [2.5, 97.5])
            rows.append(dict(arm=arm, cause=k, ibs=point[k - 1], diff_vs_base=point[k - 1] - ibs(contrib[base])[k - 1],
                             diff_lo=lo, diff_hi=hi, wolbers_c=float(np.mean(cs)), fit_s=fit_s[arm]))
    res = pl.DataFrame(rows)
    res.write_csv(OUT / "parity.csv")
    pl.Config.set_tbl_width_chars(160)
    print(res)


if __name__ == "__main__":
    main()
