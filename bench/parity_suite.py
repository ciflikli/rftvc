"""Forest-level parity suite: rftvc vs scikit-survival RSF on several datasets.

Not a merge gate (a small subset runs in ``tests/test_sksurv_forest_parity.py``).
Run from the repo root:

    .venv/bin/python -m bench.parity_suite

Writes docs/bench/parity-suite.md and docs/bench/parity-suite.csv. Protocol:
- Data: scikit-survival's bundled datasets, categoricals one-hot encoded, missing
  values median-imputed (the imputation is not fold-wise; it is identical for both
  libraries, so it cannot favour either).
- Folds: 3 repeats of KFold(5, shuffle=True). The same folds feed both libraries.
- Matched settings: bootstrap resampling, 200 trees, leaf size 10 (sksurv
  ``min_samples_split = 2 * leaf``), ``max_features="sqrt"``, ``min_events_leaf=1``.
- Metrics per fold: Harrell C on the sum of the cumulative hazard over the training
  event times (sksurv's own ``predict`` definition), and the integrated Brier score
  on the 10th-80th percentile of the test times, clipped to the training follow-up.
- Paired differences (rftvc - sksurv) are taken per fold. The interval is a 90%
  percentile bootstrap over the folds' differences; parity is claimed when it sits
  inside the equivalence margin (TOST at 5% per side). ``+`` C and ``-`` IBS favour rftvc.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sksurv import datasets
from sksurv.column import encode_categorical
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

from rftvc import SurvivalForestTV, make_survival_y

ROOT = Path(__file__).resolve().parents[1]
LEAF, N_TREES = 10, 200
MARGIN_C, MARGIN_IBS = 0.02, 0.01


def load(name):
    """Return (X ndarray, event bool, time float)."""
    X, y = getattr(datasets, f"load_{name}")()
    Xe = encode_categorical(X)
    Xe = Xe.fillna(Xe.median())
    ev, tm = y.dtype.names
    t = np.asarray(y[tm], float)
    keep = t > 0
    return Xe.to_numpy(float)[keep], np.asarray(y[ev], bool)[keep], t[keep]


DATASETS = ["veterans_lung_cancer", "whas500", "gbsg2", "aids", "breast_cancer", "flchain"]


def _fit_eval(X_tr, e_tr, t_tr, X_te, e_te, t_te, seed):
    keep = t_te < t_tr.max()  # sksurv's IPCW needs test times inside training follow-up
    X_te, e_te, t_te = X_te[keep], e_te[keep], t_te[keep]
    lo, hi = np.percentile(t_te, [10, 80])
    hi = min(hi, t_tr.max() - 1e-6)
    lo = min(lo, hi / 2)
    times = np.linspace(lo, hi, 50)
    y_tr, y_te = Surv.from_arrays(e_tr, t_tr), Surv.from_arrays(e_te, t_te)
    out = {}
    ref = RandomSurvivalForest(
        n_estimators=N_TREES, min_samples_leaf=LEAF, min_samples_split=2 * LEAF,
        max_features="sqrt", random_state=seed, n_jobs=-1,
    ).fit(X_tr, y_tr)
    out["sksurv"] = (
        ref.predict(X_te),
        np.vstack([f(times) for f in ref.predict_survival_function(X_te)]),
    )
    ours = SurvivalForestTV(
        n_estimators=N_TREES, bootstrap=True, min_ids_leaf=LEAF, min_events_leaf=1,
        random_state=seed, n_jobs=-1,
    ).fit(X_tr, make_survival_y(t_tr, e_tr))
    out["rftvc"] = (
        ours.predict_cumulative_hazard(X_te).sum(axis=1),
        ours.predict_survival_function(X_te, times),
    )
    res = {}
    for k, (risk, surv) in out.items():
        res[k] = (
            concordance_index_censored(e_te, t_te, risk)[0],
            integrated_brier_score(y_tr, y_te, surv, times),
        )
    return res


def paired(name, X, e, t, repeats=3, folds=5, seed=0):
    """Per-fold metrics for both libraries on identical folds."""
    rows = []
    for r in range(repeats):
        for f, (tr, te) in enumerate(KFold(folds, shuffle=True, random_state=seed + r).split(X)):
            res = _fit_eval(X[tr], e[tr], t[tr], X[te], e[te], t[te], seed + 100 * r + f)
            rows.append(
                dict(dataset=name, repeat=r, fold=f,
                     c_sksurv=res["sksurv"][0], ibs_sksurv=res["sksurv"][1],
                     c_rftvc=res["rftvc"][0], ibs_rftvc=res["rftvc"][1])
            )
    return rows


def summarise(rows, B=2000, seed=0):
    """Mean paired difference and 90% bootstrap interval for C and IBS."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in ("c", "ibs"):
        d = np.array([r[f"{m}_rftvc"] - r[f"{m}_sksurv"] for r in rows])
        boots = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
        out[m] = (d.mean(), *np.percentile(boots, [5, 95]))
    return out


def main():
    all_rows, lines = [], []
    for name in DATASETS:
        X, e, t = load(name)
        rows = paired(name, X, e, t)
        all_rows += rows
        s = summarise(rows)
        (dc, clo, chi), (di, ilo, ihi) = s["c"], s["ibs"]
        ok = (-MARGIN_C < clo and chi < MARGIN_C) and (-MARGIN_IBS < ilo and ihi < MARGIN_IBS)
        lines.append(
            f"| {name} | {len(t)} | {X.shape[1]} | {e.mean():.2f} | "
            f"{np.mean([r['c_sksurv'] for r in rows]):.4f} | {dc:+.4f} [{clo:+.4f}, {chi:+.4f}] | "
            f"{np.mean([r['ibs_sksurv'] for r in rows]):.4f} | {di:+.4f} [{ilo:+.4f}, {ihi:+.4f}] | "
            f"{'yes' if ok else '**no**'} |"
        )
        print(lines[-1], flush=True)
    import sksurv

    import rftvc

    md = [
        "# Forest-level parity suite: rftvc vs scikit-survival RSF",
        "",
        f"Generated by `bench/parity_suite.py`: rftvc {rftvc.__version__}, scikit-survival {sksurv.__version__}. "
        f"{N_TREES} trees, leaf {LEAF}, bootstrap, 3 x 5-fold CV on identical folds. Protocol in the script docstring.",
        "",
        f"Differences are rftvc - sksurv, mean over the 15 paired folds, with a 90% bootstrap interval. "
        f"Equivalent = both intervals inside the margin (C +/-{MARGIN_C}, IBS +/-{MARGIN_IBS}).",
        "",
        "| dataset | n | p | event rate | sksurv C | dC [90% interval] | sksurv IBS | dIBS [90% interval] | equivalent |",
        "|---|---|---|---|---|---|---|---|---|",
        *lines,
    ]
    out = ROOT / "docs" / "bench"
    (out / "parity-suite.md").write_text("\n".join(md) + "\n")
    with open(out / "parity-suite.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
