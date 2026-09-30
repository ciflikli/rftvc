"""Reproducible fit-time matrix for survival forests.

Run ``.venv/bin/python -m bench.fit_matrix --quick`` for a smoke benchmark or
omit ``--quick`` for the larger matrix. Each arm runs in a fresh process. The
generated CSV is shared by Python and R; data loading is outside fit timing.
Results are JSON lines on stdout and can be saved with ``--output PATH``.

Only one-row-per-subject cases are sent to scikit-survival and the R packages:
their ordinary survival-forest interfaces do not implement rftvc's path fit.
Leaf-size rules and feature threshold searches differ across libraries, so
compare predictive quality as well as elapsed time before drawing conclusions.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:  # Windows
    resource = None

from rftvc import SurvivalForestTV, make_survival_y


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("rftvc", "sksurv", "ranger", "rfsrc")
CASES = {
    "static": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5),
    "rare_event": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.1),
    "wide": dict(n_ids=5_000, p=100, rows_per_id=1, event_fraction=0.5),
    "tied": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, ties=True),
    "missing_1pct": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, missing=0.01),
    "missing_5pct": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, missing=0.05),
    "missing": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, missing=0.2),
    "tvc": dict(n_ids=2_000, p=10, rows_per_id=5, event_fraction=0.5),
}


def make_case(path, spec, seed=0):
    """One event per subject at most; same input rows for every arm."""
    rng = np.random.default_rng(seed)
    n, p, rpi = spec["n_ids"], spec["p"], spec["rows_per_id"]
    base = rng.normal(size=(n, p))
    scale = np.exp(-0.6 * base[:, 0] - 0.3 * base[:, 1])
    failure = rng.exponential(scale)
    # Independent censoring; use a pilot quantile to vary the event fraction.
    censor = rng.exponential(np.quantile(failure, spec["event_fraction"]), n)
    stop_id = np.minimum(failure, censor) + 0.01
    event_id = failure <= censor
    if spec.get("ties"):
        stop_id = np.maximum(0.25, np.ceil(stop_id * 4) / 4)
    if rpi == 1:
        start = np.zeros(n)
        stop = stop_id
        event = event_id
        ids = np.arange(n)
        X = base
    else:
        cuts = stop_id[:, None] * np.arange(1, rpi)[None, :] / rpi
        start = np.column_stack((np.zeros(n), cuts)).ravel()
        stop = np.column_stack((cuts, stop_id)).ravel()
        event = np.zeros((n, rpi), bool)
        event[:, -1] = event_id
        event = event.ravel()
        ids = np.repeat(np.arange(n), rpi)
        X = np.repeat(base, rpi, axis=0)
        X[:, 0] += rng.normal(0, 0.3, len(X))
    missing = spec.get("missing", 0.0)
    if missing:
        X = X.copy()
        X[rng.random(X.shape) < missing] = np.nan
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "start", "stop", "event", *(f"x{j}" for j in range(p))])
        for i in range(len(X)):
            writer.writerow((ids[i], start[i], stop[i], int(event[i]), *X[i]))
    return {"n_rows": len(X), "n_ids": n, "p": p, "event_fraction_realized": float(event_id.mean()),
            "missing_fraction": missing, "ties": bool(spec.get("ties")), "rows_per_id": rpi}


def _rss_gb():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1e9 if sys.platform == "darwin" else 1e6)


def run_python(arm, path, trees, threads, ntime):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    ids = data[:, 0].astype(np.int64)
    start, stop = data[:, 1], data[:, 2]
    event = data[:, 3].astype(bool)
    X = data[:, 4:]
    held_out = ids % 5 == 0
    is_tvc = np.unique(ids).size != len(ids)
    X_train, X_test = X[~held_out], X[held_out]
    start_train, stop_train, event_train = start[~held_out], stop[~held_out], event[~held_out]
    if arm == "rftvc":
        model = SurvivalForestTV(
            n_estimators=trees, min_ids_leaf=15, min_events_leaf=1, max_features="sqrt",
            bootstrap=True, max_samples=0.632, n_jobs=threads, ntime=ntime, random_state=0,
        )
        y = make_survival_y(stop_train, event_train, start=start_train)
        fit = lambda: model.fit(X_train, y, ids=ids[~held_out])
        predict = lambda: model.predict(X_test[:1000])
    else:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv

        model = RandomSurvivalForest(
            n_estimators=trees, min_samples_leaf=15, min_samples_split=30,
            max_features="sqrt", bootstrap=True, max_samples=0.632,
            n_jobs=threads, random_state=0, low_memory=True,
        )
        y = Surv.from_arrays(event_train, stop_train)
        fit = lambda: model.fit(X_train, y)
        predict = lambda: model.predict(X_test[:1000])
    t0 = time.perf_counter()
    fit()
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = predict()
    predict_s = time.perf_counter() - t0
    result = dict(fit_s=fit_s, predict_1000_s=predict_s, peak_rss_gb=_rss_gb(),
                  n_train=len(X_train), n_test=len(X_test), n_eval=min(1000, len(X_test)))
    if not is_tvc:
        from sksurv.metrics import concordance_index_censored

        result["test_c"] = float(concordance_index_censored(event[held_out][:1000],
                                                              stop[held_out][:1000], pred)[0])
    return result


def _run_child(args):
    if args.arm in ("rftvc", "sksurv"):
        return run_python(args.arm, args.data, args.trees, args.threads, args.ntime)
    raise ValueError(args.arm)


def _run_arm(arm, case, path, trees, threads, ntime):
    if arm in ("ranger", "rfsrc"):
        cmd = ["Rscript", str(ROOT / "bench" / "fit_matrix.R"), arm, str(path), str(trees), str(threads)]
    else:
        cmd = [sys.executable, "-m", "bench.fit_matrix", "--child", "--arm", arm,
               "--case", case, "--data", str(path), "--trees", str(trees),
               "--threads", str(threads), "--ntime", str(ntime) if ntime is not None else "none"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if proc.returncode == 2 and " is unavailable" in proc.stderr:
        return {"status": "unavailable", "reason": proc.stderr.strip()}
    if proc.returncode != 0:
        return {"status": "failed", "reason": proc.stderr.strip()[-400:]}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{\"fit_s\"")]
    if not lines:
        return {"status": "failed", "reason": proc.stdout[-400:] or proc.stderr[-400:]}
    return {"status": "ok", **json.loads(lines[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--ntime", default="none")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--data", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.ntime = None if args.ntime == "none" else int(args.ntime)
    if args.child:
        print(json.dumps(_run_child(args)))
        return
    if args.trees < 1 or args.threads < 1:
        parser.error("--trees and --threads must be positive")
    import rftvc
    import sksurv

    meta = {"kind": "metadata", "rftvc": rftvc.__version__, "sksurv": sksurv.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "trees": args.trees, "threads": args.threads, "ntime": args.ntime}
    records = [meta]
    names = args.cases or (("static", "tvc") if args.quick else tuple(CASES))
    arms = args.arms or ARMS
    with tempfile.TemporaryDirectory(prefix="rftvc-bench-") as tmp:
        for name in names:
            spec = CASES[name] if not args.quick else {**CASES[name], "n_ids": 400 if name == "static" else 100}
            path = Path(tmp) / f"{name}.csv"
            info = make_case(path, spec)
            for arm in arms:
                if info["rows_per_id"] > 1 and arm != "rftvc":
                    result = {"status": "unsupported", "reason": "path data requires counting-process fit"}
                elif info["missing_fraction"] and arm in ("ranger", "rfsrc"):
                    result = {"status": "unsupported", "reason": "R arms need a prespecified missing-value protocol"}
                else:
                    result = _run_arm(arm, name, path, args.trees, args.threads, args.ntime)
                record = {"kind": "result", "case": name, "arm": arm, **info, **result}
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(map(json.dumps, records)) + "\n")


if __name__ == "__main__":
    main()
