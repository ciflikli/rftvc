"""Reproducible fit-time matrix for survival forests.

Run ``.venv/bin/python -m bench.fit_matrix --quick --repeats 2`` for a smoke
benchmark. Every warm-up and measured repetition runs in a fresh process; the
same generated CSV is shared by Python and R. Data loading and scoring are
outside fit/predict timing. Results are JSON lines, optionally saved to a file.

Only one-row-per-subject cases are sent to competitor packages: their ordinary
forest interfaces do not implement rftvc's path fit. Competing-risk cases use
rftvc, comprisk, and randomForestSRC. Comparison is at the task/output level,
not an assertion that split rules and leaf floors are identical.
Leaf-size rules and feature threshold searches differ across libraries, so
compare predictive quality as well as elapsed time before drawing conclusions.
"""

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
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
ARMS = ("rftvc", "sksurv", "sksurv_curve", "comprisk", "ranger", "rfsrc")
CASES = {
    "static": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5),
    "rare_event": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.1),
    "wide": dict(n_ids=5_000, p=100, rows_per_id=1, event_fraction=0.5),
    "tied": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, ties=True),
    "missing_1pct": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, missing=0.01),
    "missing_5pct": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, missing=0.05),
    "missing": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, missing=0.2),
    "tvc": dict(n_ids=2_000, p=10, rows_per_id=5, event_fraction=0.5),
    "competing": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.5, competing=True),
    "competing_rare": dict(n_ids=10_000, p=10, rows_per_id=1, event_fraction=0.2, competing=True),
}


def make_case(path, spec, seed=0):
    """One event per subject at most; same input rows for every arm."""
    rng = np.random.default_rng(seed)
    n, p, rpi = spec["n_ids"], spec["p"], spec["rows_per_id"]
    base = rng.normal(size=(n, p))
    scale = np.exp(-0.6 * base[:, 0] - 0.3 * base[:, 1])
    if spec.get("competing"):
        # Two cause-specific hazards depend on different features, so neither
        # cause is just a random label assigned after observing all-cause time.
        failure_1 = rng.exponential(np.exp(-0.6 * base[:, 0]))
        failure_2 = rng.exponential(np.exp(-0.6 * base[:, 1]))
        failure = np.minimum(failure_1, failure_2)
        cause_id = np.where(failure_1 <= failure_2, 1, 2)
    else:
        failure = rng.exponential(scale)
        cause_id = np.ones(n, dtype=int)
    # Independent censoring; use a pilot quantile to vary the event fraction.
    censor = rng.exponential(np.quantile(failure, spec["event_fraction"]), n)
    stop_id = np.minimum(failure, censor) + 0.01
    event_id = np.where(failure <= censor, cause_id, 0)
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
        event = np.zeros((n, rpi), int)
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
    return {"n_rows": len(X), "n_ids": n, "p": p, "event_fraction_realized": float(np.mean(event_id != 0)),
            "missing_fraction": missing, "ties": bool(spec.get("ties")), "rows_per_id": rpi,
            "competing": bool(spec.get("competing"))}


def _rss_gb():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1e9 if sys.platform == "darwin" else 1e6)


def run_python(arm, path, trees, threads, ntime, case):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    ids = data[:, 0].astype(np.int64)
    start, stop = data[:, 1], data[:, 2]
    event = data[:, 3].astype(int)
    X = data[:, 4:]
    held_out = ids % 5 == 0
    is_tvc = np.unique(ids).size != len(ids)
    is_cr = bool(CASES[case].get("competing"))
    X_train, X_test = X[~held_out], X[held_out]
    start_train, stop_train, event_train = start[~held_out], stop[~held_out], event[~held_out]
    version = importlib.metadata.version("rftvc" if arm == "rftvc" else
                                         "scikit-survival" if arm.startswith("sksurv") else arm)
    if arm == "rftvc":
        if is_cr:
            from rftvc import CompetingRisksForestTV, make_competing_risks_y
            model = CompetingRisksForestTV(
                n_estimators=trees, min_ids_leaf=15, min_events_leaf=1, max_features="sqrt",
                bootstrap=True, max_samples=0.632, n_jobs=threads, ntime=ntime, random_state=0,
                score_cause=1,
            )
            y = make_competing_risks_y(stop_train, event_train, start=start_train)
            predict = lambda: model.predict_cumulative_incidence(X_test[:1000], [float(np.median(stop_train))], cause=1)[:, 0]
        else:
            model = SurvivalForestTV(
                n_estimators=trees, min_ids_leaf=15, min_events_leaf=1, max_features="sqrt",
                bootstrap=True, max_samples=0.632, n_jobs=threads, ntime=ntime, random_state=0,
            )
            y = make_survival_y(stop_train, event_train != 0, start=start_train)
            predict = lambda: model.predict(X_test[:1000])
        fit = lambda: model.fit(X_train, y, ids=ids[~held_out])
    elif arm.startswith("sksurv"):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv

        model = RandomSurvivalForest(
            n_estimators=trees, min_samples_leaf=15, min_samples_split=30,
            max_features="sqrt", bootstrap=True, max_samples=0.632,
            n_jobs=threads, random_state=0, low_memory=arm == "sksurv",
        )
        y = Surv.from_arrays(event_train != 0, stop_train)
        fit = lambda: model.fit(X_train, y)
        predict = lambda: model.predict(X_test[:1000])  # same mortality output in both memory modes
    elif arm == "comprisk":
        from comprisk import CompetingRiskForest
        model = CompetingRiskForest(
            n_estimators=trees, min_samples_leaf=15, max_features="sqrt",
            n_jobs=threads, random_state=0,
        )
        fit = lambda: model.fit(X_train, stop_train, event_train)
        predict = lambda: model.predict_cif(X_test[:1000], times=[float(np.median(stop_train))])[:, 0, 0]
    else:
        raise ValueError(arm)
    t0 = time.perf_counter()
    fit()
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = predict()
    predict_s = time.perf_counter() - t0
    result = dict(fit_s=fit_s, predict_1000_s=predict_s, peak_rss_gb=_rss_gb(),
                  n_train=len(X_train), n_test=len(X_test), n_eval=min(1000, len(X_test)),
                  package_version=version, prediction_kind="cif_cause1_median" if is_cr else "mortality")
    if not is_tvc:
        if is_cr:
            from rftvc import make_competing_risks_y
            from rftvc.metrics import concordance_index_cr
            y_test = make_competing_risks_y(stop[held_out][:1000], event[held_out][:1000])
            result["test_c"] = float(concordance_index_cr(y_test, pred, cause=1))
        else:
            from sksurv.metrics import concordance_index_censored
            result["test_c"] = float(concordance_index_censored(event[held_out][:1000] != 0,
                                                                  stop[held_out][:1000], pred)[0])
    return result


def _run_child(args):
    if args.arm in ("rftvc", "sksurv", "sksurv_curve", "comprisk"):
        return run_python(args.arm, args.data, args.trees, args.threads, args.ntime, args.case)
    raise ValueError(args.arm)


def _run_arm(arm, case, path, trees, threads, ntime):
    if arm in ("comprisk", "sksurv", "sksurv_curve"):
        module = "comprisk" if arm == "comprisk" else "sksurv"
        if importlib.util.find_spec(module) is None:
            return {"status": "unavailable", "reason": f"Python package {module} is unavailable"}
    if arm in ("ranger", "rfsrc"):
        cmd = ["Rscript", str(ROOT / "bench" / "fit_matrix.R"), arm, str(path), str(trees), str(threads), case]
    else:
        cmd = [sys.executable, "-m", "bench.fit_matrix", "--child", "--arm", arm,
               "--case", case, "--data", str(path), "--trees", str(trees),
               "--threads", str(threads), "--ntime", str(ntime) if ntime is not None else "none"]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if proc.returncode == 2 and " is unavailable" in proc.stderr:
        return {"status": "unavailable", "reason": proc.stderr.strip()}
    if proc.returncode != 0 and "ModuleNotFoundError: No module named 'comprisk'" in proc.stderr:
        return {"status": "unavailable", "reason": "comprisk is not installed"}
    if proc.returncode != 0:
        return {"status": "failed", "reason": proc.stderr.strip()[-400:]}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{\"fit_s\"")]
    if not lines:
        return {"status": "failed", "reason": proc.stdout[-400:] or proc.stderr[-400:]}
    result = json.loads(lines[-1])
    if "pred" in result:
        from rftvc import make_competing_risks_y
        from rftvc.metrics import concordance_index_cr

        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        test = data[data[:, 0].astype(np.int64) % 5 == 0][:1000]
        result["test_c"] = float(concordance_index_cr(
            make_competing_risks_y(test[:, 2], test[:, 3].astype(int)),
            np.asarray(result.pop("pred")), cause=1,
        ))
    return {"status": "ok", **result}


def summarize(records):
    """Median and range for every measured successful case/arm cell."""
    cells = sorted({(r["case"], r["arm"]) for r in records if r.get("kind") == "result"})
    out = []
    for case, arm in cells:
        rows = [r for r in records if r.get("kind") == "result" and r["case"] == case and r["arm"] == arm]
        ok = [r for r in rows if r["status"] == "ok"]
        row = {"kind": "summary", "case": case, "arm": arm, "n_ok": len(ok), "n_attempted": len(rows)}
        if ok:
            for key in ("fit_s", "predict_1000_s", "peak_rss_gb", "test_c"):
                values = [r[key] for r in ok if r.get(key) is not None]
                if values:
                    row[key + "_median"] = statistics.median(values)
                    row[key + "_min"] = min(values)
                    row[key + "_max"] = max(values)
        else:
            row["status"] = rows[0]["status"]
            row["reason"] = rows[0].get("reason")
        out.append(row)
    return out


def report_markdown(records, raw_name=None):
    meta = records[0]
    lines = [
        "# Repeated survival-forest fit matrix",
        "",
        f"Python {meta['python']}; {meta['platform']}; {meta['cpu_count']} logical CPUs; "
        f"{meta['trees']} trees; {meta['threads']} threads; "
        f"{meta['repeats']} measured runs and {meta['warmups']} discarded warm-up "
        f"{'run' if meta['warmups'] == 1 else 'runs'} per cell.",
        "",
        "All repetitions use the same generated rows, held-out subjects, and model seed; "
        "spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. "
        "Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. "
        "Peak RSS, where available, covers the whole process. Leaf rules and split search differ.",
        "",
        "Static arms predict mortality; competing-risk arms score cause-1 CIF at the "
        "training median follow-up. The R competing-risk prediction computes its native full "
        "CIF grid before extracting that horizon, while Python arms request one horizon. "
        "Interpret those prediction times separately. Harrell C (static) and Wolbers C "
        "(competing) are computed on the held-out subjects.",
        "",
        "| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    versions = {(r["case"], r["arm"]): r.get("package_version") for r in records
                if r.get("kind") == "result" and r.get("package_version")}
    sizes = {r["case"]: f"{r['n_train']}/{r['n_test']}" for r in records
             if r.get("kind") == "result" and r.get("n_train") is not None}
    for row in records:
        if row["kind"] != "summary":
            continue
        case, arm = row["case"], row["arm"]
        size = sizes.get(case, "—")
        version = versions.get((case, arm))
        label = f"{arm} {version}" if version else arm
        if row["n_ok"]:
            def fmt(key):
                return (f"{row[key + '_median']:.4f} "
                        f"[{row[key + '_min']:.4f}, {row[key + '_max']:.4f}]")
            c = f"{row['test_c_median']:.3f}" if "test_c_median" in row else "—"
            rss = f"{row['peak_rss_gb_median']:.3f}" if "peak_rss_gb_median" in row else "—"
            lines.append(f"| {case} | {size} | {label} | {fmt('fit_s')} | {fmt('predict_1000_s')} | "
                         f"{rss} | {c} | {row['n_ok']}/{row['n_attempted']} |")
        else:
            reason = row.get("reason") or row["status"]
            lines.append(f"| {case} | {size} | {label} | {row['status']}: {reason} | — | — | — | 0/{row['n_attempted']} |")
    raw = f"[Raw JSONL]({raw_name})" if raw_name else "Raw JSONL"
    lines.extend(["", f"{raw} retains each repetition and full configuration.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, help="write a Markdown summary of this run")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5, help="measured fresh-process runs per cell")
    parser.add_argument("--warmups", type=int, default=1, help="discarded fresh-process runs per cell")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--n-ids", type=int, help="override subjects per case (for scaled comparison runs)")
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
    if args.trees < 1 or args.threads < 1 or args.repeats < 1 or args.warmups < 0 or (args.n_ids is not None and args.n_ids < 20):
        parser.error("--trees, --threads and --repeats must be positive; --warmups must be nonnegative")
    import rftvc
    meta = {"kind": "metadata", "rftvc": rftvc.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "trees": args.trees, "threads": args.threads, "ntime": args.ntime,
            "repeats": args.repeats, "warmups": args.warmups,
            "n_ids_override": args.n_ids,
            "split": "id % 5 == 0", "seed": 0,
            "cpu_count": os.cpu_count()}
    records = [meta]
    names = args.cases or (("static", "competing", "tvc") if args.quick else tuple(CASES))
    arms = args.arms or ARMS
    with tempfile.TemporaryDirectory(prefix="rftvc-bench-") as tmp:
        for name in names:
            spec = dict(CASES[name])
            if args.quick:
                spec["n_ids"] = 400 if name != "tvc" else 100
            if args.n_ids is not None:
                spec["n_ids"] = args.n_ids
            path = Path(tmp) / f"{name}.csv"
            info = make_case(path, spec)
            for arm in arms:
                if info["rows_per_id"] > 1 and arm != "rftvc":
                    result = {"status": "unsupported", "reason": "path data requires counting-process fit"}
                elif info["competing"] and arm in ("sksurv", "sksurv_curve", "ranger"):
                    result = {"status": "unsupported", "reason": "no native competing-risk forest"}
                elif not info["competing"] and arm == "comprisk":
                    result = {"status": "unsupported", "reason": "competing-risk forest only"}
                elif info["missing_fraction"] and arm in ("ranger", "rfsrc"):
                    result = {"status": "unsupported", "reason": "R arms need a prespecified missing-value protocol"}
                else:
                    for _ in range(args.warmups):
                        warm = _run_arm(arm, name, path, args.trees, args.threads, args.ntime)
                        if warm["status"] != "ok":
                            result = warm
                            break
                    else:
                        result = None
                if result is not None:
                    record = {"kind": "result", "case": name, "arm": arm, **info, "repeat": 0, **result}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    continue
                for repeat in range(1, args.repeats + 1):
                    result = _run_arm(arm, name, path, args.trees, args.threads, args.ntime)
                    record = {"kind": "result", "case": name, "arm": arm, **info, "repeat": repeat, **result}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    if result["status"] != "ok":
                        break
    records.extend(summarize(records))
    for record in records:
        if record["kind"] == "summary":
            print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(map(json.dumps, records)) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        raw_name = args.output.name if args.output and args.output.parent == args.report.parent else None
        args.report.write_text(report_markdown(records, raw_name))


if __name__ == "__main__":
    main()
