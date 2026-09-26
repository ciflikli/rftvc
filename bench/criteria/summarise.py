"""Apply the pre-registered S8 decision rule (docs/plans/s8-plan.md, item 8) to the bake-off CSVs.

    .venv/bin/python -m bench.criteria.summarise [--out docs/bench/s8-bakeoff]

Writes ``decisions.csv``: one row per rule check with its inputs and outcome.
Differences are arm − reference, so negative favours the arm for IBS, ICI
and ISE. "Favours the reference" means the whole interval is above 0.
"""

import argparse
from pathlib import Path

import polars as pl

from .common import AGGREGATES
from .run import OUT

LANDMARK = ("pbc2", "panel")
MAX_TIME_RATIO = 2.0


def _row(cmp, dataset, arm, ref, metric):
    r = cmp.filter((pl.col("dataset") == dataset) & (pl.col("arm") == arm) & (pl.col("reference") == ref)
                   & (pl.col("metric") == metric))
    return None if r.is_empty() else r.row(0, named=True)


def challenger_checks(c, a_cmp, a_sum, sim_cmp):
    """Rule items i–iv for challenger ``c`` at each aggregation."""
    out = []
    for agg in AGGREGATES:
        arm, ref = f"{c}/{agg}", f"logrank/{agg}"
        s = _row(sim_cmp, "sim", arm, ref, "ise") if sim_cmp is not None else None
        out.append({"check": "i_sim_ise", "arm": arm, "dataset": "sim",
                    "value": None if s is None else s["diff"], "bound": None if s is None else s["diff"] + 2 * s["se"],
                    "pass": s is not None and s["diff"] + 2 * s["se"] < 0})
        ibs = [_row(a_cmp, d, arm, ref, "ibs") for d in LANDMARK]
        ici = [_row(a_cmp, d, arm, ref, "ici") for d in LANDMARK]
        for d, r in zip(LANDMARK, ibs):
            out.append({"check": "ii_ibs_not_worse", "arm": arm, "dataset": d, "value": r and r["diff"],
                        "bound": r and r["lo"], "pass": r is not None and not r["lo"] > 0})
        out.append({"check": "ii_ibs_better_somewhere", "arm": arm, "dataset": "+".join(LANDMARK), "value": None,
                    "bound": min((r["hi"] for r in ibs if r), default=None),
                    "pass": any(r is not None and r["hi"] < 0 for r in ibs)})
        for d, r in zip(LANDMARK, ici):
            out.append({"check": "iii_ici_not_worse", "arm": arm, "dataset": d, "value": r and r["diff"],
                        "bound": r and r["lo"], "pass": r is not None and not r["lo"] > 0})
        for d in LANDMARK:
            t = a_sum.filter(pl.col("dataset") == d)
            ta = t.filter(pl.col("arm") == arm)["fit_seconds"]
            tr = t.filter(pl.col("arm") == ref)["fit_seconds"]
            ratio = float(ta[0] / tr[0]) if len(ta) and len(tr) else None
            out.append({"check": "iv_fit_time", "arm": arm, "dataset": d, "value": ratio, "bound": MAX_TIME_RATIO,
                        "pass": ratio is not None and ratio <= MAX_TIME_RATIO})
    return out


def d11_checks(a_cmp):
    arm, ref = "logrank/survival", "logrank/hazard"
    out = []
    for d in LANDMARK:
        ici, ibs = _row(a_cmp, d, arm, ref, "ici"), _row(a_cmp, d, arm, ref, "ibs")
        out.append({"check": "d11_ici_better", "arm": arm, "dataset": d, "value": ici and ici["diff"],
                    "bound": ici and ici["hi"], "pass": ici is not None and ici["hi"] < 0})
        out.append({"check": "d11_ibs_not_worse", "arm": arm, "dataset": d, "value": ibs and ibs["diff"],
                    "bound": ibs and ibs["lo"], "pass": ibs is not None and not ibs["lo"] > 0})
    return out


def decide(out):
    out = Path(out)
    a_cmp, a_sum = pl.read_csv(out / "a_compare.csv"), pl.read_csv(out / "a_summary.csv")
    sim_path = out / "b_sim_compare.csv"
    sim_cmp = pl.read_csv(sim_path) if sim_path.exists() else None
    challengers = sorted({a.split("/")[0] for a in a_sum["arm"]} - {"logrank"})
    rows = []
    for c in challengers:
        rows += [{"decision": f"default -> {c}", **r} for r in challenger_checks(c, a_cmp, a_sum, sim_cmp)]
    rows += [{"decision": "D11 -> survival", **r} for r in d11_checks(a_cmp)]
    checks = pl.DataFrame(rows, schema_overrides={"value": pl.Float64, "bound": pl.Float64})
    checks.write_csv(out / "decisions.csv")
    verdict = checks.group_by("decision", maintain_order=True).agg(pl.col("pass").all().alias("adopt"))
    return checks, verdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    checks, verdict = decide(args.out)
    with pl.Config(tbl_rows=-1, tbl_width_chars=200):
        print(checks)
        print(verdict)


if __name__ == "__main__":
    main()
