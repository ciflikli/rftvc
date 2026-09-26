"""S8 bake-off driver (docs/plans/s8-plan.md).

Run from the repo root:

    .venv/bin/python -m bench.criteria.run            # full pre-registered configuration
    .venv/bin/python -m bench.criteria.run --smoke    # seconds; checks the pipeline only

Writes CSVs to ``docs/bench/s8-bakeoff/`` (``--out`` to change) and raw
out-of-fold predictions to ``<out>/raw/`` (git-ignored):

- ``a_scores.csv``: per dataset / arm / outer fold / landmark scores and selected params;
- ``a_summary.csv``: pooled IBS, Brier at w, ICI, slope, fit time per arm;
- ``a_compare.csv``: paired cluster-bootstrap differences vs the reference arm;
- ``b_sim.csv``, ``b_sim_compare.csv``: per-replication ISE and paired t-intervals;
- ``b_cl.csv``: war data C per fold and arm (supporting only);
- ``config.json``: the configuration used.
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import polars as pl

from . import protocol_a, protocol_b
from .common import FULL, SMOKE

OUT = Path(__file__).resolve().parents[2] / "docs" / "bench" / "s8-bakeoff"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--datasets", help="comma-separated subset of pbc2,panel,sim,cl")
    args = ap.parse_args(argv)
    cfg = SMOKE if args.smoke else FULL
    if args.datasets:
        cfg = replace(cfg, datasets=tuple(args.datasets.split(",")))
    run(cfg, args.out)


def run(cfg, out, log=print):
    out = Path(out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
    t0 = time.perf_counter()
    a = [protocol_a.run(d, cfg, raw, log) for d in cfg.datasets if d in protocol_a.DATASETS]
    if a:
        for i, name in enumerate(("a_scores", "a_summary", "a_compare")):
            pl.concat([r[i] for r in a], how="diagonal").write_csv(out / f"{name}.csv")
    if "sim" in cfg.datasets:
        res, compare = protocol_b.run_sim(cfg, log)
        res.write_csv(out / "b_sim.csv")
        compare.write_csv(out / "b_sim_compare.csv")
    if "cl" in cfg.datasets:
        protocol_b.run_cl(cfg, log).write_csv(out / "b_cl.csv")
    log(f"done in {time.perf_counter() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
