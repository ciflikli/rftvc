"""Regenerate tests/fixtures/survdiff.json with R survival::survdiff (needs Rscript).

Run from the repo root: python tests/fixtures/make_survdiff_fixtures.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("survdiff.json")


def datasets():
    rng = np.random.default_rng(20260925)
    yield "tiny_ties", [1, 1, 2, 3, 3, 3, 4, 5, 6, 6], [1, 0, 1, 1, 1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    n = 200
    g = rng.integers(0, 2, n)
    t = np.round(rng.exponential(1 + 0.8 * g), 1) + 0.1
    e = rng.random(n) < 0.7
    yield "rounded_n200", t.tolist(), e.astype(int).tolist(), g.tolist()
    n = 150
    g = rng.integers(0, 2, n)
    t = np.round(rng.weibull(1.5, n) * (1 + 0.5 * g), 2) + 0.01
    e = rng.random(n) < 0.3
    yield "heavy_censoring", t.tolist(), e.astype(int).tolist(), g.tolist()


def main():
    fixtures = {}
    for name, time, event, group in datasets():
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "d.csv"
            np.savetxt(csv, np.column_stack([time, event, group]), delimiter=",", header="time,event,group", comments="")
            r = f'library(survival); d <- read.csv("{csv}"); cat(sprintf("%.17g", survdiff(Surv(time, event) ~ group, data = d)$chisq))'
            chisq = float(subprocess.run(["Rscript", "-e", r], capture_output=True, text=True, check=True).stdout)
        fixtures[name] = {"time": time, "event": event, "group": group, "chisq": chisq}
    OUT.write_text(json.dumps(fixtures, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
