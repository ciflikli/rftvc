"""Regenerate tests/fixtures/concordance_cp.json with R survival::concordance (needs Rscript).

Counting-process data with time-varying risk scores, rounded times (tied
event times) and rounded scores (tied risks). Run from the repo root:
python tests/fixtures/make_concordance_fixtures.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("concordance_cp.json")


def counting_process(rng, n_ids, max_rows, digits):
    start, stop, event, risk, ids = [], [], [], [], []
    for i in range(n_ids):
        t = float(np.round(rng.uniform(0, 1), digits))  # delayed entry
        k = rng.integers(1, max_rows + 1)
        for j in range(k):
            nxt = float(np.round(t + rng.exponential(1.0) + 10.0 ** -digits, digits))
            start.append(t)
            stop.append(nxt)
            event.append(int(j == k - 1 and rng.random() < 0.6))
            risk.append(float(np.round(rng.normal(), 1)))
            ids.append(i)
            t = nxt
    return start, stop, event, risk, ids


def datasets():
    rng = np.random.default_rng(20260925)
    yield "right_censored", *counting_process(rng, 60, 1, 1)
    yield "cp_small", *counting_process(rng, 25, 3, 1)
    yield "cp_n150", *counting_process(rng, 150, 4, 2)


def main():
    fixtures = {}
    for name, start, stop, event, risk, ids in datasets():
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "d.csv"
            np.savetxt(csv, np.column_stack([start, stop, event, risk]), delimiter=",",
                       header="start,stop,event,risk", comments="")
            r = (f'library(survival); d <- read.csv("{csv}"); '
                 'cat(sprintf("%.17g", concordance(Surv(start, stop, event) ~ risk, data = d, reverse = TRUE)$concordance))')
            c = float(subprocess.run(["Rscript", "-e", r], capture_output=True, text=True, check=True).stdout)
        fixtures[name] = {"start": start, "stop": stop, "event": event, "risk": risk, "ids": ids, "concordance": c}
    OUT.write_text(json.dumps(fixtures, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
