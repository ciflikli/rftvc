"""Regenerate tests/fixtures/aj_survfit.json with R survival::survfit (needs Rscript).

Aalen–Johansen on counting-process rows with ids, delayed entry and ties:
``survfit(Surv(start, stop, factor(cause, 0:J)) ~ 1, id = id)``; ``pstate``
columns are the event-free state, then causes 1..J.

Run from the repo root: python tests/fixtures/make_aj_fixtures.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("aj_survfit.json")


def _chains(rng, n_ids, n_causes, p_event):
    """Contiguous rows per id: random entry, 1-3 rows, a terminal cause label or censoring."""
    ids, start, stop, cause = [], [], [], []
    for i in range(n_ids):
        t = float(rng.integers(0, 4)) * 0.5  # delayed entry
        n_rows = int(rng.integers(1, 4))
        for r in range(n_rows):
            end = t + float(rng.integers(1, 5)) * 0.5  # half-units: many ties
            last = r == n_rows - 1
            ids.append(i)
            start.append(t)
            stop.append(end)
            cause.append(int(rng.integers(1, n_causes + 1)) if last and rng.random() < p_event else 0)
            t = end
    return ids, start, stop, cause


def datasets():
    # Hand example: delayed entry (ids 2, 4), a cross-cause tie at 2.0, censoring
    # at an event time (id 5 at 2.0), and a time only cause 2 has (1.5).
    yield "tiny", 2, (
        [1, 1, 2, 3, 3, 4, 5, 6, 7],
        [0.0, 1.0, 0.5, 0.0, 1.0, 1.2, 0.0, 0.0, 0.0],
        [1.0, 2.0, 1.5, 1.0, 2.0, 3.0, 2.0, 2.5, 3.0],
        [0, 1, 2, 0, 2, 1, 0, 1, 0],
    )
    rng = np.random.default_rng(20260926)
    yield "random_j2", 2, _chains(rng, 80, 2, 0.6)
    yield "random_j3", 3, _chains(rng, 120, 3, 0.5)


def main():
    fixtures = {}
    for name, n_causes, (ids, start, stop, cause) in datasets():
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "d.csv"
            np.savetxt(csv, np.column_stack([ids, start, stop, cause]), delimiter=",",
                       header="id,start,stop,cause", comments="")
            r = (
                f'library(survival); d <- read.csv("{csv}"); '
                f"d$cause <- factor(d$cause, levels = 0:{n_causes}); "
                "f <- survfit(Surv(start, stop, cause) ~ 1, data = d, id = id); "
                'stopifnot(identical(f$states, c("(s0)", as.character(1:' + str(n_causes) + ")))); "
                'm <- cbind(f$time, f$pstate); cat(apply(m, 1, function(r) paste(sprintf("%.17g", r), collapse = ",")), sep = "\\n")'
            )
            out = subprocess.run(["Rscript", "-e", r], capture_output=True, text=True, check=True).stdout
        table = np.array([[float(v) for v in line.split(",")] for line in out.strip().splitlines()])
        fixtures[name] = {
            "n_causes": n_causes,
            "id": [int(i) for i in ids],
            "start": start,
            "stop": stop,
            "cause": cause,
            "time": table[:, 0].tolist(),
            "pstate": table[:, 1:].tolist(),
        }
    OUT.write_text(json.dumps(fixtures, indent=1))


if __name__ == "__main__":
    main()
