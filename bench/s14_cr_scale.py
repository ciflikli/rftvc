"""S14 scale: fit time, peak RSS and leaf bytes for J in {1, 2, 4} (not a merge gate).

    python -m bench.s14_cr_scale            # all arms, each in its own process
    python -m bench.s14_cr_scale one N J NTIME

Data: `bench.perf_fit.synth` counting-process rows (5 per id). J = 1 is
SurvivalForestTV; for J > 1 each event gets a uniformly random cause in 1..J.
100 trees, 10 threads. Leaf bytes from the pickled state: 4 E + 8 J E (+ 4 J L
leaf counts for J > 1); node bytes = forest_.nbytes - leaf bytes.
"""

import json
import re
import subprocess
import sys
import time

import numpy as np

from bench.s14_cr_sim import OUT


def one(n, J, ntime):
    from bench.perf_fit import synth
    from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y

    X, y, ids = synth(n, rows_per_id=5, seed=0)
    kw = dict(n_estimators=100, n_jobs=10, random_state=0, ntime=None if ntime == 0 else ntime)
    if J == 1:
        m = SurvivalForestTV(**kw)
    else:
        rng = np.random.default_rng(1)
        ev = np.where(y["event"], rng.integers(1, J + 1, len(y)), 0)
        y = make_competing_risks_y(y["stop"], ev, start=y["start"])
        m = CompetingRisksForestTV(**kw)
    t0 = time.perf_counter()
    m.fit(X, y, ids)
    fit_s = time.perf_counter() - t0
    state = m.forest_.__reduce__()[1][0]
    E, L = len(state["event_idx"]), int(state["leaf_offsets"][-1])
    leaf = 4 * E + 8 * J * E + (4 * J * L if J > 1 else 0)
    print(json.dumps(dict(n=n, J=J, ntime=ntime, fit_s=fit_s, entries=E, leaves=L, leaf_bytes=leaf,
                          node_bytes=m.forest_.nbytes - leaf, forest_bytes=m.forest_.nbytes)))


def main():
    rows = []
    for n in (100_000, 1_000_000):
        for ntime in (100, 0):
            for J in (1, 2, 4):
                cmd = ["/usr/bin/time", "-l", sys.executable, "-m", "bench.s14_cr_scale", "one", str(n), str(J), str(ntime)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                rec = json.loads(p.stdout.strip().splitlines()[-1])
                rec["peak_rss_gb"] = int(re.search(r"(\d+)\s+maximum resident set size", p.stderr).group(1)) / 1e9
                rows.append(rec)
                print(rec, flush=True)
    import polars as pl

    pl.DataFrame(rows).write_csv(OUT / "scale.csv")


if __name__ == "__main__":
    if sys.argv[1:2] == ["one"]:
        one(*(int(a) for a in sys.argv[2:5]))
    else:
        main()
