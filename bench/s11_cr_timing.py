"""S11 accept check: competing-risks fit time and forest size vs the single-event forest.

    python -m bench.s11_cr_timing [n_rows] [n_trees]

Data: `perf_fit.synth` counting-process rows (5 per id); for J causes, each
event gets a uniformly random cause label. Fit time is the best of 3 runs.
"""

import sys
import time

import numpy as np

from bench.perf_fit import synth
from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y


def best_fit(make, X, y, ids, reps=3):
    best, model = np.inf, None
    for _ in range(reps):
        m = make()
        t0 = time.perf_counter()
        m.fit(X, y, ids)
        best = min(best, time.perf_counter() - t0)
        model = m
    return best, model


def main(n=100_000, trees=100):
    X, y, ids = synth(n, rows_per_id=5, seed=0)
    kw = dict(n_estimators=trees, n_jobs=-1, random_state=0)
    rows = []
    t, m = best_fit(lambda: SurvivalForestTV(**kw), X, y, ids)
    rows.append(("SurvivalForestTV", 1, t, m.forest_.nbytes))
    rng = np.random.default_rng(1)
    for J in (1, 2, 4):
        ev = np.where(y["event"], rng.integers(1, J + 1, len(y)), 0)
        y_cr = make_competing_risks_y(y["stop"], ev, start=y["start"])
        t, m = best_fit(lambda: CompetingRisksForestTV(**kw), X, y_cr, ids)
        rows.append(("CompetingRisksForestTV", J, t, m.forest_.nbytes))
    base_t, base_b = rows[0][2], rows[0][3]
    print(f"n_rows={n:,} trees={trees} events={int(y['event'].sum()):,} grid={len(m.event_times_):,}")
    print("| model | J | fit s | x J=1 | forest MB | x J=1 |\n|---|---|---|---|---|---|")
    for name, J, t, b in rows:
        print(f"| {name} | {J} | {t:.2f} | {t / base_t:.2f} | {b / 1e6:.1f} | {b / base_b:.2f} |")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
