"""S16 T8: PE-score peak memory (1M rows, M = 64) and oob_cumhaz vs oob_mortality time."""

import time
import tracemalloc

import numpy as np

from bench.perf_fit import synth
from rftvc import SurvivalForestTV, make_survival_y
from rftvc.metrics import piecewise_exponential_score


def memory(n=1_000_000, M=64):
    rng = np.random.default_rng(0)
    start = rng.uniform(0, 5, n)
    y = make_survival_y(start + rng.exponential(1, n), rng.random(n) < 0.3, start=start)
    w = np.linspace(0, 8, M + 1)
    H = np.cumsum(np.c_[np.zeros(n), rng.exponential(0.1, (n, M))], axis=1)
    null = np.cumsum(np.r_[0, np.full(M, 0.1)])
    tracemalloc.start()
    piecewise_exponential_score(y, H, w, null_cumhaz=null, ids=np.arange(n) // 4)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"PE score peak extra memory {peak / 2**20:.0f} MiB vs prediction array {H.nbytes / 2**20:.0f} MiB "
          f"(ratio {peak / H.nbytes:.2f}; gate: total <= 2x, i.e. extra <= 1x)")


def timing(n_rows=100_000):
    X, y, ids = synth(n_rows, rows_per_id=4, seed=0)
    m = SurvivalForestTV(n_estimators=100, random_state=0, n_jobs=-1).fit(X, y, ids)
    d = m._rebuild_design(X, y, ids)
    w = np.quantile(m.event_times_, np.linspace(0, 1, 9))
    t = m.event_times_
    for name, f in [("oob_mortality (all event times)", lambda: m.forest_.oob_mortality(d.X, *d.oob_set, t, "hazard", 8)),
                    ("oob_cumhaz (all event times)", lambda: m.forest_.oob_cumhaz(d.X, *d.oob_set, t, "hazard", 8)),
                    ("oob_cumhaz (9 edges)", lambda: m.forest_.oob_cumhaz(d.X, *d.oob_set, w, "hazard", 8))]:
        best = min(_time(f) for _ in range(3))
        print(f"{name}: {best:.3f}s on {len(X):,} rows")


def _time(f):
    t = time.perf_counter()
    f()
    return time.perf_counter() - t


if __name__ == "__main__":
    memory()
    timing()
