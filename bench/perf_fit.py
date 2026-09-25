"""One timed fit for profiling: python -m bench.perf_fit N [ntime] [rows_per_id] [n_trees]."""

import sys
import time

import numpy as np

from rftvc import SurvivalForestTV, make_survival_y


def synth(n, p=10, rows_per_id=1, seed=0):
    """Right-censored data (rows_per_id=1) or counting-process rows with a time-varying x0."""
    rng = np.random.default_rng(seed)
    n_ids = n // rows_per_id
    base = rng.normal(size=(n_ids, p))
    t = rng.exponential(np.exp(-(0.7 * base[:, 0] + 0.5 * (base[:, 1] > 0))))
    c = rng.exponential(1.5, size=n_ids)
    u, e = np.minimum(t, c), t <= c
    if rows_per_id == 1:
        return base, make_survival_y(u, e), None
    cuts = np.sort(rng.uniform(0, 1, size=(n_ids, rows_per_id - 1)), axis=1) * u[:, None]
    start = np.column_stack([np.zeros(n_ids), cuts]).ravel()
    stop = np.column_stack([cuts, u]).ravel()
    ev = np.zeros((n_ids, rows_per_id), bool)
    ev[:, -1] = e
    X = np.repeat(base, rows_per_id, axis=0)
    X[:, 0] += rng.normal(scale=0.3, size=X.shape[0])
    ids = np.repeat(np.arange(n_ids), rows_per_id)
    return X, make_survival_y(stop, ev.ravel(), start=start), ids


def main():
    n = int(sys.argv[1])
    ntime = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "none" else None
    rpi = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    trees = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    X, y, ids = synth(n, rows_per_id=rpi)
    kw = {} if ntime is None else {"ntime": ntime}
    f = SurvivalForestTV(n_estimators=trees, n_jobs=-1, random_state=0, **kw)
    t0 = time.perf_counter()
    f.fit(X, y, ids)
    print(f"fit n={n} ntime={ntime} rows_per_id={rpi} trees={trees}: {time.perf_counter() - t0:.2f}s "
          f"K={len(f.event_times_)} leaves/tree={np.mean([f.forest_.n_leaves(b) for b in range(trees)]):.0f}")


if __name__ == "__main__":
    main()
