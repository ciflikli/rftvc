"""S20 T9: path_effect simulation with a known truth (not a merge gate).

    python -m bench.tvc_path_effect_sim pilot          # 10-rep MC-SE pilot
    python -m bench.tvc_path_effect_sim run [n_reps]   # R=50 (default), writes docs/bench/s20-effects/path_effect.csv

Reuses ``tests.sim``'s S3 generator (external ``z`` redrawn per unit interval,
Cox-type hazard with a threshold): fits ``SurvivalForestTV`` on S3 training
rows, then calls ``inspection.path_effect`` on **fresh, fully-specified**
external-covariate test paths (not S3's own ``rows()``, which truncates each
subject's path at its own event/censoring time and so cannot supply a
covariate value at every horizon -- the same construction ``tests/sim.py``'s
own ``replicate()`` already uses for its prediction check).

The simulation uses ``feature="z"``
(column 1), ``delta=1.0``, ``from_time=3.0`` (a unit-interval boundary, so the
base case needs no mid-interval row split -- that is covered separately by a
hand-built unit test), ``horizons=[4.0, 5.0, 6.0]`` (all ``>= from_time``,
inside ``[0, HORIZON]``).

Oracle: for a large i.i.d. sample, the true ``Delta risk`` at each horizon,
computed directly from the known piecewise-constant hazard (``hazard``) with
``z`` shifted by ``delta`` on every interval at or after ``from_time``.

Pass rule: over R=50 replications, ``|mean bias|
<= 0.1 * |true_delta|`` at each horizon.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rftvc import SurvivalForestTV, inspection, make_survival_y
from tests.sim import K, hazard, rows, simulate

OUT = Path(__file__).resolve().parents[1] / "docs" / "bench" / "s20-effects"
FEATURE, DELTA, FROM_TIME = 1, 1.0, 3.0
HORIZONS = np.array([4.0, 5.0, 6.0])


def _risk_at(lam, t):
    """``1 - exp(-H(t))`` from per-interval hazards ``lam`` (n, K), closed form as in tests/sim.py."""
    n = lam.shape[0]
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(lam, axis=1)], axis=1)
    k = np.minimum(np.floor(t).astype(int), K - 1)
    H = cum[:, k] + lam[:, k] * (t - k)
    return 1.0 - np.exp(-H)


def true_delta(x0, z, from_time=FROM_TIME, delta=DELTA, horizons=HORIZONS):
    """Mean true ``Delta risk`` at each horizon over the given sample (n, len(horizons))."""
    z_shift = z.copy()
    k0 = int(from_time)
    z_shift[:, k0:] += delta
    lam_orig, lam_shift = hazard(x0, z), hazard(x0, z_shift)
    orig = np.stack([_risk_at(lam_orig, h) for h in horizons], axis=1)
    shift = np.stack([_risk_at(lam_shift, h) for h in horizons], axis=1)
    return (shift - orig).mean(axis=0)


def oracle(n=100_000, seed=45678):
    rng = np.random.default_rng(seed)
    x0, z = rng.normal(size=n), rng.normal(size=(n, K))
    return true_delta(x0, z)


def _test_paths(n_test, rng):
    tx0, tz = rng.normal(size=n_test), rng.normal(size=(n_test, K))
    Xp = np.column_stack([np.repeat(tx0, K), tz.ravel()])
    starts = np.tile(np.arange(K, dtype=float), n_test)
    stops = starts + 1.0
    iv = make_survival_y(stops, np.zeros(n_test * K, dtype=bool), start=starts)
    ids = np.repeat(np.arange(n_test), K)
    return Xp, iv, ids


def replicate(seed, n_train=1000, n_test=1000, n_estimators=200):
    rng = np.random.default_rng(seed)
    x0, z, U, event = simulate(n_train, rng)
    Xtr, ytr, idtr = rows(x0, z, U, event)
    tvc = SurvivalForestTV(n_estimators=n_estimators, random_state=seed).fit(Xtr, ytr, ids=idtr)
    Xp, iv, ids = _test_paths(n_test, rng)
    res = inspection.path_effect(tvc, Xp, iv, ids, feature=FEATURE, delta=DELTA, from_time=FROM_TIME, horizons=HORIZONS)
    return {f"m_h{i}": v for i, v in enumerate(res.mean)}


def mc_se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def _check(reps, true_d, label):
    reps = pd.DataFrame(reps)
    bias = reps.values - true_d[None, :]
    mean_bias = bias.mean(axis=0)
    se = np.array([mc_se(bias[:, i]) for i in range(bias.shape[1])])
    margin = 0.1 * np.abs(true_d)
    print(f"--- {label} ---")
    for i, h in enumerate(HORIZONS):
        print(
            f"  h={h:.1f}: true_delta={true_d[i]:.4f} mean_bias={mean_bias[i]:+.4f} "
            f"MC-SE={se[i]:.4f} (margin/7={margin[i] / 7:.4f}) margin={margin[i]:.4f} "
            f"pass={abs(mean_bias[i]) <= margin[i]}"
        )
    return reps, mean_bias, se, margin


def pilot(n_reps=10):
    t0 = time.perf_counter()
    true_d = oracle()
    reps = [replicate(s) for s in range(n_reps)]
    _check(reps, true_d, f"pilot (R={n_reps})")
    print(f"pilot done in {time.perf_counter() - t0:.1f}s")


def run(n_reps=50):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    true_d = oracle()
    reps = [replicate(s) for s in range(n_reps)]
    df, mean_bias, se, margin = _check(reps, true_d, f"run (R={n_reps})")
    df.to_csv(OUT / "path_effect.csv", index=False)
    print(f"run done in {time.perf_counter() - t0:.1f}s, wrote {OUT / 'path_effect.csv'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot":
        pilot(*(int(a) for a in sys.argv[2:]))
    elif cmd == "run":
        run(*(int(a) for a in sys.argv[2:]))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
