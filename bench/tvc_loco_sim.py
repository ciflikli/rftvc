"""S18 T6: LOCO simulations with a known truth (s18-plan.md §7.1/§7.7b; not a merge gate).

    python -m bench.tvc_loco_sim pilot          # 10-rep MC-SE pilot for both sims
    python -m bench.tvc_loco_sim run [n_reps]   # R=50 (default), writes docs/bench/s18-loco/*.csv

Reuses ``bench.tvc_perm_sim``'s §7.1 trend-confounding generator (``trend_data``) and its
permutation-importance oracle margin (``trend_oracle``, already validated in the S17 note:
oracle(z1) = 0.617). The LOCO estimand differs from permutation importance (a correlated
column can compensate for a dropped one), so no separate "LOCO oracle" is computed: design
§7's rule uses the same ``oracle(z1)`` margin as a reference scale for a null check on z2's
*LOCO* importance, not a claim that LOCO recovers the permutation-importance oracle value.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bench.tvc_perm_sim import K, TrendOracle, trend_data, trend_oracle
from rftvc import SurvivalForestTV, inspection

OUT = Path(__file__).resolve().parents[1] / "docs" / "bench" / "s18-loco"


def dup_data(n, rng):
    """Rows ``X = [z1, z1_copy, n1, n2]``: an exact duplicate of z1 in place of z2, same
    hazard as ``trend_data`` (``h_k = 0.1 * 1.2**k * exp(0.8 z1)``), for the §7.7b grouped-
    vs-single-copy check."""
    from bench.tvc_perm_sim import _event_time, _grid_rows

    z1 = rng.normal(size=(n, K))
    lam = TrendOracle.RATE * TrendOracle.GROWTH ** np.arange(K) * np.exp(TrendOracle.BETA * z1)
    T = _event_time(lam, rng)
    C = np.minimum(rng.uniform(2, 8.0, size=n), 8.0)
    U, event = np.minimum(T, C), T <= C
    n1, n2 = rng.normal(size=n), rng.normal(size=n)
    Xk = np.stack([z1, z1, np.broadcast_to(n1[:, None], (n, K)), np.broadcast_to(n2[:, None], (n, K))], axis=-1)
    from rftvc import make_survival_y

    return _grid_rows(Xk, U, event, lambda stop, ev, start: make_survival_y(stop, ev.astype(bool), start=start))


N_ESTIMATORS = 100
CV = 5


def loco_trend_replicate(seed, n=1000):
    rng = np.random.default_rng(seed)
    X, y, ids = trend_data(n, rng)
    m = SurvivalForestTV(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1)
    r = inspection.drop_column_importance(m, X, y, ids=ids, cv=CV, features=[0, 1], random_state=seed, n_jobs=-1)
    return dict(imp_z1=r.importances_mean[0], imp_z2=r.importances_mean[1])


def loco_noise_replicate(seed, n=1000):
    rng = np.random.default_rng(seed)
    X, y, ids = trend_data(n, rng)
    m = SurvivalForestTV(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1)
    r = inspection.drop_column_importance(
        m, X, y, ids=ids, cv=CV, features=[0, 1], add_noise_control=True, random_state=seed, n_jobs=-1
    )
    return dict(imp_noise=r.importances_mean[-1])


def loco_group_replicate(seed, n=1000):
    rng = np.random.default_rng(seed)
    X, y, ids = dup_data(n, rng)
    m = SurvivalForestTV(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1)
    single = inspection.drop_column_importance(
        m, X, y, ids=ids, cv=CV, features=[0, 1], random_state=seed, n_jobs=-1
    )
    grouped = inspection.drop_column_importance(
        m, X, y, ids=ids, cv=CV, groups={"z1_both": [0, 1]}, random_state=seed, n_jobs=-1
    )
    return dict(imp_z1=single.importances_mean[0], imp_z1_copy=single.importances_mean[1], imp_grouped=grouped.importances_mean[0])


def mc_se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def one_sided_t(x):
    n = len(x)
    t = np.mean(x) / (np.std(x, ddof=1) / np.sqrt(n))
    return stats.t.sf(t, n - 1)


def pilot(n_reps=10):
    print(f"--- pilot ({n_reps} reps) ---")
    t0 = time.perf_counter()
    trend = pd.DataFrame([loco_trend_replicate(s) for s in range(n_reps)])
    oz1, _ = trend_oracle()
    delta = 0.1 * oz1
    print(f"§7.1 LOCO: oracle(z1)={oz1:.4f} delta={delta:.4f}")
    print(f"  imp_z1 mean={trend.imp_z1.mean():.4f}; imp_z2 mean={trend.imp_z2.mean():.4f} "
          f"MC-SE={mc_se(trend.imp_z2):.4f} (delta/7={delta / 7:.4f})")

    noise = pd.DataFrame([loco_noise_replicate(1000 + s) for s in range(n_reps)])
    null_margin = 0.03 * oz1
    print(f"§7.7b null control: imp_noise mean={noise.imp_noise.mean():.4f} "
          f"MC-SE={mc_se(noise.imp_noise):.4f} (margin~{null_margin:.4f}, margin/7={null_margin / 7:.4f})")

    group = pd.DataFrame([loco_group_replicate(2000 + s) for s in range(n_reps)])
    print(f"§7.7b grouped vs single: single(z1)={group.imp_z1.mean():.4f} "
          f"single(z1_copy)={group.imp_z1_copy.mean():.4f} grouped={group.imp_grouped.mean():.4f} "
          f"MC-SE(grouped)={mc_se(group.imp_grouped):.4f}")
    print(f"pilot done in {time.perf_counter() - t0:.1f}s")


def run(n_reps=50):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    trend = pd.DataFrame([loco_trend_replicate(s) for s in range(n_reps)])
    oz1, _ = trend_oracle()
    delta = 0.1 * oz1
    trend.to_csv(OUT / "trend.csv", index=False)
    pass1 = abs(trend.imp_z2.mean()) <= delta
    print(f"§7.1 LOCO part: oracle(z1)={oz1:.4f} delta={delta:.4f}")
    print(f"  mean imp(z1)={trend.imp_z1.mean():.4f} (for context, cf. S17's permutation-importance mean)")
    print(f"  mean imp(z2)={trend.imp_z2.mean():.4f} (|.| <= {delta:.4f}? {pass1})")

    noise = pd.DataFrame([loco_noise_replicate(1000 + s) for s in range(n_reps)])
    noise.to_csv(OUT / "noise.csv", index=False)
    se_mean = mc_se(noise.imp_noise)
    pass2 = abs(noise.imp_noise.mean()) <= 2 * se_mean
    print(f"§7.7b null control: mean imp(_noise)={noise.imp_noise.mean():.4f} MC-SE={se_mean:.4f} "
          f"(within +/-2 MC-SE of 0? {pass2})")

    group = pd.DataFrame([loco_group_replicate(2000 + s) for s in range(n_reps)])
    group.to_csv(OUT / "group.csv", index=False)
    g = group.imp_grouped.mean()
    pass3a = one_sided_t(group.imp_grouped.values) <= 0.05
    print(f"§7.7b grouped vs single: single(z1)={group.imp_z1.mean():.4f} "
          f"single(z1_copy)={group.imp_z1_copy.mean():.4f} grouped={g:.4f}")
    print(f"  grouped > 0 (one-sided t)? {pass3a} [the design's own pass rule]")
    print(
        f"  single/grouped ratio (reported, no rule; design says 'each single copy ~0', not "
        f"a literal 0: dropping one of two identical columns is measurably cheaper than "
        f"dropping both, since the surviving copy remains available to every split, but is "
        f"not a free substitute under max_features subsampling): "
        f"z1={group.imp_z1.mean() / g:.3f}, z1_copy={group.imp_z1_copy.mean() / g:.3f}"
    )

    print(f"\nrun done in {time.perf_counter() - t0:.1f}s, wrote csvs to {OUT}")
    print(f"\nPASS SUMMARY: §7.1 z2<={pass1}; §7.7b null={pass2}; §7.7b grouped>0={pass3a}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot":
        pilot(*(int(a) for a in sys.argv[2:]))
    else:
        run(*(int(a) for a in sys.argv[2:]))
