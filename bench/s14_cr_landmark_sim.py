"""S14 landmark competing-risks simulation (reported, not used for model selection).

    python -m bench.s14_cr_landmark_sim [n_reps]    # writes docs/bench/s14-cr/landmark_sim.csv

Scenario A of `bench.s14_cr_sim` (no delayed entry). Landmarks s in {1, 2, 3},
horizon w = 2; `LandmarkCompetingRisksForest` with z (last), x0, x1. A row is
known at s iff its start <= s, so z_s (in force on (s, s+1]) is history and
only z_{s+1} is unknown: the true F_k(s + w | T > s, H(s)) is the Monte Carlo
mean over 4000 draws of z_{s+1}. Reported per cause and aggregation:
- `mse_truth`: mean over test subjects of (F_hat - F_true)^2 at s + w;
- `brier_ipcw`: cause-specific IPCW Brier at w on the censored test outcomes;
- `brier_oracle`: the Brier on the same subjects' uncensored outcomes (known in
  simulation). IPCW should track the oracle.
"""

import sys

import numpy as np
import polars as pl

from bench.s14_cr_sim import END, OUT, covariates, event_times, hazards, true_cif
from rftvc import CompetingRisksForestTV, LandmarkCompetingRisksForest, landmark_features, make_competing_risks_y
from rftvc.metrics import brier_landmark

LANDMARKS, W, N_MC = (1.0, 2.0, 3.0), 2.0, 4000


def frame(n, rng, first_id=0):
    """Counting-process rows (one per unit interval) and the uncensored truth."""
    x0, x1, _, z = covariates(n, rng)
    t, cause = event_times(hazards("A", x0, x1, z), rng)
    c = np.minimum(rng.uniform(2, 8, n), END)
    rows = []
    for i in range(n):
        u = min(t[i], c[i])
        label = int(cause[i]) if t[i] <= c[i] else 0
        a = 0.0
        while a < u:
            b = min(a + 1.0, u)
            rows.append((first_id + i, a, b, label if b == u else 0, z[i, int(a)], x0[i], x1[i]))
            a = b
    df = pl.DataFrame(rows, schema=["id", "start", "stop", "event", "z", "x0", "x1"], orient="row")
    return df, dict(t=t, cause=cause, x0=x0, x1=x1, z=z)


def true_landmark_cif(truth, rows, s, rng):
    """F_k(s + W | T > s, H(s)) by Monte Carlo over z_{s+1} (z_s is known at s)."""
    k = int(s)
    x0, x1, zs = truth["x0"][rows], truth["x1"][rows], truth["z"][rows, k]
    out = np.zeros((rows.size, 2))
    for i in range(rows.size):
        z_next = rng.normal(size=N_MC)
        z = np.zeros((N_MC, 8))
        z[:, 0], z[:, 1] = zs[i], z_next  # intervals (s, s+1], (s+1, s+2] on the reset clock
        lam = hazards("A", np.full(N_MC, x0[i]), np.full(N_MC, x1[i]), z)
        out[i] = true_cif(lam, np.array([W]))[:, :, 0].mean(axis=0)
    return out


def replicate(seed, n_train=600, n_test=400):
    rng = np.random.default_rng(seed)
    train, _ = frame(n_train, rng)
    test, truth = frame(n_test, rng, first_id=10**6)
    rows = []
    for agg in ("hazard", "cif"):
        m = LandmarkCompetingRisksForest(
            horizon=W, landmarks=list(LANDMARKS), history_features=["z", "x0", "x1"],
            forest=CompetingRisksForestTV(n_estimators=200, aggregate=agg, causes=[1, 2], random_state=seed),
        ).fit(train)
        mc_rng = np.random.default_rng(seed + 10_000)
        for s in LANDMARKS:
            ids, X = landmark_features(test, s, history_features=["z", "x0", "x1"], event="event")
            idx = ids - 10**6
            U = np.minimum(truth["t"][idx], test.group_by("id").agg(pl.col("stop").max()).sort("id")["stop"].to_numpy()[idx])
            keep = U > s  # the training definition of the risk set (U = s gives a zero-length row)
            idx, X = idx[keep], X[keep]
            F = m.forest_.predict_cumulative_incidence(X, [W])[:, :, 0]
            F_true = true_landmark_cif(truth, idx, s, mc_rng)
            obs = test.group_by("id").agg(pl.col("stop").max(), pl.col("event").max()).sort("id")
            stop, lab = obs["stop"].to_numpy()[idx] - s, obs["event"].to_numpy()[idx]
            y_cens = make_competing_risks_y(np.minimum(stop, W), np.where(stop <= W, lab, 0))
            t_true = truth["t"][idx] - s
            y_orac = make_competing_risks_y(np.minimum(t_true, W), np.where(t_true <= W, truth["cause"][idx], 0))
            for k in (1, 2):
                rows.append(dict(rep=seed, aggregate=agg, landmark=s, cause=k, n=int(idx.size),
                                 mse_truth=float(np.mean((F[:, k - 1] - F_true[:, k - 1]) ** 2)),
                                 brier_ipcw=brier_landmark(y_cens, F[:, k - 1], W, cause=k, y_censor=y_cens),
                                 brier_oracle=brier_landmark(y_orac, F[:, k - 1], W, cause=k)))
    return rows


def run(n_reps=10):
    rows = []
    for seed in range(n_reps):
        rows += replicate(seed)
        print("rep", seed, flush=True)
    pl.DataFrame(rows).write_csv(OUT / "landmark_sim.csv")
    print(pl.DataFrame(rows).group_by("aggregate", "cause").agg(
        pl.col("mse_truth").mean(), pl.col("brier_ipcw").mean(), pl.col("brier_oracle").mean()).sort("aggregate", "cause"))


if __name__ == "__main__":
    run(*(int(a) for a in sys.argv[1:]))
