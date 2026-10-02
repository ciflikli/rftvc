"""S19 T8: landmark-importance simulations with a known truth (not a merge gate).

    python -m bench.tvc_landmark_sim pilot          # 10-rep MC-SE pilot for §7.2
    python -m bench.tvc_landmark_sim run [n_reps]   # R=50 (default), writes docs/bench/s19-landmark/*.csv

§7.2 (level vs history): the true hazard depends on the rolling 2-unit mean of an external
``z`` (the "history" scenario) or on the instantaneous ``z`` (the "Markov" control) — a
featurization mismatch is deliberate: the fitted model only has ``z`` (last value) and its
*cumulative* mean (``history_features=["z", ("z", "mean")]``, the framework's own "mean"
semantics), not the true rolling-2 driver. The oracle is computed on a model whose raw
columns are ``[z, true_hist]`` (``true_hist`` = the rolling-2-unit mean itself, precomputed
per row) with a stub ``forest_`` giving the true (fixed-profile) hazard, so the oracle's own
``features=`` / ``conditional_on=`` query can target the true driver directly.

§7.5b (landmark specifics): repeated-landmark-copy bootstrap SE stability across ``step``,
and a PE-vs-Brier importance-ranking concordance check under heavy censoring.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from bench.tvc_perm_sim import _event_time, _grid_rows
from rftvc import LandmarkSurvivalForest, SurvivalForestTV, inspection, make_survival_y

OUT = Path(__file__).resolve().parents[1] / "docs" / "bench" / "s19-landmark"
K, END = 8, 8.0
RATE, BETA = 0.15, 0.8


def _true_hist(z):
    """Rolling 2-unit mean per interval: ``z_0`` at k=0, ``mean(z_{k-1}, z_k)`` at k >= 1."""
    out = z.copy()
    out[:, 1:] = (z[:, :-1] + z[:, 1:]) / 2.0
    return out


def level_history_data(n, rng, scenario):
    """Rows ``[z, true_hist]`` for the §7.2 generator; ``scenario`` picks the true driver."""
    z = rng.normal(size=(n, K))
    hist = _true_hist(z)
    driver = hist if scenario == "history" else z
    lam = RATE * np.exp(BETA * driver)
    T = _event_time(lam, rng)
    C = np.minimum(rng.uniform(2, END, size=n), END)
    U, event = np.minimum(T, C), T <= C
    Xk = np.stack([z, hist], axis=-1)
    X, y, ids = _grid_rows(Xk, U, event, lambda stop, ev, start: make_survival_y(stop, ev.astype(bool), start=start))
    return pl.DataFrame(
        {"id": ids, "start": y["start"], "stop": y["stop"], "event": y["event"], "z": X[:, 0], "true_hist": X[:, 1]}
    )


class LevelHistoryOracle:
    """True fixed-profile cumhaz: constant rate ``RATE * exp(BETA * driver)``, ``driver`` frozen
    at the row's value. ``column`` selects the driver in the oracle carrier's own ``[z,
    true_hist]`` layout: 1 for the history scenario, 0 for the Markov control."""

    def __init__(self, column):
        self.column = column

    def predict_cumhaz(self, X, times, aggregate, n_jobs):
        driver = X[:, [self.column]]
        return RATE * np.exp(BETA * driver) * np.asarray(times, dtype=float)[None, :]


def _carrier(df, seed):
    sel = df["id"].to_numpy() < 4000  # whole subjects only
    return LandmarkSurvivalForest(
        horizon=2.0, step=1.0, history_features=["z", "true_hist"], forest=SurvivalForestTV(n_estimators=20, random_state=seed)
    ).fit(df.filter(pl.Series(sel)))


def _stubbed(model, stub):
    """Swap the *inner* forest's Rust-engine level for ``stub``, keeping ``model.forest_``'s
    own ``event_times_`` / ``baseline_cumhaz_`` / ``aggregate`` (from the real carrier fit) —
    ``inspection._pe_landmark`` reads windows/null from ``model.forest_``, not from the engine."""
    import copy

    m = copy.copy(model)
    m.forest_ = copy.copy(model.forest_)
    m.forest_.forest_ = stub
    return m


def oracle(scenario, n=100_000, seed=12345, n_repeats=5):
    """The oracle's own history-given-level (``scenario="history"``) or level-given-history
    (``scenario="markov"``) importance, on the true driver, at n = 10^5."""
    rng = np.random.default_rng(seed)
    df = level_history_data(n, rng, scenario)
    m = _stubbed(_carrier(df, seed), LevelHistoryOracle(1 if scenario == "history" else 0))
    if scenario == "history":
        r = inspection.permutation_importance(
            m, df, features=["true_hist"], conditional_on=["z"], n_repeats=n_repeats, n_bootstrap=0
        )
    else:
        r = inspection.permutation_importance(
            m, df, features=["z"], conditional_on=["true_hist"], n_repeats=n_repeats, n_bootstrap=0
        )
    return float(r.importances_mean[0])


def replicate(seed, scenario, n_train=1000, n_eval=1000, n_estimators=200, step=1.0):
    """The fitted model's history-given-level statistic (``features=["z_mean"],
    conditional_on=["z"]``) on data generated under ``scenario``."""
    rng = np.random.default_rng(seed)
    df_train = level_history_data(n_train, rng, scenario)
    df_eval = level_history_data(n_eval, rng, scenario)
    m = LandmarkSurvivalForest(
        horizon=2.0, step=step, history_features=["z", ("z", "mean")],
        forest=SurvivalForestTV(n_estimators=n_estimators, random_state=seed, n_jobs=-1),
    ).fit(df_train)
    r = inspection.permutation_importance(
        m, df_eval, features=["z_mean"], conditional_on=["z"], n_bootstrap=0, random_state=seed
    )
    return float(r.importances_mean[0])


def mc_se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


# --- §7.5b: repeated landmark copies (bootstrap SE clustering) --------------------------


def copies_replicate(seed, step, n_train=500, n_eval=500, n_estimators=100):
    rng = np.random.default_rng(seed)
    df_train = level_history_data(n_train, rng, "history")
    df_eval = level_history_data(n_eval, rng, "history")
    m = LandmarkSurvivalForest(
        horizon=2.0, step=step, history_features=["z", ("z", "mean")],
        forest=SurvivalForestTV(n_estimators=n_estimators, random_state=seed, n_jobs=-1),
    ).fit(df_train)
    r = inspection.permutation_importance(m, df_eval, features=["z_mean"], n_bootstrap=50, n_repeats=3, random_state=seed)
    return float(r.importances_se[0])


# --- §7.5b: heavy censoring, PE vs Brier ranking concordance -----------------------------


def censoring_data(n, rng):
    """Two raw units: ``z1`` (real signal), ``z2`` (noise); heavy admin + random censoring."""
    z1 = rng.normal(size=(n, K))
    z2 = rng.normal(size=(n, K))
    lam = RATE * np.exp(BETA * z1)
    T = _event_time(lam, rng)
    C = np.minimum(rng.exponential(0.4, size=n), rng.uniform(0.3, END, size=n))  # heavy censoring (g_min active)
    U, event = np.minimum(T, C), T <= C
    Xk = np.stack([z1, z2], axis=-1)
    X, y, ids = _grid_rows(Xk, U, event, lambda stop, ev, start: make_survival_y(stop, ev.astype(bool), start=start))
    return pl.DataFrame(
        {"id": ids, "start": y["start"], "stop": y["stop"], "event": y["event"], "z1": X[:, 0], "z2": X[:, 1]}
    )


def censoring_replicate(seed, n_train=800, n_eval=800, n_estimators=150):
    rng = np.random.default_rng(seed)
    df_train = censoring_data(n_train, rng)
    df_eval = censoring_data(n_eval, rng)
    m = LandmarkSurvivalForest(
        horizon=2.0, step=1.0, history_features=["z1", "z2"],
        forest=SurvivalForestTV(n_estimators=n_estimators, random_state=seed, n_jobs=-1),
    ).fit(df_train)
    r_pe = inspection.permutation_importance(m, df_eval, n_bootstrap=0, random_state=seed)
    r_brier = inspection.permutation_importance(m, df_eval, scoring="brier", n_bootstrap=0, random_state=seed)
    return {
        "pe_z1": r_pe.importances_mean[0], "pe_z2": r_pe.importances_mean[1],
        "brier_z1": r_brier.importances_mean[0], "brier_z2": r_brier.importances_mean[1],
    }


# --- pilot / run ---------------------------------------------------------------------------


def pilot(n_reps=10):
    print(f"--- pilot ({n_reps} reps) ---")
    t0 = time.perf_counter()
    oh, ol = oracle("history"), oracle("markov")
    print(f"§7.2 oracle: history={oh:.4f} (markov) level={ol:.4f}")
    hist = np.array([replicate(s, "history") for s in range(n_reps)])
    mark = np.array([replicate(s, "markov") for s in range(n_reps)])
    print(f"  history scenario: mean={hist.mean():.4f} MC-SE={mc_se(hist):.4f} (margin/7={0.25*oh/7:.4f})")
    print(f"  markov scenario:  mean={mark.mean():.4f} MC-SE={mc_se(mark):.4f} (margin/7={0.05*ol/7:.4f})")
    print(f"pilot done in {time.perf_counter() - t0:.1f}s")


def run(n_reps=50):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    oh = oracle("history")
    ol = oracle("markov")
    hist = pd.DataFrame({"history_given_level": [replicate(s, "history") for s in range(n_reps)]})
    mark = pd.DataFrame({"history_given_level": [replicate(s, "markov") for s in range(n_reps)]})
    hist.to_csv(OUT / "level_history_history.csv", index=False)
    mark.to_csv(OUT / "level_history_markov.csv", index=False)
    pass1 = hist.history_given_level.mean() >= 0.25 * oh
    pass2 = mark.history_given_level.mean() <= 0.05 * ol
    print(f"§7.2 level vs history: oracle(history)={oh:.4f} oracle(markov-level)={ol:.4f}")
    print(f"  history scenario mean={hist.history_given_level.mean():.4f} (>= {0.25*oh:.4f}? {pass1})")
    print(f"  markov scenario mean={mark.history_given_level.mean():.4f} (<= {0.05*ol:.4f}? {pass2})")

    small_se = np.array([copies_replicate(s, step=0.5) for s in range(10)])
    large_se = np.array([copies_replicate(s, step=4.0) for s in range(10)])
    ratio = small_se.mean() / large_se.mean()
    pass3 = ratio >= 0.5
    print(f"§7.5b copies: se(step=0.5) mean={small_se.mean():.4f}, se(step=4.0) mean={large_se.mean():.4f}, "
          f"ratio={ratio:.3f} (>= 0.5? {pass3})")

    cens = pd.DataFrame([censoring_replicate(s) for s in range(10)])
    cens.to_csv(OUT / "censoring.csv", index=False)
    pe_rank_z1_larger = cens.pe_z1.mean() > cens.pe_z2.mean()
    brier_rank_z1_larger = cens.brier_z1.mean() > cens.brier_z2.mean()
    pass4 = pe_rank_z1_larger == brier_rank_z1_larger
    print(f"§7.5b censoring: mean PE z1={cens.pe_z1.mean():.4f} z2={cens.pe_z2.mean():.4f}; "
          f"mean Brier z1={cens.brier_z1.mean():.4f} z2={cens.brier_z2.mean():.4f}")
    print(f"  rankings agree (z1 largest in both)? {pass4}")

    print(f"\nrun done in {time.perf_counter() - t0:.1f}s, wrote csvs to {OUT}")
    print(f"\nPASS SUMMARY: §7.2 history={pass1} markov={pass2}; §7.5b copies={pass3} censoring={pass4}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot":
        pilot(*(int(a) for a in sys.argv[2:]))
    else:
        run(*(int(a) for a in sys.argv[2:]))
