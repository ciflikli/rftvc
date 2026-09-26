"""S8 bake-off: arms, configurations and the pre-specified metrics (docs/plans/s8-plan.md, items 5–8).

Everything that decides the outcome is fixed here before any full run:
the arms, the tuning grid, the seeds, the horizons, the calibration summary
and the bootstrap.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from rftvc import make_survival_y
from rftvc.metrics import KaplanMeierCensoring, _ipcw, calibration_table

CRITERIA = ("logrank", "grouped_lik", "poisson", "km_gini")
AGGREGATES = ("hazard", "survival")
BASELINE = ("logrank", "hazard")
G_MIN = 0.05


def arms(criteria=CRITERIA):
    return [(c, a) for c in criteria for a in AGGREGATES]


def arm_name(arm):
    return f"{arm[0]}/{arm[1]}"


def comparisons(criteria):
    """(arm, reference) pairs: every challenger vs log-rank at the same
    aggregation (rule items i–iv), and log-rank survival vs hazard (D11)."""
    out = [((c, a), ("logrank", a)) for c in criteria if c != "logrank" for a in AGGREGATES]
    return out + [(("logrank", "survival"), BASELINE)]


@dataclass(frozen=True)
class Config:
    n_estimators: int = 300
    grid_min_ids_leaf: tuple = (5, 15, 50)
    grid_max_features: tuple = ("sqrt", None)
    outer_folds: int = 5
    inner_folds: int = 3
    n_times: int = 10  # IBS grid on (0, w]
    n_bins: int = 10  # calibration deciles
    n_boot: int = 500
    sim_reps: int = 20
    sim_n_train: int = 400
    sim_n_test: int = 200
    panel_units: int = 2000
    panel_periods: int = 96
    panel_test_size: float = 6.0  # two landmarks per rolling-origin test window
    fit_time_reps: int = 3
    seed: int = 0
    n_jobs: int = -1
    datasets: tuple = ("pbc2", "panel", "sim", "cl")
    criteria: tuple = field(default=CRITERIA)

    def grid(self):
        return {"min_ids_leaf": list(self.grid_min_ids_leaf), "max_features": list(self.grid_max_features)}


FULL = Config()
SMOKE = replace(
    FULL, n_estimators=5, grid_min_ids_leaf=(5, 15), grid_max_features=("sqrt",), outer_folds=2,
    inner_folds=2, n_times=4, n_bins=5, n_boot=20, sim_reps=2, sim_n_train=150, sim_n_test=50,
    panel_units=150, panel_periods=60, fit_time_reps=1, datasets=("panel", "sim"),
)


# --- landmark metrics from out-of-fold predictions ---------------------------------


def row_losses(preds, w, n_times):
    """Per-row IPCW Brier at ``w`` and integrated Brier over the ``n_times`` grid.

    Censoring weights come from a reverse KM fitted on each (fold, landmark)
    test risk set, as in ``landmark_cross_validate``, so the mean over a
    group's rows equals that group's score. Returns ``preds`` with columns
    ``brier_row`` and ``ibs_row``.
    """
    times = np.linspace(0, w, n_times + 1)[1:]
    out = []
    for (_, _), g in preds.group_by(["fold", "landmark"], maintain_order=True):
        stop, event = g["time"].to_numpy(), g["event"].to_numpy()
        cens = KaplanMeierCensoring().fit(make_survival_y(stop, event))
        S = np.vstack(g["survival"].to_list())
        per_t = np.empty((len(g), times.size))
        for j, t in enumerate(times):
            case, _, wt, _ = _ipcw(stop, event, t, None, cens, G_MIN)
            per_t[:, j] = wt * np.square(case - (1.0 - S[:, j]))
        ibs = np.trapezoid(per_t, times, axis=1) / (times[-1] - times[0])
        out.append(g.with_columns(pl.Series("brier_row", per_t[:, -1]), pl.Series("ibs_row", ibs)))
    return pl.concat(out)


def calibration_summary(time, event, risk, w, n_bins):
    """ICI (bin-size-weighted mean |KM-observed − predicted|) and the calibration
    slope (weighted least squares of observed on predicted over bins; NaN when
    the bins' predictions do not vary)."""
    y = make_survival_y(time, event)
    cal = calibration_table(y, risk, w, n_bins=n_bins)
    n = cal["n"].to_numpy().astype(float)
    pred, obs = cal["mean_risk"].to_numpy(), cal["observed_risk"].to_numpy()
    ici = float(np.sum(n * np.abs(obs - pred)) / n.sum())
    pm = np.average(pred, weights=n)
    var = np.sum(n * (pred - pm) ** 2)
    slope = float(np.sum(n * (pred - pm) * (obs - np.average(obs, weights=n))) / var) if var > 1e-12 else np.nan
    return ici, slope


def landmark_summary(p, w, n_bins):
    """Pooled out-of-fold metrics of one arm; ``p`` has ``brier_row`` / ``ibs_row``."""
    ici, slope = calibration_summary(p["time"].to_numpy(), p["event"].to_numpy(), p["risk"].to_numpy(), w, n_bins)
    return {"ibs": float(p["ibs_row"].mean()), "brier": float(p["brier_row"].mean()), "ici": ici, "slope": slope}


def cluster_bootstrap(preds_by_arm, pairs, w, n_bins, n_boot, seed):
    """Percentile 95% intervals of paired differences (arm − reference) in pooled
    IBS, Brier at ``w``, ICI and slope, resampling subjects (``id``) with replacement.

    ``preds_by_arm`` must be ``align``-ed (same rows in the same order), so each
    replicate draws one set of ids and evaluates every arm on it (paired).
    Censoring weights are those of the original fit (``row_losses``).
    """
    first = next(iter(preds_by_arm.values()))
    ids = first["id"].to_numpy()
    uniq, inverse = np.unique(ids, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(uniq.size)]
    point = {arm: landmark_summary(p, w, n_bins) for arm, p in preds_by_arm.items()}
    rng = np.random.default_rng(seed)
    draws = {pair: [] for pair in pairs}
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[k] for k in rng.integers(0, uniq.size, uniq.size)])
        stats = {arm: landmark_summary(p[rows], w, n_bins) for arm, p in preds_by_arm.items()}
        for a, r in pairs:
            draws[(a, r)].append({k: stats[a][k] - stats[r][k] for k in stats[a]})
    out = []
    for (a, r), ds in draws.items():
        for metric in ("ibs", "brier", "ici", "slope"):
            v = np.array([d[metric] for d in ds], dtype=float)
            v = v[np.isfinite(v)]
            lo, hi = np.quantile(v, [0.025, 0.975]) if v.size else (np.nan, np.nan)
            out.append({
                "arm": arm_name(a), "reference": arm_name(r), "metric": metric,
                "diff": point[a][metric] - point[r][metric], "lo": float(lo), "hi": float(hi),
                "n_boot": int(v.size),
            })
    return pl.DataFrame(out), point


def align(preds_by_arm):
    """Sort every arm's predictions identically and check they cover the same rows."""
    out = {arm: p.sort(["fold", "landmark", "id"]) for arm, p in preds_by_arm.items()}
    keys = None
    for arm, p in out.items():
        k = p.select("fold", "landmark", "id")
        if keys is None:
            keys = k
        elif not k.equals(keys):
            raise ValueError(f"arm {arm_name(arm)} predicts different rows")
    return out
