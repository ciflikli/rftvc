"""Protocol A: nested landmark CV on PBC2 (new patients) and the simulated panel (future periods)."""

import time

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from rftvc import LandmarkSurvivalForest, SurvivalForestTV
from rftvc.model_selection import RollingOriginSplit, landmark_cross_validate
from tests.fixtures.pbcseq import pbcseq_counting_process
from tests.sim_panel import simulate_panel

from .common import align, arm_name, arms, cluster_bootstrap, comparisons, row_losses

YEAR = 365.25


def pbc2(cfg):
    """PBC2 (``pbcseq``): landmarks 1–4 y, horizon 2 y, new-patient CV."""
    base = ["age", "sex", "trt", "edema", "ascites"]
    markers = ["log_bili", "albumin", "protime"]
    return dict(
        df=pbcseq_counting_process(),
        w=2 * YEAR,
        model=dict(landmarks=np.arange(1, 5) * YEAR, step=None,
                   history_features=base + markers + [("log_bili", "max"), ("log_bili", "first")]),
        outer=GroupKFold(cfg.outer_folds),
        inner=GroupKFold(cfg.inner_folds),
    )


def panel(cfg):
    """Simulated panel (tests/sim_panel.py): landmarks every 3 periods, horizon 6, rolling origin."""
    h = 6.0
    return dict(
        df=simulate_panel(n_units=cfg.panel_units, n_periods=cfg.panel_periods, seed=cfg.seed),
        w=h,
        model=dict(landmarks=None, step=3.0, history_features=["x", ("x", "mean"), ("x", "max"), "z"]),
        outer=RollingOriginSplit(cfg.outer_folds, test_size=cfg.panel_test_size, gap=h),
        inner=RollingOriginSplit(cfg.inner_folds, test_size=cfg.panel_test_size, gap=h),
    )


DATASETS = {"pbc2": pbc2, "panel": panel}


def model_for(spec, arm, cfg, **forest_kw):
    criterion, aggregate = arm
    forest = SurvivalForestTV(
        n_estimators=cfg.n_estimators, split_criterion=criterion, aggregate=aggregate,
        criterion_horizon=spec["w"] if criterion == "km_gini" else None,
        random_state=cfg.seed, n_jobs=cfg.n_jobs, **forest_kw,
    )
    return LandmarkSurvivalForest(horizon=spec["w"], forest=forest, **spec["model"])


def fit_seconds(spec, arm, cfg):
    """Median wall time of one fit on all landmark rows at fixed settings."""
    m = model_for(spec, arm, cfg, min_ids_leaf=15, max_features="sqrt")
    ts = []
    for _ in range(cfg.fit_time_reps):
        t0 = time.perf_counter()
        m.fit(spec["df"])
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run(name, cfg, raw_dir=None, log=print):
    spec = DATASETS[name](cfg)
    w = spec["w"]
    grid = {f"forest__{k}": v for k, v in cfg.grid().items()}
    all_arms = arms(cfg.criteria)
    scores, preds = [], {}
    for arm in all_arms:
        t0 = time.perf_counter()
        s, p = landmark_cross_validate(
            model_for(spec, arm, cfg), spec["df"], spec["outer"], scoring=["brier", "integrated_brier"],
            n_times=cfg.n_times, param_grid=grid, inner_cv=spec["inner"], refit="integrated_brier",
            return_predictions=True,
        )
        elapsed = time.perf_counter() - t0
        scores.append(s.with_columns(pl.lit(name).alias("dataset"), pl.lit(arm_name(arm)).alias("arm")))
        preds[arm] = row_losses(p, w, cfg.n_times)
        if raw_dir is not None:
            p.write_parquet(raw_dir / f"{name}_{arm[0]}_{arm[1]}.parquet")
        log(f"[{name}] {arm_name(arm)}: CV {elapsed:.0f} s, IBS {preds[arm]['ibs_row'].mean():.4f}")
    preds = align(preds)
    compare, point = cluster_bootstrap(preds, comparisons(cfg.criteria), w, cfg.n_bins, cfg.n_boot, cfg.seed)
    summary = pl.DataFrame([
        {"dataset": name, "arm": arm_name(arm), **point[arm], "n_rows": preds[arm].height,
         "n_ids": preds[arm]["id"].n_unique(), "fit_seconds": fit_seconds(spec, arm, cfg)}
        for arm in all_arms
    ])
    return (pl.concat(scores, how="diagonal"), summary,
            compare.with_columns(pl.lit(name).alias("dataset")))

