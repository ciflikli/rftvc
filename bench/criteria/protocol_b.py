"""Protocol B: the core forest on counting-process rows.

- ``sim``: tests/sim.py (known truth). Per replication, each arm is tuned by an
  inner GroupKFold on counting-process C and scored by the integrated squared
  error of its path predictions against the true S. Replications are
  independent, so paired differences get a t-interval.
- ``cl``: Cunningham & Lemke war data, new-war GroupKFold, counting-process C
  on held-out war-years. Supporting evidence only (no landmark endpoint).

``km_gini`` needs a landmark clock and is not run here.
"""

import time
from itertools import product

import numpy as np
import polars as pl
from scipy import stats
from sklearn.model_selection import GroupKFold

from rftvc import SurvivalForestTV, make_survival_y
from rftvc.metrics import concordance_index_cp
from tests import sim

from .common import arm_name, arms, comparisons

CRITERIA_B = ("logrank", "grouped_lik", "poisson")


def _forest(arm, cfg, seed, **kw):
    return SurvivalForestTV(n_estimators=cfg.n_estimators, split_criterion=arm[0], aggregate=arm[1],
                            random_state=seed, n_jobs=cfg.n_jobs, **kw)


def _candidates(cfg):
    g = cfg.grid()
    return [dict(zip(g, v)) for v in product(*g.values())]


def tune_and_fit(arm, cfg, X, y, ids, seed, fit_kw=None):
    """Pick the grid point with the best mean inner-fold C (GroupKFold by id), refit on all rows."""
    fit_kw = fit_kw or {}
    best, best_c = None, -np.inf
    for params in _candidates(cfg):
        cs = []
        for tr, te in GroupKFold(cfg.inner_folds).split(X, groups=ids):
            f = _forest(arm, cfg, seed, **params).fit(X[tr], y[tr], ids[tr], **fit_kw)
            cs.append(concordance_index_cp(y[te], f.predict(X[te]), ids=ids[te]))
        if np.mean(cs) > best_c:
            best, best_c = params, float(np.mean(cs))
    t0 = time.perf_counter()
    model = _forest(arm, cfg, seed, **best).fit(X, y, ids, **fit_kw)
    return model, best, time.perf_counter() - t0


def _sim_rep(seed, cfg, arm_list):
    rng = np.random.default_rng(seed)
    x0, z, U, event = sim.simulate(cfg.sim_n_train, rng)
    X, y, ids = sim.rows(x0, z, U, event)
    tx0, tz = rng.normal(size=cfg.sim_n_test), rng.normal(size=(cfg.sim_n_test, sim.K))
    S_true = sim.true_survival(tx0, tz)
    n_int = int(np.ceil(sim.HORIZON))
    Xp = np.column_stack([np.repeat(tx0, n_int), tz[:, :n_int].ravel()])
    starts = np.tile(np.arange(n_int, dtype=float), cfg.sim_n_test)
    iv = make_survival_y(starts + 1.0, np.zeros(len(starts), bool), start=starts)
    path_ids = np.repeat(np.arange(cfg.sim_n_test), n_int)
    out = []
    for arm in arm_list:
        model, params, secs = tune_and_fit(arm, cfg, X, y, ids, seed)
        S = model.predict_survival_function(Xp, sim.GRID, intervals=iv, ids=path_ids)
        out.append({"rep": seed, "arm": arm_name(arm), "ise": sim.ise(S, S_true), "params": repr(params),
                    "fit_seconds": secs})
    return out


def _paired_t(res, pairs):
    wide = res.pivot(on="arm", index="rep", values="ise")
    out = []
    for a, r in pairs:
        d = (wide[arm_name(a)] - wide[arm_name(r)]).to_numpy()
        n = d.size
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        half = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
        out.append({"dataset": "sim", "arm": arm_name(a), "reference": arm_name(r), "metric": "ise",
                    "diff": float(d.mean()), "se": float(se), "lo": float(d.mean() - half),
                    "hi": float(d.mean() + half), "n": n})
    return pl.DataFrame(out)


def run_sim(cfg, log=print):
    crit = tuple(c for c in cfg.criteria if c in CRITERIA_B)
    arm_list = arms(crit)
    rows = []
    for seed in range(cfg.sim_reps):
        t0 = time.perf_counter()
        rows += _sim_rep(seed, cfg, arm_list)
        log(f"[sim] rep {seed}: {time.perf_counter() - t0:.0f} s")
    res = pl.DataFrame(rows)
    return res, _paired_t(res, comparisons(crit))


def run_cl(cfg, log=print):
    from examples.data.cunningham_lemke import COVARIATES, load_counting_process

    pdf, _ = load_counting_process()
    X = pdf[COVARIATES].to_numpy()
    y = make_survival_y(pdf["stop"].to_numpy(), pdf["event"].to_numpy(), start=pdf["start"].to_numpy())
    ids = pdf["CLID"].to_numpy()
    crit = tuple(c for c in cfg.criteria if c in CRITERIA_B)
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(cfg.outer_folds).split(X, groups=ids)):
        for arm in arms(crit):
            model, params, secs = tune_and_fit(arm, cfg, X[tr], y[tr], ids[tr], cfg.seed,
                                               fit_kw={"gap_policy": "split_id"})
            c = concordance_index_cp(y[te], model.predict(X[te]), ids=ids[te])
            rows.append({"dataset": "cl", "fold": fold, "arm": arm_name(arm), "cindex": c,
                         "params": repr(params), "fit_seconds": secs})
        log(f"[cl] fold {fold} done")
    return pl.DataFrame(rows)
