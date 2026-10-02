"""Slice 11: TVC vs. `lifelines.CoxTimeVaryingFitter` on a known-truth DGP.

Closes the TVC half of the validation audit's closing-summary gap: "no
landmark-estimator or TVC cross-check against an external tool exists at all" — the only existing
`CoxTimeVaryingFitter`-vs-`rftvc` comparison (the Rossi case study, the
Rossi dataset) is real data with no known ground truth.

Data-generating process (new for this slice — deliberately *not* a reuse of ``tests.sim``'s
existing external-TVC DGP): a single external covariate ``z_k ~ N(0, 1)`` redrawn on each unit
interval ``(k, k+1]``, ``k = 0..7``, with hazard ``rate0 * exp(beta * z_k)`` — genuinely log-linear,
so ``CoxTimeVaryingFitter`` is correctly specified, the same role Slice 2's static single-covariate
DGP gave ``CoxPHFitter``. ``tests.sim``'s existing DGP adds a ``0.8 * 1{z>1}`` threshold
non-linearity that ``rows()`` never exposes to any fitted model, which would leave
``CoxTimeVaryingFitter`` misspecified too — not a fair "home turf" comparator (the same reasoning
behind Slice 2's own plan-review correction against subsetting an existing TVC DGP). Censoring
``C ~ Uniform(2, 8)``, administrative end at 8, evaluation grid ``[0, 6]`` (61 points) — reusing
``tests.sim``/Slice 2's established conventions rather than inventing new ones.

Comparator: ``lifelines.CoxTimeVaryingFitter`` has no built-in ``predict_survival_function`` for
time-varying covariates (unlike ``CoxPHFitter``). Reconstructing ``S(t | known covariate path)``
combines ``predict_partial_hazard`` with ``baseline_cumulative_hazard_``'s non-parametric
(Breslow-type) step function — the same piecewise-constant-hazard convention used throughout this
codebase (e.g. ``bench/pe_score_convergence_sim.py::true_cumhaz``). Two real bugs were caught
during review, both in this reconstruction, neither in ``rftvc`` itself:

1. **Caught during research:** naively assuming the baseline lands on exact integer times
   (``reindex``) silently gives ``NaN`` here, because this DGP's event times are continuous, not
   boundary-aligned except for non-terminal rows — fixed with a proper step-function lookup
   (``_baseline_step_at``, ``np.searchsorted`` on the baseline's own index).
2. **Caught by Codex diff review on this slice's first version:** using ``exp(beta_hat * z)``
   directly for the per-row partial hazard, instead of ``cox.predict_partial_hazard`` — lifelines
   mean-centers covariates internally (``exp((x - x_bar)' beta)``), and
   ``baseline_cumulative_hazard_`` is defined relative to that centered scale, so the uncentered
   version silently mis-scaled every partial hazard by a constant factor of
   ``exp(-beta_hat * mean(z_train))``. Fixed by calling ``cox.predict_partial_hazard`` directly.
   Verified impact: at this DGP's scale (``z ~ N(0,1)``, ``n_train=400``, so ``mean(z_train)`` is
   already close to 0), the numeric effect on the reported gap was small (Codex found a ~9% shift
   in one seed's cox ISE, 0.00537 vs 0.00589 — not enough to change this gate's pass/fail at its
   loose 0.18 epsilon), but the uncentered version was conceptually wrong regardless of scale and
   would matter more with a larger true effect size or non-centered covariates.

Metric: integrated squared error (ISE) of predicted ``S(t | path)`` vs. the true ``S(t | path)``
over ``t ∈ [0, 6]`` (trapezoid rule, 61 points), averaged over test subjects — same convention as
``tests/sim.py``/Slice 2.

Pass rule (declared here, using an epsilon calibrated from an out-of-band pilot, not from the
gate's own seeds): over R=10 replications (seeds 0-9), ``mean(rftvc_ISE) <= mean(cox_ISE) +
0.18``. Calibration (seeds 10000-10004 and 20010-20014, two independent out-of-band batches):
mean rftvc ISE ≈ 0.093-0.097, mean cox ISE ≈ 0.005-0.007, gap ≈ 0.086-0.092 — stable across both
batches and the gate's own seeds (gap ≈ 0.092 there too). 0.18 is roughly double the observed gap,
matching Slice 2's exact discipline (an additive tolerance, not a ratio, since the reference ISE is
small and a ratio would be unstable near a near-zero denominator). The absolute ISE and gap are
both larger than Slice 2's static case (0.041/0.0038 there vs. ~0.095/0.005 here) — expected, since
per-interval TVC estimation is a harder problem than a single static covariate, not a regression.

Scope: single-event TVC, single covariate. Does not cover competing risks (needs
``randomForestSRC``, a separate R-only gap) or landmark estimators (no external tool exists for
that shape at all).
"""

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from rftvc import SurvivalForestTV, make_survival_y

RATE0 = 0.15
BETA = 0.8
K, END, HORIZON = 8, 8.0, 6.0
GRID = np.linspace(0.0, HORIZON, 61)
EPSILON = 0.18  # calibrated from an out-of-band pilot (seeds >= 10_000), see module docstring


def true_cumhaz(z, times):
    """Λ(t | covariate path z), piecewise-constant hazard per unit interval. z: (n, K)."""
    lam = RATE0 * np.exp(BETA * z)
    cum = np.concatenate([np.zeros((len(z), 1)), np.cumsum(lam, axis=1)], axis=1)
    times = np.asarray(times, dtype=float)
    k = np.clip(np.floor(times).astype(int), 0, K - 1)
    return cum[:, k] + lam[:, k] * (times[None, :] - k[None, :])


def simulate(n, rng):
    """Event time by piecewise-constant-hazard inversion, same pattern as ``tests.sim.simulate``."""
    z = rng.normal(size=(n, K))
    lam = RATE0 * np.exp(BETA * z)
    e_draw = rng.exponential(size=n)
    cum = np.cumsum(lam, axis=1)
    k_evt = (cum < e_draw[:, None]).sum(axis=1)  # interval containing the event (K = none)
    prev = np.where(k_evt > 0, cum[np.arange(n), np.maximum(k_evt - 1, 0)], 0.0)
    within = (e_draw - prev) / lam[np.arange(n), np.minimum(k_evt, K - 1)]
    T = np.where(k_evt < K, k_evt + within, np.inf)
    C = np.minimum(rng.uniform(2, END, size=n), END)
    return z, np.minimum(T, C), T <= C


def rows(z, U, event):
    """Counting-process rows: X (n_rows, 1), y (rftvc survival y), ids (n_rows,)."""
    X, start, stop, ev, ids = [], [], [], [], []
    for i in range(len(z)):
        k = 0
        while k < U[i]:
            X.append([z[i, k]])
            start.append(float(k))
            stop.append(min(k + 1.0, U[i]))
            ev.append(bool(event[i] and k + 1.0 >= U[i]))
            ids.append(i)
            k += 1
    X, stop, start, ev = np.array(X), np.array(stop), np.array(start), np.array(ev)
    return X, make_survival_y(stop, ev, start=start), np.array(ids)


def ise(s_hat, s_true):
    return np.trapezoid((s_hat - s_true) ** 2, GRID, axis=1).mean()


def _rftvc_survival(X, y, ids, z_test, seed, n_estimators):
    m = SurvivalForestTV(n_estimators=n_estimators, random_state=seed).fit(X, y, ids=ids)
    n_int = int(np.ceil(HORIZON))
    n_test = z_test.shape[0]
    Xp = z_test[:, :n_int].reshape(-1, 1)
    starts = np.tile(np.arange(n_int, dtype=float), n_test)
    intervals = make_survival_y(starts + 1.0, np.zeros(len(starts), bool), start=starts)
    pred_ids = np.repeat(np.arange(n_test), n_int)
    H = m.predict_cumulative_hazard(Xp, GRID, intervals=intervals, ids=pred_ids)
    return np.exp(-H)


def _baseline_step_at(cox, query_times):
    """Baseline cumulative hazard (right-continuous step function) at arbitrary times.

    Does NOT assume the baseline's index lands on exact query times (event times in this DGP
    are continuous, not interval-boundary-aligned) -- uses a searchsorted step lookup instead of
    ``reindex``, which silently gives ``NaN`` at any query time absent from the index.
    """
    base = cox.baseline_cumulative_hazard_.iloc[:, 0]
    idx = base.index.to_numpy(dtype=float)
    vals = base.to_numpy(dtype=float)
    pos = np.searchsorted(idx, query_times, side="right") - 1
    return np.where(pos < 0, 0.0, vals[np.clip(pos, 0, len(vals) - 1)])


def _cox_survival(X, y, ids, z_test):
    df = pd.DataFrame(
        {"id": ids, "start": y["start"], "stop": y["stop"], "event": y["event"].astype(int), "z": X[:, 0]}
    )
    cox = CoxTimeVaryingFitter().fit(df, id_col="id", start_col="start", stop_col="stop", event_col="event")
    base_at = np.concatenate([[0.0], _baseline_step_at(cox, np.arange(1.0, K + 1.0))])
    n_test = z_test.shape[0]
    # cox.predict_partial_hazard applies lifelines' own mean-centering (exp((x - x_bar)' beta)) --
    # NOT the same as exp(beta_hat * z) directly, since baseline_cumulative_hazard_ is itself
    # computed relative to the training mean covariate. Using the raw (uncentered) exponential
    # here would silently mis-scale every partial hazard by a constant factor of
    # exp(-beta_hat * mean(z_train)) -- a real bug caught by Codex review on this slice's first
    # version (verified: changes seed-0 cox ISE from 0.005372 (partial-hazard, correct) to
    # 0.005894 (uncentered, wrong) -- close enough at this scale to have passed the pass rule
    # silently, but wrong in every other regard: it doesn't correspond to what
    # `baseline_cumulative_hazard_` is defined relative to).
    partial_z = cox.predict_partial_hazard(pd.DataFrame({"z": z_test.ravel()})).to_numpy().reshape(z_test.shape)
    H = np.zeros((n_test, GRID.size))
    for j, t in enumerate(GRID):
        k = min(int(np.floor(t)), K - 1)
        incs = np.diff(base_at[: k + 1])
        cum = (partial_z[:, :k] * incs[None, :]).sum(axis=1) if k > 0 else np.zeros(n_test)
        frac_inc = _baseline_step_at(cox, np.array([t]))[0] - base_at[k]
        H[:, j] = cum + partial_z[:, k] * frac_inc
    return np.exp(-H)


def replicate(seed, n_train=400, n_test=200, n_estimators=200):
    """One replication's (rftvc_ISE, cox_ISE) against the closed-form truth."""
    rng = np.random.default_rng(seed)
    z_tr, U, event = simulate(n_train, rng)
    z_te, _, _ = simulate(n_test, rng)
    X, y, ids = rows(z_tr, U, event)
    s_true = np.exp(-true_cumhaz(z_te, GRID))
    s_rftvc = _rftvc_survival(X, y, ids, z_te, seed, n_estimators)
    s_cox = _cox_survival(X, y, ids, z_te)
    return ise(s_rftvc, s_true), ise(s_cox, s_true)


def run(n_reps=10, seed0=0, **kw):
    """(n_reps, 2) array of (rftvc_ISE, cox_ISE) per replication, seeds seed0..seed0+n_reps-1."""
    return np.array([replicate(seed0 + r, **kw) for r in range(n_reps)])
