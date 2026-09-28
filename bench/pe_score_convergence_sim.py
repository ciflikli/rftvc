"""Slice 1 (docs/plans/plan.md): PE-score oracle-gap convergence.

Reuses ``tests.sim``'s DGP unchanged: subjects have a baseline covariate
``x0 ~ N(0, 1)`` and an external covariate ``z_k`` redrawn on each unit
interval, hazard ``0.15 * exp(0.8 z_k + 0.8 * 1{z_k > 1} + 0.4 x0)``. See
``tests/sim.py``'s module docstring for the full DGP.

For a sweep of training sizes ``n``, this checks that ``SurvivalForestTV``'s
piecewise-exponential score (``rftvc.metrics.piecewise_exponential_score``)
gets closer to an *oracle* score as ``n`` grows. The oracle is computed under
the exact same alpha-mixing, window grid and null convention as the model's
own score — it differs only in using the DGP's true cumulative hazard in
place of the model's prediction (``oracle_pe_score``, a thin wrapper around
``piecewise_exponential_score`` itself, not a reimplementation of its
log-likelihood — see docs/plans/plan.md Slice 1's correction of the original
"unmixed true-hazard ceiling" framing, which was wrong: an unmixed
log-likelihood is not the ceiling of the mixed, windowed quantity actually
being scored).

Both the model and the oracle are scored on the same fixed evaluation set
(``_EVAL``, drawn once at import time from a fixed seed, reused across every
``n`` and every replication) and the same fixed window grid (``WINDOWS``,
derived once from a large deterministic reference draw, independent of any
fitted model or training size) — so the comparison isolates the fitted
model's quality rather than confounding it with a moving eval set or a
moving window grid (``rftvc.metrics.event_windows``'s grid is itself
n-dependent via training event-time quantiles, which would otherwise
interact with ``zero_rate_share``'s "windows too fine" warning as ``n``
changes).

Pass rule (declared here, before this file's sim code was written): over
R=10 replications at ``n_values=(200, 5000)``, the lower bound of the
one-sided 95% CI on the mean paired difference (gap at n=200 minus gap at
n=5000) is > 0 — i.e. the oracle-to-model gap is significantly smaller at
the larger training size.
"""

import numpy as np
from scipy import stats

from rftvc import SurvivalForestTV, make_survival_y
from rftvc.metrics import piecewise_exponential_score
from tests.sim import END, HORIZON, K, hazard, rows, simulate

ALPHA = 0.01
N_WINDOWS = 8
N_EVAL = 300
EVAL_SEED = 999
REF_SEED = 2026
REF_N = 50_000


def true_cumhaz(x0, z, times):
    """Closed-form Λ(t | x0, covariate path z) at arbitrary ``times``.

    Generalizes ``tests.sim.true_survival``'s ``H`` computation (same DGP,
    same piecewise-constant-hazard integration) to any time grid instead of
    only its fixed ``GRID``.
    """
    lam = hazard(x0, z)
    cum = np.concatenate([np.zeros((len(x0), 1)), np.cumsum(lam, axis=1)], axis=1)
    times = np.asarray(times, dtype=float)
    k = np.clip(np.floor(times).astype(int), 0, K - 1)
    return cum[:, k] + lam[:, k] * (times[None, :] - k[None, :])


def _reference_windows(n_windows=N_WINDOWS, horizon=HORIZON, ref_n=REF_N, ref_seed=REF_SEED):
    """Event-time quantile windows on ``[0, horizon]``.

    Computed once from a large deterministic reference draw (fixed seed,
    independent of any replication or training size), so the grid never
    moves across the n-sweep or across replications — the fixed-window
    requirement from docs/plans/plan.md Slice 1's review correction. Same
    quantile-bucketing algorithm as ``rftvc.metrics.event_windows``, applied
    directly to a raw array of observed event times instead of a fitted
    estimator's ``_event_counts_``.
    """
    rng = np.random.default_rng(ref_seed)
    _, _, U, event = simulate(ref_n, rng)
    times = np.sort(U[event & (U <= horizon)])
    total = times.size
    ranks = np.clip(np.ceil(np.arange(1, n_windows) * total / n_windows).astype(int) - 1, 0, total - 1)
    interior = times[ranks]
    return np.unique(np.r_[0.0, interior[interior > 0], horizon])


WINDOWS = _reference_windows()


def _eval_set(n_eval=N_EVAL, seed=EVAL_SEED):
    """Fixed evaluation set: drawn once, reused across every ``n`` and every replication."""
    rng = np.random.default_rng(seed)
    x0, z, U, event = simulate(n_eval, rng)
    n_int = int(np.ceil(HORIZON))
    Xp = np.column_stack([np.repeat(x0, n_int), z[:, :n_int].ravel()])
    starts = np.tile(np.arange(n_int, dtype=float), n_eval)
    intervals = make_survival_y(starts + 1.0, np.zeros(len(starts), bool), start=starts)
    ids = np.repeat(np.arange(n_eval), n_int)
    u_trunc = np.minimum(U, HORIZON)
    event_trunc = event & (U <= HORIZON)
    y_eval = make_survival_y(u_trunc, event_trunc)
    h_true = true_cumhaz(x0, z, WINDOWS)
    return Xp, intervals, ids, y_eval, h_true


_EVAL = _eval_set()


def replicate(seed, n_values, n_estimators=200):
    """One replication's oracle-minus-model PEScore gap at each ``n`` in ``n_values``."""
    rng = np.random.default_rng(seed)
    Xp, intervals, ids, y_eval, h_true = _EVAL
    gaps = []
    for n in n_values:
        x0, z, U, event = simulate(n, rng)
        X, y, tr_ids = rows(x0, z, U, event)
        tvc = SurvivalForestTV(n_estimators=n_estimators, random_state=seed).fit(X, y, ids=tr_ids)
        h_model = tvc.predict_cumulative_hazard(Xp, WINDOWS, intervals=intervals, ids=ids)
        null = np.interp(WINDOWS, tvc.event_times_, tvc.baseline_cumhaz_)
        model_score = piecewise_exponential_score(y_eval, h_model, WINDOWS, null_cumhaz=null, alpha=ALPHA)
        oracle_score = piecewise_exponential_score(y_eval, h_true, WINDOWS, null_cumhaz=null, alpha=ALPHA)
        gaps.append(oracle_score.total - model_score.total)
    return np.array(gaps)


def run(n_values=(200, 5000), n_estimators=200, n_reps=10, seed0=0):
    """Gap at each ``n``, per replication, plus the paired-difference CI bound
    for the pass rule declared in this module's docstring (compares the
    first and last entries of ``n_values``)."""
    res = np.array([replicate(seed0 + r, n_values, n_estimators) for r in range(n_reps)])
    diff = res[:, 0] - res[:, -1]
    lower = diff.mean() - stats.t.ppf(0.975, n_reps - 1) * diff.std(ddof=1) / np.sqrt(n_reps)
    return res, diff, lower
