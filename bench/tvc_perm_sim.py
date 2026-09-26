"""S17 T6: permutation-importance simulations with a known truth (s17-plan.md §7.1/7.3/7.4; not a merge gate).

    python -m bench.tvc_perm_sim pilot          # 10-rep MC-SE pilot for all three sims
    python -m bench.tvc_perm_sim run [n_reps]   # R=50 (default), writes docs/bench/s17-perm/*.csv

Each sim compares a trained forest's permutation importance (``R`` replications,
1000 training ids, 1000 evaluation ids, 200 trees, defaults otherwise) against an
**oracle** importance: the same ``permutation_importance`` call, but with the
forest's ``forest_`` swapped for a stub whose ``predict_cumhaz`` /
``predict_cause_cumhaz`` is the true (fixed-profile) hazard, scored on a single
10^5-subject evaluation sample under the same ``strata="time", n_strata=10`` and
``windows`` (design §7). Rows are built by a vectorized K-interval grid (mask by
each subject's follow-up), not a Python per-subject loop, so the 10^5-subject
oracle sample is cheap.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bench.s14_cr_sim import covariates as cr_covariates
from bench.s14_cr_sim import hazards as cr_hazards
from rftvc import CompetingRisksForestTV, SurvivalForestTV, inspection, make_competing_risks_y, make_survival_y

OUT = Path(__file__).resolve().parents[1] / "docs" / "bench" / "s17-perm"
K, END = 8, 8.0  # unit intervals (0, 1] .. (7, 8]; shared by sims 1 and 2


def _grid_rows(Xk, U, label, y_builder):
    """Counting-process rows from a dense ``(n, K, p)`` per-interval covariate grid.

    ``U`` (n,) is each subject's follow-up end; ``label`` (n,) is 0 (censored) or
    the event/cause code, applied to the interval containing ``U``. Vectorized:
    no per-subject Python loop, so this scales to n = 10^5+.
    """
    n, kk, p = Xk.shape
    start = np.broadcast_to(np.arange(kk, dtype=float), (n, kk))
    stop = np.minimum(start + 1.0, U[:, None])
    keep = start < U[:, None]
    is_last = stop == U[:, None]
    lbl = np.zeros((n, kk), dtype=np.asarray(label).dtype)
    lbl[is_last] = np.broadcast_to(np.asarray(label)[:, None], (n, kk))[is_last]
    ids = np.broadcast_to(np.arange(n)[:, None], (n, kk))
    m = keep.ravel()
    X = Xk.reshape(n * kk, p)[m]
    y = y_builder(stop.ravel()[m], lbl.ravel()[m], start.ravel()[m])
    return X, y, ids.ravel()[m]


def _event_time(lam, rng):
    """First-event time from piecewise-constant per-interval rates ``lam`` (n, K)."""
    n = lam.shape[0]
    e_draw = rng.exponential(size=n)
    cum = np.cumsum(lam, axis=1)
    k_evt = (cum < e_draw[:, None]).sum(axis=1)
    prev = np.where(k_evt > 0, cum[np.arange(n), np.maximum(k_evt - 1, 0)], 0.0)
    within = (e_draw - prev) / lam[np.arange(n), np.minimum(k_evt, lam.shape[1] - 1)]
    return np.where(k_evt < lam.shape[1], k_evt + within, np.inf)


def _carrier(X, y, ids, n_estimators=20, seed=0, competing=False, causes=None):
    """A lightly-fitted forest, kept only for its ``event_times_`` / ``baseline_cumhaz_`` /
    ``aggregate`` metadata; its ``forest_`` is swapped for the true-hazard oracle stub."""
    if competing:
        return CompetingRisksForestTV(n_estimators=n_estimators, causes=causes, random_state=seed).fit(X, y, ids)
    return SurvivalForestTV(n_estimators=n_estimators, random_state=seed).fit(X, y, ids)


def _stubbed(model, stub):
    import copy

    m = copy.copy(model)
    m.forest_ = stub
    return m


# --- §7.1 trend confounding (M2 part) --------------------------------------------------


class TrendOracle:
    """True fixed-profile cumhaz: ``h_k = 0.1 * 1.2**k * exp(0.8 z1)``, k = 0..K-1 (z2 unused)."""

    RATE, GROWTH, BETA = 0.1, 1.2, 0.8

    def _H0(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, K)
        inc = self.RATE * self.GROWTH ** np.arange(K)
        cum = np.concatenate([[0.0], np.cumsum(inc)])
        k = np.minimum(np.floor(t).astype(int), K - 1)
        return cum[k] + inc[k] * (t - k)

    def predict_cumhaz(self, X, times, aggregate, n_jobs):
        return self._H0(times)[None, :] * np.exp(self.BETA * X[:, [0]])


def trend_data(n, rng):
    """Rows ``X = [z1, z2, n1, n2]``, ``y``, ``ids`` for the §7.1 generator."""
    z1 = rng.normal(size=(n, K))
    e = rng.normal(size=(n, K))
    k = np.arange(K)
    z2 = 0.6 * z1 + 0.8 * e + 0.5 * (k - 3.5)
    lam = TrendOracle.RATE * TrendOracle.GROWTH**k * np.exp(TrendOracle.BETA * z1)
    T = _event_time(lam, rng)
    C = np.minimum(rng.uniform(2, END, size=n), END)
    U, event = np.minimum(T, C), T <= C
    n1, n2 = rng.normal(size=n), rng.normal(size=n)
    Xk = np.stack([z1, z2, np.broadcast_to(n1[:, None], (n, K)), np.broadcast_to(n2[:, None], (n, K))], axis=-1)
    return _grid_rows(Xk, U, event, lambda stop, ev, start: make_survival_y(stop, ev.astype(bool), start=start))


def trend_replicate(seed, n_train=1000, n_eval=1000, n_estimators=200):
    rng = np.random.default_rng(seed)
    X, y, ids = trend_data(n_train, rng)
    Xt, yt, idt = trend_data(n_eval, rng)
    m = SurvivalForestTV(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(X, y, ids)
    r = inspection.permutation_importance(m, Xt, yt, ids=idt, features=[0, 1], n_bootstrap=0, random_state=seed)
    rc = inspection.permutation_importance(
        m, Xt, yt, ids=idt, features=[1], conditional_on=[0], n_bootstrap=0, random_state=seed
    )
    m1 = inspection.permutation_importance(
        m, Xt, yt, ids=idt, features=[0, 1], strata=None, n_bootstrap=0, random_state=seed
    )
    return dict(
        imp_z1=r.importances_mean[0], imp_z2=r.importances_mean[1],
        imp_z2_cond_z1=rc.importances_mean[0], m1_z1=m1.importances_mean[0], m1_z2=m1.importances_mean[1],
    )


def trend_oracle(n=100_000, seed=12345, n_repeats=5):
    rng = np.random.default_rng(seed)
    X, y, ids = trend_data(n, rng)
    sel = ids < 4000  # whole subjects only: rows are grouped by id in ascending order
    carrier = _carrier(X[sel], y[sel], ids[sel], seed=seed)
    m = _stubbed(carrier, TrendOracle())
    r = inspection.permutation_importance(m, X, y, ids=ids, features=[0, 1], n_repeats=n_repeats, n_bootstrap=0)
    return float(r.importances_mean[0]), float(r.importances_mean[1])


# --- §7.3 timing -------------------------------------------------------------------------

TIMING_WINDOWS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
TIMING_INSIDE, TIMING_OUTSIDE = [2, 3], [0, 1, 4, 5]  # 0-based positions in importances_window (windows 3-4 / rest)


class TimingOracle:
    """True fixed-profile cumhaz: ``h(t) = 0.15 exp(z 1{2<t<=4} + 0.4 x0)``."""

    def predict_cumhaz(self, X, times, aggregate, n_jobs):
        t = np.asarray(times, dtype=float)
        x0, z = X[:, 0], X[:, 1]
        base = (0.15 * np.exp(0.4 * x0))[:, None]
        seg1 = np.clip(t, 0.0, 2.0)[None, :]
        seg2 = (np.clip(t, 2.0, 4.0) - 2.0)[None, :] * np.exp(z)[:, None]
        seg3 = np.clip(t - 4.0, 0.0, None)[None, :]
        return base * (seg1 + seg2 + seg3)


def timing_data(n, rng):
    """Rows ``X = [x0, z]``, ``y``, ``ids`` for the §7.3 generator (z on t in (2, 4] only)."""
    x0 = rng.normal(size=n)
    z = rng.normal(size=(n, K))
    ind = np.isin(np.arange(K), [2, 3]).astype(float)
    lam = 0.15 * np.exp(z * ind + 0.4 * x0[:, None])
    T = _event_time(lam, rng)
    C = np.minimum(rng.uniform(2, END, size=n), END)
    U, event = np.minimum(T, C), T <= C
    Xk = np.stack([np.broadcast_to(x0[:, None], (n, K)), z], axis=-1)
    return _grid_rows(Xk, U, event, lambda stop, ev, start: make_survival_y(stop, ev.astype(bool), start=start))


def timing_replicate(seed, n_train=1000, n_eval=1000, n_estimators=200):
    rng = np.random.default_rng(seed)
    X, y, ids = timing_data(n_train, rng)
    Xt, yt, idt = timing_data(n_eval, rng)
    m = SurvivalForestTV(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(X, y, ids)
    r = inspection.permutation_importance(
        m, Xt, yt, ids=idt, features=[1], windows=TIMING_WINDOWS, n_bootstrap=0, random_state=seed
    )
    return {f"w{m_}": v for m_, v in enumerate(r.importances_window[0])}


def timing_oracle(n=100_000, seed=23456, n_repeats=5):
    rng = np.random.default_rng(seed)
    X, y, ids = timing_data(n, rng)
    sel = ids < 4000
    carrier = _carrier(X[sel], y[sel], ids[sel], seed=seed)
    m = _stubbed(carrier, TimingOracle())
    r = inspection.permutation_importance(
        m, X, y, ids=ids, features=[1], windows=TIMING_WINDOWS, n_repeats=n_repeats, n_bootstrap=0
    )
    return r.importances_window[0]  # (6,)


# --- §7.4 competing risks ------------------------------------------------------------------


class CROracle:
    """True fixed-profile cause-specific cumhaz for S14 scenario A (z acts on cause 1 only)."""

    def predict_cause_cumhaz(self, X, times, n_jobs):
        t = np.asarray(times, dtype=float)
        z, x0, x1 = X[:, 0], X[:, 1], X[:, 2]
        h1 = 0.12 * np.exp(0.8 * z + 0.5 * x0)
        h2 = 0.08 * np.exp(0.7 * x1 - 0.3 * x0)
        return np.stack([h1[:, None] * t[None, :], h2[:, None] * t[None, :]], axis=1)


def cr_data(n, rng):
    """Rows ``X = [z, x0, x1, n1, n2, n3]``, ``y``, ``ids`` for the S14 scenario-A generator."""
    x0, x1, noise, z = cr_covariates(n, rng)
    lam = cr_hazards("A", x0, x1, z)  # (n, K, 2)
    T = _event_time(lam.sum(axis=2), rng)
    p1 = lam[:, :, 0] / lam.sum(axis=2)
    k_at_T = np.minimum(np.floor(np.where(np.isfinite(T), T, 0)).astype(int), K - 1)
    C = np.minimum(rng.uniform(2, END, size=n), END)
    U, event = np.minimum(T, C), T <= C
    cause = np.where(rng.random(n) < p1[np.arange(n), k_at_T], 1, 2)
    Xk = np.stack(
        [z, np.broadcast_to(x0[:, None], (n, K)), np.broadcast_to(x1[:, None], (n, K)), *[
            np.broadcast_to(noise[:, [j]], (n, K)) for j in range(noise.shape[1])
        ]],
        axis=-1,
    )
    label = np.where(event, cause, 0)
    return _grid_rows(Xk, U, label, lambda stop, lbl, start: make_competing_risks_y(stop, lbl, start=start))


def cr_replicate(seed, n_train=1000, n_eval=1000, n_estimators=200):
    rng = np.random.default_rng(seed)
    X, y, ids = cr_data(n_train, rng)
    Xt, yt, idt = cr_data(n_eval, rng)
    m = CompetingRisksForestTV(n_estimators=n_estimators, causes=[1, 2], random_state=seed, n_jobs=-1).fit(X, y, ids)
    r = inspection.permutation_importance(
        m, Xt, yt, ids=idt, features=[0], cause=1, n_bootstrap=0, random_state=seed
    )
    r2 = inspection.permutation_importance(
        m, Xt, yt, ids=idt, features=[0], cause=2, n_bootstrap=0, random_state=seed
    )
    return dict(d_s1=r.importances_mean[0], d_s2=r2.importances_mean[0])


def cr_oracle(n=100_000, seed=34567, n_repeats=5):
    rng = np.random.default_rng(seed)
    X, y, ids = cr_data(n, rng)
    sel = ids < 4000
    carrier = _carrier(X[sel], y[sel], ids[sel], seed=seed, competing=True, causes=[1, 2])
    m = _stubbed(carrier, CROracle())
    r = inspection.permutation_importance(
        m, X, y, ids=ids, features=[0], cause=1, n_repeats=n_repeats, n_bootstrap=0
    )
    return float(r.importances_mean[0])


# --- stats helpers -----------------------------------------------------------------------


def holm_reject(pvals, alpha=0.05):
    """Holm–Bonferroni: booleans, ``True`` = reject at family-wise ``alpha``."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = pvals.size
    thresh = alpha / (m - np.arange(m))
    passed = pvals[order] <= thresh
    stop = np.argmin(passed) if not passed.all() else m
    reject = np.zeros(m, dtype=bool)
    reject[order[:stop]] = True
    return reject


def one_sided_t(x, alpha=0.05):
    """p-value for H0: mean(x) <= 0 vs H1: mean(x) > 0."""
    n = len(x)
    t = np.mean(x) / (np.std(x, ddof=1) / np.sqrt(n))
    return stats.t.sf(t, n - 1)


def upper_bound(x, alpha_per_test):
    """One-sided upper confidence bound of ``mean(x)`` at ``1 - alpha_per_test``."""
    n = len(x)
    return np.mean(x) + stats.t.ppf(1 - alpha_per_test, n - 1) * np.std(x, ddof=1) / np.sqrt(n)


def mc_se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


# --- pilot / run ---------------------------------------------------------------------------


def pilot(n_reps=10):
    print(f"--- pilot ({n_reps} reps) ---")
    t0 = time.perf_counter()
    trend = pd.DataFrame([trend_replicate(s) for s in range(n_reps)])
    oz1, oz2 = trend_oracle()
    print(f"§7.1 trend: oracle(z1)={oz1:.4f} oracle(z2)={oz2:.4f}")
    print(f"  imp_z1 mean={trend.imp_z1.mean():.4f} MC-SE={mc_se(trend.imp_z1):.4f} (margin/7={0.5*oz1/7:.4f})")
    print(f"  imp_z2 mean={trend.imp_z2.mean():.4f} MC-SE={mc_se(trend.imp_z2):.4f} (margin/7={0.1*oz1/7:.4f})")
    print(f"  imp_z2|z1 mean={trend.imp_z2_cond_z1.mean():.4f}; M1 z1={trend.m1_z1.mean():.4f} z2={trend.m1_z2.mean():.4f}")

    timing = pd.DataFrame([timing_replicate(s) for s in range(n_reps)])
    ow = timing_oracle()
    O = ow[TIMING_INSIDE].sum()
    delta = 0.1 * O
    print(f"§7.3 timing: oracle windows={np.round(ow, 4).tolist()} O(inside)={O:.4f} delta={delta:.4f}")
    for m_ in range(6):
        col = timing[f"w{m_}"]
        print(f"  window {m_ + 1}: mean={col.mean():.4f} MC-SE={mc_se(col):.4f} (delta/7={delta / 7:.4f})")

    cr = pd.DataFrame([cr_replicate(s) for s in range(n_reps)])
    O1 = cr_oracle()
    delta1 = 0.1 * O1
    print(f"§7.4 CR: oracle(cause1)={O1:.4f} delta={delta1:.4f}")
    print(f"  d_s1 mean={cr.d_s1.mean():.4f} MC-SE={mc_se(cr.d_s1):.4f}")
    print(f"  d_s2 mean={cr.d_s2.mean():.4f} MC-SE={mc_se(cr.d_s2):.4f} (delta/7={delta1 / 7:.4f})")
    print(f"pilot done in {time.perf_counter() - t0:.1f}s")


def run(n_reps=50):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    trend = pd.DataFrame([trend_replicate(s) for s in range(n_reps)])
    oz1, oz2 = trend_oracle()
    trend.to_csv(OUT / "trend.csv", index=False)
    pass1a = abs(trend.imp_z2.mean()) <= 0.1 * oz1
    pass1b = trend.imp_z1.mean() >= 0.5 * oz1
    print(f"§7.1 trend confounding: oracle(z1)={oz1:.4f} oracle(z2)={oz2:.4f}")
    print(f"  mean imp(z1)={trend.imp_z1.mean():.4f} (>= {0.5 * oz1:.4f}? {pass1b})")
    print(f"  mean imp(z2)={trend.imp_z2.mean():.4f} (|.| <= {0.1 * oz1:.4f}? {pass1a})")
    print(f"  estimand caveat: imp(z2 | z1)={trend.imp_z2_cond_z1.mean():.4f}, "
          f"M1 imp(z1)={trend.m1_z1.mean():.4f}, M1 imp(z2)={trend.m1_z2.mean():.4f}")

    timing = pd.DataFrame([timing_replicate(s) for s in range(n_reps)])
    ow = timing_oracle()
    O = ow[TIMING_INSIDE].sum()
    delta = 0.1 * O
    timing.to_csv(OUT / "timing.csv", index=False)
    pin = [timing[f"w{m_}"].values for m_ in range(6)]
    p_inside = [one_sided_t(pin[m_]) for m_ in TIMING_INSIDE]
    reject_inside = holm_reject(p_inside)
    p_all_holm_positions = holm_reject([one_sided_t(pin[m_]) for m_ in range(6)])
    outside_ub = [upper_bound(pin[m_], 0.05 / len(TIMING_OUTSIDE)) for m_ in TIMING_OUTSIDE]
    pass2a = bool(np.any([p_all_holm_positions[m_] for m_ in TIMING_INSIDE]))
    pass2b = all(ub <= delta for ub in outside_ub)
    print(f"§7.3 timing: oracle windows={np.round(ow, 4).tolist()} O(inside)={O:.4f} delta={delta:.4f}")
    print(f"  window means={[round(float(np.mean(c)), 4) for c in pin]}")
    print(f"  Holm-reject (all 6 windows, one-sided>0): {p_all_holm_positions.tolist()}")
    print(f"  (a) reject in >=1 inside window (3,4)? {pass2a}")
    print(f"  outside upper 95% bounds (Bonferroni/4)={[round(u, 4) for u in outside_ub]}")
    print(f"  (b) all outside bounds <= delta? {pass2b}")

    cr = pd.DataFrame([cr_replicate(s) for s in range(n_reps)])
    O1 = cr_oracle()
    delta1 = 0.1 * O1
    cr.to_csv(OUT / "cr.csv", index=False)
    p_cr = holm_reject([one_sided_t(cr.d_s1.values), one_sided_t(cr.d_s2.values)])
    ub_s2 = upper_bound(cr.d_s2.values, 0.05)
    pass3a = bool(p_cr[0])
    pass3b = ub_s2 <= delta1
    print(f"§7.4 CR: oracle(cause1)={O1:.4f} delta={delta1:.4f}")
    print(f"  mean ΔS1={cr.d_s1.mean():.4f} ΔS2={cr.d_s2.mean():.4f}")
    print(f"  Holm-reject [S1, S2] (one-sided>0): {p_cr.tolist()}")
    print(f"  upper 95% bound of mean ΔS2={ub_s2:.4f} (<= delta? {pass3b}); "
          f"design's mean ΔS2 <= 0.1 mean ΔS1? {cr.d_s2.mean() <= 0.1 * cr.d_s1.mean()}")

    print(f"\nrun done in {time.perf_counter() - t0:.1f}s, wrote csvs to {OUT}")
    print(f"\nPASS SUMMARY: §7.1 z1>={pass1b} z2<={pass1a}; §7.3 (a)={pass2a} (b)={pass2b}; §7.4 (a)={pass3a} (b)={pass3b}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot":
        pilot(*(int(a) for a in sys.argv[2:]))
    else:
        run(*(int(a) for a in sys.argv[2:]))
