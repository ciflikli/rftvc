"""S17 T6: the permutation-importance simulations' closed-form truth (bench/tvc_perm_sim.py).

Fast (default): the oracle stubs' closed-form cumulative hazards match numerical
integration of the literal design hazard (docs/plans/s17-plan.md §7), and one
smoke replication (R=1) of each sim runs end to end. Slow (``-m slow``): a
10-replication pilot loosely checks the declared pass rules point the right way
(the predeclared R=50 statistical gate itself runs via ``bench.tvc_perm_sim``,
not pytest; results are recorded in the S17 note).
"""

import numpy as np
import pytest

from bench.tvc_perm_sim import (
    K,
    TIMING_INSIDE,
    TIMING_OUTSIDE,
    TIMING_WINDOWS,
    CROracle,
    TrendOracle,
    TimingOracle,
    cr_data,
    cr_oracle,
    cr_replicate,
    holm_reject,
    one_sided_t,
    timing_data,
    timing_oracle,
    timing_replicate,
    trend_data,
    trend_oracle,
    trend_replicate,
    upper_bound,
)
from bench.s14_cr_sim import hazards as cr_hazards
from bench.tvc_path_effect_sim import HORIZONS as PE_HORIZONS
from bench.tvc_path_effect_sim import oracle as path_effect_oracle
from bench.tvc_path_effect_sim import replicate as path_effect_replicate

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*extrapolate:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*windows are too fine:UserWarning"),
]


def _numerical_H(rate_fn, t_grid, dt=1e-4):
    """Trapezoid-free left-Riemann cumulative integral of ``rate_fn(t)`` at ``t_grid``."""
    t = np.arange(0.0, t_grid[-1] + dt / 2, dt)
    h = rate_fn(t)  # (n, len(t))
    H = np.cumsum(h * dt, axis=1)
    idx = np.round(t_grid / dt).astype(int)
    return np.concatenate([np.zeros((h.shape[0], 1)), H], axis=1)[:, idx]


# --- closed-form oracle formulas match numerical integration of the design hazard -------


def test_trend_oracle_matches_numerical_integration():
    assert (TrendOracle.RATE, TrendOracle.GROWTH, TrendOracle.BETA) == (0.1, 1.2, 0.8)
    rng = np.random.default_rng(0)
    z1 = rng.normal(size=5)
    t_grid = np.linspace(0.0, K, 17)

    def rate(t):
        k = np.minimum(np.floor(t).astype(int), K - 1)
        return 0.1 * 1.2**k[None, :] * np.exp(0.8 * z1[:, None])

    H_num = _numerical_H(rate, t_grid)
    X = np.zeros((5, 2))
    X[:, 0] = z1
    H_closed = TrendOracle().predict_cumhaz(X, t_grid, None, 1)
    np.testing.assert_allclose(H_closed, H_num, atol=1e-6)


def test_timing_oracle_matches_numerical_integration():
    rng = np.random.default_rng(1)
    x0, z = rng.normal(size=5), rng.normal(size=5)
    t_grid = np.linspace(0.0, 6.0, 13)

    def rate(t):
        ind = ((t > 2.0) & (t <= 4.0)).astype(float)
        return 0.15 * np.exp(z[:, None] * ind[None, :] + 0.4 * x0[:, None])

    H_num = _numerical_H(rate, t_grid)
    X = np.column_stack([x0, z])
    H_closed = TimingOracle().predict_cumhaz(X, t_grid, None, 1)
    # the rate jumps at t = 2, 4: a left-Riemann sum at dt = 1e-4 is O(dt) off there, exact elsewhere
    np.testing.assert_allclose(H_closed, H_num, atol=2e-4)
    # the design's declared inside/outside window split
    assert TIMING_INSIDE == [2, 3] and TIMING_OUTSIDE == [0, 1, 4, 5]
    np.testing.assert_array_equal(TIMING_WINDOWS, [0, 1, 2, 3, 4, 5, 6])


def test_cr_oracle_matches_the_scenario_a_generator_at_every_interval():
    """Scenario A's cause-specific hazard has no time dependence: the oracle's fixed rate
    must equal ``cr_hazards("A", ...)`` at every unit interval, not just a re-derived formula."""
    rng = np.random.default_rng(2)
    x0, x1 = rng.normal(size=6), rng.integers(0, 2, 6).astype(float)
    z = rng.normal(size=6)
    lam_k = cr_hazards("A", x0, x1, np.broadcast_to(z[:, None], (6, K)))  # (6, K, 2), same at every k by construction
    for k in range(K):
        np.testing.assert_allclose(lam_k[:, k, 0], 0.12 * np.exp(0.8 * z + 0.5 * x0))
        np.testing.assert_allclose(lam_k[:, k, 1], 0.08 * np.exp(0.7 * x1 - 0.3 * x0))
    X = np.column_stack([z, x0, x1])
    t = np.array([0.0, 1.0, 3.0])
    H = CROracle().predict_cause_cumhaz(X, t, 1)
    np.testing.assert_allclose(H[:, 0, :], lam_k[:, 0, 0][:, None] * t[None, :])
    np.testing.assert_allclose(H[:, 1, :], lam_k[:, 0, 1][:, None] * t[None, :])


# --- oracle exact-zero facts (a wrong formula would generally not give exactly 0) --------


def test_trend_oracle_ignores_z2_exactly():
    _, oz2 = trend_oracle(n=2000, seed=0, n_repeats=1)
    assert oz2 == 0.0


def test_timing_oracle_is_exactly_zero_outside_the_signal_window():
    w = timing_oracle(n=2000, seed=0, n_repeats=1)
    np.testing.assert_allclose(w[TIMING_OUTSIDE], np.zeros(4), atol=1e-9)  # exact but for fp cancellation noise
    assert (w[TIMING_INSIDE] > 0).all()


# --- generator sanity (rows, events) ------------------------------------------------------


@pytest.mark.parametrize("gen", [trend_data, timing_data, cr_data])
def test_generator_rows_are_well_formed(gen):
    rng = np.random.default_rng(0)
    X, y, ids = gen(500, rng)
    assert X.shape[0] == y.shape[0] == ids.shape[0]
    np.testing.assert_array_equal(np.unique(ids), np.arange(500))  # every id appears (>= 1 row)
    assert np.all(y["stop"] > y["start"])
    # one event at most per id, and only on that id's last row
    order = np.argsort(ids, kind="stable")
    last = np.r_[np.diff(ids[order]) != 0, True]
    assert np.array_equal(np.flatnonzero(y["event"][order] != 0), np.flatnonzero(last & (y["event"][order] != 0)))


# --- smoke (fast, R=1) and pilot regression (slow, R~10) ---------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::RuntimeWarning")
@pytest.mark.parametrize("fn", [trend_replicate, timing_replicate, cr_replicate])
def test_replicate_smoke(fn):
    r = fn(0, n_train=400, n_eval=400, n_estimators=20)
    assert all(np.isfinite(v) for v in r.values())


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::RuntimeWarning")
def test_path_effect_replicate_smoke():
    r = path_effect_replicate(0, n_train=200, n_test=100, n_estimators=20)
    assert len(r) == len(PE_HORIZONS)
    assert all(np.isfinite(v) for v in r.values())


@pytest.mark.slow
def test_pilot_pass_rules_point_the_right_way():
    n_reps = 10
    trend = [trend_replicate(s) for s in range(n_reps)]
    oz1, oz2 = trend_oracle()
    imp_z1 = np.array([r["imp_z1"] for r in trend])
    imp_z2 = np.array([r["imp_z2"] for r in trend])
    assert oz2 == 0.0 and oz1 > 0
    assert imp_z1.mean() >= 0.25 * oz1  # loose at R=10 (design rule: >= 0.5*oracle at R=50)
    assert abs(imp_z2.mean()) <= 0.3 * oz1  # loose at R=10 (design rule: <= 0.1*oracle at R=50)

    timing = [timing_replicate(s) for s in range(n_reps)]
    ow = timing_oracle()
    assert (ow[TIMING_OUTSIDE] == 0.0).all() and (ow[TIMING_INSIDE] > 0).all()
    win = np.array([[r[f"w{m}"] for m in range(6)] for r in timing])
    p = [one_sided_t(win[:, m]) for m in TIMING_INSIDE]
    assert holm_reject(p).any()  # at least one inside window significant, even at R=10

    cr = [cr_replicate(s) for s in range(n_reps)]
    O1 = cr_oracle()
    d_s1 = np.array([r["d_s1"] for r in cr])
    d_s2 = np.array([r["d_s2"] for r in cr])
    assert holm_reject([one_sided_t(d_s1), one_sided_t(d_s2)])[0]
    assert upper_bound(d_s2, 0.05) <= 0.3 * O1  # loose at R=10 (design rule: <= 0.1*oracle at R=50)

    # path_effect: FAILS the design's tight |mean bias| <= 0.1*|true_delta| rule even at the
    # full R=50 run (docs/bench/s20-effects/path_effect.csv; s20-plan.md T9) -- a real,
    # well-powered (MC-SE << margin/7) finite-sample forest bias, not a code defect (checked
    # against an exact-hazard stub separately). As S19 did for its own accepted §7.2 deviation
    # (this same file's sibling test_landmark_sim_truth.py: "assert hist.mean() > 0", not the
    # real rule's tight bound), this pilot-regression check only confirms the machinery points
    # the right way -- positive, same order of magnitude as the truth -- not the tight margin,
    # so a known, documented, user-pending deviation doesn't leave the suite permanently red.
    pe = [path_effect_replicate(s) for s in range(n_reps)]
    true_d = path_effect_oracle()
    est = np.array([[r[f"m_h{i}"] for i in range(len(PE_HORIZONS))] for r in pe])
    mean_est = est.mean(axis=0)
    assert (mean_est > 0).all()
    assert (mean_est >= 0.5 * true_d).all()
