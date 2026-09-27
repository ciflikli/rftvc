"""S19 T8: the landmark-importance simulations' generator truth (bench/tvc_landmark_sim.py).

Fast (default): ``_true_hist`` matches a hand-built rolling-2-unit mean, the oracle stub's
closed-form rate matches its literal formula, and generator rows are well-formed. Slow
(``-m slow``): a small-scale smoke run of the three §7.2/§7.5b replicate functions.
"""

import numpy as np
import pytest

from bench.tvc_landmark_sim import (
    BETA,
    RATE,
    LevelHistoryOracle,
    _true_hist,
    censoring_data,
    censoring_replicate,
    copies_replicate,
    level_history_data,
    oracle,
    replicate,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*extrapolate:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*windows are too fine:UserWarning"),
]


def test_true_hist_is_the_rolling_2_unit_mean():
    z = np.array([[1.0, 3.0, 5.0, 9.0]])
    got = _true_hist(z)
    np.testing.assert_allclose(got, [[1.0, 2.0, 4.0, 7.0]])  # z0, (1+3)/2, (3+5)/2, (5+9)/2


def test_oracle_stub_matches_its_literal_formula():
    X = np.array([[0.5, -1.0], [2.0, 0.0]])
    t = np.array([0.0, 1.0, 2.0])
    H_hist = LevelHistoryOracle(1).predict_cumhaz(X, t, None, 1)  # driver = column 1 (true_hist)
    np.testing.assert_allclose(H_hist, RATE * np.exp(BETA * X[:, [1]]) * t[None, :])
    H_markov = LevelHistoryOracle(0).predict_cumhaz(X, t, None, 1)  # driver = column 0 (z)
    np.testing.assert_allclose(H_markov, RATE * np.exp(BETA * X[:, [0]]) * t[None, :])


@pytest.mark.parametrize("scenario", ["history", "markov"])
def test_level_history_rows_are_well_formed(scenario):
    rng = np.random.default_rng(0)
    df = level_history_data(300, rng, scenario)
    assert set(df.columns) >= {"id", "start", "stop", "event", "z", "true_hist"}
    assert (df["stop"] > df["start"]).all()
    assert df["id"].n_unique() == 300


def test_censoring_data_rows_are_well_formed_and_heavily_censored():
    rng = np.random.default_rng(0)
    df = censoring_data(300, rng)
    assert (df["stop"] > df["start"]).all()
    censoring_rate = 1.0 - df["event"].cast(int).to_numpy().mean()
    assert censoring_rate > 0.5  # "heavy" censoring, the point of this fixture


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_replicate_smoke():
    assert np.isfinite(replicate(0, "history", n_train=100, n_eval=100, n_estimators=10))
    assert np.isfinite(replicate(0, "markov", n_train=100, n_eval=100, n_estimators=10))


def test_copies_replicate_smoke():
    se = copies_replicate(0, step=1.0, n_train=100, n_eval=100, n_estimators=10)
    assert np.isfinite(se)


def test_censoring_replicate_smoke():
    out = censoring_replicate(0, n_train=200, n_eval=200, n_estimators=10)
    assert all(np.isfinite(v) for v in out.values())


def test_oracle_smoke():
    assert np.isfinite(oracle("history", n=2000, n_repeats=1))
    assert np.isfinite(oracle("markov", n=2000, n_repeats=1))


@pytest.mark.slow
def test_pilot_pass_rules_point_the_right_way():
    n_reps = 10
    oh, ol = oracle("history"), oracle("markov")
    assert oh > 0 and ol > 0
    hist = np.array([replicate(s, "history") for s in range(n_reps)])
    mark = np.array([replicate(s, "markov") for s in range(n_reps)])
    assert hist.mean() > 0  # history matters when the true driver is the rolling mean
    assert mark.mean() <= 0.05 * ol + 3 * (mark.std(ddof=1) / np.sqrt(n_reps))  # near the null, loosely

    small_se = np.array([copies_replicate(s, step=0.5, n_train=300, n_eval=300, n_estimators=50) for s in range(5)])
    large_se = np.array([copies_replicate(s, step=4.0, n_train=300, n_eval=300, n_estimators=50) for s in range(5)])
    assert small_se.mean() / large_se.mean() >= 0.3  # loose at 5 reps (design rule: >= 0.5 at the full run)

    cens = [censoring_replicate(s, n_train=400, n_eval=400, n_estimators=60) for s in range(5)]
    pe_z1 = np.mean([c["pe_z1"] for c in cens])
    pe_z2 = np.mean([c["pe_z2"] for c in cens])
    assert pe_z1 > pe_z2
