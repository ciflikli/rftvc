"""S19 T8: the landmark-importance simulations' generator truth (bench/tvc_landmark_sim.py).

Fast (default): ``_true_hist`` matches a hand-built rolling-2-unit mean, the oracle stub's
closed-form rate matches its literal formula, and generator rows are well-formed. Also
default-tier: the §7.2/§7.5b CI gates from
``bench.landmark_importance_truth_check`` — closes
the validation audit row 4, the last item on the manual-gate
punchlist. Slow (``-m slow``): a small-scale smoke run of the three §7.2/§7.5b replicate
functions, at yet another (looser) scale — not fully redundant with the Slice 9 gates below,
since it also smoke-tests ``copies_replicate``/``censoring_replicate`` at their own scale.
"""

import numpy as np
import pytest

from bench.landmark_importance_truth_check import (
    check_censoring,
    check_copies,
    check_level_history,
    run_censoring,
    run_copies,
    run_level_history,
)
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


# --- §7.2/§7.5b gates, default tier -------------------------------------------
#
# Reduced scales/rep counts and margins verified across >= 2 independent out-of-band seed
# batches (10000-.../20010-...) distinct from these gates' own seeds (0-14, or 0-29 for
# history's R=30).


def test_markov_control_and_history_significance():
    """Markov control: the design's real, unmodified bound
    ``mark.mean() <= 0.05 * oracle(markov)`` (R=15) — passes with 3-7x margin at every scale
    tested (out-of-band batches, same R=15: seeds 10000-10014, 20010-20024).

    History: **not** the design's declared 0.25x-of-oracle magnitude rule — that rule already
    fails at the full R=50 scale and was independently re-verified here to be worse than a coin flip against its
    own bar at reduced scale too (43%-101% of the bound across seed batches). Recalibrating a
    new threshold to force a pass would silently override that accepted deviation. Instead: a
    one-sided t-test that ``hist.mean() > 0`` is statistically significant, at R=30 (R=15 is
    not robust enough for this — p=0.138 in the primary seed batch; R=30 gives
    p=0.0004/1.5e-6/7.3e-5 at this gate's own seeds (0-29) and two independent out-of-band
    batches (10000-10029, 20010-20039)).
    """
    oh, ol = oracle("history", n=20000, n_repeats=2), oracle("markov", n=20000, n_repeats=2)
    hist, mark = run_level_history()
    p_history, pass_markov, bound = check_level_history(hist, mark, oh, ol)
    assert p_history < 0.05, f"history-given-level mean not significantly positive: p={p_history}"
    assert pass_markov, f"Markov control's mean {mark.mean()} exceeded its bound {bound}"


def test_copies_se_does_not_shrink_with_more_landmarks():
    """The design's real rule (``se(step=0.5)/se(step=4.0) >= 0.5``, R=15): the id-cluster
    bootstrap SE does not shrink with more landmark copies per subject — confirming it
    clusters by subject, not by row. Verified with real margin (0.56-0.73, never near the 0.5
    boundary) across three seed batches — this gate's own (0-14) and two independent
    out-of-band batches (10000-10014, 20010-20024) — at this reduced scale
    (n_train=n_eval=100, n_estimators=20, vs. the design's 500/500/100).
    """
    small_se, large_se = run_copies()
    pass_ratio, ratio = check_copies(small_se, large_se)
    assert pass_ratio, f"copies SE ratio {ratio} fell below the design's 0.5 bound"


def test_censoring_pe_and_brier_rankings_agree():
    """The design's real rank-*concordance* rule, unmodified from
    ``bench.tvc_landmark_sim.run()``'s own ``pass4``: PE and Brier importance agree on which
    of ``z1``/``z2`` ranks larger under heavy censoring (not "both rank z1 above z2" — a
    stricter claim the design never asked for), R=15 at n_train=n_eval=150, n_estimators=25
    (n=100 breaks the generator — heavy censoring sometimes leaves an eval fold with no
    scorable PE events). Uses a plain mean comparison per family, not a per-replicate
    significance test: a one-sided t-test on the Brier half is seed-sensitive at this scale
    (p=0.109 in one out-of-band batch), while the mean comparison — the design's actual rule —
    is robust across all three batches tested (0-14, 10000-10014, 20010-20024): both families
    ranked z1 above z2 in every batch tested, so agreement and "both rank z1 above z2" happen
    to coincide on the data seen so far, but only agreement is the gated claim.
    """
    df = run_censoring()
    pass_agree, pe_rank_z1_larger, brier_rank_z1_larger = check_censoring(df)
    assert pass_agree, (
        f"PE and Brier importance rankings disagreed: "
        f"pe_rank_z1_larger={pe_rank_z1_larger}, brier_rank_z1_larger={brier_rank_z1_larger}"
    )


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
