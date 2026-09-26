"""S13: cause-specific IPCW Brier / integrated Brier and dynamic Wolbers C."""

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rftvc import make_competing_risks_y, make_survival_y
from rftvc.metrics import (
    KaplanMeierCensoring,
    UndefinedMetricError,
    brier_landmark,
    cindex_dynamic,
    integrated_brier,
)

ORACLE = json.loads((Path(__file__).parent / "fixtures" / "cr_brier.json").read_text())


# --- Brier / integrated Brier --------------------------------------------------------


@pytest.mark.parametrize("cause", [1, 2])
def test_brier_matches_comprisk(cause):
    y = make_competing_risks_y(ORACLE["time"], ORACLE["event"])
    t = np.array(ORACLE["eval_times"])
    fx = ORACLE["causes"][str(cause)]
    probs = np.array(fx["probs"])
    want = np.array(fx["brier"])
    kw = dict(cause=cause, y_censor=y, g_min=1e-12)
    got = [brier_landmark(y, probs[:, j], tj, **kw) for j, tj in enumerate(t)]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    _, info = integrated_brier(y, probs, t, return_info=True, **kw)
    np.testing.assert_allclose(info["brier"], want, rtol=0, atol=1e-12)


# Boundary fixture at w = 2 for cause 1 (the comprisk oracle has no times at w).
#   id  stop label  role                               risk
#    1  1.0   1     case (tie with censoring of id 8)  0.6
#    2  1.5   2     competing before w                 0.3
#    3  2.0   0     censored at w -> control           0.2
#    4  2.0   2     competing at w                     0.4
#    5  2.0   1     case at w                          0.5
#    6  0.5   0     censored before w -> weight 0      0.9
#    7  3.0   0     control                            0.1
#    8  1.0   0     censored before w (tie at 1.0)     0.7
# Reverse KM, events before censoring at ties: G(0.5) = 7/8; at 1.0, 7 at risk,
# 1 event, 1 censored -> hazard 1/6, G(1) = 35/48; G(2) = 35/96.
# Weights: 1/G(1-) = 8/7 (id 1); 1/G(1.5-) = 48/35 (id 2); 1/G(2-) = 48/35 (ids 3, 4, 5, 7).
# Brier = [8/7 * 0.16 + 48/35 * (0.09 + 0.04 + 0.16 + 0.25 + 0.01)] / 8 = 41/350.
BOUNDARY_STOP = [1.0, 1.5, 2.0, 2.0, 2.0, 0.5, 3.0, 1.0]
BOUNDARY_LABEL = [1, 2, 0, 2, 1, 0, 0, 0]
BOUNDARY_RISK = [0.6, 0.3, 0.2, 0.4, 0.5, 0.9, 0.1, 0.7]


def test_brier_boundary_fixture():
    y = make_competing_risks_y(BOUNDARY_STOP, BOUNDARY_LABEL)
    score, info = brier_landmark(y, BOUNDARY_RISK, 2.0, cause=1, y_censor=y, return_info=True)
    assert score == pytest.approx(41 / 350, abs=1e-15)
    assert (info["n_cases"], info["n_competing"], info["n_controls"], info["n_censored"]) == (2, 2, 2, 2)
    G = KaplanMeierCensoring().fit(y)
    np.testing.assert_allclose(G.predict([0.5, 1.0, 1.5, 2.0], left=False), [7 / 8, 35 / 48, 35 / 48, 35 / 96])
    # Censoring-first ordering at t = 1 would give G(1) = 3/4 and a different score.
    assert G.predict([1.5], left=True)[0] != pytest.approx(3 / 4)


def _survival_data(seed, censor):
    rng = np.random.default_rng(seed)
    n = 80
    t = np.round(rng.exponential(2.0, n), 1) + 0.1
    c = np.round(rng.exponential(3.0, n), 1) + 0.1 if censor else np.full(n, np.inf)
    stop = np.minimum(t, np.minimum(c, 6.0))
    event = (t <= c) & (t <= 6.0)
    return make_survival_y(stop, event), rng.uniform(size=(n, 4))


@pytest.mark.parametrize("censor", [False, True])
def test_one_cause_equals_the_survival_metrics(censor):
    y, p = _survival_data(1, censor)
    times = np.array([0.5, 1.0, 2.0, 3.0])
    kw = dict(y_censor=y) if censor else {}
    for j, w in enumerate(times):
        assert brier_landmark(y, p[:, j], w, cause=1, **kw) == brier_landmark(y, p[:, j], w, **kw)
        assert cindex_dynamic(y, p[:, j], w, kind="incident", cause=1, **kw) == cindex_dynamic(
            y, p[:, j], w, kind="incident", **kw
        )
    # integrated_brier takes F = 1 - S with a cause, S without.
    S = 1.0 - np.cumsum(p, axis=1) / 4
    assert integrated_brier(y, 1.0 - S, times, cause=1, **kw) == pytest.approx(integrated_brier(y, S, times, **kw),
                                                                               abs=1e-15)


# --- dynamic Wolbers C -------------------------------------------------------------------


def test_type_a_weight_is_the_squared_left_limit():
    # Case A at 1.0 ties with a censoring (B) at 1.0: G(1-) = 1, G(1) = 3/4.
    # Uno weights 1/G(T-)^2: A (4 of 4 pairs, weight 1), C at 2.0 (1 of 2, weight 16/9)
    # -> (4 + 16/9) / (4 + 32/9) = 13/17; the 1/(G(T-) G(T)) form would give 0.8.
    y = make_competing_risks_y([1.0, 1.0, 2.0, 3.0, 3.0], [1, 0, 1, 0, 0])
    risk = [0.9, 0.5, 0.2, 0.4, 0.1]
    c = cindex_dynamic(y, risk, 2.5, kind="incident", cause=1, y_censor=y)
    assert c == pytest.approx(13 / 17, abs=1e-15)


def _brute_dynamic(stop, labels, risk, w, cause, g_min):
    """Weighted type A / B pairs with the same reverse KM (events first)."""
    n = stop.size
    case = (labels == cause) & (stop <= w)
    comp = (labels != 0) & (labels != cause) & (stop <= w)
    control = (stop >= w) & ~((labels != 0) & (stop <= w))
    if np.all(case | comp | control):
        wt = np.ones(n)
    else:
        G = KaplanMeierCensoring().fit(make_competing_risks_y(stop, labels))
        wt = 1.0 / np.maximum(G.predict(stop, left=True), g_min)
    num = den = 0.0
    for i in np.flatnonzero(case):
        for j in range(n):
            if j == i:
                continue
            if stop[j] >= stop[i] and not (labels[j] != 0 and stop[j] == stop[i]):
                pw = wt[i] ** 2
            elif labels[j] not in (0, cause) and stop[j] <= stop[i]:
                pw = wt[i] * wt[j]
            else:
                continue
            den += pw
            num += pw * (1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0)
    return num, den


@settings(max_examples=300, deadline=None)
@given(n=st.integers(2, 25), data=st.data())
def test_dynamic_wolbers_matches_brute_force(n, data):
    ints = lambda lo, hi: np.array(data.draw(st.lists(st.integers(lo, hi), min_size=n, max_size=n)))
    stop = ints(1, 8) * 0.5
    labels = ints(0, 3)
    risk = ints(0, 4).astype(float)
    cause = data.draw(st.integers(1, 3))
    w = data.draw(st.sampled_from([1.0, 2.0, 2.5, 5.0]))
    g_min = data.draw(st.sampled_from([0.05, 0.5]))
    y = make_competing_risks_y(stop, labels)
    num, den = _brute_dynamic(stop, labels, risk, w, cause, g_min)
    kw = dict(kind="incident", cause=cause, y_censor=y, g_min=g_min)
    exact = np.all(((labels != 0) & (stop <= w)) | (stop >= w))
    if exact:
        kw.pop("y_censor")
    if den == 0:
        with pytest.raises(UndefinedMetricError):
            cindex_dynamic(y, risk, w, **kw)
    else:
        assert cindex_dynamic(y, risk, w, **kw) == pytest.approx(num / den, rel=1e-12, abs=1e-12)


def test_cause_specific_auc_is_not_available():
    y = make_competing_risks_y([1.0, 2.0, 3.0], [1, 2, 0])
    with pytest.raises(NotImplementedError, match="competing-risks AUC"):
        cindex_dynamic(y, [0.3, 0.2, 0.1], 2.5, kind="cumulative", cause=1, y_censor=y)


def test_labels_need_a_cause():
    y = make_competing_risks_y([1.0, 2.0], [1, 2])
    with pytest.raises(TypeError, match="CompetingRisksForestTV"):
        brier_landmark(y, [0.1, 0.2], 1.5)
    with pytest.raises(ValueError, match="positive integer"):
        brier_landmark(y, [0.1, 0.2], 1.5, cause=0)
