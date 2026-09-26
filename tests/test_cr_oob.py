"""S12: competing-risks OOB, score, and the Wolbers cause-specific concordance."""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rftvc import CompetingRisksForestTV, make_competing_risks_y, make_survival_y
from rftvc.metrics import UndefinedMetricError, concordance_index_cp, concordance_index_cr
from tests.ref.coarsen_ref import coarsen_ref
from tests.test_cr_core import _cr_data
from tests.test_tvc import _cp_data


def _manual_oob_cif(model, X, group, aggregate):
    """F_j at the last event time from leaf profiles of the trees whose bag lacks the row's id."""
    core, J, t_end = model.forest_, model.n_causes_, model.event_times_[-1]
    leaves = model.apply(X)
    bags = [set(core.in_bag_ids(b)) for b in range(core.n_trees)]
    out = np.full((X.shape[0], J), np.nan)
    for r in range(X.shape[0]):
        per_tree = []
        for b in range(core.n_trees):
            if group[r] in bags[b]:
                continue
            lt, ch = core.leaf_profile(b, int(leaves[r, b]))
            keep = np.asarray(lt) <= t_end
            per_tree.append(dict(zip(np.asarray(lt)[keep], np.diff(np.asarray(ch), axis=0, prepend=0.0)[keep])))
        if not per_tree:
            continue
        grid = sorted(set().union(*per_tree))
        dL = np.array([[inc.get(t, np.zeros(J)) for t in grid] for inc in per_tree])  # (trees, V, J)

        def final_cif(d):
            s, f = 1.0, np.zeros(J)
            for row in d:
                f, s = f + s * row, s * (1.0 - row.sum())
            return f

        out[r] = final_cif(dL.mean(axis=0)) if aggregate == "hazard" else np.mean([final_cif(d) for d in dL], axis=0)
    return out


@pytest.mark.parametrize("aggregate", ["hazard", "cif"])
def test_oob_prediction_matches_manual_reference(aggregate):
    X, y, ids = _cr_data(60, seed=13, labels=[2, 5])
    m = CompetingRisksForestTV(
        n_estimators=15, min_ids_leaf=3, aggregate=aggregate, oob_score=True, score_cause=5, random_state=0
    ).fit(X, y, ids)
    assert m.oob_prediction_.shape == (len(y), 2)
    np.testing.assert_allclose(m.oob_prediction_, _manual_oob_cif(m, X, ids, aggregate), rtol=0, atol=1e-12)
    ok = np.isfinite(m.oob_prediction_).all(axis=1)
    assert ok.sum() > 0.8 * len(y)
    want = concordance_index_cr(y[ok], m.oob_prediction_[ok, 1], cause=5, ids=ids[ok])
    assert m.oob_score_ == pytest.approx(want, abs=1e-15)
    # The OOB trees are counted per row.
    assert (m.oob_n_trees_[ok] > 0).all() and (m.oob_n_trees_[~ok] == 0).all()


def test_oob_with_coarsening_keeps_labels_of_moved_events():
    X, y, ids = _cr_data(150, seed=14, labels=[2, 5])
    ntime = 4
    codes = np.searchsorted([2, 5], y["event"]) + 1
    codes[y["event"] == 0] = 0
    kept, s, t, e, _, _ = coarsen_ref(y["start"], y["stop"], codes, ids, ntime)
    assert ((e > 0) & (codes[kept] == 0)).any()  # an event moved onto an earlier row
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = CompetingRisksForestTV(
            n_estimators=12, min_ids_leaf=3, ntime=ntime, oob_score=True, score_cause=5, random_state=1
        ).fit(X, y, ids)
    dropped = np.setdiff1d(np.arange(len(y)), kept)
    assert dropped.size and np.isnan(m.oob_prediction_[dropped]).all() and (m.oob_n_trees_[dropped] == 0).all()
    pred = m.oob_prediction_[kept]
    ok = np.isfinite(pred).all(axis=1)
    labels = np.where(e > 0, np.array([2, 5])[np.maximum(e, 1) - 1], 0)
    y_c = make_competing_risks_y(t[ok], labels[ok], start=s[ok])
    assert m.oob_score_ == pytest.approx(concordance_index_cr(y_c, pred[ok, 1], cause=5, ids=ids[kept][ok]), abs=1e-15)


def test_score_is_wolbers_concordance_of_predict():
    X, y, ids = _cr_data(100, seed=15)
    m = CompetingRisksForestTV(n_estimators=10, min_ids_leaf=3, score_cause=2, random_state=0).fit(X, y, ids)
    np.testing.assert_array_equal(m.predict(X), m.predict_cumulative_incidence(X, m.event_times_[-1:], cause=2)[:, 0])
    assert m.score(X, y, ids) == concordance_index_cr(y, m.predict(X), cause=2, ids=ids)


def test_oob_score_tracks_the_score_cause_signal():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, 3))
    t1 = rng.exponential(np.exp(-1.2 * X[:, 0]))  # cause 1 driven by x0
    t2 = rng.exponential(np.exp(-1.2 * X[:, 1]))  # cause 2 driven by x1
    c = rng.exponential(2.0, n)
    stop = np.minimum(np.minimum(t1, t2), c)
    ev = np.where(stop == c, 0, np.where(t1 < t2, 1, 2))
    y = make_competing_risks_y(stop, ev)
    kw = dict(n_estimators=100, oob_score=True, random_state=0)
    s1 = CompetingRisksForestTV(score_cause=1, **kw).fit(X, y).oob_score_
    s2 = CompetingRisksForestTV(score_cause=2, **kw).fit(X, y).oob_score_
    assert s1 > 0.65 and s2 > 0.65
    # x1 raises the competing cause: its high values are type-B controls (competing
    # events) for cause 1, so as a cause-1 score it is worse than chance.
    assert concordance_index_cr(y, X[:, 1], cause=1) < 0.45 < 0.55 < concordance_index_cr(y, X[:, 0], cause=1)


# --- Wolbers C ---------------------------------------------------------------------

# Hand-worked fixture: (id, start, stop, label, risk). Cause 1 cases: A (T=2),
# G (T=3), C (T=4, delayed entry). D and E have cause 2 at 1.5 and at 2 (a tie
# with A); F enters at 2.5.
HAND = [
    ("A", 0.0, 2.0, 1, 0.9),
    ("B", 0.0, 1.0, 0, 0.1),
    ("B", 1.0, 3.0, 0, 0.5),
    ("C", 1.0, 4.0, 1, 0.2),
    ("D", 0.0, 1.5, 2, 0.95),
    ("E", 0.0, 2.0, 2, 0.9),
    ("F", 2.5, 5.0, 0, 0.2),
    ("G", 0.0, 3.0, 1, 0.3),
]
# A: type A {B .5, C .2, G .3} -> 3 of 3; type B {D .95: 0, E .9 (tie at T): 1/2}.
# G: type A {B .5: 0, C .2: 1, F .2: 1}; type B {D: 0, E: 0}.
# C: type A {F .2: 1/2}; type B {D: 0, E: 0}.
# 13 pairs, 6 concordant.
HAND_C = 6 / 13


def test_wolbers_hand_worked_fixture():
    ids, start, stop, label, risk = (np.array(c) for c in zip(*HAND))
    y = make_competing_risks_y(stop.astype(float), label.astype(int), start=start.astype(float))
    assert concordance_index_cr(y, risk.astype(float), cause=1, ids=ids) == pytest.approx(HAND_C, abs=1e-15)
    assert concordance_index_cr(y, risk.astype(float), cause=1) == pytest.approx(HAND_C, abs=1e-15)


def _brute_wolbers(start, stop, labels, risk, cause, groups):
    num = den = 0.0
    for i in np.flatnonzero(labels == cause):
        T = stop[i]
        for j in range(stop.size):
            if j == i or (groups is not None and groups[j] == groups[i]):
                continue
            type_a = start[j] < T <= stop[j] and not (labels[j] != 0 and stop[j] == T)
            type_b = labels[j] not in (0, cause) and stop[j] <= T
            if type_a or type_b:
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num, den


@settings(max_examples=300, deadline=None)
@given(
    n=st.integers(2, 25),
    data=st.data(),
)
def test_wolbers_matches_brute_force(n, data):
    ints = lambda lo, hi: np.array(data.draw(st.lists(st.integers(lo, hi), min_size=n, max_size=n)))
    start = ints(0, 4) * 0.5
    stop = start + ints(1, 5) * 0.5
    labels = ints(0, 3)
    risk = ints(0, 4).astype(float)
    groups = ints(0, n // 2) if data.draw(st.booleans()) else None
    cause = data.draw(st.integers(1, 3))
    num, den = _brute_wolbers(start, stop, labels, risk, cause, groups)
    y = make_competing_risks_y(stop, labels, start=start)
    if den == 0:
        with pytest.raises(UndefinedMetricError):
            concordance_index_cr(y, risk, cause=cause, ids=groups)
    else:
        assert concordance_index_cr(y, risk, cause=cause, ids=groups) == pytest.approx(num / den, abs=1e-12)


def test_wolbers_with_one_cause_is_the_counting_process_concordance():
    X, y, ids = _cp_data(80, seed=1)
    for risk in [X[:, 0], np.round(X[:, 0])]:
        assert concordance_index_cr(y, risk, cause=1, ids=ids) == concordance_index_cp(y, risk, ids=ids)
    y2 = make_survival_y(y["stop"], y["event"], start=y["start"])
    assert concordance_index_cr(y2, X[:, 0], cause=1) == concordance_index_cp(y2, X[:, 0])


def test_wolbers_rejects_bad_causes():
    y = make_competing_risks_y([1.0, 2.0], [1, 2])
    for cause in [0, -1, True, 1.5, "1"]:
        with pytest.raises(ValueError, match="positive integer"):
            concordance_index_cr(y, [0.1, 0.2], cause=cause)
