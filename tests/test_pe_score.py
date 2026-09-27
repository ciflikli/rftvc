"""Piecewise-exponential score (S16; docs/plans/tvc-deviance.md)."""

import types

import numpy as np
import pytest

from rftvc import SurvivalForestTV, make_competing_risks_y, make_survival_y
from rftvc.metrics import UndefinedMetricError, _baseline_at, _window_exposure_1, event_windows, piecewise_exponential_score

A, B = 0.3, 0.7
W = np.array([0.0, 1.0, 2.0])


def pe(y, H, w=W, null=None, **kw):
    null = np.zeros(len(w)) if null is None else null
    return piecewise_exponential_score(y, H, w, null_cumhaz=null, **kw)


def _const(n, a=A, b=B):
    return np.tile([0.0, a, a + b], (n, 1))


def _ref_score(start, stop, event, H, w, null, alpha):
    """Row-by-row, window-by-window reference in plain Python."""
    total, tau = 0.0, w[-1]
    for r in range(len(stop)):
        for m in range(len(w) - 1):
            lo, hi = w[m], w[m + 1]
            rate = ((1 - alpha) * (H[r, m + 1] - H[r, m]) + alpha * (null[m + 1] - null[m])) / (hi - lo)
            e = max(0.0, min(stop[r], hi) - max(start[r], lo))
            total -= rate * e
            if event[r] and lo < stop[r] <= hi and 0 < stop[r] <= tau:
                total += np.log(rate)
    return total


def test_boundary_literals():
    # rows: event at edge w1; event at tau; event after tau; start == tau; (-2, 1] censored; event at 0
    start = np.array([0.0, 0.5, 1.5, 2.0, -2.0, -3.0])
    stop = np.array([1.0, 2.0, 3.0, 3.0, 1.0, 0.0])
    event = np.array([True, True, True, False, False, True])
    r = pe(make_survival_y(stop, event, start=start), _const(6), alpha=0.0, reduce="sum")
    w1 = np.log(A) - A - 0.5 * A - A
    w2 = np.log(B) - B - 0.5 * B
    np.testing.assert_allclose(r.by_window, [w1, w2], rtol=1e-14)
    assert r.total == pytest.approx(w1 + w2, rel=1e-14)
    assert (r.n_events, r.n_truncated_events) == (2, 2)
    per_event = pe(make_survival_y(stop, event, start=start), _const(6), alpha=0.0)
    assert per_event.total == pytest.approx((w1 + w2) / 2, rel=1e-14)


def test_zero_exposure_zero_rate_cells_are_not_nan():
    y = make_survival_y(np.array([1.0, 2.0]), np.array([True, False]), start=np.array([0.0, 1.0]))
    H = np.array([[0.0, A, A], [0.0, A, A]])  # zero rate in window 2, where row 2 is censored
    r = pe(y, H, alpha=0.0, reduce="sum")
    np.testing.assert_allclose(r.by_window, [np.log(A) - A, 0.0], rtol=1e-14)  # row 1 only; row 2 adds -0 * 1


def test_empty_window_scores_zero():
    y = make_survival_y(np.array([1.0]), np.array([True]))
    r = pe(y, np.array([[0.0, A, A + B, A + B + 1]]), w=np.array([0.0, 1.0, 2.0, 3.0]), alpha=0.0, reduce="sum")
    np.testing.assert_allclose(r.by_window, [np.log(A) - A, 0.0, 0.0], rtol=1e-14)


def test_matches_reference_and_decomposes():
    rng = np.random.default_rng(0)
    n = 200
    start = rng.uniform(-0.5, 2.0, n)
    stop = start + rng.exponential(1.5, n)
    event = rng.random(n) < 0.6
    w = np.array([0.0, 0.4, 1.1, 2.5, 3.0])
    H = np.cumsum(np.c_[np.zeros(n), rng.exponential(0.5, (n, 4))], axis=1)
    null = np.cumsum(np.r_[0.0, rng.exponential(0.5, 4)])
    ids = rng.integers(0, 40, n)
    y = make_survival_y(stop, event, start=start)
    for alpha in (0.0, 0.2):
        r = pe(y, H, w, null, alpha=alpha, ids=ids, reduce="sum")
        assert r.total == pytest.approx(_ref_score(start, stop, event, H, w, null, alpha), rel=1e-12)
        assert r.by_window.sum() == pytest.approx(r.total, rel=1e-12)
        assert r.by_id.sum() == pytest.approx(r.total, rel=1e-12)
        per = pe(y, H, w, null, alpha=alpha, ids=ids)
        assert per.total == pytest.approx(r.total / r.n_events, rel=1e-12)
        np.testing.assert_allclose(per.by_id, r.by_id / r.n_events, rtol=1e-12)
    # id labels in order of first appearance
    first = list(dict.fromkeys(ids.tolist()))
    assert r.id_labels.tolist() == first


def test_deviance_relation():
    rng = np.random.default_rng(1)
    n = 100
    stop = rng.uniform(0.2, 2.0, n)
    event = rng.random(n) < 0.5
    y = make_survival_y(stop, event)
    H = np.cumsum(np.c_[np.zeros(n), rng.exponential(0.5, (n, 2))], axis=1)
    r = pe(y, H, alpha=0.0, reduce="sum")
    # Poisson deviance of cells, computed densely
    D, extra = 0.0, 0.0
    for i in range(n):
        for m in range(2):
            e = max(0.0, min(stop[i], W[m + 1]) - W[m])
            N = float(event[i] and W[m] < stop[i] <= W[m + 1])
            mu = (H[i, m + 1] - H[i, m]) * e
            if e > 0:
                D += 2 * ((N * np.log(N / mu) if N else 0.0) - (N - mu))
                extra += N * np.log(e)
    assert D == pytest.approx(-2 * r.total - 2 * extra - 2 * r.n_events, rel=1e-12)


def test_proper_on_constants_and_improper_v0():
    rng = np.random.default_rng(2)
    n, a = 200_000, 2.5
    T = rng.exponential(size=n) ** (1 / a)  # Weibull, Λ(t) = t^a
    C = rng.uniform(0, 2, n)
    stop = np.minimum(np.minimum(T, C), 1.0)
    event = (T <= C) & (T <= 1)
    y = make_survival_y(stop, event)
    w = np.array([0.0, 1.0])
    grid = np.linspace(0.4, 1.1, 36)
    scores = [pe(y, np.tile([0.0, c], (n, 1)), w, alpha=0.0).total for c in grid]
    c_star = event.sum() / stop.sum()
    assert abs(grid[int(np.argmax(scores))] - c_star) <= grid[1] - grid[0]
    # v0 (log of the row-own-exposure expectation) prefers a constant hazard to the truth:
    # the reason the score uses window rates (tvc-deviance.md §2.3).
    v0 = {b: np.mean(event * np.log(stop**b) - stop**b) for b in (1.0, a)}
    assert v0[1.0] > v0[a]


def test_zero_rate_share_and_alpha():
    stop = np.array([0.5, 0.6, 1.5, 1.6, 2.5])
    event = np.array([True, True, True, True, True])
    y = make_survival_y(stop, event)
    H = np.array([[0.0, 0.0, 1.0]] * 2 + [[0.0, 1.0, 2.0]] * 3)  # rows 0, 1: zero rate in window 1
    null = np.array([0.0, 1.0, 2.0])
    shares = set()
    for alpha in (0.0, 0.01, 0.5):
        with pytest.warns(UserWarning, match="too fine"):
            r = pe(y, H, null=null, alpha=alpha)
        shares.add(r.zero_rate_share)
        assert (r.n_events, r.n_truncated_events) == (4, 1)
        assert np.isfinite(r.total) == (alpha > 0)
        assert r.null_total == pytest.approx(pe(y, np.tile(null, (5, 1)), null=null, alpha=0.0).total)
    assert shares == {2 / 4}


def test_competing_risks_labels_and_causes():
    rng = np.random.default_rng(3)
    n = 120
    stop = rng.uniform(0.1, 2.0, n)
    labels = rng.choice([0, 2, 5], n)
    y = make_competing_risks_y(stop, labels)
    H = np.cumsum(np.concatenate([np.zeros((n, 2, 1)), rng.exponential(0.4, (n, 2, 2))], axis=2), axis=2)
    null = np.array([[0.0, 0.3, 0.6], [0.0, 0.2, 0.5]])
    r = pe(y, H, null=null, causes=[2, 5], reduce="sum", alpha=0.1)
    for k, c in enumerate((2, 5)):
        single = pe(make_survival_y(stop, labels == c), H[:, k], null=null[k], reduce="sum", alpha=0.1)
        assert r.by_cause[k] == pytest.approx(single.total, rel=1e-12)
        sel = pe(y, H, null=null, causes=[2, 5], cause=c, reduce="sum", alpha=0.1)
        assert sel.total == pytest.approx(single.total, rel=1e-12)
        assert sel.by_cause is None
    np.testing.assert_allclose(r.by_cause_window.sum(axis=1), r.by_cause, rtol=1e-12)
    for c in (2, 5):  # per-event normalisation with one cause divides by that cause's events
        sel = pe(y, H, null=null, causes=[2, 5], cause=c, alpha=0.1)
        sel_sum = pe(y, H, null=null, causes=[2, 5], cause=c, alpha=0.1, reduce="sum")
        assert sel.n_events == int(np.sum(labels == c))
        assert sel.total == pytest.approx(sel_sum.total / np.sum(labels == c), rel=1e-12)
    assert r.by_cause.sum() == pytest.approx(r.total, rel=1e-12)
    # a fitted cause absent from y is scored by its compensator only
    no5 = make_competing_risks_y(stop, np.where(labels == 5, 0, labels))
    with pytest.raises(UndefinedMetricError):
        pe(no5, H, null=null, causes=[2, 5], cause=5, alpha=0.1)
    both = pe(no5, H, null=null, causes=[2, 5], reduce="sum", alpha=0.1)
    rate = 0.9 * np.diff(H[:, 1], axis=1) + 0.1 * np.diff(null[1])  # windows have width 1
    e = np.c_[np.minimum(stop, 1.0), np.clip(stop - 1.0, 0, 1.0)]
    assert both.by_cause[1] == pytest.approx(-(rate * e).sum(), rel=1e-12)
    with pytest.raises(ValueError, match="not in causes"):
        pe(make_competing_risks_y(stop, np.where(labels == 5, 7, labels)), H, null=null, causes=[2, 5])
    with pytest.raises(ValueError, match="causes"):
        pe(y, H, null=null)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(w=np.array([0.0, 2.0, 1.0])), "strictly increasing"),
        (dict(w=np.array([0.5, 1.0, 2.0])), "starting at 0"),
        (dict(alpha=1.0), "alpha"),
        (dict(alpha=-0.1), "alpha"),
        (dict(reduce="mean"), "reduce"),
    ],
)
def test_errors(kw, match):
    y = make_survival_y(np.array([1.0, 1.5]), np.array([True, False]))
    with pytest.raises(ValueError, match=match):
        pe(y, _const(2), **kw)


def test_shape_and_nan_errors():
    y = make_survival_y(np.array([1.0, 1.5]), np.array([True, False]))
    with pytest.raises(ValueError, match="edge columns"):
        pe(y, _const(2)[:, :2])
    H = _const(2)
    H[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pe(y, H)
    with pytest.raises(ValueError, match="null_cumhaz"):
        pe(y, _const(2), null=np.zeros(4))
    with pytest.raises(ValueError, match="causes and cause"):
        pe(y, _const(2), cause=1)


def test_event_windows_inverse_cdf_and_ties():
    est = types.SimpleNamespace(event_times_=np.array([1.0, 2.0, 3.0, 4.0]), _event_counts_=np.array([1.0, 1, 1, 5]))
    np.testing.assert_array_equal(event_windows(est, 4), [0.0, 2.0, 4.0])  # ranks 2, 4, 6 -> t=2, 4, 4
    np.testing.assert_array_equal(event_windows(est, 1), [0.0, 4.0])
    with pytest.raises(AttributeError, match="refit"):
        event_windows(types.SimpleNamespace(event_times_=np.array([1.0])))
    with pytest.raises(ValueError, match="n_windows"):
        event_windows(est, 0)


def test_window_exposure_matches_hand_computation():
    start = np.array([0.0, 0.5, 1.5, 2.0])
    stop = np.array([1.0, 2.0, 3.0, 2.0])
    # window (0.5, 1.5]: row 0 contributes (1.0-0.5)=0.5; row 1 (1.5-0.5)=1.0;
    # row 2 has start=1.5 == hi, no exposure; row 3 (stop==start) is a zero-length row, no exposure.
    e = _window_exposure_1(start, stop, 0.5, 1.5)
    np.testing.assert_allclose(e, [0.5, 1.0, 0.0, 0.0])


def test_window_exposure_feeds_the_score_identically_to_the_pre_refactor_loop():
    # a direct per-window scalar computation, as the loop wrote inline before extraction
    rng = np.random.default_rng(0)
    start = rng.uniform(0, 2, size=50)
    stop = start + rng.uniform(0.1, 2, size=50)
    for lo, hi in [(0.0, 1.0), (1.0, 2.5), (2.5, 4.0)]:
        expected = np.clip(np.minimum(stop, hi) - np.maximum(start, lo), 0.0, None)
        np.testing.assert_allclose(_window_exposure_1(start, stop, lo, hi), expected)


def test_baseline_at_is_a_step_function():
    est = types.SimpleNamespace(event_times_=np.array([1.0, 2.0]), baseline_cumhaz_=np.array([0.1, 0.3]))
    np.testing.assert_allclose(_baseline_at(est, [0.0, 0.5, 1.0, 1.5, 2.0, 9.0]), [0, 0, 0.1, 0.1, 0.3, 0.3])
    cr = types.SimpleNamespace(event_times_=np.array([1.0, 2.0]), baseline_cumhaz_=np.array([[0.1, 0.3], [0.0, 0.2]]))
    np.testing.assert_allclose(_baseline_at(cr, [0.0, 1.5, 2.0]), [[0, 0.1, 0.3], [0, 0, 0.2]])


def test_on_a_fitted_forest():
    from tests.sim import rows, simulate

    rng = np.random.default_rng(0)
    X, y, ids = rows(*simulate(400, rng))
    Xt, yt, it = rows(*simulate(400, rng))
    m = SurvivalForestTV(n_estimators=100, random_state=0).fit(X, y, ids)
    w = event_windows(m)
    H = np.c_[np.zeros(len(Xt)), m.predict_cumulative_hazard(Xt, times=w[1:])]
    r = piecewise_exponential_score(yt, H, w, null_cumhaz=_baseline_at(m, w), ids=it)
    assert r.zero_rate_share == 0
    # strong covariate signal (S3): the forest beats the covariate-free null by a clear margin
    assert r.total > r.null_total + 0.2
