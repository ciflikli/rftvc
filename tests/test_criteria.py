"""S8 candidate split criteria: parity with naive references, edge cases and invariants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lifelines import KaplanMeierFitter

from rftvc import _core
from tests.ref.criteria_ref import grouped_lik_ref, km_gini_ref, km_ref, poisson_ref
from tests.ref.logrank_ref import logrank_ref


def _rows(n, seed, ties=True):
    """Delayed-entry counting-process rows; ties on a 0.5 grid unless ``ties=False``."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, 4, n) * 0.5 if ties else rng.uniform(0, 2, n)
    stop = start + (rng.integers(1, 8, n) * 0.5 if ties else rng.uniform(0.1, 4, n))
    event = rng.random(n) < 0.6
    event[0] = True
    left = rng.random(n) < 0.5
    left[:2] = [True, False]
    units = np.sort(rng.integers(0, max(n // 2, 1), n)).astype(np.uint32)
    return start, stop, event, left, units


def _score(name, start, stop, event, left, **kw):
    return _core.criterion_score(start, stop, event, left, name, **kw)


@settings(max_examples=80, deadline=None)
@given(n=st.integers(2, 60), seed=st.integers(0, 10_000), ties=st.booleans(),
       horizon=st.sampled_from([0.25, 1.0, 1.5, 2.5, 4.0, 99.0]))
def test_criteria_match_naive_references(n, seed, ties, horizon):
    start, stop, event, left, units = _rows(n, seed, ties)
    args = (start, stop, event, left)
    assert _score("logrank", *args) == pytest.approx(logrank_ref(*args), rel=1e-9, abs=1e-12)
    assert _score("grouped_lik", *args) == pytest.approx(grouped_lik_ref(*args), rel=1e-9, abs=1e-9)
    assert _score("poisson", *args) == pytest.approx(poisson_ref(*args), rel=1e-9, abs=1e-9)
    got = _score("km_gini", *args, horizon=horizon, units=units)
    assert got == pytest.approx(km_gini_ref(*args, horizon, units), rel=1e-9, abs=1e-12)


@settings(max_examples=80, deadline=None)
@given(n=st.integers(2, 60), seed=st.integers(0, 10_000), ties=st.booleans())
def test_likelihood_gains_are_nonnegative_and_side_symmetric(n, seed, ties):
    start, stop, event, left, units = _rows(n, seed, ties)
    for name, kw in [("grouped_lik", {}), ("poisson", {}), ("km_gini", {"horizon": 2.0, "units": units})]:
        a = _score(name, start, stop, event, left, **kw)
        b = _score(name, start, stop, event, ~left, **kw)
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12)
        if name != "km_gini":  # the Gini decrease can be negative (straddling units)
            assert a >= -1e-9


@pytest.mark.parametrize("name,kw", [("grouped_lik", {}), ("poisson", {}), ("km_gini", {"horizon": 2.0})])
def test_identical_children_give_zero_gain(name, kw):
    start, stop, event, _, _ = _rows(30, seed=4)
    two = lambda a: np.concatenate([a, a])  # noqa: E731
    left = np.repeat([True, False], 30)
    got = _score(name, two(start), two(stop), two(event), left, **kw)
    assert got == pytest.approx(0.0, abs=1e-10)


def test_all_event_and_tied_nodes_use_zero_log_zero():
    # Every row fails, all at the same two times: d == y at the last time.
    start, stop = np.zeros(6), np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    event, left = np.ones(6, bool), np.array([1, 0, 1, 0, 1, 0], bool)
    for name, ref in [("grouped_lik", grouped_lik_ref), ("poisson", poisson_ref)]:
        got = _score(name, start, stop, event, left)
        assert np.isfinite(got)
        assert got == pytest.approx(ref(start, stop, event, left), abs=1e-12)
    # An eventless left child.
    left = np.array([0, 0, 0, 0, 0, 1], bool)
    event = np.array([1, 1, 1, 1, 1, 0], bool)
    for name, ref in [("grouped_lik", grouped_lik_ref), ("poisson", poisson_ref)]:
        assert _score(name, start, stop, event, left) == pytest.approx(ref(start, stop, event, left), abs=1e-12)


def test_km_gini_horizon_edges():
    # Event times 1, 2 (tied: two events at 2), 3.
    start = np.array([0.0, 0.0, 0.5, 0.0, 1.5, 0.0])
    stop = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
    event = np.array([1, 1, 1, 1, 0, 0], bool)
    left = np.array([1, 1, 0, 0, 1, 0], bool)
    score = lambda h: _score("km_gini", start, stop, event, left, horizon=h)  # noqa: E731
    ref = lambda h: km_gini_ref(start, stop, event, left, h)  # noqa: E731
    assert score(0.5) == 0.0  # before the first event: S = 1 everywhere
    # At a tie the events at tau count (right-continuous).
    assert score(2.0) == pytest.approx(ref(2.0), abs=1e-12)
    assert score(2.0) != pytest.approx(score(2.0 - 1e-9), abs=1e-6)
    # After the last event the score no longer changes.
    assert score(3.0) == pytest.approx(ref(3.0), abs=1e-12)
    assert score(50.0) == score(3.0)


def test_km_gini_counts_straddling_units_in_both_children():
    start, stop = np.array([0.0, 1.0, 0.0, 0.0]), np.array([1.0, 2.0, 2.5, 3.0])
    event, left = np.array([0, 1, 1, 1], bool), np.array([1, 0, 1, 0], bool)
    units = np.array([0, 0, 1, 2], np.uint32)  # unit 0 has a row on each side
    got = _score("km_gini", start, stop, event, left, horizon=2.5, units=units)
    assert got == pytest.approx(km_gini_ref(start, stop, event, left, 2.5, units), abs=1e-12)
    assert got != pytest.approx(_score("km_gini", start, stop, event, left, horizon=2.5), abs=1e-6)


def test_km_reference_matches_lifelines_with_delayed_entry():
    start, stop, event, *_ = _rows(80, seed=9, ties=False)
    km = KaplanMeierFitter().fit(stop, event, entry=start)
    for h in [0.5, 1.5, 3.0]:
        want = float(km.survival_function_at_times(h).iloc[0])
        assert km_ref(start, stop, event, h) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("name,kw,match", [
    ("gini", {}, "split_criterion must be"),
    ("km_gini", {}, "requires criterion_horizon"),
    ("km_gini", {"horizon": 0.0}, "finite and > 0"),
    ("km_gini", {"horizon": float("nan")}, "finite and > 0"),
    ("logrank", {"horizon": 1.0}, "only used by"),
])
def test_criterion_score_validates_name_and_horizon(name, kw, match):
    start, stop, event, left, _ = _rows(5, seed=0)
    with pytest.raises(ValueError, match=match):
        _score(name, start, stop, event, left, **kw)


def test_criterion_score_logrank_equals_logrank_score():
    start, stop, event, left, _ = _rows(40, seed=3)
    assert _score("logrank", start, stop, event, left) == _core.logrank_score(start, stop, event, left)
