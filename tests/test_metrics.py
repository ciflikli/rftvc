"""S5: survival metrics against scikit-survival, R and naive references."""

import json
from pathlib import Path

import numpy as np
import pytest
from lifelines import KaplanMeierFitter
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from rftvc import make_survival_y
from rftvc.metrics import (
    KaplanMeierCensoring,
    brier_landmark,
    calibration_table,
    cindex_dynamic,
    concordance_index_cp,
    integrated_brier,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "concordance_cp.json").read_text())


def concordance_ref(start, stop, event, risk, ids):
    """Naive O(n^2) counting-process concordance (independent of metrics.py)."""
    num = den = 0.0
    for i in np.flatnonzero(event):
        t = stop[i]
        for j in range(stop.size):
            if j == i or ids[j] == ids[i] or not (start[j] < t <= stop[j]):
                continue
            if event[j] and stop[j] == t:
                continue
            den += 1
            num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den


def _sk(y):
    return Surv.from_arrays(y["event"], y["stop"])


def _parity_data(seed=0, n_train=300, n_test=200):
    """Continuous times, mild censoring: G stays well above g_min up to w."""
    rng = np.random.default_rng(seed)

    def draw(n):
        x = rng.normal(size=n)
        t = rng.exponential(np.exp(-0.7 * x))
        c = rng.exponential(4.0, size=n)
        return x, make_survival_y(np.minimum(t, c), t <= c)

    _, y_train = draw(n_train)
    x, y_test = draw(n_test)
    keep = y_test["stop"] < y_train["stop"].max()  # sksurv needs test times within training follow-up
    return x[keep], y_train, y_test[keep]


# --- concordance -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_concordance_cp_matches_r_and_naive_reference(name):
    f = FIXTURES[name]
    start, stop = np.array(f["start"]), np.array(f["stop"])
    event, risk, ids = np.array(f["event"], bool), np.array(f["risk"]), np.array(f["ids"])
    y = make_survival_y(stop, event, start=start)
    c = concordance_index_cp(y, risk, ids=ids)
    assert c == pytest.approx(f["concordance"], abs=1e-12)
    assert c == pytest.approx(concordance_ref(start, stop, event, risk, ids), abs=1e-12)


def test_concordance_cp_is_harrell_for_right_censored_data():
    rng = np.random.default_rng(3)
    n = 120
    stop = np.round(rng.exponential(size=n), 1) + 0.1  # tied times
    event = rng.random(n) < 0.6
    risk = np.round(rng.normal(size=n), 1)  # tied risks
    expected = concordance_index_censored(event, stop, risk, tied_tol=1e-12)[0]
    assert concordance_index_cp(make_survival_y(stop, event), risk) == pytest.approx(expected, abs=1e-12)


def test_concordance_cp_excludes_same_id_rows_in_stacked_data():
    # Two overlapping rows of id 0 (stacked landmarks): never compared with each other.
    y = make_survival_y([2.0, 3.0, 4.0], [True, False, False])
    risk = [1.0, 5.0, 0.0]
    assert concordance_index_cp(y, risk, ids=[0, 0, 1]) == 1.0
    assert concordance_index_cp(y, risk) == 0.5


# --- censoring model -------------------------------------------------------------


def test_censoring_km_matches_sksurv_and_left_limits():
    rng = np.random.default_rng(1)
    stop = np.round(rng.exponential(size=80), 1) + 0.1
    event = rng.random(80) < 0.5
    y = make_survival_y(stop, event)
    km = KaplanMeierCensoring().fit(y)
    ref = CensoringDistributionEstimator().fit(_sk(y))
    grid = np.unique(stop)[:-1]
    np.testing.assert_allclose(km.predict(grid), ref.predict_proba(grid), atol=1e-12)
    # Left limit at t equals the value just before t.
    np.testing.assert_allclose(km.predict(grid[1:], left=True), km.predict(grid[:-1]), atol=1e-12)
    assert km.predict([0.0], left=True)[0] == 1.0


# --- Brier -------------------------------------------------------------------------


def test_brier_matches_sksurv():
    x, y_train, y_test = _parity_data()
    w = 0.8
    assert not np.any(np.isin(np.r_[y_test["stop"], w], y_train["stop"]))
    surv = np.exp(-0.9 * w * np.exp(0.7 * x))
    _, expected = brier_score(_sk(y_train), _sk(y_test), surv, w)
    score, info = brier_landmark(y_test, 1 - surv, w, y_censor=y_train, return_info=True)
    assert KaplanMeierCensoring().fit(y_train).predict([w])[0] > 0.5  # far from g_min
    assert info["n_clipped"] == 0 and not info["exact"]
    assert score == pytest.approx(expected[0], abs=1e-10)


def test_integrated_brier_matches_sksurv():
    x, y_train, y_test = _parity_data(seed=2)
    times = np.linspace(0.1, 1.0, 10)
    surv = np.exp(-0.9 * times[None, :] * np.exp(0.7 * x)[:, None])
    expected = integrated_brier_score(_sk(y_train), _sk(y_test), surv, times)
    assert integrated_brier(y_test, surv, times, y_censor=y_train) == pytest.approx(expected, abs=1e-10)


def test_exact_path_under_complete_follow_up_uses_no_censoring_model():
    class Exploding:
        def predict(self, times, left=False):
            raise AssertionError("censoring model consulted on the exact path")

    # Every subject has an event by w or is followed through w (stop == w: administrative).
    y = make_survival_y([0.5, 1.0, 2.0, 2.0, 1.5], [True, True, False, False, True])
    risk = np.array([0.9, 0.6, 0.2, 0.1, 0.3])
    case = np.array([1, 1, 0, 0, 1])
    expected = np.mean((case - risk) ** 2)
    score, info = brier_landmark(y, risk, 2.0, censoring_estimator=Exploding(), return_info=True)
    assert info["exact"] and score == pytest.approx(expected, abs=1e-15)
    assert brier_landmark(y, risk, 2.0) == pytest.approx(expected, abs=1e-15)


def test_administrative_censoring_at_w_is_a_control_weighted_by_left_limit():
    # Subject 3 is censored before w; subjects 1-2 are administratively censored at w.
    y = make_survival_y([2.0, 2.0, 0.5, 1.0], [False, False, True, False])
    risk = np.array([0.1, 0.2, 0.7, 0.4])
    km = KaplanMeierCensoring().fit(y)
    g_w = km.predict([2.0], left=True)[0]  # G(2-) = 2/3, not G(2) = 0
    g_case = km.predict([0.5], left=True)[0]
    expected = (0.1**2 / g_w + 0.2**2 / g_w + (1 - 0.7) ** 2 / g_case) / 4
    assert g_w == pytest.approx(2 / 3)
    assert brier_landmark(y, risk, 2.0, y_censor=y) == pytest.approx(expected, abs=1e-15)


def test_truncation_diagnostic_counts_clipped_weights():
    x, y_train, y_test = _parity_data(seed=4)
    w, g_min = 2.0, 0.8
    km = KaplanMeierCensoring().fit(y_train)
    stop, event = y_test["stop"], y_test["event"]
    case = event & (stop <= w)
    control = (stop >= w) & ~case
    g = np.where(case, km.predict(stop, left=True), km.predict([w], left=True)[0])
    expected = int(((case | control) & (g < g_min)).sum())
    assert expected > 0
    _, info = brier_landmark(y_test, np.full(stop.size, 0.3), w, y_censor=y_train, g_min=g_min, return_info=True)
    assert info["n_clipped"] == expected


def test_censoring_arguments_are_exclusive():
    _, y_train, y_test = _parity_data()
    risk = np.full(y_test.size, 0.5)
    with pytest.raises(ValueError, match="exactly one"):
        brier_landmark(y_test, risk, 0.8)
    with pytest.raises(ValueError, match="exactly one"):
        brier_landmark(y_test, risk, 0.8, y_censor=y_train, censoring_estimator=KaplanMeierCensoring().fit(y_train))
    with pytest.raises(ValueError, match="start == 0"):
        brier_landmark(make_survival_y([2.0], [True], start=[1.0]), [0.5], 1.0)


# --- time-dependent discrimination --------------------------------------------


def test_cumulative_auc_matches_sksurv():
    x, y_train, y_test = _parity_data(seed=5)
    w = 0.8
    risk = x + 0.3 * np.random.default_rng(0).normal(size=x.size)
    expected, _ = cumulative_dynamic_auc(_sk(y_train), _sk(y_test), risk, [w])
    assert cindex_dynamic(y_test, risk, w, y_censor=y_train) == pytest.approx(expected[0], abs=1e-10)


def test_incident_cindex_matches_uno_truncated_c():
    x, y_train, y_test = _parity_data(seed=6)
    w = 1.2
    risk = x + 0.3 * np.random.default_rng(1).normal(size=x.size)
    expected = concordance_index_ipcw(_sk(y_train), _sk(y_test), risk, tau=w)[0]
    got = cindex_dynamic(y_test, risk, w, kind="incident", y_censor=y_train)
    assert got == pytest.approx(expected, abs=1e-10)


# --- calibration -------------------------------------------------------------------


def test_calibration_table_matches_per_bin_kaplan_meier():
    x, _, y_test = _parity_data(seed=7)
    w = 1.0
    risk = 1 - np.exp(-w * np.exp(0.7 * x))
    table = calibration_table(y_test, risk, w, n_bins=4)
    assert table["n"].sum() == y_test.size and table.height == 4
    edges = np.quantile(risk, [0, 0.25, 0.5, 0.75, 1])
    for row in table.iter_rows(named=True):
        lo, hi = edges[row["bin"]], edges[row["bin"] + 1]
        m = (risk >= lo) & ((risk < hi) if row["bin"] < 3 else (risk <= hi))
        km = KaplanMeierFitter().fit(y_test["stop"][m], y_test["event"][m])
        assert row["n"] == m.sum()
        assert row["mean_risk"] == pytest.approx(risk[m].mean())
        assert row["observed_risk"] == pytest.approx(1 - km.survival_function_at_times(w).iloc[0], abs=1e-12)
