"""S17: ``inspection.permutation_importance`` for the counting-process forests."""

import numpy as np
import pandas as pd
import pytest

from bench.perf_fit import synth
from rftvc import (
    CompetingRisksForestTV,
    LandmarkSurvivalForest,
    SurvivalForestTV,
    inspection,
    make_competing_risks_y,
    make_survival_y,
)
from rftvc._inspection import _boot, _score, _strata
from rftvc.metrics import _baseline_at, event_windows, piecewise_exponential_score
from tests.sim import rows, simulate

# Small test forests and stubs trigger these diagnostics; tests that need them use pytest.warns.
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*windows are too fine:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*rows have no tree:UserWarning"),
]


def _data(n, seed):
    """S3 rows ``[x0, z, noise]`` with ids."""
    rng = np.random.default_rng(seed)
    x0, z, U, ev = simulate(n, rng)
    X, y, ids = rows(x0, z, U, ev)
    return np.c_[X, rng.normal(size=len(X))], y, ids


X, Y, IDS = _data(300, 0)
XT, YT, IDT = _data(200, 1)
FOREST = SurvivalForestTV(n_estimators=30, random_state=0).fit(X, Y, IDS)


class _Stub:
    """A forest whose cumulative hazard is ``scale * exp(x0) * t`` (column 0 only);
    cause 1 follows it, cause 2 has the constant rate 0.1."""

    def __init__(self, scale=0.15):
        self.scale = scale

    def predict_cumhaz(self, X, times, aggregate, n_jobs):
        return self.scale * np.exp(X[:, [0]]) * times[None, :]

    def predict_cause_cumhaz(self, X, times, n_jobs):
        return np.stack([self.predict_cumhaz(X, times, None, n_jobs), np.broadcast_to(0.1 * times, (len(X), times.size))], 1)


def _stubbed(model, stub=None):
    import copy

    m = copy.copy(model)
    m.forest_ = stub or _Stub()
    return m


def _cr_data():
    y = make_competing_risks_y(Y["stop"], np.where(Y["event"], 1 + (IDS % 2), 0), start=Y["start"])
    yt = make_competing_risks_y(YT["stop"], np.where(YT["event"], 1 + (IDT % 2), 0), start=YT["start"])
    return y, yt


Y_CR, YT_CR = _cr_data()
CR_FOREST = CompetingRisksForestTV(n_estimators=30, random_state=0).fit(X, Y_CR, IDS)


def _pi(model, Xe=XT, ye=YT, **kw):
    kw.setdefault("ids", IDT)
    kw.setdefault("n_bootstrap", 0)
    kw.setdefault("random_state", 0)
    return inspection.permutation_importance(model, Xe, ye, **kw)


# --- values ---------------------------------------------------------------------------


def test_ignored_column_is_exactly_zero_and_the_used_column_positive():
    r = _pi(_stubbed(FOREST))
    assert r.importances_mean[0] > 0.05
    assert np.array_equal(r.importances[1:], np.zeros((2, 5)))


def test_cr_stub_splits_by_cause():
    r = _pi(_stubbed(CR_FOREST), ye=YT_CR)
    assert r.importances_cause[0, 0] > 0.01
    assert r.importances_cause[0, 1] == 0.0  # cause 2 does not depend on x0
    assert np.array_equal(r.importances_cause[1:], np.zeros((2, 2)))


def _literal_setup():
    Xe = np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0]])
    ye = make_survival_y(np.array([1.0, 1, 2, 2]), np.array([True, False, True, False]), start=np.array([0.0, 0, 1, 1]))
    return Xe, ye


def test_literal_known_hazard_case():
    """Rows (start, stop, event, x0): (0,1,1,0) (0,1,0,1) (1,2,1,0) (1,2,0,2); rate exp(x0).

    Intact S = -1 - e - 1 - e^2. Swapping stratum {0,1} gives (1 - e) - 1, a
    drop of -1; swapping {2,3} gives (2 - e^2) - 1, a drop of -2; N = 2.
    """
    Xe, ye = _literal_setup()
    for seed in range(8):
        r = _pi(_stubbed(FOREST, _Stub(1.0)), Xe, ye, ids=None, windows=[0.0, 1.0, 2.0], alpha=0.0, n_repeats=1,
                features=[0], random_state=seed)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(0,)).spawn(1)[0]))
        src = _strata.donors(np.array([0, 0, 1, 1]), rng)
        s1, s2 = src[0] == 1, src[2] == 3
        np.testing.assert_allclose(r.importances_mean, [(-1.0 * s1 - 2.0 * s2) / 2], atol=1e-12)
        np.testing.assert_allclose(r.importances_window[0], [-0.5 * s1, -1.0 * s2], atol=1e-12)
        np.testing.assert_allclose(r.baseline_score, (-2 - np.e - np.e**2) / 2, atol=1e-12)


def _permuted_X(Xe, start, cols, seed, first_col, n_strata=10):
    """The evaluation X after the first repeat's permutation of unit ``cols``."""
    stream = np.random.SeedSequence(seed, spawn_key=(first_col,))
    rng = np.random.Generator(np.random.PCG64(stream.spawn(1)[0]))
    lab = _strata.bin_codes(start, n_strata)
    src = _strata.donors(lab, rng)
    Xp = Xe.copy()
    Xp[:, cols] = Xe[src][:, cols]
    return Xp


def test_one_permutation_equals_direct_scoring():
    w = event_windows(FOREST, 8)
    null = _baseline_at(FOREST, w)
    r = _pi(FOREST, n_repeats=1, features=[1], random_state=3)
    Xp = _permuted_X(XT, YT["start"], [1], 3, 1)
    s0 = piecewise_exponential_score(YT, FOREST.predict_cumulative_hazard(XT, times=w), w, null_cumhaz=null, reduce="sum")
    s1 = piecewise_exponential_score(YT, FOREST.predict_cumulative_hazard(Xp, times=w), w, null_cumhaz=null, reduce="sum")
    np.testing.assert_allclose(r.importances[0, 0], (s0.total - s1.total) / s0.n_events, rtol=1e-12)
    np.testing.assert_allclose(r.importances_window[0], (s0.by_window - s1.by_window) / s0.n_events, atol=1e-12)


def test_cr_single_cause_equals_direct_scoring():
    w = event_windows(CR_FOREST, 4)
    null = _baseline_at(CR_FOREST, w)
    r = _pi(CR_FOREST, ye=YT_CR, n_repeats=1, features=[1], cause=2, windows=4, random_state=4)
    Xp = _permuted_X(XT, YT_CR["start"], [1], 4, 1)
    kw = dict(null_cumhaz=null, causes=CR_FOREST.causes_, cause=2, reduce="sum")
    H0 = CR_FOREST.predict_cumulative_hazard(XT, times=w)
    H1 = CR_FOREST.predict_cumulative_hazard(Xp, times=w)
    s0, s1 = piecewise_exponential_score(YT_CR, H0, w, **kw), piecewise_exponential_score(YT_CR, H1, w, **kw)
    np.testing.assert_allclose(r.importances[0, 0], (s0.total - s1.total) / s0.n_events, rtol=1e-12)
    assert r.importances_cause is None and r.n_events == s0.n_events


def test_decompositions_sum_to_the_mean():
    r = _pi(FOREST)
    np.testing.assert_allclose(r.importances_window.sum(1), r.importances_mean, atol=1e-12)
    np.testing.assert_allclose(r.importances_id.sum(1), r.importances_mean, atol=1e-12)
    np.testing.assert_allclose(r.importances.mean(1), r.importances_mean, atol=1e-15)
    np.testing.assert_allclose(r.importances_std, r.importances.std(1))
    assert r.importances_id.shape == (3, np.unique(IDT).size) and r.window_edges.size == 9
    rc = _pi(CR_FOREST, ye=YT_CR)
    np.testing.assert_allclose(rc.importances_cause.sum(1), rc.importances_mean, atol=1e-12)
    np.testing.assert_allclose(rc.importances_window.sum(1), rc.importances_mean, atol=1e-12)


def test_share_of_gain():
    r = _pi(FOREST)
    np.testing.assert_allclose(r.share_of_gain, r.importances_mean / (r.baseline_score - r.null_score))


class _NullStub:
    def __init__(self, model):
        self.model = model

    def predict_cumhaz(self, X, times, aggregate, n_jobs):
        return np.broadcast_to(_baseline_at(self.model, times), (len(X), times.size)).copy()


def test_share_of_gain_is_nan_when_the_model_does_not_beat_the_null():
    with pytest.warns(UserWarning, match="does not beat the training null"):
        r = _pi(_stubbed(FOREST, _NullStub(FOREST)))
    assert np.isnan(r.share_of_gain).all()


# --- strata, groups, conditioning -----------------------------------------------------


def test_donors_stay_in_their_stratum():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 5, 200)
    src = _strata.donors(lab, rng)
    assert np.array_equal(lab[src], lab)
    assert np.array_equal(np.sort(src), np.arange(200))  # a permutation
    assert (src != np.arange(200)).any()
    src1 = _strata.donors(np.zeros(200, dtype=int), rng)  # one stratum: a global shuffle
    assert np.array_equal(np.sort(src1), np.arange(200)) and (src1 != np.arange(200)).mean() > 0.9


def _recording_eval(Xe, ye):
    calls = []

    def predict(Xr, rows):
        calls.append((Xr.copy(), rows.copy()))
        return FOREST.forest_.predict_cumhaz(np.ascontiguousarray(Xr), W, "hazard", 1)

    ev = _score.Evaluation(Xe, ye, ye["start"], IDT, W, _baseline_at(FOREST, W), 0.01, None, None, predict)
    return ev, calls


W = event_windows(FOREST, 8)


def test_permuted_rows_stay_in_their_time_stratum_and_groups_move_jointly():
    Xe = XT.copy()
    Xe[:, 2] = Xe[:, 1] * 3 + 1  # column 2 is a function of column 1: the pair must survive
    ev, calls = _recording_eval(Xe, YT)
    st = _score.Strata("time", None, 4, [], 4)
    lab = _score.labels(Xe, YT["start"], st, np.array([1, 2]))
    H = FOREST.forest_.predict_cumhaz(Xe, W, "hazard", 1)
    _score._permuted(ev, H, Xe, np.array([1, 2]), lab, np.random.default_rng(0), np.arange(len(Xe)))
    Xc, rows_ = calls[-1]
    np.testing.assert_array_equal(Xc[:, 2], Xc[:, 1] * 3 + 1)  # joint
    np.testing.assert_array_equal(Xc[:, 0], Xe[rows_, 0])  # untouched column
    for s in np.unique(lab):  # each received value comes from the same stratum
        inside = lab[rows_] == s
        assert np.isin(Xc[inside, 1], Xe[lab == s, 1]).all()


def test_binning_rule():
    np.testing.assert_array_equal(_strata.bin_codes([1, 2, 3, 4], 2), [0, 0, 1, 1])  # q = [2.5]
    np.testing.assert_array_equal(_strata.bin_codes([1, 2, 2, 3, 9], 2), [0, 0, 0, 1, 1])  # q = [2]: 2 goes low
    skewed = np.r_[np.zeros(90), np.ones(10)]
    np.testing.assert_array_equal(_strata.bin_codes(skewed, 4), skewed)  # by value, not collapsed
    v = np.random.default_rng(0).normal(size=1000)
    assert np.bincount(_strata.bin_codes(v, 4)).tolist() == [250, 250, 250, 250]


def test_user_strata_labels():
    np.testing.assert_array_equal(_strata.user_labels(np.array(["b", "a", "b"]), 3), [1, 0, 1])
    with pytest.raises(ValueError, match="NaN"):
        _strata.user_labels(np.array([0.0, np.nan]), 2)
    with pytest.raises(ValueError, match="3 entries"):
        _strata.user_labels(np.array([0, 1]), 3)
    with pytest.raises(ValueError, match="sortable"):
        _strata.user_labels(np.array([1, "a", 2], dtype=object), 3)


def test_strata_module_does_not_require_pandas():
    """pandas is a dev-only dependency (pyproject.toml's dependency-groups.dev), not a
    runtime one: _strata.py must not import it unconditionally (regression: it briefly
    did, which would break `import rftvc` for a normal, non-dev install)."""
    import subprocess
    import sys

    script = (
        "import sys, builtins\n"
        "real_import = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name == 'pandas' or name.startswith('pandas.'):\n"
        "        raise ModuleNotFoundError(name)\n"
        "    return real_import(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "import rftvc.inspection\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_user_strata_labels_reject_nan_in_an_object_array():
    """NaN detection must not be skipped for object-dtype strata (regression: the check
    used to look at ``s.dtype.kind == "f"`` only, so a float NaN or a None boxed in an
    object array silently got its own distinct stratum instead of raising)."""
    with pytest.raises(ValueError, match="NaN"):
        _strata.user_labels(np.array([1, "a", None], dtype=object), 3)
    with pytest.raises(ValueError, match="NaN"):
        _strata.user_labels(np.array([1.0, 2.0, float("nan")], dtype=object), 3)


def test_user_strata_labels_reject_nat_datetime64():
    """A datetime64/timedelta64 array's own NaT (regression: an earlier fix's dtype-kind
    dispatch only special-cased "f"/"c"/"O", leaving datetime64's "M" kind unhandled, so a
    NaT silently got its own stratum code instead of raising)."""
    d = np.array(["2020-01-01", "NaT"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="NaN"):
        _strata.user_labels(d, 2)


def test_user_strata_labels_reject_boxed_nat_and_decimal_nan():
    """NaT or Decimal("NaN") boxed in an object array (regression: an earlier fix's
    object-array branch only special-cased None and float/np.floating NaN)."""
    import decimal

    with pytest.raises(ValueError, match="NaN"):
        _strata.user_labels(np.array([np.datetime64("2020-01-01"), np.datetime64("NaT")], dtype=object), 2)
    with pytest.raises(ValueError, match="NaN"):
        _strata.user_labels(np.array([decimal.Decimal("1"), decimal.Decimal("NaN")], dtype=object), 2)


def test_user_strata_labels_reject_pandas_na():
    """pandas' pd.NA (a nullable-dtype missing marker) boxed in an object array (regression:
    ``x != x`` is itself an ``NA``, not a bool, for ``pd.NA``, so ``bool(x != x)`` raises
    TypeError there by design rather than returning True/False; must be treated as missing,
    not left to propagate as an opaque TypeError)."""
    _strata.user_labels(np.array([1, 2], dtype=object), 2)  # sanity: no false positive
    with pytest.raises(ValueError, match="NaN"):
        _strata.user_labels(np.array([1, pd.NA], dtype=object), 2)


def test_user_strata_restrict_the_permutation():
    labels = (XT[:, 0] > 0).astype(int)
    ev, calls = _recording_eval(XT, YT)
    st = _score.Strata("user", labels, 10, [], 4)
    lab = _score.labels(XT, YT["start"], st, np.array([1]))
    H = FOREST.forest_.predict_cumhaz(XT, W, "hazard", 1)
    _score._permuted(ev, H, XT, np.array([1]), lab, np.random.default_rng(1), np.arange(len(XT)))
    Xc, rows_ = calls[-1]
    for s in (0, 1):
        assert np.isin(Xc[labels[rows_] == s, 1], XT[labels == s, 1]).all()
    r = _pi(FOREST, strata=labels, features=[1])
    assert np.isfinite(r.importances_mean).all()


def test_singleton_strata_are_counted_and_warn():
    with pytest.warns(UserWarning, match="alone in their stratum"):
        r = _pi(FOREST, features=[1], conditional_on=[0, 2], n_bins=30, n_strata=10)
    Xe = XT
    lab = _strata.combine(_strata.bin_codes(YT["start"], 10), _strata.bin_codes(Xe[:, 0], 30), _strata.bin_codes(Xe[:, 2], 30))
    assert r.n_unpermuted[0] == _strata.n_singletons(lab) > 0.1 * len(Xe)


def _discrete_setup():
    rng = np.random.default_rng(5)
    Xd = XT.copy()
    Xd[:, 0] = rng.integers(0, 3, len(Xd))
    Xd[:, 2] = Xd[:, 0]  # an exact copy
    return Xd


def test_conditioning_on_a_copy_gives_exactly_zero():
    Xd = _discrete_setup()
    m = _stubbed(FOREST)
    assert _pi(m, Xd, features=[0]).importances_mean[0] > 0
    r = _pi(m, Xd, features=[0], conditional_on=[2])
    assert r.importances_mean[0] == 0.0


def test_own_columns_are_dropped_from_the_conditioning_set():
    a = _pi(FOREST, features=[0], conditional_on=[0], random_state=2)
    b = _pi(FOREST, features=[0], random_state=2)
    np.testing.assert_array_equal(a.importances, b.importances)


def test_naive_shuffle_warns_and_is_documented():
    with pytest.warns(UserWarning, match="extrapolate"):
        _pi(FOREST, strata=None, features=[1])
    assert "extrapolat" in inspection.permutation_importance.__doc__


# --- reproducibility ------------------------------------------------------------------


def test_reproducible_across_n_jobs_and_unit_order():
    a = _pi(FOREST, n_jobs=1, random_state=7)
    b = _pi(FOREST, n_jobs=4, random_state=7)
    for k in ("importances", "importances_window", "importances_id"):
        np.testing.assert_array_equal(a[k], b[k])
    c = _pi(FOREST, features=[2, 0], random_state=7)
    d = _pi(FOREST, features=[0], random_state=7)
    np.testing.assert_array_equal(c.importances, a.importances[[2, 0]])
    np.testing.assert_array_equal(d.importances[0], a.importances[0])
    assert not np.array_equal(_pi(FOREST, random_state=8).importances, a.importances)


# --- groups and names -----------------------------------------------------------------


def test_groups_and_feature_names():
    r = _pi(FOREST, groups={"x": [0], "rest": [1, 2]})
    assert r.feature_names.tolist() == ["x", "rest"]
    assert [u.tolist() for u in r.units] == [[0], [1, 2]]
    single = _pi(FOREST, features=[0])
    np.testing.assert_array_equal(r.importances[0], single.importances[0])  # same stream key, same unit


def test_dataframe_input_matches_arrays():
    cols = ["x0", "z", "noise"]
    df = pd.DataFrame(X, columns=cols).assign(id=IDS)
    m = SurvivalForestTV(n_estimators=30, random_state=0).fit(df, Y, ids="id")
    dft = pd.DataFrame(XT, columns=cols).assign(id=IDT)
    a = inspection.permutation_importance(m, dft, YT, ids="id", features=["z", "x0"], n_bootstrap=0, random_state=0)
    b = inspection.permutation_importance(m, dft, YT, features=["z", "x0"], n_bootstrap=0, random_state=0)
    c = _pi(FOREST, features=[1, 0])
    np.testing.assert_array_equal(a.importances, c.importances)
    np.testing.assert_array_equal(b.importances, c.importances)  # the fit-time id column is found in X
    assert a.feature_names.tolist() == ["z", "x0"]
    with pytest.raises(ValueError, match="ids column"):
        inspection.permutation_importance(m, dft, YT, ids="id", features=["id"], n_bootstrap=0)
    with pytest.raises(ValueError, match="unknown feature 'bogus'"):
        inspection.permutation_importance(m, dft, YT, ids="id", features=["bogus"], n_bootstrap=0)


# --- OOB ------------------------------------------------------------------------------


def test_subset_csr_composes():
    """OOB re-prediction subsets the CSR set twice (once to the OOB rows, again to the
    permuted rows within them); the composition must equal subsetting by the combined index directly."""
    offsets = np.array([0, 2, 2, 5, 7, 9])
    units = np.array([10, 11, 20, 21, 22, 30, 31, 40, 41])
    off1, uni1 = inspection._subset_csr(offsets, units, np.array([0, 2, 3, 4]))
    off2, uni2 = inspection._subset_csr(off1, uni1, np.array([1, 3]))  # positions within the first subset
    off_direct, uni_direct = inspection._subset_csr(offsets, units, np.array([2, 4]))  # the same rows directly
    np.testing.assert_array_equal(off2, off_direct)
    np.testing.assert_array_equal(uni2, uni_direct)


XS, YS, IDSS = synth(1200, p=4, rows_per_id=4, seed=1)
OOB_MODES = {
    "id": {},
    "coarse": {"ntime": 15},
    "block": {"resample_unit": "block", "block_length": 2.0, "oob_buffer": 1},
}


@pytest.mark.parametrize("mode", OOB_MODES)
def test_oob_baseline_equals_direct_oob_scoring(mode):
    m = SurvivalForestTV(n_estimators=20, oob_score=True, random_state=0, **OOB_MODES[mode]).fit(XS, YS, IDSS)
    r = inspection.permutation_importance(m, XS, YS, ids=IDSS, oob=True, n_repeats=2, random_state=0)
    d = m._rebuild_design(XS, YS, IDSS)
    w = event_windows(m, 8)
    H, n = m.forest_.oob_cumhaz(d.X, *d.oob_set, w, m.aggregate, 1)
    ok = n > 0
    y_d = make_survival_y(d.stop, d.event, start=d.start)[ok]
    s = piecewise_exponential_score(y_d, H[ok], w, null_cumhaz=_baseline_at(m, w))
    np.testing.assert_allclose(r.baseline_score, s.total, rtol=1e-12)
    # independent reference: these rows' OOB mortality is the fit's oob_prediction_
    Hk, _ = m.forest_.oob_cumhaz(d.X, *d.oob_set, m.event_times_, m.aggregate, 1)
    ref = m.oob_prediction_ if d.kept is None else m.oob_prediction_[d.kept]
    np.testing.assert_array_equal(np.cumsum(Hk, 1)[:, -1], ref)
    dropped = 0 if d.kept is None else len(XS) - d.kept.size
    assert r.n_rows_excluded == dropped + int((~ok).sum())
    assert np.isnan(r.importances_se).all()
    assert r.importances_mean[0] > 0  # synth's first column carries signal


def test_oob_counts_rows_without_an_oob_tree():
    m = SurvivalForestTV(n_estimators=3, max_samples=0.9, random_state=0).fit(XS, YS, IDSS)
    r = inspection.permutation_importance(m, XS, YS, ids=IDSS, oob=True, n_repeats=1, random_state=0)
    d = m._rebuild_design(XS, YS, IDSS)
    _, n = m.forest_.oob_cumhaz(d.X, *d.oob_set, m.event_times_[:1], m.aggregate, 1)
    assert r.n_rows_excluded == int((n == 0).sum()) > 0


def test_oob_checks_the_training_data():
    m = SurvivalForestTV(n_estimators=10, random_state=0).fit(XS, YS, IDSS)
    X2 = XS.copy()
    X2[0, 0] += 1.0
    with pytest.raises(ValueError, match="do not match"):
        inspection.permutation_importance(m, X2, YS, ids=IDSS, oob=True)
    ma = XS[:, 0] * 0 + YS["start"]
    m2 = SurvivalForestTV(n_estimators=10, random_state=0).fit(XS, YS, IDSS, measured_at=ma)
    with pytest.raises(ValueError, match="do not match"):
        inspection.permutation_importance(m2, XS, YS, ids=IDSS, oob=True)
    r = inspection.permutation_importance(m2, XS, YS, ids=IDSS, oob=True, measured_at=ma, n_repeats=1)
    assert np.isfinite(r.importances_mean).all()
    with pytest.raises(ValueError, match="only used with oob=True"):
        inspection.permutation_importance(m2, XS, YS, ids=IDSS, measured_at=ma)


def test_oob_user_strata_follow_the_input_rows():
    m = SurvivalForestTV(n_estimators=10, ntime=15, random_state=0).fit(XS, YS, IDSS)
    labels = (XS[:, 1] > 0).astype(int)
    r = inspection.permutation_importance(m, XS, YS, ids=IDSS, oob=True, strata=labels, n_repeats=1)
    assert np.isfinite(r.importances_mean).all()
    with pytest.raises(ValueError, match="entries"):
        inspection.permutation_importance(m, XS, YS, ids=IDSS, oob=True, strata=labels[:-1])


# --- bootstrap ------------------------------------------------------------------------


def test_resample_ids_keeps_whole_ids_in_order():
    ids = np.array([5, 5, 7, 9, 9, 9, 7])
    for seed in range(20):
        ri, bid = _boot._resample_ids(ids, np.random.default_rng(seed))
        assert np.array_equal(np.unique(bid), np.arange(3))  # one label per drawn copy
        for c in range(3):
            rows_ = ri[bid == c]
            orig = np.flatnonzero(ids == ids[rows_[0]])
            np.testing.assert_array_equal(rows_, orig)  # all rows of the id, in order
        drawn = ids[ri][np.r_[True, np.diff(bid) != 0]]  # the id of each copy
        assert ri.size == sum(int((ids == i).sum()) for i in drawn)


def test_bootstrap_copies_are_distinct_rows_in_strata():
    ids = np.array([0, 1])
    ri, bid = _boot._resample_ids(ids, np.random.default_rng(0))
    while np.unique(ids[ri]).size != 1:  # find a draw with one id twice
        ri, bid = _boot._resample_ids(ids, np.random.default_rng(int(ri.sum() + bid.sum() + 1)))
    assert bid.tolist() == [0, 1]  # two copies, two subjects
    lab = _strata.bin_codes(np.zeros(2), 10)
    assert _strata.n_singletons(lab) == 0  # the copies form a 2-row stratum


def test_bootstrap_se():
    r = _pi(FOREST, features=[1], n_bootstrap=5, n_repeats=2)
    assert np.isfinite(r.importances_se).all() and r.importances_se[0] > 0
    for b in (0, 1):
        assert np.isnan(_pi(FOREST, features=[1], n_bootstrap=b).importances_se).all()


def test_noise_column_interval_covers_zero():
    covered = 0
    for seed in range(20):
        Xe, ye, ide = _data(100, 100 + seed)
        r = _pi(FOREST, Xe, ye, ids=ide, features=[2], n_bootstrap=20, n_repeats=2, random_state=seed)
        covered += abs(r.importances_mean[0]) <= 1.96 * r.importances_se[0]
    assert covered >= 18


# --- errors ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"features": ["x0"]}, "not fitted on named columns"),
        ({"features": [5]}, "out of range"),
        ({"features": [0, 0]}, "twice"),
        ({"groups": {"a": [0, 1], "b": [1]}}, "is in groups"),
        ({"groups": {"a": []}}, "empty"),
        ({"features": [0], "groups": {"a": [1]}}, "not both"),
        ({"strata": np.zeros(3)}, "entries"),
        ({"strata": "calendar"}, "strata must be"),
        ({"scoring": "brier"}, "scoring"),
        ({"windows": [0.0, 1.0, 1e6]}, "beyond the last training event"),
        ({"windows": [1.0, 2.0]}, "starting at 0"),
        ({"n_repeats": 0}, "n_repeats"),
        ({"n_bootstrap": -1}, "n_bootstrap"),
        ({"n_strata": 0}, "n_strata"),
        ({"n_bins": 1.5}, "n_bins"),
        ({"block_time": np.zeros(3)}, "only used with oob=True"),
    ],
)
def test_errors(kw, match):
    with pytest.raises(ValueError, match=match):
        _pi(FOREST, **kw)


def test_estimator_errors():
    with pytest.raises(ValueError, match="y is required"):
        inspection.permutation_importance(FOREST, XT)
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):  # landmark dispatch also requires a fitted estimator
        inspection.permutation_importance(LandmarkSurvivalForest(), XT, YT)
    with pytest.raises(TypeError):
        inspection.permutation_importance(object(), XT, YT)
    with pytest.raises(NotFittedError):
        inspection.permutation_importance(SurvivalForestTV(), XT, YT)
