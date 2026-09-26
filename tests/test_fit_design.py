"""The fit-time design (S16): pure extraction, fingerprint, rebuild, training null."""

import numpy as np
import pytest

import rftvc._estimator as est_mod
from bench.perf_fit import synth
from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y, make_survival_y
from rftvc.metrics import concordance_index_cp

X, Y, IDS = synth(800, rows_per_id=4, seed=2)


def _cr_y(y, seed=0):
    rng = np.random.default_rng(seed)
    return make_competing_risks_y(y["stop"], np.where(y["event"], rng.integers(1, 3, len(y)), 0), start=y["start"])


def _pooled_na(times, start, stop, event):
    """Reference pooled Nelson–Aalen by brute force."""
    out, cum = [], 0.0
    for t in times:
        Y_t = np.sum((start < t) & (t <= stop))
        d = np.sum(event & (stop == t))
        cum += d / Y_t if Y_t else 0.0
        out.append(cum)
    return np.array(out)


def test_call_order(monkeypatch):
    calls = []

    def spy(name, fn):
        def wrapped(*a, **k):
            calls.append(name)
            return fn(*a, **k)

        return wrapped

    monkeypatch.setattr(est_mod, "check_counting_process", spy("check_counting_process", est_mod.check_counting_process))
    monkeypatch.setattr(est_mod, "check_random_state", spy("check_random_state", est_mod.check_random_state))
    monkeypatch.setattr(est_mod._core, "coarsen", spy("coarsen", est_mod._core.coarsen))
    monkeypatch.setattr(est_mod._core, "fit_forest", spy("fit_forest", est_mod._core.fit_forest))
    for cls, y in ((SurvivalForestTV, Y), (CompetingRisksForestTV, _cr_y(Y))):
        for kw, step in (({"ntime": 20}, "coarsen"), ({"resample_unit": "block", "block_length": 2.0}, "_block_design")):

            class Spy(cls):
                pass

            for name in ("_check_y", "_validate_params", "_block_design", "_resolve_min_ids_leaf", "_resolve_n_draw"):
                setattr(Spy, name, _method_spy(name, getattr(cls, name), calls))
            calls.clear()
            Spy(n_estimators=3, random_state=0, **kw).fit(X, y, IDS)
            expected = ["_check_y", "check_counting_process", "_validate_params", step,
                        "_resolve_min_ids_leaf", "_resolve_n_draw", "check_random_state", "fit_forest"]
            assert [c for c in calls if c in expected] == expected, (cls.__name__, kw, calls)


def _method_spy(name, fn, calls):
    def wrapped(self, *a, **k):
        calls.append(name)
        return fn(self, *a, **k)

    return wrapped


def test_design_is_deterministic_and_rng_free(monkeypatch):
    m = SurvivalForestTV(n_estimators=3, ntime=20, random_state=None)
    monkeypatch.setattr(est_mod, "check_random_state", lambda *_: pytest.fail("_fit_design must not draw"))
    state = np.random.get_state()[1].copy()
    d1 = m._fit_design(X, Y, IDS, None, "error", "counting_process", None)
    d2 = m._fit_design(X, Y, IDS, None, "error", "counting_process", None)
    np.testing.assert_array_equal(np.random.get_state()[1], state)
    for a, b in zip(d1.fit_rows + d1.oob_set, d2.fit_rows + d2.oob_set):
        np.testing.assert_array_equal(a, b)
    assert d1.fingerprint == d2.fingerprint


def _stacked():
    start = np.zeros(len(Y))
    block_time = Y["start"].copy()
    return make_survival_y(Y["stop"] - Y["start"], Y["event"], start=start), block_time


REBUILD_CASES = {
    "id": ({}, {}),
    "block": ({"resample_unit": "block", "block_length": 2.0}, {}),
    "coarse": ({"ntime": 20}, {}),
    "stacked_coarse_block": ({"ntime": 20, "resample_unit": "block", "block_length": 2.0}, {"layout": "stacked"}),
}


@pytest.mark.parametrize("case", REBUILD_CASES)
def test_rebuild_reproduces_the_fit_design(case):
    kw, fit_kw = REBUILD_CASES[case]
    y, block_time = (Y, None) if "layout" not in fit_kw else _stacked()
    m = SurvivalForestTV(n_estimators=3, random_state=0, **kw).fit(X, y, IDS, block_time=block_time, **fit_kw)
    d = m._rebuild_design(X, y, IDS, block_time=block_time)
    ref = m._fit_design(X, y, IDS, None, fit_kw.get("gap_policy", "error"), fit_kw.get("layout", "counting_process"),
                        block_time)
    for a, b in zip(d.fit_rows + d.oob_set + (d.units,), ref.fit_rows + ref.oob_set + (ref.units,)):
        np.testing.assert_array_equal(a, b)
    # the training null is the pooled Nelson–Aalen of the design rows
    _, start, stop, event = d.fit_rows
    np.testing.assert_allclose(m.baseline_cumhaz_, _pooled_na(m.event_times_, start, stop, event != 0), rtol=1e-12)


def test_fingerprint_detects_changes():
    m = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, Y, IDS)
    m._rebuild_design(X, Y, list(IDS))  # list vs array: same ids
    m._rebuild_design(X, Y, IDS.astype(np.int32))  # integer width
    X2 = X.copy()
    X2[0, 0] += 1
    y2 = Y.copy()
    y2["stop"][np.flatnonzero(IDS == IDS[0])[-1]] += 1e-9  # the id's last row: no overlap created
    relabel = IDS + 1000  # same structure, new labels
    moved = IDS.copy()
    moved[np.flatnonzero(IDS == IDS[0])[-1]] = IDS.max() + 1  # the first id's last row becomes its own id
    changes = [
        dict(X=X2),
        dict(y=y2),
        dict(ids=relabel),
        dict(ids=moved),  # membership change
        dict(measured_at=Y["start"]),
    ]
    for ch in changes:
        args = dict(X=X, y=Y, ids=IDS) | ch
        with pytest.raises(ValueError, match="do not match"):
            m._rebuild_design(**args)
    m.set_params(ntime=10)
    with pytest.raises(ValueError, match="do not match"):
        m._rebuild_design(X, Y, IDS)


def test_fingerprint_block_time_and_split_id():
    y, bt = _stacked()
    m = SurvivalForestTV(n_estimators=3, resample_unit="block", block_length=2.0, random_state=0)
    m.fit(X, y, IDS, layout="stacked", block_time=bt)
    with pytest.raises(ValueError, match="do not match"):
        m._rebuild_design(X, y, IDS, block_time=bt + 1)
    # split_id: drop each id's second row to create gaps -> two chains per id
    keep = np.ones(len(Y), bool)
    first = np.r_[True, IDS[1:] != IDS[:-1]]
    second = np.r_[False, first[:-1]] & ~first
    keep[second] = False
    Xg, yg, ig = X[keep], Y[keep], IDS[keep]
    mg = SurvivalForestTV(n_estimators=3, random_state=0).fit(Xg, yg, ig, gap_policy="split_id")
    d = mg._rebuild_design(Xg, yg, ig)
    assert d.fingerprint == mg._fit_fingerprint_


def test_rebuild_with_an_ids_column_in_a_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])]).assign(id=IDS)
    m = SurvivalForestTV(n_estimators=10, oob_score=True, random_state=0).fit(df, Y, ids="id")
    d = m._rebuild_design(df, Y)  # ids default to the fit-time column
    assert d.fingerprint == m._fit_fingerprint_
    np.testing.assert_array_equal(d.X, X)  # the id column is not a feature
    H, _ = m.forest_.oob_cumhaz(d.X, *d.oob_set, m.event_times_, m.aggregate, 1)
    assert np.array_equal(np.cumsum(H, axis=1)[:, -1], m.oob_prediction_, equal_nan=True)
    with pytest.raises(ValueError, match="do not match"):
        m._rebuild_design(df.assign(id=IDS + 1), Y)


def test_competing_baseline_per_cause():
    y = _cr_y(Y)
    m = CompetingRisksForestTV(n_estimators=3, random_state=0).fit(X, y, IDS)
    assert m.baseline_cumhaz_.shape == (2, m.event_times_.size)
    d = m._rebuild_design(X, y, IDS)
    _, start, stop, event = d.fit_rows
    for j in range(2):
        ref = _pooled_na(m.event_times_, start, stop, event == j + 1)
        np.testing.assert_allclose(m.baseline_cumhaz_[j], ref, rtol=1e-12)
    np.testing.assert_allclose(
        m.baseline_cumhaz_.sum(axis=0), _pooled_na(m.event_times_, start, stop, event != 0), rtol=1e-12
    )


def test_rebuild_does_not_mutate_on_mismatch():
    y = _cr_y(Y)
    m = CompetingRisksForestTV(n_estimators=3, random_state=0).fit(X, y, IDS)
    before = m.causes_.copy()
    bad = make_competing_risks_y(y["stop"], np.where(y["event"] > 0, 7, 0), start=y["start"])
    with pytest.raises(ValueError, match="do not match"):
        m._rebuild_design(X, bad, IDS)
    np.testing.assert_array_equal(m.causes_, before)


def test_score_and_oob_score_stay_concordance():
    m = SurvivalForestTV(n_estimators=20, oob_score=True, random_state=0).fit(X, Y, IDS)
    assert m.score(X, Y, ids=IDS) == pytest.approx(concordance_index_cp(Y, m.predict(X), ids=IDS))
    ok = np.isfinite(m.oob_prediction_)
    assert m.oob_score_ == pytest.approx(concordance_index_cp(Y[ok], m.oob_prediction_[ok], ids=IDS[ok]))


def test_ambiguous_string_ids_do_not_collide():
    from rftvc._estimator import _canonical_ids

    assert _canonical_ids(np.array(["a\x1fb", "c"], dtype=object)) != _canonical_ids(np.array(["a", "b\x1fc"], dtype=object))
    assert _canonical_ids(np.array(["1", "2"], dtype=object)) != _canonical_ids(np.array([1, 2], dtype=object))
    assert _canonical_ids([3, 4]) == _canonical_ids(np.array([3, 4], dtype=np.int32))


def test_string_and_categorical_ids_rebuild():
    pd = pytest.importorskip("pandas")
    sid = np.array([f"s{i}" for i in IDS], dtype=object)
    m = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, Y, sid)
    m._rebuild_design(X, Y, sid)
    cat = pd.Categorical(sid)
    mc = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, Y, cat)
    mc._rebuild_design(X, Y, cat)
    with pytest.raises(ValueError, match="do not match"):
        m._rebuild_design(X, Y, np.array([f"t{i}" for i in IDS], dtype=object))


def test_refit_without_ntime_drops_coarse_attributes():
    m = SurvivalForestTV(n_estimators=3, ntime=20, random_state=0).fit(X, Y, IDS)
    assert hasattr(m, "coarse_grid_")
    m.set_params(ntime=None).fit(X, Y, IDS)
    for name in ("coarse_grid_", "n_coarsen_dropped_rows_", "n_coarsen_lost_events_"):
        assert not hasattr(m, name)


def test_failed_refit_updates_input_attributes_as_before():
    pd = pytest.importorskip("pandas")
    m = SurvivalForestTV(n_estimators=3, random_state=0).fit(X, Y, IDS)
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(X.shape[1])])
    bad = Y.copy()
    bad["stop"] = bad["start"]  # invalid target: fails after the input is parsed
    with pytest.raises(ValueError):
        m.fit(df, bad, IDS)
    assert list(m.feature_names_in_) == list(df.columns)  # main's behaviour: set before target validation
    assert m.ids_column_ is None
