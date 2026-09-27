"""S19: landmark permutation and drop-column importance."""

import contextlib
import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from rftvc import (
    CompetingRisksForestTV,
    LandmarkCompetingRisksForest,
    LandmarkSurvivalForest,
    SurvivalForestTV,
    inspection,
)
from rftvc._inspection._units import resolve_landmark_units
from rftvc.landmark import _raw_groups
from tests.sim import rows, simulate


@contextlib.contextmanager
def _no_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

# --- units (T2) -------------------------------------------------------------------------

NAMES = ["z", "z_mean", "z_slope", "x", "landmark"]
RAW = _raw_groups(["z", ("z", "mean"), ("z", "slope"), "x"])


def test_default_units_are_raw_columns():
    units, names = resolve_landmark_units(None, None, RAW, NAMES, len(NAMES))
    assert names == ["z", "x"]
    assert [u.tolist() for u in units] == [[0, 1, 2], [3]]


def test_features_raw_column_expands_to_its_group():
    units, names = resolve_landmark_units(["z"], None, RAW, NAMES, len(NAMES))
    assert names == ["z"]
    assert units[0].tolist() == [0, 1, 2]


def test_features_derived_name_is_a_singleton():
    units, names = resolve_landmark_units(["z_mean"], None, RAW, NAMES, len(NAMES))
    assert names == ["z_mean"]
    assert units[0].tolist() == [1]


def test_features_mixed_list_of_raw_and_derived():
    units, names = resolve_landmark_units(["z_mean", "x"], None, RAW, NAMES, len(NAMES))
    assert names == ["z_mean", "x"]
    assert [u.tolist() for u in units] == [[1], [3]]


@pytest.mark.parametrize("ref", ["landmark", 4])
def test_landmark_is_not_a_permutable_unit(ref):
    with pytest.raises(ValueError, match="not a permutable unit"):
        resolve_landmark_units([ref], None, RAW, NAMES, len(NAMES))


def test_overlapping_features_raise():
    with pytest.raises(ValueError, match="named by both"):
        resolve_landmark_units(["z", "z_mean"], None, RAW, NAMES, len(NAMES))


def test_unknown_feature_name_raises():
    with pytest.raises(ValueError, match="unknown feature"):
        resolve_landmark_units(["bogus"], None, RAW, NAMES, len(NAMES))


def test_groups_naming_landmark_raises():
    with pytest.raises(ValueError, match="ids column"):
        resolve_landmark_units(None, {"g": ["landmark"]}, RAW, NAMES, len(NAMES))


def test_features_and_groups_together_raise():
    with pytest.raises(ValueError, match="not both"):
        resolve_landmark_units(["z"], {"g": ["x"]}, RAW, NAMES, len(NAMES))


def test_groups_still_works_as_arbitrary_column_sets():
    units, names = resolve_landmark_units(None, {"zx": ["z", "x"]}, RAW, NAMES, len(NAMES))
    assert names == ["zx"]
    assert units[0].tolist() == [0, 3]


# --- PE dispatch (T3) --------------------------------------------------------------------


def _df(n, seed):
    """S3 rows ``[x0, z, noise]`` as a long counting-process frame."""
    rng = np.random.default_rng(seed)
    x0, z, U, ev = simulate(n, rng)
    X, y, ids = rows(x0, z, U, ev)
    noise = np.random.default_rng(seed + 1).normal(size=len(X))
    return pl.DataFrame(
        {"id": ids, "start": y["start"], "stop": y["stop"], "event": y["event"], "x0": X[:, 0], "z": X[:, 1], "n1": noise}
    )


DF = _df(400, 0)
LM = LandmarkSurvivalForest(
    horizon=2.0, step=1.0, history_features=["x0", ("z", "mean"), "n1"], forest=SurvivalForestTV(n_estimators=40, random_state=0)
).fit(DF)

CR_DF = DF.with_columns(pl.when(pl.col("event")).then(1 + (pl.col("id") % 2)).otherwise(0).alias("event"))
LM_CR = LandmarkCompetingRisksForest(
    horizon=2.0, step=1.0, history_features=["x0", ("z", "mean"), "n1"], forest=CompetingRisksForestTV(n_estimators=40, random_state=0)
).fit(CR_DF)


def _pi(model=LM, df=DF, **kw):
    kw.setdefault("n_bootstrap", 0)
    kw.setdefault("random_state", 0)
    return inspection.permutation_importance(model, df, **kw)


def test_strata_time_and_landmark_alias_agree():
    a, b = _pi(strata="time"), _pi(strata="landmark")
    np.testing.assert_array_equal(a.importances, b.importances)


@pytest.mark.parametrize("bad", [None, np.zeros(10), "calendar"])
def test_strata_other_than_time_or_landmark_raises(bad):
    with pytest.raises(ValueError, match="fixed to the landmark partition"):
        _pi(strata=bad)


def test_feature_names_and_units_are_raw_columns_by_default():
    r = _pi()
    assert r.feature_names.tolist() == ["x0", "z", "n1"]
    assert [u.tolist() for u in r.units] == [[0], [1], [2]]


def test_own_column_conditioning_is_a_no_op():
    a = _pi(features=["z_mean"], random_state=3)
    b = _pi(features=["z_mean"], conditional_on=["z_mean"], random_state=3)
    np.testing.assert_array_equal(a.importances, b.importances)


def test_conditioning_on_landmark_itself_is_allowed():
    """decision 5: conditioning on 's' is meaningful (unlike permuting it); it must not be
    rejected the way naming 'landmark' as a *unit* is."""
    r = _pi(features=["z_mean"], conditional_on=["landmark"], random_state=3)
    assert np.isfinite(r.importances_mean).all()


def test_decompositions_sum_to_the_mean():
    r = _pi()
    np.testing.assert_allclose(r.importances_window.sum(1), r.importances_mean, atol=1e-12)
    np.testing.assert_allclose(r.importances_id.sum(1), r.importances_mean, atol=1e-12)
    rc = _pi(LM_CR, CR_DF)
    np.testing.assert_allclose(rc.importances_cause.sum(1), rc.importances_mean, atol=1e-12)


def test_windows_and_null_come_from_the_inner_forest():
    from rftvc.metrics import _baseline_at, event_windows

    w = inspection._windows(LM.forest_, 8)
    np.testing.assert_array_equal(w, event_windows(LM.forest_, 8))
    with pytest.raises(ValueError, match="beyond"):
        _pi(windows=[0.0, LM.horizon + 100.0])


def test_groups_permute_jointly_and_named():
    r = _pi(groups={"both": ["x0", "z_mean"]})
    assert r.feature_names.tolist() == ["both"]
    assert r.units[0].tolist() == [0, 1]


def test_reproducible_across_seeds_and_differs_with_another_seed():
    a = _pi(random_state=7)
    b = _pi(random_state=7)
    np.testing.assert_array_equal(a.importances, b.importances)
    assert not np.array_equal(_pi(random_state=8).importances, a.importances)


# --- errors (reverse-direction checks of decision 1) --------------------------------------


def test_landmark_rejects_counting_process_style_arguments():
    y = DF.select("start", "stop", "event").to_numpy()
    with pytest.raises(ValueError, match="y must be None"):
        _pi(y=y)
    with pytest.raises(ValueError, match="ids must be None"):
        _pi(ids=DF["id"].to_numpy())
    with pytest.raises(ValueError, match="oob=True is only"):
        _pi(oob=True)
    with pytest.raises(ValueError, match="measured_at and block_time"):
        _pi(measured_at=np.zeros(DF.height))


def test_unfitted_landmark_estimator_raises_not_fitted():
    with pytest.raises(NotFittedError):
        inspection.permutation_importance(LandmarkSurvivalForest(), DF)


def test_scoring_outside_pe_brier_ibs_raises():
    with pytest.raises(ValueError, match="scoring must be"):
        _pi(scoring="bogus")


def test_censoring_kwargs_only_valid_for_brier_ibs():
    with pytest.raises(ValueError, match="only used with scoring"):
        _pi(g_min=0.2)
    with pytest.raises(ValueError, match="only used with scoring"):
        _pi(n_times=5)


def test_censoring_kwargs_reject_counting_process_estimators():
    rng = np.random.default_rng(0)
    x0, z, U, ev = simulate(100, rng)
    Xn, yn, idn = rows(x0, z, U, ev)
    forest = SurvivalForestTV(n_estimators=5, random_state=0).fit(Xn, yn, idn)
    with pytest.raises(ValueError, match="only used with scoring"):
        inspection.permutation_importance(forest, Xn, yn, ids=idn, g_min=0.2)


# --- Brier / IBS scoring path (T4) ---------------------------------------------------------


def test_brier_baseline_equals_landmark_cross_validate_pooled_by_n():
    """tvc-design's explicit requirement: the same per-landmark censoring fit, pooled the
    same way, for a fold refit exactly as ``landmark_cross_validate`` does its own."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    from rftvc.model_selection import landmark_cross_validate

    template = LandmarkSurvivalForest(
        horizon=2.0, step=1.0, history_features=["x0", ("z", "mean")], forest=SurvivalForestTV(n_estimators=30, random_state=0)
    )
    cv = GroupKFold(2)
    ref = landmark_cross_validate(template, DF, cv, ("brier",))
    n_ref, b_ref = ref["n"].to_numpy(), ref["brier"].to_numpy()
    pooled_ref = float((n_ref * b_ref).sum() / n_ref.sum())

    data = template._landmark_data(DF)
    total_n = total_nb = 0.0
    for train_idx, test_idx in cv.split(data.s, groups=data.groups):
        train_ids, test_ids = np.unique(data.ids[train_idx]), np.unique(data.ids[test_idx])
        df_train = DF.filter(pl.col("id").is_in(train_ids))
        df_test = DF.filter(pl.col("id").is_in(test_ids))
        m_fold = clone(template).set_params(landmarks=np.unique(data.s[train_idx]), step=None).fit(df_train)
        r = inspection.permutation_importance(m_fold, df_test, scoring="brier", n_bootstrap=0)
        # r.baseline_score is already the n-weighted pool of this fold's own landmarks;
        # its total n is the sum of landmark_cross_validate's own risk-set sizes for this fold's landmarks.
        fold_n = ref.filter(pl.col("landmark").is_in(np.unique(m_fold._landmark_data(df_test).s).tolist()))["n"].sum()
        total_n += fold_n
        total_nb += fold_n * r.baseline_score
    pooled_mine = total_nb / total_n
    np.testing.assert_allclose(pooled_mine, pooled_ref, rtol=1e-9)


def test_sign_convention_positive_when_permuting_hurts():
    r = _pi(scoring="brier", features=["z_mean"], n_repeats=3)
    assert (r.importances > 0).all()
    r_ibs = _pi(scoring="ibs", features=["z_mean"], n_times=6, n_repeats=2)
    assert (r_ibs.importances > 0).all()


def test_cr_requires_a_cause_for_brier_ibs():
    with pytest.raises(ValueError, match="cause is required"):
        _pi(LM_CR, CR_DF, scoring="brier")
    r = _pi(LM_CR, CR_DF, scoring="brier", cause=1)
    assert r.share_of_gain is None
    assert np.isfinite(r.baseline_score)


def test_brier_bootstrap_se_finite():
    r = _pi(scoring="brier", features=["x0"], n_bootstrap=5, n_repeats=2)
    assert np.isfinite(r.importances_se).all()
    assert np.isnan(_pi(scoring="brier", features=["x0"], n_bootstrap=0).importances_se).all()


def test_brier_result_omits_pe_only_fields():
    r = _pi(scoring="brier")
    for field in ("importances_window", "importances_cause", "importances_id", "id_labels", "zero_rate_share",
                  "n_truncated_events", "n_unpermuted", "window_edges", "null_score", "n_events"):
        assert r[field] is None


# --- strata / groups / M3 equivalence, level-history diagnostic (T6 audit) --------------

from rftvc._inspection import _score as _sc  # noqa: E402


def test_permutation_stays_within_each_landmark_and_groups_move_jointly():
    data = LM._landmark_data(DF)
    Xe = data.X.copy()
    Xe[:, 2] = Xe[:, 0] * 3 + 1  # n1 := f(x0): the pair must survive a joint permutation
    codes = np.unique(data.s, return_inverse=True)[1]
    st = _sc.Strata("user", codes, 10, [], 4)
    cols = np.array([0, 2])
    lab = _sc.labels(Xe, np.zeros(Xe.shape[0]), st, cols)
    calls = []
    w = np.array([0.0, LM.horizon])

    def predict(Xr, rows):
        calls.append((Xr.copy(), rows.copy()))
        return LM.forest_.forest_.predict_cumhaz(np.ascontiguousarray(Xr), w, LM.forest_.aggregate, 1)

    H = predict(Xe, np.arange(Xe.shape[0]))
    ev = _sc.Evaluation(Xe, data.y, np.zeros(Xe.shape[0]), data.ids, w, np.zeros(2), 0.01, None, None, predict)
    _sc._permuted(ev, H, Xe, cols, lab, np.random.default_rng(0), np.arange(Xe.shape[0]))
    Xc, rows_ = calls[-1]
    np.testing.assert_array_equal(Xc[:, 2], Xc[:, 0] * 3 + 1)  # joint
    np.testing.assert_array_equal(Xc[:, 1], Xe[rows_, 1])  # untouched column
    for lm in np.unique(codes):  # a received value comes from a row at the same landmark
        inside = codes[rows_] == lm
        assert np.isin(Xc[inside, 0], Xe[codes == lm, 0]).all()
    assert np.unique(codes).size >= 2 and np.bincount(codes).min() != np.bincount(codes).max()  # distinct risk-set sizes


def test_m3_equivalence_permutation_equals_recomputing_from_the_donor_raw_history():
    """tvc-design §3's constructed test: a 2-subject risk set where permuting the ``z``
    group with a forced donor gives the same row as recomputing ``z``/``z_mean`` by hand
    from the donor's raw history up to ``s``."""
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "start": [0.0, 1.0, 0.0, 1.0],
            "stop": [1.0, 2.0, 1.0, 2.0],
            "event": [False, False, False, True],
            "z": [1.0, 3.0, 10.0, 20.0],
        }
    )
    lm = LandmarkSurvivalForest(
        horizon=1.0, landmarks=[1.5], history_features=["z", ("z", "mean")], forest=SurvivalForestTV(n_estimators=5, random_state=0)
    ).fit(df)
    data = lm._landmark_data(df)  # rows known at 1.5: a's [1,3] -> z=3, mean=2; b's [10,20] -> z=20, mean=15
    row_a, row_b = np.flatnonzero(data.ids == "a")[0], np.flatnonzero(data.ids == "b")[0]
    np.testing.assert_allclose(data.X[row_a, :2], [3.0, 2.0])
    np.testing.assert_allclose(data.X[row_b, :2], [20.0, 15.0])
    codes = np.zeros(2, dtype=np.intp)  # one landmark: one stratum
    st = _sc.Strata("user", codes, 10, [], 4)
    cols = np.array([0, 1])
    lab = _sc.labels(data.X, np.zeros(2), st, cols)

    class _ForcedSwap:
        """A fake RNG that forces ``_strata.donors`` (one stratum, ``lexsort`` on this key) to
        swap the two rows: ascending order of ``[1.0, 0.0]`` puts row 1 first, row 0 second."""

        def random(self, n):
            return np.array([1.0, 0.0])

    src = _sc._strata.donors(lab, _ForcedSwap())
    assert src[row_a] == row_b and src[row_b] == row_a  # a receives b's group, and vice versa
    Xp = data.X.copy()
    Xp[row_a, cols] = data.X[row_b, cols]
    np.testing.assert_allclose(Xp[row_a, :2], [20.0, 15.0])  # a's permuted row equals b's own (z, z_mean)


def test_level_history_diagnostic_fast_loose():
    """A generator whose hazard depends on ``z_mean`` (history) only: history-given-level
    importance is positive and level-given-history is small (fast, loose; the strict version
    is the §7.2 sim)."""
    rng = np.random.default_rng(0)
    n, K = 500, 8
    z = rng.normal(size=(n, K))
    zmean = np.cumsum(z, axis=1) / np.arange(1, K + 1)
    lam = 0.15 * np.exp(0.8 * zmean)
    e = rng.exponential(size=n)
    cum = np.cumsum(lam, axis=1)
    k_evt = (cum < e[:, None]).sum(axis=1)
    prev = np.where(k_evt > 0, cum[np.arange(n), np.maximum(k_evt - 1, 0)], 0.0)
    within = (e - prev) / lam[np.arange(n), np.minimum(k_evt, K - 1)]
    T = np.where(k_evt < K, k_evt + within, np.inf)
    C = np.minimum(rng.uniform(2, 8, size=n), 8.0)
    U, ev = np.minimum(T, C), T <= C
    rows_, ids_ = [], []
    for i in range(n):
        k = 0
        while k < U[i]:
            rows_.append((i, float(k), min(k + 1.0, U[i]), bool(ev[i] and k + 1.0 >= U[i]), z[i, k]))
            k += 1
    df = pl.DataFrame(rows_, schema=["id", "start", "stop", "event", "z"], orient="row")
    # a single, later landmark: at k=0 a cumulative mean equals the instantaneous z exactly,
    # which would confound "level" and "history" by construction, not by a modelling failure.
    lm = LandmarkSurvivalForest(
        horizon=2.0, landmarks=[4.0], history_features=["z", ("z", "mean")], forest=SurvivalForestTV(n_estimators=80, random_state=0)
    ).fit(df)
    with _no_warn():
        history_given_level = inspection.permutation_importance(
            lm, df, features=["z_mean"], conditional_on=["z"], n_bootstrap=0, random_state=0
        ).importances_mean[0]
        level_given_history = inspection.permutation_importance(
            lm, df, features=["z"], conditional_on=["z_mean"], n_bootstrap=0, random_state=0
        ).importances_mean[0]
    # fast/loose sanity only (small n, one landmark): the true history effect is detected;
    # the strict history-vs-level comparison, with enough power to bound the confound, is §7.2.
    assert history_given_level > 0
    assert np.isfinite(level_given_history)


def test_bootstrap_resample_spans_every_landmark_a_subject_is_at_risk_at():
    data = LM._landmark_data(DF)
    a_id = data.ids[0]
    a_rows = np.flatnonzero(data.ids == a_id)
    ri, boot_ids = _sc._boot._resample_ids(data.ids, np.random.default_rng(0))
    first_copy_rows = ri[boot_ids == 0]
    a_landmarks_in_data = set(data.s[a_rows].tolist())
    # whichever id copy 0 corresponds to, its rows span exactly that id's own landmarks
    copy_id = data.ids[first_copy_rows[0]]
    expected = set(data.s[data.ids == copy_id].tolist())
    assert set(data.s[first_copy_rows].tolist()) == expected


# --- LOCO (T5) --------------------------------------------------------------------------

from rftvc._inspection import _loco  # noqa: E402


def _loco_kw(**kw):
    kw.setdefault("cv", 3)
    kw.setdefault("n_seeds", 1)
    kw.setdefault("random_state", 0)
    return kw


def test_loco_oracle_noise_near_zero_signal_positive():
    r = inspection.drop_column_importance(LM, DF, **_loco_kw())
    idx = {n: i for i, n in enumerate(r.feature_names)}
    assert abs(r.importances_mean[idx["n1"]]) <= 3 * r.importances_se[idx["n1"]]
    assert r.importances_mean[idx["x0"]] > 0
    assert r.importances_mean[idx["z"]] > 0


def test_loco_dropping_removes_every_derived_feature_of_its_raw_column(monkeypatch):
    seen = []
    orig = _loco._fit_landmark

    def spy(stack_template, df_train, train_s, seed, history_features):
        seen.append(list(history_features))
        return orig(stack_template, df_train, train_s, seed, history_features)

    monkeypatch.setattr(_loco, "_fit_landmark", spy)
    fixture = _df(200, 5)
    lm3 = LandmarkSurvivalForest(
        horizon=2.0, step=1.0, history_features=["x0", ("z", "mean"), ("z", "max"), "n1"],
        forest=SurvivalForestTV(n_estimators=10, random_state=0),
    ).fit(fixture)
    inspection.drop_column_importance(lm3, fixture, **_loco_kw(cv=2))
    full_calls = [h for h in seen if len(h) == 4]
    z_dropped_calls = [h for h in seen if len(h) == 2]  # x0, n1 remain; z's two derived features gone
    assert full_calls and z_dropped_calls
    for h in z_dropped_calls:
        assert "z" not in h and ("z", "mean") not in h and ("z", "max") not in h
        assert "x0" in h and "n1" in h


def test_loco_windows_and_null_come_from_the_folds_own_full_model(monkeypatch):
    seen = []
    orig = _loco._windows_for

    def spy(fitted, windows):
        seen.append(fitted)
        return orig(fitted, windows)

    monkeypatch.setattr(_loco, "_windows_for", spy)
    inspection.drop_column_importance(LM, DF, **_loco_kw())
    assert seen and all(hasattr(f, "event_times_") for f in seen)  # the inner counting-process forest


def test_loco_folds_reuse_split_checks_and_disjoint_check(monkeypatch):
    calls = {"split_checks": 0, "disjoint": 0}
    orig_split, orig_disjoint = _loco._split_checks, _loco._disjoint_check

    def spy_split(*a, **k):
        calls["split_checks"] += 1
        return orig_split(*a, **k)

    def spy_disjoint(*a, **k):
        calls["disjoint"] += 1
        return orig_disjoint(*a, **k)

    monkeypatch.setattr(_loco, "_split_checks", spy_split)
    monkeypatch.setattr(_loco, "_disjoint_check", spy_disjoint)
    inspection.drop_column_importance(LM, DF, **_loco_kw())
    assert calls["split_checks"] >= 1 and calls["disjoint"] >= 1


def test_loco_time_split_gap_below_horizon_raises():
    from rftvc.model_selection import RollingOriginSplit

    cv = RollingOriginSplit(n_splits=2, test_size=1.0, gap=0.0)
    with pytest.raises(ValueError, match="gap"):
        inspection.drop_column_importance(LM, DF, cv=cv, windows=[0.0, 1.0, 2.0])


def test_loco_time_split_se_nan_id_split_se_finite():
    from rftvc.model_selection import RollingOriginSplit

    cv = RollingOriginSplit(n_splits=2, test_size=1.0, gap=2.0)
    r_time = inspection.drop_column_importance(LM, DF, cv=cv, windows=[0.0, 1.0, 2.0], random_state=0)
    assert np.isnan(r_time.importances_se).all()
    assert np.isfinite(r_time.importances).all()
    r_id = inspection.drop_column_importance(LM, DF, **_loco_kw())
    assert np.isfinite(r_id.importances_se).all()


def test_loco_add_noise_control_column_is_a_raw_row_level_draw(monkeypatch):
    """The noise column is drawn once per row of the raw frame (decision 8), not per id: a
    spy confirms every fold's training frame carries the SAME per-row values (``df.filter``
    never changes them), and two independent runs with the same seed agree exactly."""
    seen_rows = {}
    orig = _loco._fit_landmark

    def spy(stack_template, df_train, train_s, seed, history_features):
        if "_noise" in df_train.columns:
            keys = zip(df_train["id"].to_list(), df_train["start"].to_list(), df_train["stop"].to_list())
            for key, v in zip(keys, df_train["_noise"].to_list()):
                seen_rows.setdefault(key, set()).add(v)
        return orig(stack_template, df_train, train_s, seed, history_features)

    monkeypatch.setattr(_loco, "_fit_landmark", spy)
    inspection.drop_column_importance(LM, DF, **_loco_kw(add_noise_control=True, cv=3))
    assert seen_rows and all(len(v) == 1 for v in seen_rows.values())  # never re-drawn per fold

    r1 = inspection.drop_column_importance(LM, DF, **_loco_kw(add_noise_control=True, cv=3, random_state=5))
    r2 = inspection.drop_column_importance(LM, DF, **_loco_kw(add_noise_control=True, cv=3, random_state=5))
    assert "_noise" in r1.feature_names
    np.testing.assert_array_equal(r1.importances_mean, r2.importances_mean)


def test_loco_cost_guard_fit_count(monkeypatch):
    calls = {"n": 0}
    orig = _loco._fit_landmark

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_loco, "_fit_landmark", spy)
    n_folds, n_seeds = 3, 2
    inspection.drop_column_importance(LM, DF, cv=n_folds, n_seeds=n_seeds, random_state=0)
    p_units = len(LM.feature_names_) - 1  # excludes "landmark"
    assert calls["n"] == (p_units + 1) * n_folds * n_seeds


def test_loco_landmark_needs_at_least_one_remaining_feature():
    with pytest.raises(ValueError, match="no history_features"):
        inspection.drop_column_importance(LM, DF, groups={"all": ["x0", "z_mean", "n1"]})


def test_loco_brier_ibs_pooled_and_se_nan():
    r = inspection.drop_column_importance(LM, DF, scoring="brier", **_loco_kw())
    assert np.isfinite(r.baseline_score)
    assert np.isnan(r.importances_se).all()
    assert r.share_of_gain is None and r.importances_window is None


class _GroupFixedSplit:
    """A splitter yielding one pre-set train/test split by group (id) membership: forces a
    fold whose training data lacks a cause entirely, for a deterministic regression test."""

    def __init__(self, train_ids, test_ids):
        self._train_ids = np.asarray(train_ids)
        self._test_ids = np.asarray(test_ids)

    def split(self, X, y=None, groups=None):
        groups = np.asarray(groups)
        yield np.flatnonzero(np.isin(groups, self._train_ids)), np.flatnonzero(np.isin(groups, self._test_ids))

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def test_loco_brier_survives_a_cause_absent_from_a_training_fold():
    """As the counting-process LOCO fix: a landmark competing-risks Brier/IBS refit
    (``run_landmark_loss``) must fix the cause vocabulary from the full data before
    cross-fitting, not leave each fold to infer its own (regression: this used to raise
    "cause=... is not a fitted cause label" when a fold's training ids never saw a cause)."""
    n_ids = 6
    id_ = np.repeat(np.arange(n_ids), 4)
    start = np.tile([0.0, 1.0, 2.0, 3.0], n_ids)
    stop = start + 1.0
    event = np.zeros(n_ids * 4, dtype=int)
    event[3::4] = [1, 1, 1, 1, 1, 2]  # only id 5 ever has a cause-2 event
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=n_ids * 4)
    x1 = rng.normal(size=n_ids * 4)
    df = pl.DataFrame({"id": id_, "start": start, "stop": stop, "event": event, "x0": x0, "x1": x1})
    model = LandmarkCompetingRisksForest(
        horizon=1.0, step=1.0, history_features=["x0", "x1"],
        forest=CompetingRisksForestTV(n_estimators=5, random_state=0),
    )
    cv = _GroupFixedSplit(train_ids=np.arange(5), test_ids=[5])
    r = inspection.drop_column_importance(model, df, cv=cv, scoring="brier", cause=2, n_seeds=1)
    assert np.isfinite(r.baseline_score)
