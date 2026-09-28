"""S18: ``inspection.drop_column_importance`` (LOCO, cross-fitted)."""

import numpy as np
import pandas as pd
import pytest

from rftvc import (
    CompetingRisksForestTV,
    LandmarkSurvivalForest,
    SurvivalForestTV,
    inspection,
    make_competing_risks_y,
)
from rftvc._inspection import _loco
from rftvc.model_selection import RollingOriginSplit
from tests.sim import rows, simulate

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*windows are too fine:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
]


def _data(n, seed):
    """S3 rows ``[x0, z, noise]`` with ids (as ``test_inspection_perm.py``'s convention)."""
    rng = np.random.default_rng(seed)
    x0, z, U, ev = simulate(n, rng)
    X, y, ids = rows(x0, z, U, ev)
    return np.c_[X, rng.normal(size=len(X))], y, ids


def _dci(estimator, X, y, **kw):
    kw.setdefault("random_state", 0)
    kw.setdefault("cv", 3)
    return inspection.drop_column_importance(estimator, X, y, **kw)


def _cr_data(n_ids, seed, effect=0.8):
    """A small competing-risks panel: cause 1 driven by column 0, cause 2 by nothing.

    4 rows per id (``start`` = 0..3); only the last row of an id may carry an event
    (``check_counting_process`` requires the event on the id's last row).
    """
    rng = np.random.default_rng(seed)
    n = n_ids * 4
    ids = np.repeat(np.arange(n_ids), 4)
    start = np.tile([0.0, 1.0, 2.0, 3.0], n_ids)
    stop = start + 1.0
    X = rng.normal(size=(n, 2))
    z = X[3::4, 0]
    p1 = 1.0 / (1.0 + np.exp(-(effect * z)))
    u = rng.random(n_ids)
    last = np.where(u < 0.5 * p1, 1, np.where(u < 0.5, 2, 0))
    labels = np.zeros(n, dtype=int)
    labels[3::4] = last
    y = make_competing_risks_y(stop, labels, start=start)
    return X, y, ids


# --- oracle / groups (S3 generator, small, seeded) --------------------------------------


def test_oracle_noise_is_null_and_x0_matters():
    X, y, ids = _data(400, 0)
    est = SurvivalForestTV(n_estimators=30, random_state=0, n_jobs=1)
    res = _dci(est, X, y, ids=ids, cv=4, n_seeds=2)
    assert abs(res.importances_mean[2]) <= 3 * res.importances_se[2]  # noise column
    assert res.importances_mean[0] > 0  # x0


def test_random_state_accepts_generator():
    X, y, ids = _data(200, 3)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    a = _dci(est, X, y, ids=ids, random_state=np.random.default_rng(0))
    b = _dci(est, X, y, ids=ids, random_state=np.random.default_rng(0))
    np.testing.assert_array_equal(a.importances_mean, b.importances_mean)


def test_groups_single_copy_is_near_zero_grouped_is_positive():
    X, y, ids = _data(300, 1)
    z_copy = X[:, 1].copy()
    X2 = np.c_[X, z_copy]  # columns: x0, z, noise, z_copy
    est = SurvivalForestTV(n_estimators=30, random_state=0, n_jobs=1)
    single = _dci(est, X2, y, ids=ids, features=[1, 3])
    grouped = _dci(est, X2, y, ids=ids, groups={"z_both": [1, 3]})
    g = grouped.importances_mean[0]
    assert g > 0
    # "single ~ 0" holds relative to grouped, not close to a literal 0: dropping one of
    # two identical columns is measurably (~15-25%, observed) cheaper than dropping both,
    # since the other copy remains available to every split, but not free (max_features
    # subsampling makes the surviving copy a competitor, not a guaranteed substitute).
    assert abs(single.importances_mean[0]) <= 0.5 * g
    assert abs(single.importances_mean[1]) <= 0.5 * g


# --- folds -------------------------------------------------------------------------------


class SpyForest(SurvivalForestTV):
    seen = []

    def fit(self, X, y, ids=None, **kw):
        SpyForest.seen.append((np.asarray(ids).copy(), np.asarray(X).copy()))
        return super().fit(X, y, ids, **kw)


def test_id_splitter_every_id_scored_exactly_once():
    X, y, ids = _data(200, 2)
    est = SurvivalForestTV(n_estimators=20, random_state=0, n_jobs=1)
    res = _dci(est, X, y, ids=ids, cv=4)
    assert res.n_ids == np.unique(ids).size
    assert np.isfinite(res.importances_se).all()


def test_repeated_splitter_n_ids_counts_distinct_ids_once():
    """A valid repeated group splitter (e.g. GroupShuffleSplit) can put the same id in more
    than one fold's test set. ``n_ids``/the cluster SE must treat that id as one cluster,
    not once per occurrence (regression: previously summed per-fold test-id counts)."""
    from sklearn.model_selection import GroupShuffleSplit

    X, y, ids = _data(200, 5)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    cv = GroupShuffleSplit(n_splits=4, test_size=0.4, random_state=0)
    test_ids = set()
    for _, test_idx in cv.split(X, groups=ids):
        test_ids.update(np.unique(ids[test_idx]).tolist())
    res = _dci(est, X, y, ids=ids, cv=cv, n_seeds=1)
    assert res.n_ids == len(test_ids)
    assert np.isfinite(res.importances_se).all()


def test_se_invariant_to_duplicating_an_identical_fold(monkeypatch):
    """Repeating a byte-identical fold must not change the reported SE: redundant,
    duplicate data carries no new information about spread (regression: the id-cluster SE
    was averaged per id while N still summed every fold occurrence, silently halving the
    reported SE for an exact duplicate fold)."""
    from sklearn.model_selection import GroupKFold

    import rftvc._inspection._loco as loco_mod

    X, y, ids = _data(200, 9)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    train_idx, test_idx = next(GroupKFold(4).split(X, groups=ids))
    monkeypatch.setattr(loco_mod, "seed_for", lambda entropy, *keys: 0)  # every fold/seed fits identically
    once = inspection.drop_column_importance(
        est, X, y, ids=ids, cv=_FixedSplit([(train_idx, test_idx)]), n_seeds=1
    )
    twice = inspection.drop_column_importance(
        est, X, y, ids=ids, cv=_FixedSplit([(train_idx, test_idx), (train_idx, test_idx)]), n_seeds=1
    )
    np.testing.assert_allclose(once.importances_se, twice.importances_se, rtol=1e-10)


def test_id_cluster_se_centers_by_scored_event_share():
    """An id scored on more events naturally has a larger raw total drop without being any
    more variable per event: the cluster SE must center each id's pooled total by its own
    share of the pooled per-event mean (mean * n_i), not treat the raw pooled total as the
    observation (regression: summing raw per-id totals directly gave a spurious nonzero SE
    even when every id's per-event rate was identical, merely because one id had more
    scored events than the others).

    3 ids: "a" appears in two fold-occurrences (1 scored event each, drops 1 and 3);
    "b" and "c" appear once each (1 scored event, drop 2). Every id's rate is 2/event
    (a: (1+3)/2 = 2; b, c: 2/1 = 2), so the true cluster spread is exactly zero.
    """
    id_values = [np.array(["a", "b"]), np.array(["a", "c"])]
    id_n = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    id_drop = [[np.array([1.0, 2.0]), np.array([3.0, 2.0])]]  # one unit, per-fold raw drops
    mean = np.array([2.0])  # total_drop = 8, N = 4 -> mean = 2
    n_ids, se = _loco._id_cluster_se(id_values, id_n, id_drop, mean, N=4.0, n_units=1, time_split=False)
    assert n_ids == 3
    assert se[0] == pytest.approx(0.0, abs=1e-12)


def test_cause_breakdown_survives_disagreeing_window_edges():
    """Cause labels are a fixed, fold-independent vocabulary, so a fold disagreement in
    window edges should drop the window decomposition but not the (unrelated) cause
    decomposition (regression: cause_drop's accumulation and reporting were both gated on
    window_ok, silently discarding a valid, independently-computed cause breakdown)."""
    import rftvc._inspection._loco as loco_mod

    X, y, ids = _cr_data(150, 15)
    est = CompetingRisksForestTV(n_estimators=10, random_state=0, n_jobs=1)
    orig = loco_mod.event_windows
    calls = {"n": 0}

    def fake(fitted, windows):
        edges = orig(fitted, windows)
        if calls["n"] % 2 == 1:
            edges = edges.copy()
            edges[1:] *= 2.0
        calls["n"] += 1
        return edges

    loco_mod.event_windows = fake
    try:
        res = _dci(est, X, y, ids=ids, cv=3, n_seeds=1)
    finally:
        loco_mod.event_windows = orig
    assert res.importances_window is None
    assert res.importances_cause is not None
    np.testing.assert_allclose(res.importances_cause.sum(axis=1), res.importances_mean, atol=1e-8)


def test_window_pooling_requires_matching_edges_not_just_shape():
    """Folds with the same number of windows but different edges (expected under the
    default int ``windows``, since each fold's own event_windows are quantiles of that
    fold's own training data) must not be pooled by position (regression: previously only
    the shape was checked, and the first fold's edges were reported for the pooled sum)."""
    import rftvc._inspection._loco as loco_mod

    X, y, ids = _data(150, 8)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    orig = loco_mod.event_windows
    calls = {"n": 0}

    def fake(fitted, windows):
        edges = orig(fitted, windows)
        if calls["n"] % 2 == 1:
            edges = edges.copy()
            edges[1:] *= 2.0  # same shape and starts at 0, but different interior edges
        calls["n"] += 1
        return edges

    loco_mod.event_windows = fake
    try:
        res = _dci(est, X, y, ids=ids, cv=3, n_seeds=1)
    finally:
        loco_mod.event_windows = orig
    assert res.importances_window is None
    assert res.window_edges is None


@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_time_splitter_censors_training_and_gives_nan_se():
    X, y, ids = _data(400, 3)
    est = SurvivalForestTV(n_estimators=20, random_state=0, n_jobs=1)
    SpyForest.seen = []
    spy = SpyForest(**est.get_params(deep=False))
    cv = RollingOriginSplit(n_splits=2, test_size=3.0, gap=0.0)
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    with pytest.warns(UserWarning):
        res = _dci(spy, X, y, ids=ids, cv=cv, windows=edges)
    # the censoring mechanics themselves (training rows clipped at the cutoff) are
    # tests/test_model_selection.py's test_cv_folds_time_split_censors_training_rows;
    # this test only checks drop_column_importance's own SE/shape contract under a time
    # splitter (every fold's fits see the same training ids, one full + one per unit).
    n_units = X.shape[1]
    per_fold = len(SpyForest.seen) // 2
    assert per_fold == n_units + 1
    assert np.isnan(res.importances_se).all()
    assert res.importances.shape == (n_units, 2)


def test_int_windows_under_time_splitter_raises():
    X, y, ids = _data(100, 4)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    cv = RollingOriginSplit(n_splits=2, test_size=3.0, gap=0.0)
    with pytest.raises(ValueError, match="RollingOriginSplit"):
        _dci(est, X, y, ids=ids, cv=cv, windows=8)


def test_other_splitter_with_shared_ids_raises():
    from sklearn.model_selection import KFold

    X, y, ids = _data(100, 5)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    with pytest.raises(ValueError, match="disjoint"):
        _dci(est, X, y, ids=ids, cv=KFold(3))


# --- consistency: windows/null come from the full model ---------------------------------


def test_dropped_model_scored_with_full_models_windows_and_null():
    import rftvc.metrics as metrics_mod

    X, y, ids = _data(150, 6)
    est = SurvivalForestTV(n_estimators=15, random_state=0, n_jobs=1)
    seen_baselines = []
    orig = metrics_mod._baseline_at

    def spy(estimator, windows):
        out = orig(estimator, windows)
        seen_baselines.append((estimator, out))
        return out

    import rftvc._inspection._loco as loco_mod

    loco_mod._baseline_at = spy
    try:
        _dci(est, X, y, ids=ids, cv=3)
    finally:
        loco_mod._baseline_at = orig
    # exactly one _baseline_at call per (fold, seed): the null is computed once from
    # the full model and reused (not recomputed) for every dropped unit's score.
    assert len(seen_baselines) == 3  # n_folds * n_seeds (default n_seeds=1)


# --- seeds / cost guard -------------------------------------------------------------------


def test_seeds_reproducible_and_fit_count():
    X, y, ids = _data(150, 7)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    a = _dci(est, X, y, ids=ids, cv=3, n_seeds=2, random_state=5)
    b = _dci(est, X, y, ids=ids, cv=3, n_seeds=2, random_state=5)
    np.testing.assert_allclose(a.importances_mean, b.importances_mean)

    SpyForest.seen = []
    spy = SpyForest(**est.get_params(deep=False))
    n_units = X.shape[1]
    n_folds = 3
    _dci(spy, X, y, ids=ids, cv=n_folds, n_seeds=2, random_state=5)
    assert len(SpyForest.seen) == (n_units + 1) * n_folds * 2


@pytest.mark.parametrize("add_noise", [False, True])
def test_cost_guard(add_noise):
    X, y, ids = _data(120, 8)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    SpyForest.seen = []
    spy = SpyForest(**est.get_params(deep=False))
    n_units = X.shape[1] + (1 if add_noise else 0)
    _dci(spy, X, y, ids=ids, cv=3, n_seeds=2, add_noise_control=add_noise)
    assert len(SpyForest.seen) == (n_units + 1) * 3 * 2


# --- noise control ------------------------------------------------------------------------


def test_noise_control_reported_and_shared_across_folds():
    X, y, ids = _data(150, 9)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    SpyForest.seen = []
    spy = SpyForest(**est.get_params(deep=False))
    n_units = X.shape[1] + 1  # + "_noise"
    res = _dci(spy, X, y, ids=ids, cv=3, add_noise_control=True, random_state=11)
    assert res.feature_names[-1] == "_noise"
    assert res.units[-1].tolist() == [X.shape[1]]
    # the noise column's value for a given original row (identified by its other, unchanged
    # covariates, since a row's own id can repeat with different covariates across rows) is
    # identical everywhere it is seen: drawn once, before any fold split, not per fold/unit.
    # Only the full-model fits (first of each (fold, seed) batch) keep every column,
    # including the appended "_noise" one, in the same last position.
    full_fits = SpyForest.seen[0 :: n_units + 1]
    seen_by_row = {}
    for ids_arr, Xarr in full_fits:
        for i, row in zip(ids_arr, Xarr):
            key = (int(i), *np.round(row[:-1], 9).tolist())
            seen_by_row.setdefault(key, []).append(row[-1])
    assert seen_by_row  # sanity: the slicing above actually selected full-model fits
    for vals in seen_by_row.values():
        assert np.allclose(vals, vals[0])


# --- DataFrame / ids column ---------------------------------------------------------------


def test_dataframe_ids_column():
    X, y, ids = _data(150, 10)
    df = pd.DataFrame({"x0": X[:, 0], "z": X[:, 1], "noise": X[:, 2], "id": ids})
    est = SurvivalForestTV(n_estimators=15, random_state=0, n_jobs=1)
    arr_res = _dci(est, X, y, ids=ids, cv=3, features=[0, 1])
    df_res = _dci(est, df, y, ids="id", cv=3, features=["x0", "z"])
    np.testing.assert_allclose(df_res.importances_mean, arr_res.importances_mean)
    default = _dci(est, df, y, ids="id", cv=3)
    assert "id" not in list(default.feature_names)
    with pytest.raises(ValueError, match="ids column"):
        _dci(est, df, y, ids="id", cv=3, features=["id"])
    with pytest.raises(ValueError, match="ids column"):
        _dci(est, df, y, ids="id", cv=3, groups={"g": ["id"]})

    # the grouping by ids is identical across the full and every dropped fit of a fold
    # (one ids_values array, sliced once, never re-derived per refit from the DataFrame).
    SpyForest.seen = []
    spy = SpyForest(**est.get_params(deep=False))
    n_units = len(df.columns) - 1
    _dci(spy, df, y, ids="id", cv=3, n_seeds=1)
    per_fold = len(SpyForest.seen) // 3
    assert per_fold == n_units + 1
    for f in range(3):
        block = SpyForest.seen[f * per_fold : (f + 1) * per_fold]
        first_ids = block[0][0]
        for other_ids, _ in block[1:]:
            np.testing.assert_array_equal(other_ids, first_ids)


# --- competing risks -----------------------------------------------------------------------


class _FixedSplit:
    """A splitter yielding pre-set ``(train_idx, test_idx)`` pairs, to force a fold whose
    training data lacks a cause entirely (a CV/`GroupKFold` split could do this by chance
    with a rare cause; here it's forced for a deterministic regression test)."""

    def __init__(self, folds):
        self._folds = folds

    def split(self, X, y=None, groups=None):
        yield from self._folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._folds)


def test_competing_risks_loco_survives_a_cause_absent_from_a_training_fold():
    """A cause vocabulary fixed from the fold's own training data (rather than the full
    data, as ``landmark_cross_validate`` already does) makes scoring raise once a fold's
    test set contains a cause its training fold never saw (regression: this used to crash
    with "y has cause labels [...] not in causes")."""
    n_ids = 6
    ids = np.repeat(np.arange(n_ids), 4)
    start = np.tile([0.0, 1.0, 2.0, 3.0], n_ids)
    stop = start + 1.0
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_ids * 4, 2))
    labels = np.zeros(n_ids * 4, dtype=int)
    labels[3::4] = [1, 1, 1, 1, 1, 2]  # only id 5 ever has a cause-2 event
    y = make_competing_risks_y(stop, labels, start=start)

    train_idx = np.flatnonzero(np.isin(ids, np.arange(5)))  # ids 0-4: cause 1 only
    test_idx = np.flatnonzero(ids == 5)  # id 5: the only cause-2 event
    cv = _FixedSplit([(train_idx, test_idx)])

    est = CompetingRisksForestTV(n_estimators=10, random_state=0, n_jobs=1)
    res = inspection.drop_column_importance(est, X, y, ids=ids, cv=cv, n_seeds=1)
    assert res.n_events > 0


def test_competing_risks_cause_selection():
    X, y, ids = _cr_data(150, 12)
    est = CompetingRisksForestTV(n_estimators=20, random_state=0, n_jobs=1)
    r1 = _dci(est, X, y, ids=ids, cv=3, cause=1)
    assert r1.importances_cause is None
    rboth = _dci(est, X, y, ids=ids, cv=3)
    assert rboth.importances_cause is not None
    np.testing.assert_allclose(rboth.importances_cause.sum(axis=1), rboth.importances_mean, atol=1e-8)
    with pytest.raises(ValueError, match="cause"):
        _dci(est, X, y, ids=ids, cv=3, cause=99)


# --- errors ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"features": ["bogus"]}, "not fitted on named columns"),
        ({"features": [99]}, "out of range"),
        ({"groups": {"a": [0, 1], "b": [1]}}, "is in groups"),
        ({"groups": {"a": []}}, "empty"),
        ({"features": [0], "groups": {"a": [1]}}, "not both"),
        ({"scoring": "brier"}, "scoring"),
        ({"n_seeds": 0}, "n_seeds"),
        ({"cv": 1}, "cv must be"),
        ({"cv": object()}, "cv must be"),
    ],
)
def test_errors(kw, match):
    X, y, ids = _data(100, 13)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    with pytest.raises(ValueError, match=match):
        _dci(est, X, y, ids=ids, **kw)


def test_estimator_errors():
    X, y, ids = _data(100, 14)
    est = SurvivalForestTV(n_estimators=10, random_state=0, n_jobs=1)
    with pytest.raises(ValueError, match="y is required"):
        inspection.drop_column_importance(est, X)
    with pytest.raises(ValueError, match="y must be None"):
        inspection.drop_column_importance(LandmarkSurvivalForest(), X, y)
    with pytest.raises(TypeError):
        inspection.drop_column_importance(object(), X, y)
