"""S19: landmark permutation and drop-column importance."""

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
