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
