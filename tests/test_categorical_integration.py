"""Categorical columns remain whole units in model inspection."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from rftvc import (
    CompetingRisksForestTV, LandmarkSurvivalForest, SurvivalForestTV,
    make_competing_risks_y, make_survival_y,
)
from rftvc.inspection import drop_column_importance, hazard_effect, path_effect, permutation_importance


@pytest.mark.parametrize("oob", [False, True])
def test_permutation_importance_moves_all_category_indicators_together(oob):
    n = 90
    group = np.array(["high", "low", "mid"])[np.arange(n) % 3]
    X = pd.DataFrame({"group": pd.Categorical(group), "noise": np.arange(n) / n})
    y = make_survival_y(1 + np.arange(n) % 8, np.arange(n) % 3 != 0)
    model = SurvivalForestTV(n_estimators=30, min_ids_leaf=3, min_events_leaf=1,
                             max_features=None, oob_score=oob, random_state=8).fit(X, y)
    result = permutation_importance(model, X, y, oob=oob, features=["group"],
                                    windows=2, n_repeats=2, n_bootstrap=0, random_state=4)
    assert result.feature_names.tolist() == ["group"]
    np.testing.assert_array_equal(result.units[0], [0, 1, 2])
    assert np.isfinite(result.importances).all()
    effect = hazard_effect(model, X, y, feature="group", windows=2)
    assert effect["values"].tolist() == ["high", "low", "mid"]
    assert effect.hazard.shape == (3, 2)
    intervals = make_survival_y(np.full(n, 8.0), np.zeros(n, dtype=bool))
    path = path_effect(model, X, intervals, np.arange(n), feature="group",
                       delta=lambda values, start: np.full(values.shape, "low"),
                       from_time=2.0, horizons=[4.0, 8.0])
    assert path.per_subject.shape == (n, 2)


def test_drop_column_importance_refits_numeric_category_dtype():
    n = 72
    X = pd.DataFrame({"group": pd.Categorical([1, 4, 9] * (n // 3)),
                      "noise": np.arange(n) / n})
    y = make_survival_y(1 + np.arange(n) % 8, np.arange(n) % 4 != 0)
    model = SurvivalForestTV(n_estimators=4, min_ids_leaf=2,
                             min_events_leaf=1, random_state=1)
    result = drop_column_importance(model, X, y, cv=3, features=["group"],
                                    windows=2, n_seeds=1, n_jobs=1)
    assert result.feature_names.tolist() == ["group"]
    assert np.isfinite(result.importances_mean).all()


def test_landmark_last_string_feature_is_encoded():
    n = 30
    df = pl.DataFrame({"id": np.repeat(np.arange(n), 2),
                       "start": np.tile([0.0, 1.0], n),
                       "stop": np.tile([1.0, 2.0], n),
                       "event": np.tile([False, True], n),
                       "status": np.repeat(np.array(["A", "B", "C"])[np.arange(n) % 3], 2),
                       "marker": np.repeat(np.arange(n) / n, 2)})
    model = LandmarkSurvivalForest(
        horizon=1.0, history_features=["status", "marker"], landmarks=[0.0, 1.0],
        forest=SurvivalForestTV(n_estimators=4, min_ids_leaf=2,
                                min_events_leaf=1, random_state=0),
    ).fit(df)
    assert model.forest_.n_encoded_features_ == 5  # 3 levels, marker, landmark time
    assert model.predict_risk(df, s=0.0).height == n
    result = permutation_importance(model, df, features=["status"], scoring="pe",
                                    windows=1, n_repeats=2, n_bootstrap=0, random_state=3)
    assert result.feature_names.tolist() == ["status"]
    np.testing.assert_array_equal(result.units[0], [0, 1, 2])
    loss = permutation_importance(model, df, features=["status"], scoring="brier",
                                  n_repeats=2, n_bootstrap=0, random_state=3)
    assert np.isfinite(loss.importances_mean).all()
    effect = hazard_effect(model, df, None, feature="status", windows=1)
    assert effect["values"].tolist() == ["A", "B", "C"]
    loco = drop_column_importance(model, df, cv=3, features=["status"],
                                  windows=1, n_seeds=1, n_jobs=1)
    assert np.isfinite(loco.importances_mean).all()


def test_competing_risks_categorical_inspection():
    n = 72
    X = pd.DataFrame({"group": pd.Categorical(["A", "B", "C"] * (n // 3)),
                      "marker": np.arange(n) / n})
    y = make_competing_risks_y(1 + np.arange(n) % 8,
                               np.where(np.arange(n) % 4 == 0, 0, 1 + np.arange(n) % 2))
    model = CompetingRisksForestTV(n_estimators=8, min_ids_leaf=2,
                                   min_events_leaf=1, random_state=2).fit(X, y)
    imp = permutation_importance(model, X, y, features=["group"], cause=1,
                                 windows=2, n_repeats=2, n_bootstrap=0, random_state=3)
    np.testing.assert_array_equal(imp.units[0], [0, 1, 2])
    effect = hazard_effect(model, X, y, feature="group", cause=1, windows=2)
    assert effect.hazard.shape == (3, 2)
    intervals = make_survival_y(np.full(n, 8.0), np.zeros(n, dtype=bool))
    path = path_effect(model, X, intervals, np.arange(n), feature="group", cause=1,
                       delta=lambda values, start: np.full(values.shape, "B"),
                       from_time=2.0, horizons=[4.0, 8.0])
    assert path.per_subject.shape == (n, 2)
