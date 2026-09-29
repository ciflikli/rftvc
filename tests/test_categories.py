"""The fitted category vocabulary is shared by all prediction routes."""

import pickle

import numpy as np
import pandas as pd
import pytest

from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y, make_survival_y


@pytest.mark.parametrize("cls", [SurvivalForestTV, CompetingRisksForestTV])
def test_mixed_dataframe_matches_explicit_indicators(cls):
    n = 72
    levels = np.array(["red", "blue", "green"])[np.arange(n) % 3]
    levels[5] = "red"
    numeric = np.arange(n, dtype=float) / n
    df = pd.DataFrame({"id": np.arange(n), "value": numeric, "group": pd.Categorical(levels)})
    df.loc[4, "group"] = None
    stop = 1 + (np.arange(n) % 9).astype(float)
    event = np.arange(n) % 4 != 0
    if cls is CompetingRisksForestTV:
        y = make_competing_risks_y(stop, np.where(event, 1 + np.arange(n) % 2, 0))
    else:
        y = make_survival_y(stop, event)
    model = cls(n_estimators=5, max_features=None, min_ids_leaf=2,
                min_events_leaf=1, random_state=3).fit(df, y, ids="id")
    d = model._rebuild_design(df, y)
    assert d.X.shape == (n, 4)
    assert model.n_features_in_ == 2
    assert model.n_encoded_features_ == 4
    assert list(model.export_tree(0).feature_names) == [
        "value", "group=blue", "group=green", "group=red"
    ]
    assert np.isnan(d.X[4, 1:]).all()
    assert np.array_equal(model.apply(df), model.apply(df.drop(columns="id")))
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.apply(df), model.apply(df))
    with pytest.raises(ValueError, match="unseen category.*yellow"):
        model.apply(df.assign(group=["yellow"] * n))
    with pytest.raises(ValueError, match="feature names"):
        model.apply(df[["group", "value"]])


def test_string_array_and_changed_training_category():
    X = np.array([["a"], ["b"]] * 20)
    y = make_survival_y(np.arange(40) % 8 + 1, np.ones(40, dtype=bool))
    model = SurvivalForestTV(n_estimators=2, min_ids_leaf=2,
                             min_events_leaf=1, random_state=0).fit(X, y)
    assert model.n_encoded_features_ == 2
    np.testing.assert_array_equal(model._rebuild_design(X, y).X[:2], [[1, 0], [0, 1]])
    changed = X.copy()
    changed[0, 0] = "b"
    with pytest.raises(ValueError, match="do not match"):
        model._rebuild_design(changed, y)
    renamed = np.where(X == "a", "c", "d")
    with pytest.raises(ValueError, match="do not match"):
        model._rebuild_design(renamed, y)


def test_numeric_pandas_category_is_not_treated_as_continuous():
    df = pd.DataFrame({"dose": pd.Categorical([1, 3, 8] * 15)})
    y = make_survival_y(np.arange(45) % 9 + 1, np.ones(45, dtype=bool))
    model = SurvivalForestTV(n_estimators=2, min_ids_leaf=2,
                             min_events_leaf=1, random_state=0).fit(df, y)
    assert model.n_features_in_ == 1
    assert model.n_encoded_features_ == 3
    np.testing.assert_array_equal(model._rebuild_design(df, y).X[0], [1, 0, 0])
