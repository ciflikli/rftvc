"""S4: landmark data building, look-ahead guards, and the landmark super-model."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rftvc import LandmarkSurvivalForest, SurvivalForestTV, landmark_features, make_landmark_data
from rftvc.landmark import _raw_groups
from tests.fixtures.pbcseq import pbcseq_counting_process, visits_to_counting_process


def _frame(rows):
    return pl.DataFrame(rows, schema=["id", "start", "stop", "event", "z"], orient="row")


# Hand-built subjects (stop = U on the last row):
#   a: enters 0, z changes at 2, event at 5
#   b: enters 1 (delayed), censored at 3
#   c: enters 0, event exactly at 4
HAND = _frame(
    [
        ("a", 0.0, 2.0, False, 1.0),
        ("a", 2.0, 5.0, True, 3.0),
        ("b", 1.0, 3.0, False, 7.0),
        ("c", 0.0, 4.0, True, 2.0),
    ]
)


def _rows(data):
    return {
        (i, s): (tuple(x[:-1]), stop, ev)
        for i, s, x, stop, ev in zip(data.ids, data.s, data.X, data.y["stop"], data.y["event"])
    }


def test_row_formulas_on_hand_built_cases():
    data = make_landmark_data(HAND, horizon=2.0, landmarks=[0.0, 1.0, 2.0, 3.0, 4.0], history_features=["z"])
    got = _rows(data)
    expected = {
        # s = 0: a and c at risk (b has not entered)
        ("a", 0.0): ((1.0,), 2.0, False),
        ("c", 0.0): ((2.0,), 2.0, False),
        # s = 1: b enters at 1 (entered iff first start <= s)
        ("a", 1.0): ((1.0,), 2.0, False),
        ("b", 1.0): ((7.0,), 2.0, False),  # censored at 3 = s + w: no event
        ("c", 1.0): ((2.0,), 2.0, False),
        # s = 2: a's z switches to 3 (row starting at 2 is known at 2); c's event at 4 = s + w counts
        ("a", 2.0): ((3.0,), 2.0, False),
        ("b", 2.0): ((7.0,), 1.0, False),  # censored before s + w
        ("c", 2.0): ((2.0,), 2.0, True),
        # s = 3: b censored at 3 is not at risk (U > s required); a's event at 5 = s + w
        ("a", 3.0): ((3.0,), 2.0, True),
        ("c", 3.0): ((2.0,), 1.0, True),
        # s = 4: c's event at exactly 4 means it is not event-free at 4
        ("a", 4.0): ((3.0,), 1.0, True),
    }
    assert got == expected
    assert data.feature_names == ["z", "landmark"]
    np.testing.assert_array_equal(data.X[:, -1], data.s)
    np.testing.assert_array_equal(data.groups, data.ids)


def test_aggregations_use_only_rows_known_at_landmark():
    df = _frame([("a", 0.0, 1.0, False, 1.0), ("a", 1.0, 2.0, False, 5.0), ("a", 2.0, 9.0, True, 100.0)])
    data = make_landmark_data(
        df,
        horizon=1.0,
        landmarks=[1.5],
        history_features=["z", ("z", "mean"), ("z", "max"), ("z", "first"), ("z", "count")],
    )
    np.testing.assert_allclose(data.X[0], [5.0, 3.0, 5.0, 1.0, 2.0, 1.5])


def test_slope_and_std_aggregations():
    df = _frame(
        [("a", 0.0, 1.0, False, 1.0), ("a", 1.0, 2.0, False, 3.0), ("a", 2.0, 5.0, True, 5.0)]
    )
    data = make_landmark_data(df, horizon=1.0, landmarks=[2.5], history_features=[("z", "slope"), ("z", "std")])
    slope, std = np.polyfit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], 1)[0], np.std([1.0, 3.0, 5.0], ddof=1)
    np.testing.assert_allclose(data.X[0, :2], [slope, std])


def test_slope_nan_with_one_distinct_time_std_nan_with_one_row():
    one_row = _frame([("a", 0.0, 2.0, True, 1.0)])  # only one row known at s: one distinct time
    data = make_landmark_data(one_row, horizon=1.0, landmarks=[0.5], history_features=[("z", "slope"), ("z", "std")])
    assert np.isnan(data.X[0, 0]) and np.isnan(data.X[0, 1])


def test_slope_uses_measured_at_when_given_else_start():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a"],
            "start": [0.0, 1.0, 2.0],
            "stop": [1.0, 2.0, 5.0],
            "event": [False, False, True],
            "z": [1.0, 3.0, 5.0],
            "m": [-2.0, -1.0, 1.0],  # a different (but valid, m <= start) clock
        }
    )
    by_start = make_landmark_data(df, horizon=1.0, landmarks=[2.5], history_features=[("z", "slope")])
    by_measured = make_landmark_data(
        df, horizon=1.0, landmarks=[2.5], history_features=[("z", "slope")], measured_at="m"
    )
    np.testing.assert_allclose(by_start.X[0, 0], 2.0)  # z = 1 + 2 * start
    np.testing.assert_allclose(by_measured.X[0, 0], 9.0 / 7.0)
    assert not np.isclose(by_start.X[0, 0], by_measured.X[0, 0])


def test_raw_groups_worked_example():
    assert _raw_groups(["z", ("z", "mean"), ("z", "slope"), "x"]) == {"z": ["z", "z_mean", "z_slope"], "x": ["x"]}


def test_raw_groups_column_and_name_order_preserved():
    got = _raw_groups([("b", "mean"), "a", ("b", "std"), ("a", "max")])
    assert list(got) == ["b", "a"]
    assert got == {"b": ["b_mean", "b_std"], "a": ["a", "a_max"]}


def test_raw_groups_propagates_empty_history_features_error():
    with pytest.raises(ValueError, match="at least one feature"):
        _raw_groups([])


def test_rows_are_known_by_start_not_by_measurement_time():
    df = pl.DataFrame(
        {"id": [1, 1], "start": [0.0, 1.0], "stop": [1.0, 3.0], "event": [False, True], "z": [1.0, 9.0], "m": [0.0, 1.0]}
    )
    base = make_landmark_data(df, horizon=1.0, landmarks=[1.0], history_features=["z"])
    assert base.X[0, 0] == 9.0  # the row starting at 1 is known at 1
    # Measured at 0.5, but the row (1, 3] exists only if the subject survives to 1:
    # at s = 0.75 it must not be used.
    early = make_landmark_data(df.with_columns(pl.col("m") - 0.5), horizon=1.0, landmarks=[0.75], history_features=["z"], measured_at="m")
    assert early.X[0, 0] == 1.0
    with pytest.raises(ValueError, match="look-ahead"):
        make_landmark_data(df.with_columns(pl.col("m") + 0.5), horizon=1.0, landmarks=[1.0], history_features=["z"], measured_at="m")


@pytest.mark.parametrize("feature", ["stop", "event", ("stop", "max"), ("event", "sum"), "id", ("stop", "slope"), ("event", "std")])
def test_history_features_on_outcome_columns_raise(feature):
    with pytest.raises(ValueError, match="look ahead"):
        make_landmark_data(HAND, horizon=1.0, landmarks=[1.0], history_features=[feature])


def test_invalid_arguments():
    with pytest.raises(ValueError, match="exactly one"):
        make_landmark_data(HAND, horizon=1.0, history_features=["z"])
    with pytest.raises(ValueError, match="horizon"):
        make_landmark_data(HAND, horizon=0.0, landmarks=[1.0], history_features=["z"])
    with pytest.raises(ValueError, match="aggregation"):
        make_landmark_data(HAND, horizon=1.0, landmarks=[1.0], history_features=[("z", "median")])
    with pytest.raises(ValueError, match="missing"):
        make_landmark_data(HAND.drop("event"), horizon=1.0, landmarks=[1.0], history_features=["z"])


def test_step_grid_and_pandas_input():
    a = make_landmark_data(HAND, horizon=1.0, step=1.0, history_features=["z"])
    hand_pd = pd.DataFrame({c: HAND[c].to_list() for c in HAND.columns})
    b = make_landmark_data(hand_pd, horizon=1.0, landmarks=[0.0, 1.0, 2.0, 3.0, 4.0], history_features=["z"])
    assert _rows(a) == _rows(b)


# ---------------------------------------------------------------- independent pandas reference


def _reference(df, landmarks, horizon, column):
    rows = []
    for s in landmarks:
        for i, g in df.groupby("id", sort=False):
            g = g.sort_values("start")
            entry, U, ev = g["start"].min(), g["stop"].max(), bool(g["event"].any())
            if not (entry <= s < U):
                continue
            known = g[g["start"] <= s]
            rows.append((i, float(s), float(known[column].iloc[-1]), float(known[column].mean()), min(U, s + horizon) - s, ev and U <= s + horizon))
    return sorted(rows)


def _random_cp(seed, n_ids):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_ids):
        t = float(rng.integers(0, 3))
        k = int(rng.integers(1, 4))
        for j in range(k):
            dur = float(rng.integers(1, 4))
            out.append((i, t, t + dur, j == k - 1 and rng.random() < 0.6, float(rng.normal())))
            t += dur
    df = pd.DataFrame(out, columns=["id", "start", "stop", "event", "z"])
    # Measurement lags of 0-1.5 (never after start); they must not change what is known.
    df["m"] = df["start"] - np.random.default_rng(seed + 1).choice([0.0, 0.5, 1.5], size=len(df))
    return df


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n_ids=st.integers(2, 25),
    horizon=st.sampled_from([0.5, 1.0, 2.5, 4.0]),
    measured=st.booleans(),
)
def test_matches_pandas_reference(seed, n_ids, horizon, measured):
    df = _random_cp(seed, n_ids)
    landmarks = [0.0, 1.0, 1.5, 3.0, 5.0]
    ref = _reference(df, landmarks, horizon, "z")
    kwargs = dict(horizon=horizon, landmarks=landmarks, history_features=["z", ("z", "mean")], measured_at="m" if measured else None)
    if not ref:
        with pytest.raises(ValueError):
            make_landmark_data(df, **kwargs)
        return
    data = make_landmark_data(df, **kwargs)
    got = sorted(
        (int(i), float(s), float(x[0]), float(x[1]), float(t), bool(e))
        for i, s, x, t, e in zip(data.ids, data.s, data.X, data.y["stop"], data.y["event"])
    )
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g[:2] == r[:2] and g[5] == r[5]
        np.testing.assert_allclose(g[2:5], r[2:5], rtol=1e-12)


# ---------------------------------------------------------------- landmark super-model on pbcseq


FEATURES = ["log_bili", "albumin", "protime", "edema", "ascites", "age", ("log_bili", "max")]


def test_pbcseq_end_to_end():
    df = pbcseq_counting_process()
    model = LandmarkSurvivalForest(
        horizon=3 * 365.25,
        landmarks=[0.0, 365.25, 730.5, 1095.75],
        history_features=FEATURES,
        forest=SurvivalForestTV(n_estimators=50, random_state=0),
    ).fit(df)
    assert model.forest_.n_ids_ == df["id"].n_unique()
    for s in (0.0, 730.5):
        out = model.predict_risk(df, s)
        assert out.height > 0
        assert out["risk"].is_between(0.0, 1.0).all()


def test_resampling_is_by_subject_not_landmark_row():
    df = pbcseq_counting_process()
    model = LandmarkSurvivalForest(
        horizon=365.25, step=365.25, history_features=["log_bili"], forest=SurvivalForestTV(n_estimators=3, random_state=0)
    ).fit(df)
    assert model.forest_.n_ids_ == df["id"].n_unique()
    assert model.n_rows_ > model.forest_.n_ids_


def test_prediction_excludes_subjects_not_event_free_or_not_observed():
    model = LandmarkSurvivalForest(
        horizon=2.0, landmarks=[0.0, 1.0, 2.0], history_features=["z"], forest=SurvivalForestTV(n_estimators=5, min_ids_leaf=1, min_events_leaf=1, random_state=0)
    ).fit(HAND)
    ids, _ = model.predict_survival_function(HAND, 4.0, [1.0])
    assert list(ids) == ["a"]  # c had its event at 4; b left at 3
    no_event = HAND.drop("event")
    ids, _ = model.predict_survival_function(no_event, 4.0, [1.0])
    assert list(ids) == ["a", "c"]  # without outcomes, c is observed through 4
    with pytest.raises(ValueError, match="horizon"):
        model.predict_survival_function(HAND, 1.0, [3.0])


def test_landmark_features_match_training_rows():
    data = make_landmark_data(HAND, horizon=2.0, landmarks=[2.0], history_features=["z"])
    ids, X = landmark_features(HAND, 2.0, history_features=["z"], event="event")
    assert list(ids) == list(data.ids)
    np.testing.assert_array_equal(X, data.X)


def test_censored_exactly_at_landmark_is_predicted_but_not_trained():
    df = _frame([("a", 0.0, 4.0, True, 1.0), ("b", 0.0, 2.0, False, 5.0), ("c", 0.0, 6.0, True, 2.0)])
    data = make_landmark_data(df, horizon=1.0, landmarks=[2.0], history_features=["z"])
    assert sorted(data.ids) == ["a", "c"]  # b's follow-up ends at s: zero-length row
    ids, _ = landmark_features(df, 2.0, history_features=["z"], event="event")
    assert sorted(ids) == ["a", "b", "c"]  # b is event-free at s: in the prediction population


@pytest.mark.parametrize(
    ("rows", "expected_last"),
    [
        # final visit exactly at futime: dropped; the previous interval ends at futime with the death
        ([(1, 10, 2, 0), (1, 10, 2, 4), (1, 10, 2, 10)], (4.0, 10.0, True)),
        # visit after futime: ignored, not used to extend follow-up
        ([(1, 10, 2, 0), (1, 10, 2, 12)], (0.0, 10.0, True)),
        # transplant is censoring
        ([(1, 10, 1, 0), (1, 10, 1, 5)], (5.0, 10.0, False)),
    ],
)
def test_pbcseq_conversion_edge_cases(rows, expected_last):
    d = pl.DataFrame(rows, schema=["id", "futime", "status", "day"], orient="row")
    cp = visits_to_counting_process(d)
    last = cp.sort("start").tail(1)
    assert (last["start"][0], last["stop"][0], last["event"][0]) == expected_last
    assert cp["event"].sum() == int(expected_last[2])
    assert (cp["stop"] <= 10).all()
