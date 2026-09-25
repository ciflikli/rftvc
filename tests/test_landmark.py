"""S4: landmark data building, look-ahead guards, and the landmark super-model."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rftvc import LandmarkSurvivalForest, SurvivalForestTV, landmark_features, make_landmark_data

PBCSEQ = Path(__file__).parent / "fixtures" / "pbcseq.csv"


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


def test_measured_at_controls_what_is_known():
    df = pl.DataFrame(
        {"id": [1, 1], "start": [0.0, 1.0], "stop": [1.0, 3.0], "event": [False, True], "z": [1.0, 9.0], "m": [0.0, 1.0]}
    )
    base = make_landmark_data(df, horizon=1.0, landmarks=[1.0], history_features=["z"])
    lagged = make_landmark_data(df.with_columns(pl.col("m") - 0.5), horizon=1.0, landmarks=[0.75], history_features=["z"], measured_at="m")
    assert base.X[0, 0] == 9.0  # the row starting at 1 is known at 1
    assert lagged.X[0, 0] == 9.0  # measured at 0.5 <= 0.75
    with pytest.raises(ValueError, match="look-ahead"):
        make_landmark_data(df.with_columns(pl.col("m") + 0.5), horizon=1.0, landmarks=[1.0], history_features=["z"], measured_at="m")


@pytest.mark.parametrize("feature", ["stop", "event", ("stop", "max"), ("event", "sum"), "id"])
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
    return pd.DataFrame(out, columns=["id", "start", "stop", "event", "z"])


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), n_ids=st.integers(2, 25), horizon=st.sampled_from([0.5, 1.0, 2.5, 4.0]))
def test_matches_pandas_reference(seed, n_ids, horizon):
    df = _random_cp(seed, n_ids)
    landmarks = [0.0, 1.0, 1.5, 3.0, 5.0]
    ref = _reference(df, landmarks, horizon, "z")
    if not ref:
        with pytest.raises(ValueError):
            make_landmark_data(df, horizon=horizon, landmarks=landmarks, history_features=["z", ("z", "mean")])
        return
    data = make_landmark_data(df, horizon=horizon, landmarks=landmarks, history_features=["z", ("z", "mean")])
    got = sorted(
        (int(i), float(s), float(x[0]), float(x[1]), float(t), bool(e))
        for i, s, x, t, e in zip(data.ids, data.s, data.X, data.y["stop"], data.y["event"])
    )
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g[:2] == r[:2] and g[5] == r[5]
        np.testing.assert_allclose(g[2:5], r[2:5], rtol=1e-12)


# ---------------------------------------------------------------- landmark super-model on pbcseq


def pbcseq_counting_process():
    """pbcseq visits -> counting process: covariates from visit j apply until the next visit."""
    d = pl.read_csv(PBCSEQ).sort(["id", "day"])
    d = d.with_columns(pl.col("day").shift(-1).over("id").alias("next_day"))
    d = d.with_columns(
        pl.col("day").cast(pl.Float64).alias("start"),
        pl.coalesce("next_day", "futime").cast(pl.Float64).alias("stop"),
        (pl.col("next_day").is_null() & (pl.col("status") == 2)).alias("event"),
    ).filter(pl.col("start") < pl.col("stop"))
    return d.with_columns(pl.col("bili").log().alias("log_bili"))


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
    # Discrimination at s = 1 year on the training data should clearly beat chance.
    ids, S = model.predict_survival_function(df, 365.25, [3 * 365.25])
    lm = make_landmark_data(df, horizon=3 * 365.25, landmarks=[365.25], history_features=FEATURES)
    order = {i: k for k, i in enumerate(ids)}
    risk = 1 - S[[order[i] for i in lm.ids], 0]
    ev, t = lm.y["event"], lm.y["stop"]
    conc = tot = 0.0
    for i in np.flatnonzero(ev):
        later = t > t[i]
        tot += later.sum()
        conc += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    assert conc / tot > 0.75


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
