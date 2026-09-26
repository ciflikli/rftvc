"""S10: id × time-block resampling (resample_unit="block") and buffered block OOB."""

import warnings

import numpy as np
import polars as pl
import pytest

from rftvc import LandmarkSurvivalForest, SurvivalForestTV, make_landmark_data, make_survival_y
from rftvc._blocks import split_at_blocks
from tests.ref.logrank_ref import nelson_aalen_ref
from tests.test_tvc import _cp_data

# Small forests leave some rows with no qualifying OOB tree; that warning is tested in test_oob.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*no tree whose bag leaves out:UserWarning")


def _block(**kw):
    return SurvivalForestTV(**{"resample_unit": "block", "block_length": 1.0, "min_ids_leaf": 3, **kw})


# ------------------------------------------------------------------ splitting rows at block boundaries


def test_split_hand_cases():
    start = np.array([0.0, 1.0, -0.5, 2.25])
    stop = np.array([2.5, 2.0, 0.5, 2.75])
    event = np.array([True, False, True, True])
    row, block, s, t, e = split_at_blocks(start, stop, event, 1.0)
    np.testing.assert_array_equal(row, [0, 0, 0, 1, 2, 2, 3])
    np.testing.assert_array_equal(block, [0, 1, 2, 1, -1, 0, 2])
    np.testing.assert_array_equal(s, [0.0, 1.0, 2.0, 1.0, -0.5, 0.0, 2.25])
    np.testing.assert_array_equal(t, [1.0, 2.0, 2.5, 2.0, 0.0, 0.5, 2.75])
    np.testing.assert_array_equal(e, [False, False, True, False, False, True, True])


@pytest.mark.parametrize("length", [0.1, 0.3, 0.7, 2.5])
def test_pieces_tile_rows_and_lie_in_their_block(length):
    _, y, _ = _cp_data(80, seed=1, round_to=2)
    start, stop, event = (np.ascontiguousarray(y[k]) for k in ("start", "stop", "event"))
    row, block, s, t, e = split_at_blocks(start, stop, event, length)
    assert np.all(s < t)
    assert np.all((block * length <= s) & (t <= (block + 1) * length))
    for r in range(start.size):
        p = np.flatnonzero(row == r)
        assert s[p[0]] == start[r] and t[p[-1]] == stop[r]
        np.testing.assert_array_equal(s[p[1:]], t[p[:-1]])  # contiguous
        assert e[p].sum() == event[r] and (not event[r] or e[p[-1]])
        assert np.all(np.diff(block[p]) == 1)


def test_splitting_leaves_the_nelson_aalen_estimate_unchanged():
    X, y, ids = _cp_data(60, seed=2)
    m = _block(n_estimators=1, max_depth=0, max_samples=1.0, block_length=0.4).fit(X, y, ids)
    assert m.n_units_ > m.n_ids_  # rows were split
    times, cumhaz = map(np.asarray, m.forest_.leaf_profile(0, 0))
    ref_t, ref_c = nelson_aalen_ref(y["start"], y["stop"], y["event"])
    np.testing.assert_allclose(times, ref_t)
    np.testing.assert_allclose(cumhaz, ref_c, atol=1e-12)


# ------------------------------------------------------------------ oracles against other resampling units


@pytest.mark.parametrize("aggregate", ["hazard", "survival"])
def test_one_block_per_id_equals_id_resampling(aggregate):
    X, y, ids = _cp_data(50, seed=3)
    kw = dict(n_estimators=12, min_ids_leaf=3, aggregate=aggregate, oob_score=True, random_state=4)
    by_id = SurvivalForestTV(**kw).fit(X, y, ids)
    by_block = SurvivalForestTV(resample_unit="block", block_length=1 + y["stop"].max(), **kw).fit(X, y, ids)
    assert by_block.n_units_ == by_id.n_ids_
    np.testing.assert_array_equal(by_block.predict_cumulative_hazard(X), by_id.predict_cumulative_hazard(X))
    np.testing.assert_array_equal(by_block.oob_prediction_, by_id.oob_prediction_)
    np.testing.assert_array_equal(by_block.oob_n_trees_, by_id.oob_n_trees_)
    assert by_block.oob_score_ == by_id.oob_score_


def test_one_block_per_row_equals_row_units():
    # Rows (k, k+1] of 30 ids: each row is exactly one block, so nothing is split.
    rng = np.random.default_rng(5)
    n_ids, k = 30, 4
    start = np.tile(np.arange(k, dtype=float), n_ids)
    event = np.zeros(n_ids * k, bool)
    event[k - 1 :: k] = rng.random(n_ids) < 0.7
    y = make_survival_y(start + 1, event, start=start)
    X = rng.normal(size=(n_ids * k, 3))
    ids = np.repeat(np.arange(n_ids), k)
    kw = dict(n_estimators=10, min_ids_leaf=3, oob_score=True, random_state=1)
    rows = SurvivalForestTV(**kw).fit(X, y)  # ids=None: every row its own unit
    blocks = _block(oob_buffer=0, **kw).fit(X, y, ids)
    assert blocks.n_units_ == n_ids * k
    np.testing.assert_array_equal(blocks.predict_cumulative_hazard(X), rows.predict_cumulative_hazard(X))
    np.testing.assert_array_equal(blocks.oob_prediction_, rows.oob_prediction_)


# ------------------------------------------------------------------ buffered block OOB vs a reference


def _reference_oob(forest, X, y, ids, length, buffer, aggregate):
    """OOB mortality and tree counts from first principles.

    Units are (id, block) pairs numbered by first appearance over the rows' blocks
    in row order; a row needs every unit of its id within [first block - buffer,
    last block + buffer] out of the bag.
    """
    covered = []
    for s, t in zip(y["start"], y["stop"]):
        k = int(np.floor(s / length))
        blocks = []
        while k * length < t:
            if (k + 1) * length > s:
                blocks.append(k)
            k += 1
        covered.append(blocks)
    label = {}
    for i, blocks in zip(ids, covered):
        for k in blocks:
            label.setdefault((i, k), len(label))
    core, times = forest.forest_, forest.event_times_
    bags = [set(core.in_bag_ids(b)) for b in range(core.n_trees)]
    leaves = forest.apply(X)
    out, n_trees = np.full(len(X), np.nan), np.zeros(len(X), int)
    for r, (i, blocks) in enumerate(zip(ids, covered)):
        need = {label[(i, k)] for k in range(blocks[0] - buffer, blocks[-1] + buffer + 1) if (i, k) in label}
        H = []
        for b in range(core.n_trees):
            if need & bags[b]:
                continue
            lt, cumhaz = core.leaf_profile(b, int(leaves[r, b]))
            H.append(np.r_[0.0, cumhaz][np.searchsorted(lt, times, side="right")])
        n_trees[r] = len(H)
        if H:
            H = np.array(H)
            out[r] = (H.mean(0) if aggregate == "hazard" else -np.log(np.exp(-H).mean(0))).sum()
    return out, n_trees


@pytest.mark.parametrize("buffer", [0, 1, 2])
@pytest.mark.parametrize("aggregate", ["hazard", "survival"])
def test_buffered_oob_matches_reference(buffer, aggregate):
    X, y, ids = _cp_data(40, seed=6, max_rows=5)
    length = 0.6
    f = _block(
        n_estimators=40, block_length=length, oob_buffer=buffer, aggregate=aggregate, oob_score=True, random_state=2
    )
    f.fit(X, y, ids)
    ref, n_ref = _reference_oob(f, X, y, ids, length, buffer, aggregate)
    np.testing.assert_array_equal(f.oob_n_trees_, n_ref)
    np.testing.assert_allclose(f.oob_prediction_, ref, rtol=1e-12)


def test_wider_buffer_uses_fewer_trees():
    X, y, ids = _cp_data(40, seed=7, max_rows=5)
    n = [
        _block(n_estimators=30, block_length=0.5, oob_buffer=h, oob_score=True, random_state=0)
        .fit(X, y, ids)
        .oob_n_trees_
        for h in (0, 1)
    ]
    assert np.all(n[1] <= n[0]) and np.any(n[1] < n[0])


# ------------------------------------------------------------------ layouts, block_time, coarsening


def test_split_id_segments_share_their_ids_blocks():
    # One id with a gap inside block 0: both segments are the same unit.
    X = np.array([[0.0], [1.0], [2.0]])
    y = make_survival_y(np.array([0.3, 0.9, 1.5]), np.array([False, False, True]), start=np.array([0.0, 0.5, 0.9]))
    m = _block(n_estimators=1, min_ids_leaf=1, min_events_leaf=1).fit(X, y, np.zeros(3), gap_policy="split_id")
    assert (m.n_ids_, m.n_units_) == (1, 2)  # blocks 0 and 1 of the one id


def test_block_time_forms_blocks_without_splitting():
    X, y, ids = _cp_data(30, seed=8)
    calendar = np.floor(np.arange(len(ids)) / 2.0)  # an unrelated clock
    m = _block(n_estimators=3, block_length=2.0).fit(X, y, ids, block_time=calendar)
    expected = len({(i, int(np.floor(c / 2.0))) for i, c in zip(ids, calendar)})
    assert m.n_units_ == expected


def test_stacked_layout_needs_block_time():
    X, y, ids = _cp_data(20, seed=9)
    with pytest.raises(ValueError, match="needs block_time"):
        _block().fit(X, y, ids, layout="stacked")


def test_landmark_forest_blocks_on_landmark_time():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(40):
        u = float(rng.uniform(1, 8))
        rows += [(i, 0.0, u / 2, False, rng.normal()), (i, u / 2, u, bool(rng.random() < 0.8), rng.normal())]
    df = pl.DataFrame(rows, schema=["id", "start", "stop", "event", "z"], orient="row")
    landmarks = [0.0, 1.0, 2.0, 3.0, 4.0]
    forest = _block(n_estimators=5, block_length=2.0, min_ids_leaf=2, min_events_leaf=1)
    lm = LandmarkSurvivalForest(horizon=2.0, landmarks=landmarks, history_features=["z"], forest=forest).fit(df)
    data = make_landmark_data(df, horizon=2.0, landmarks=landmarks, history_features=["z"])
    assert lm.forest_.n_units_ == len({(i, int(s // 2.0)) for i, s in zip(data.ids, data.s)})
    assert lm.forest_.n_units_ > lm.forest_.n_ids_


def test_landmark_forest_warns_when_block_oob_would_leak():
    rng = np.random.default_rng(4)
    rows = [(i, 0.0, float(u), bool(rng.random() < 0.8), rng.normal()) for i, u in enumerate(rng.uniform(1, 8, 40))]
    df = pl.DataFrame(rows, schema=["id", "start", "stop", "event", "z"], orient="row")

    def fit(length, buffer):
        forest = _block(n_estimators=5, block_length=length, oob_buffer=buffer, oob_score=True, min_ids_leaf=2)
        lm = LandmarkSurvivalForest(horizon=2.0, landmarks=[0.0, 1.0, 2.0, 3.0], history_features=["z"], forest=forest)
        return lm.fit(df)

    with pytest.warns(UserWarning, match="block OOB leaks"):
        fit(1.0, 1)  # landmarks 2 blocks apart are 1 < horizon apart
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*no tree whose bag leaves out")
        fit(2.0, 1)
        fit(1.0, 2)


def test_coarsening_then_blocks_keeps_oob_on_original_rows():
    X, y, ids = _cp_data(50, seed=10, round_to=2)
    m = _block(n_estimators=20, block_length=0.5, ntime=8, oob_score=True, random_state=0)
    m.fit(X, y, ids)
    assert m.n_coarsen_dropped_rows_ > 0
    assert m.oob_prediction_.shape == m.oob_n_trees_.shape == (len(X),)
    dropped = np.isnan(m.oob_prediction_) & (m.oob_n_trees_ == 0)
    assert dropped.sum() >= m.n_coarsen_dropped_rows_


def test_unit_counts_resolve_from_blocks():
    X, y, ids = _cp_data(60, seed=11, max_rows=6)
    m = _block(n_estimators=2, block_length=0.25, min_ids_leaf="auto", max_samples=0.5).fit(X, y, ids)
    assert m.n_units_ > m.n_ids_
    assert m.min_ids_leaf_ == max(15, int(np.sqrt(m.n_units_)))
    assert m.n_draw_ == round(0.5 * m.n_units_)


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (dict(resample_unit="block"), "positive finite block_length"),
        (dict(resample_unit="block", block_length=0.0), "positive finite block_length"),
        (dict(resample_unit="block", block_length=np.inf), "positive finite block_length"),
        (dict(resample_unit="block", block_length=True), "positive finite block_length"),
        (dict(block_length=1.0), "only used with resample_unit='block'"),
        (dict(resample_unit="block", block_length=1.0, oob_buffer=-1), "oob_buffer"),
        (dict(resample_unit="row"), "resample_unit must be"),
        (dict(resample_unit="block", block_length=1e-6), "splits a row into"),
        (dict(resample_unit="block", block_length=1e-16), "below 2\\*\\*52"),
        (dict(resample_unit="block", block_length=1.0, oob_buffer=2**63), "oob_buffer must be <="),
    ],
)
def test_invalid_block_parameters(params, match):
    X, y, ids = _cp_data(10, seed=12)
    with pytest.raises(ValueError, match=match):
        SurvivalForestTV(n_estimators=1, **params).fit(X, y, ids)


@pytest.mark.parametrize("scale", [1e20, -1e20])
def test_huge_times_are_rejected_not_split_into_empty_pieces(scale):
    start = np.array([scale])
    stop = np.nextafter(start, np.inf)
    with pytest.raises(ValueError, match="below 2"):
        split_at_blocks(start, stop, np.array([True]), 1000.0)


def test_block_time_is_validated():
    X, y, ids = _cp_data(10, seed=13)
    with pytest.raises(ValueError, match="only used with resample_unit='block'"):
        SurvivalForestTV(n_estimators=1).fit(X, y, ids, block_time=np.zeros(len(X)))
    with pytest.raises(ValueError, match="block_time must be finite"):
        _block(n_estimators=1).fit(X, y, ids, block_time=np.full(len(X), np.nan))
    with pytest.raises(ValueError, match="block_time must be finite"):
        _block(n_estimators=1).fit(X, y, ids, block_time=np.zeros(3))
    with pytest.raises(ValueError, match="below 2"):  # would all collapse to one int64 block
        _block(n_estimators=1).fit(X, y, ids, block_time=np.full(len(X), 1e307))
