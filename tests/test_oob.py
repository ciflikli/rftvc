"""S5: id-level out-of-bag predictions and score."""

import numpy as np
import pytest

from rftvc import SurvivalForestTV, make_survival_y
from rftvc.metrics import concordance_index_cp
from tests.test_tvc import _cp_data


def _manual_oob_mortality(forest, X, ids):
    """Mean over trees whose bag lacks the row's id of sum_k Λ_b(t_k), from leaf profiles."""
    core, times = forest.forest_, forest.event_times_
    leaves = forest.apply(X)
    _, group = np.unique(ids, return_inverse=True)
    out = np.full(X.shape[0], np.nan)
    for r in range(X.shape[0]):
        vals = []
        for b in range(core.n_trees):
            if group[r] in set(core.in_bag_ids(b)):
                continue
            lt, _, _, cumhaz = core.leaf_profile(b, int(leaves[r, b]))
            pos = np.searchsorted(lt, times, side="right")
            vals.append(np.where(pos > 0, np.r_[0.0, cumhaz][pos], 0.0).sum())
        if vals:
            out[r] = np.mean(vals)
    return out


def test_oob_prediction_uses_only_trees_without_the_id():
    X, y, ids = _cp_data(40, seed=3)
    f = SurvivalForestTV(n_estimators=15, min_ids_leaf=3, oob_score=True, random_state=0).fit(X, y, ids)
    np.testing.assert_allclose(f.oob_prediction_, _manual_oob_mortality(f, X, ids), rtol=1e-12)
    assert f.oob_score_ == pytest.approx(concordance_index_cp(y, f.oob_prediction_, ids=ids), abs=1e-15)


def test_oob_score_tracks_signal():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    t = rng.exponential(np.exp(-1.0 * X[:, 0]))
    c = rng.exponential(2.0, size=n)
    y = make_survival_y(np.minimum(t, c), t <= c)
    f = SurvivalForestTV(n_estimators=100, oob_score=True, random_state=0).fit(X, y)
    assert 0.65 < f.oob_score_ < 0.85
    noise = SurvivalForestTV(n_estimators=100, oob_score=True, random_state=0).fit(X[:, 1:], y)
    assert abs(noise.oob_score_ - 0.5) < 0.08


def test_oob_survival_aggregation_and_no_oob_error():
    X, y, ids = _cp_data(30, seed=5)
    f = SurvivalForestTV(n_estimators=10, min_ids_leaf=3, aggregate="survival", oob_score=True, random_state=1)
    assert np.isfinite(f.fit(X, y, ids).oob_score_)
    with pytest.raises(ValueError, match="out of bag"):
        SurvivalForestTV(n_estimators=5, max_samples=1.0, oob_score=True, min_ids_leaf=3).fit(X, y, ids)


def test_oob_warns_for_ids_in_every_bag():
    X, y, ids = _cp_data(30, seed=6)
    f = SurvivalForestTV(n_estimators=2, max_samples=0.9, min_ids_leaf=3, oob_score=True, random_state=0)
    with pytest.warns(UserWarning, match="every bag"):
        f.fit(X, y, ids)
    assert np.isnan(f.oob_prediction_).any() and np.isfinite(f.oob_score_)


def test_oob_rejects_split_segments():
    X = np.zeros((3, 1))
    y = make_survival_y([1.0, 3.0, 2.0], [False, True, True], start=[0.0, 2.0, 0.0])  # id a has a gap
    with pytest.raises(ValueError, match="split_id"):
        SurvivalForestTV(n_estimators=2, min_ids_leaf=1, min_events_leaf=1, oob_score=True).fit(
            X, y, ["a", "a", "b"], gap_policy="split_id"
        )
