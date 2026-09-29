"""Single-tree parity with scikit-survival's ``SurvivalTree``.

With integer features of at most 12 levels, rftvc's histogram bins are lossless,
so a tree grown on all rows with every feature tried at each node should make
the same splits as sksurv's exact-value search. This checks the leaf partition
and every leaf's Nelson-Aalen cumulative hazard, with and without tied times.
It is the exact counterpart of the statistical forest parity in
``bench/parity_suite.py``.
"""

import numpy as np
import pytest
from sksurv.tree import SurvivalTree
from sksurv.util import Surv

from rftvc import SurvivalForestTV, make_survival_y


def _data(seed, n=400, p=4, ties=False):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 12, (n, p)).astype(float)
    t = rng.exponential(np.exp(-0.15 * X[:, 0] + 0.1 * (X[:, 1] > 5)))
    if ties:
        t = np.round(t, 2) + 0.01
    return X, t, rng.random(n) < 0.7


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("depth", [None, 3])
@pytest.mark.parametrize("seed", range(4))
def test_single_tree_matches_sksurv(seed, depth, ties):
    leaf = 10
    X, t, e = _data(seed, ties=ties)
    ref = SurvivalTree(
        min_samples_leaf=leaf, min_samples_split=2 * leaf, max_depth=depth, random_state=0
    ).fit(X, Surv.from_arrays(e, t))
    ours = SurvivalForestTV(
        n_estimators=1, max_samples=1.0, max_features=None, min_ids_leaf=leaf,
        min_events_leaf=1, max_depth=depth, random_state=0,
    ).fit(X, make_survival_y(t, e))

    a = ref.apply(X.astype(np.float32))
    b = ours.apply(X)[:, 0]
    # identical partition: a bijection between the two leaf labelings
    assert len(set(zip(a, b))) == len(set(a)) == len(set(b))

    times = ref.unique_times_
    chf_ref = ref.predict_cumulative_hazard_function(X.astype(np.float32), return_array=True)
    chf = ours.predict_cumulative_hazard(X, times)
    np.testing.assert_allclose(chf, chf_ref, rtol=0, atol=1e-9)
