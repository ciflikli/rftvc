"""Forest-level statistical parity with scikit-survival's RSF (fast subset).

The full suite is ``bench/parity_suite.py``. Here: two small datasets, one
5-fold repeat, and the mean paired difference in C and IBS must sit inside the
suite's equivalence margins. The margins are wide against the observed gaps
(about 0.001 in C and 0.006 in IBS on these two), so this catches a real
regression without flaking on fold noise.
"""

import numpy as np
import pytest

pytest.importorskip("sksurv")

from bench import parity_suite as ps  # noqa: E402


@pytest.mark.parametrize("name", ["veterans_lung_cancer", "whas500"])
def test_forest_matches_sksurv_within_margin(name, monkeypatch):
    monkeypatch.setattr(ps, "N_TREES", 100)
    X, e, t = ps.load(name)
    rows = ps.paired(name, X, e, t, repeats=1)
    dc = np.mean([r["c_rftvc"] - r["c_sksurv"] for r in rows])
    dibs = np.mean([r["ibs_rftvc"] - r["ibs_sksurv"] for r in rows])
    assert abs(dc) < ps.MARGIN_C
    assert abs(dibs) < ps.MARGIN_IBS
