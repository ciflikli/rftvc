"""Slice 7 (docs/plans/plan.md): timing-window permutation-importance CI gate.

Closes another part of the gap in docs/plans/simulation-validation-findings.md
row 3: the timing scenario's real design rule (Holm-rejects inside, all
outside-window upper bounds under delta) never ran against a fitted forest
in CI — only the oracle's own exact-zero fact was checked (a different,
weaker claim). Pass rule, unmodified from the design, declared before this
gate was run (see bench/timing_importance_truth_check.py's module
docstring, including the out-of-band pilot at seeds 10000-10009 and
20010-20024 that verified it holds at this reduced scale): over R=15
replications (seeds 0-14), Holm-corrected one-sided t-tests across all 6
windows reject in >= 1 of the 2 inside windows, AND every outside window's
Bonferroni-corrected upper 95% bound is <= 0.1 * oracle-inside-sum.

Deliberately NOT marked ``slow``.
"""

import numpy as np
import pytest

from bench.timing_importance_truth_check import check, run

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*extrapolate:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*windows are too fine:UserWarning"),
]


def test_replicate_smoke():
    windows = run(n_reps=1, n_estimators=15)
    assert windows.shape == (1, 6)
    assert np.all(np.isfinite(windows))


def test_timing_signal_detected_inside_and_absent_outside():
    windows = run()
    pass_inside, pass_outside, outside_ub, delta = check(windows)
    assert pass_inside, "no inside window rejected the Holm-corrected one-sided test"
    assert pass_outside, f"an outside window's upper bound exceeded delta: {outside_ub} vs {delta}"
