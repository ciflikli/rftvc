"""Slice 8 (docs/plans/plan.md): competing-risks permutation-importance CI gate.

Closes the last part of the row-3 gap in docs/plans/simulation-validation-findings.md
(S17's third oracle scenario): the real design rule never ran against a
fitted forest in CI — only a loosened 0.3x-of-oracle pilot bound did (design:
0.1x). Pass rule, unmodified from the design, declared before this gate was
run (see bench/cr_importance_truth_check.py's module docstring, including
the out-of-band pilots at seeds 10000-10009 and 20010-20024): over R=15
replications (seeds 0-14), Holm-corrected one-sided test on d_s1 rejects,
AND d_s2's upper 95% bound is <= 0.1 * oracle(cause 1).

Deliberately NOT marked ``slow``.
"""

import numpy as np
import pytest

from bench.cr_importance_truth_check import check, run

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*extrapolate:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*does not beat the training null:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*windows are too fine:UserWarning"),
]


def test_replicate_smoke():
    res = run(n_reps=1, n_estimators=15)
    assert res.shape == (1, 2)
    assert np.all(np.isfinite(res))


def test_cr_importance_detected_on_cause1_absent_on_cause2():
    res = run()
    pass_cause1, pass_cause2, ub_s2, delta1 = check(res)
    assert pass_cause1, "cause-1 importance did not reject the Holm-corrected one-sided test"
    assert pass_cause2, f"cause-2 importance's upper bound exceeded delta1: {ub_s2} vs {delta1}"
