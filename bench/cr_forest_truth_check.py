"""Slice 5: CompetingRisksForestTV fitted-model-vs-truth check.

Closes the gap flagged in the validation audit row 2:
`bench/s14_cr_sim.py`'s `run()` (the bake-off across many hyperparameter
"arms") writes an ISE record but computes no pass/fail decision anywhere —
no CI-gated check exists that `CompetingRisksForestTV`'s own predictions get
close to a known competing-risks truth, only that the DGP's own closed-form
truth is internally consistent (`tests/test_cr_sim_truth.py`, untouched
here).

Reuses `bench.s14_cr_sim`'s scenario-A DGP and its `training_rows`/
`test_paths`/`true_cif`/`ise` helpers unchanged — no new DGP, no new
closed-form-truth math (that's already independently truth-checked). This
script adds a single `CompetingRisksForestTV` fit at its actual **library
defaults** (`n_estimators=500`, `criterion="composite"`, `aggregate="cif"`,
`split_cause=None` — matching `CompetingRisksForestTV.__init__`'s own
defaults exactly, one of the arms the S14 bake-off already compared) against
the closed-form `true_cif`. It does not re-run the bake-off or re-litigate
which criterion/aggregate is best — that was already decided; this is a
plain regression-style truth check on what ships by default.

This gate runs in the **default (fast) test tier, not `slow`** — deliberate:
the whole point of this slice is a check that actually runs in CI on every
push, and at ~2.5s for the full R=10 gate (500-tree competing-risks fits are
cheap at this DGP's scale) there is no speed reason to exclude it from the
merge gate the way the R=50 bake-off/importance-pilot sims are (those cost
much more per replication).

Pass rule (declared here, before the gate's own R=10 replications were run,
threshold calibrated from an out-of-band pilot, seeds `10_000..10_004`, 5
replications at this module's own defaults — n_train=500, n_test=200,
n_estimators=500, the library default: per-seed per-cause ISE = [0.0524,
0.0346], [0.0528, 0.0409], [0.0501, 0.0514], [0.0469, 0.0398], [0.0435,
0.0365]; overall mean ≈ 0.0449, max single value ≈ 0.0528): over R=10
replications (seeds 0-9) on scenario A, mean per-cause ISE (averaged over
both causes and all replications) <= 0.10 — roughly double the pilot mean
and well above the pilot's max single value, enough headroom for
run-to-run variability while still failing on a real regression (e.g. an
ISE blowup to several times the observed scale).
"""

import numpy as np

from bench.s14_cr_sim import GRID, ise, test_paths, training_rows, true_cif  # noqa: F401 (true_cif re-exported)
from rftvc import CompetingRisksForestTV

ISE_THRESHOLD = 0.10
N_ESTIMATORS = 500  # the library default (CompetingRisksForestTV.__init__)


def replicate(seed, n_train=500, n_test=200, n_estimators=N_ESTIMATORS):
    """One replication's per-cause ISE (shape ``(2,)``) for scenario A."""
    rng = np.random.default_rng(seed)
    x, y, ids = training_rows("A", n_train, rng)
    xt, iv, ids_t, f_true = test_paths("A", n_test, rng)
    m = CompetingRisksForestTV(causes=[1, 2], n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(x, y, ids)
    f_hat = m.predict_cumulative_incidence(xt, GRID, intervals=iv, ids=ids_t)
    return ise(f_hat, f_true)


def run(n_reps=10, seed0=0, **kw):
    """``(n_reps, 2)`` array of per-cause ISE, seeds ``seed0..seed0+n_reps-1``."""
    return np.array([replicate(seed0 + r, **kw) for r in range(n_reps)])
