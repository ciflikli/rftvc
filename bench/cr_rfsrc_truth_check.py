"""Slice 12 (docs/plans/plan.md): competing risks vs. ``randomForestSRC`` on a known-truth DGP.

Closes the last item on `docs/plans/simulation-validation-findings.md`'s external-tool-parity
punchlist: no competing-risks comparison against any external tool with a known ground truth
existed (`bench/s14_cr_parity.{R,py}`'s pbc comparison is real data, no ground truth).

Data-generating process (new for this slice, deliberately *not* a reuse of `bench/s14_cr_sim.py`'s
TVC DGP): a single static covariate ``x ~ N(0, 1)``, two competing causes with constant
cause-specific hazards conditional on ``x`` (``lam_k(x) = RATE_k * exp(BETA_k * x)``), independent
exponential censoring. Chosen because research (`docs/plans/cr-rfsrc-research.md`, Q1) found
``randomForestSRC``'s competing-risks `predict()` returns one CIF per input *row*, not per
subject-path -- `subj.unique.count` is literally aliased to `nrow(xvar)` in its source, with no
found mechanism to chain a subject's time-varying-covariate rows into one curve the way Slice 11
chained `CoxTimeVaryingFitter`'s Cox baseline against per-row partial hazards. A one-row-per-subject,
non-TVC DGP with plain ``Surv(time, status)`` rows (S14's own real-data comparison's exact shape)
sidesteps this by construction -- there is no subject-path aggregation to get wrong.

Closed-form CIF (standard constant-hazard competing-risks result, cause k):
``F_k(t | x) = (lam_k(x) / lam_tot(x)) * (1 - exp(-lam_tot(x) * t))``, where
``lam_tot(x) = lam_1(x) + lam_2(x)``.

Comparator: ``randomForestSRC::rfsrc`` (splitrule ``"random"`` -- research found this is the only
splitrule confirmed to run error-free on this build's competing-risks mode at small-to-medium scale;
``"logrankCR"`` (the documented default) and the true default (``splitrule=NULL``) both errored on a
60-id dataset with ``argument is of length zero``, a pre-existing library issue not diagnosed
further here). ``randomForestSRC`` is not runnable in CI (R-only, no guarantee of presence), so its
predictions are generated **once**, this session, against a scratch install
(`tests/fixtures/make_cr_rfsrc_fixture.py`, mirrors the existing `tests/fixtures/make_*.py`
R-fixture pattern) and checked into `tests/fixtures/cr_rfsrc_truth.json`. That fixture stores the
test set's own covariates (`x`) alongside rfsrc's CIF predictions -- `rfsrc_ise_from_fixture` below
recomputes ``true_cif`` from those *same* stored covariates, never from an independently-regenerated
test set, ruling out a cross-language sample mismatch silently producing an invalid ISE (Codex plan
review finding, applied).

**Disclosed asymmetry**: the gate compares `rftvc`'s own resampled mean ISE (R=10 replications) to
`randomForestSRC`'s single fixed-fixture ISE (R=1, no replication) -- not a like-for-like bootstrap
comparison, because rfsrc never runs inside CI and the fixture is generated once. This inflates the
apparent stability of the rfsrc arm relative to rftvc's; the epsilon below is calibrated to account
for rftvc's own seed-to-seed variability against that single fixed point, not to claim rfsrc has none.

Pass rule (declared here, epsilon calibrated from an out-of-band pilot, not the gate's own seeds):
over R=10 replications (seeds 0-9), per-cause mean(rftvc_ISE) <= rfsrc_fixture_ISE + EPSILON.
Calibration (rfsrc fixture per-cause ISE: [0.0430, 0.0513]; two independent out-of-band batches,
seeds 10000-10009 and 20010-20019, plus the gate's own seeds 0-9, all n_train=400/n_test=200/
n_estimators=500): per-cause gap (rftvc mean ISE - rfsrc fixture ISE) ranged [0.0074, 0.0051] (gate
seeds), [0.0152, 0.0013] (batch 1), [0.0159, 0.0054] (batch 2) -- stable across all three, max
observed gap ~0.0159. EPSILON = 0.03, roughly double the largest observed gap, matching Slice 2/11's
discipline (additive tolerance, not a ratio, since the reference ISE isn't near zero here anyway).

Scope: static (non-TVC) competing risks only, one covariate, two causes. Does not attempt a
TVC-competing-risks comparison (the row-vs-subject-path gap above) or landmark estimators (no
external tool exists for that shape at all).
"""

import json
from pathlib import Path

import numpy as np

from rftvc import CompetingRisksForestTV, make_competing_risks_y

RATE1, BETA1 = 0.15, 0.8
RATE2, BETA2 = 0.10, -0.6
CENS_RATE = 0.06
HORIZON = 8.0
GRID = np.linspace(0.0, HORIZON, 41)
N_ESTIMATORS = 500  # matches the rfsrc fixture's ntree (tests/fixtures/make_cr_rfsrc_fixture.py)
EPSILON = 0.03  # calibrated from an out-of-band pilot, see module docstring

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "cr_rfsrc_truth.json"


def _lam(x):
    lam1 = RATE1 * np.exp(BETA1 * x)
    lam2 = RATE2 * np.exp(BETA2 * x)
    return lam1, lam2, lam1 + lam2


def true_cif(x, t):
    """Closed-form CIF, shape ``(len(x), 2, len(t))`` (cause 1, cause 2)."""
    lam1, lam2, lam_tot = _lam(np.asarray(x, dtype=float))
    t = np.asarray(t, dtype=float)
    surv_gap = 1.0 - np.exp(-lam_tot[:, None] * t[None, :])  # (n, T)
    f1 = (lam1 / lam_tot)[:, None] * surv_gap
    f2 = (lam2 / lam_tot)[:, None] * surv_gap
    return np.stack([f1, f2], axis=1)


def simulate(n, rng):
    """``(x, time, status)``: status in {0 (censored), 1, 2}."""
    x = rng.normal(size=n)
    lam1, lam2, lam_tot = _lam(x)
    event_time = rng.exponential(1.0 / lam_tot)
    is_cause1 = rng.random(n) < (lam1 / lam_tot)
    cens_time = rng.exponential(1.0 / CENS_RATE, size=n)
    time = np.minimum(event_time, cens_time)
    status = np.where(event_time <= cens_time, np.where(is_cause1, 1, 2), 0)
    return x, time, status.astype(np.int64)


def ise(f_hat, f_true):
    """Per-cause ISE (shape ``(2,)``), trapezoid over ``GRID``, mean over subjects."""
    return np.trapezoid((f_hat - f_true) ** 2, GRID, axis=2).mean(axis=0)


def replicate(seed, n_train=400, n_test=200, n_estimators=N_ESTIMATORS):
    """One replication's per-cause rftvc ISE (shape ``(2,)``) against the closed-form truth."""
    rng = np.random.default_rng(seed)
    x_tr, t_tr, s_tr = simulate(n_train, rng)
    x_te, _, _ = simulate(n_test, rng)
    y = make_competing_risks_y(t_tr, s_tr)
    m = CompetingRisksForestTV(causes=[1, 2], n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    m.fit(x_tr.reshape(-1, 1), y)
    f_hat = m.predict_cumulative_incidence(x_te.reshape(-1, 1), GRID)
    f_true = true_cif(x_te, GRID)
    return ise(f_hat, f_true)


def run(n_reps=10, seed0=0, **kw):
    """``(n_reps, 2)`` array of per-cause rftvc ISE, seeds ``seed0..seed0+n_reps-1``."""
    return np.array([replicate(seed0 + r, **kw) for r in range(n_reps)])


def rfsrc_ise_from_fixture(path=FIXTURE):
    """Per-cause rfsrc ISE (shape ``(2,)``) from the checked-in fixture.

    Recomputes ``true_cif`` from the fixture's own stored test covariates (not a freshly-simulated
    test set) -- see module docstring's "Codex plan review finding, applied" note.
    """
    d = json.loads(Path(path).read_text())
    x = np.array(d["x_test"], dtype=float)
    grid = np.array(d["grid"], dtype=float)
    assert grid.shape == GRID.shape and np.allclose(grid, GRID), "fixture grid must match GRID"
    f_hat = np.array(d["cif"], dtype=float)  # (n_test, 2, len(GRID))
    assert f_hat.shape == (x.shape[0], 2, GRID.shape[0]), f"unexpected fixture cif shape {f_hat.shape}"
    assert np.all(np.isfinite(f_hat)), "fixture contains non-finite CIF values"
    f_true = true_cif(x, grid)
    return ise(f_hat, f_true)
