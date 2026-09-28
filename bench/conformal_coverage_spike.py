"""Slice 4 (docs/plans/plan.md): conformal-coverage spike.

EXPLORATORY ONLY — no public API, no change to ``src/rftvc/``. This is an
empirical investigation of whether an OOB-calibrated, censoring-weighted
interval around ``SurvivalForestTV.predict_risk`` achieves roughly nominal
coverage, not a shipped feature.

Target: ``P(event by horizon | X)``, the DGP's true conditional risk (a
*probability*, not the realized binary outcome) at a fixed horizon — for
single-event ``SurvivalForestTV`` only, non-TVC. Reuses
``bench.lifelines_truth_check``'s static, closed-form-truth DGP, whose exact
``true_survival`` gives this target directly.

Calibration set: the fitted estimator's own **training-set OOB rows**, not
an external calibration split. ``estimator.forest_.oob_cumhaz(...)`` (the
same private call ``rftvc.inspection`` already relies on internally, via
``estimator._rebuild_design(X_train, y_train, ids_train)``) only returns
predictions for the fitted training rows, not for arbitrary held-out data —
so ``weighted_conformal_risk_interval`` takes the original training
``(X_train, y_train)`` and re-derives OOB predictions from them, rather than
a separate ``X_calib`` (a deviation from the plan's originally sketched
signature, forced by how ``_rebuild_design`` actually works — it fingerprint
-checks against the exact fit-time data). Rows with no OOB tree
(``n_trees == 0``) are excluded, mirroring how ``oob_score_`` already
excludes them.

Weighting: IPCW, reusing this library's own existing case/control/weight
convention — ``rftvc.metrics._ipcw``, the private helper ``brier_landmark``/
``integrated_brier`` already build on. A training row observed to have the
event by its own stop time is weighted ``1/G(stop-)``; a row observed
event-free through the horizon is weighted ``1/G(horizon-)``; a row censored
*before* the horizon has unknown status there and is excluded (weight 0).
``G`` is the reverse-KM censoring-survival estimate
(``rftvc.metrics.KaplanMeierCensoring``), fit on the same training outcomes.

No formal split-conformal guarantee is claimed. OOB rows are not an
independent, freshly-drawn calibration split in the classical exchangeable
sense (they come from the same bootstrap-resampled trees as the fit), so
this is an empirical investigation, not a proof. "Coverage does not hold" or
"inconclusive" is an acceptable, reportable outcome — see
``docs/plans/conformal-prediction-investigation.md`` for the results.

Method (deliberately the simplest defensible weighted-conformal
construction, not the most sophisticated one in the literature — see
``docs/plans/research.md`` Q8): one symmetric interval half-width per fit,
the weighted ``(1 - alpha)`` quantile of ``|case_i - risk_hat_oob_i|`` over
resolved (case or control) OOB training rows. Test-row intervals are
``[predict_risk(X_test, horizon) -+ q]``, clipped to ``[0, 1]``.

``check_coverage`` compares this interval, across many independent
replications, against the DGP's **true conditional probability**
``P(event by horizon | X)`` on a fresh test draw — not against the realized
0/1 outcome. Checking probability coverage is only possible because the
ground truth is known in this synthetic setting; it is a stronger and
non-standard target relative to the literature's usual (and only generally
checkable) outcome-coverage guarantee, and the two are reported separately
in the write-up rather than conflated.
"""

import numpy as np
from joblib import effective_n_jobs

from bench.lifelines_truth_check import HORIZON, simulate, true_survival
from rftvc import SurvivalForestTV, make_survival_y
from rftvc.metrics import _ipcw

N_ESTIMATORS = 200


def _weighted_quantile(values, weights, q):
    """The weighted ``q``-quantile of ``values`` (simple, not finite-sample-corrected
    — see module docstring's "no formal guarantee" note)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    idx = min(np.searchsorted(cw, q), v.size - 1)
    return float(v[idx])


def weighted_conformal_risk_interval(fitted_estimator, X_train, y_train, X_test, horizon, alpha, g_min=0.05):
    """``(lower, upper)`` risk-at-``horizon`` bounds for ``X_test`` rows.

    Calibrated on ``fitted_estimator``'s own training-set OOB rows (from
    ``X_train``/``y_train``, the exact data it was fit on — see module
    docstring). No ``X_calib``: there is no separable calibration split in
    this design.
    """
    n_jobs = effective_n_jobs(fitted_estimator.n_jobs)
    d = fitted_estimator._rebuild_design(X_train, y_train, None)
    offsets, units = d.oob_set
    h, n_trees = fitted_estimator.forest_.oob_cumhaz(
        d.X, offsets, units, np.array([horizon]), fitted_estimator.aggregate, n_jobs
    )
    ok = n_trees[:, 0] > 0 if n_trees.ndim == 2 else n_trees > 0
    if not ok.any():
        raise ValueError("no training row has an out-of-bag tree at this horizon")
    risk_oob = 1.0 - np.exp(-h[ok, 0])
    stop, event = d.stop[ok], d.event[ok]
    y_censor = make_survival_y(stop, event)
    case, control, weights, _ = _ipcw(stop, event, horizon, y_censor, None, g_min)
    resolved = case | control
    if not resolved.any():
        raise ValueError("no training row is resolved (case or control) at this horizon")
    resid = np.abs(case[resolved].astype(float) - risk_oob[resolved])
    q = _weighted_quantile(resid, weights[resolved], 1.0 - alpha)
    risk_test = fitted_estimator.predict_risk(X_test, horizon)
    return np.clip(risk_test - q, 0.0, 1.0), np.clip(risk_test + q, 0.0, 1.0)


def check_coverage(n_train=400, n_test=300, alpha=0.1, n_reps=30, seed=0, horizon=HORIZON, n_estimators=N_ESTIMATORS, g_min=0.05):
    """Empirical coverage of the true conditional risk, over ``n_reps`` independent
    replications, plus a Monte Carlo CI on the mean (not a single-seed point estimate)."""
    per_rep = np.empty(n_reps)
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r)
        x_tr, u, event = simulate(n_train, rng)
        y_tr = make_survival_y(u, event)
        m = SurvivalForestTV(n_estimators=n_estimators, random_state=seed + r).fit(x_tr[:, None], y_tr)
        x_te, _, _ = simulate(n_test, rng)
        true_risk = 1.0 - true_survival(x_te, np.array([horizon]))[:, 0]
        lower, upper = weighted_conformal_risk_interval(m, x_tr[:, None], y_tr, x_te[:, None], horizon, alpha, g_min)
        per_rep[r] = float(np.mean((true_risk >= lower) & (true_risk <= upper)))
    mean_coverage = float(per_rep.mean())
    se = float(per_rep.std(ddof=1) / np.sqrt(n_reps))
    return {
        "nominal": 1.0 - alpha,
        "mean_coverage": mean_coverage,
        "mc_ci": (mean_coverage - 1.96 * se, mean_coverage + 1.96 * se),
        "per_replication": per_rep,
    }
