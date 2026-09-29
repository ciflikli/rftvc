# Design: TVC vs. `CoxTimeVaryingFitter` on a known-truth DGP

## Executive summary

Mirrors Slice 2 (`bench/lifelines_truth_check.py`) exactly, but for the TVC case: a new, genuinely
log-linear external-TVC DGP (`hazard = rate0 * exp(beta * z_k)`, single covariate, dropped
`tests/sim.py`'s threshold nonlinearity so `CoxTimeVaryingFitter` is correctly specified), comparing
`rftvc`'s and `CoxTimeVaryingFitter`'s ISE against the closed-form truth. Research found a real,
stable gap (rftvc ISE ≈0.09-0.10, Cox ISE ≈0.003-0.008, gap ≈0.086-0.092 across 3 independent
batches) — same qualitative shape as Slice 2, confirming `rftvc` isn't a statistical outlier versus
a correctly-specified competitor on its own model's data, just less statistically efficient (as
expected for a nonparametric method vs. the exactly-correct parametric one). One real implementation
bug was caught and fixed during research: `CoxTimeVaryingFitter.baseline_cumulative_hazard_` cannot
be assumed to land on exact integer times — event times in this DGP are continuous — so a
step-function lookup (`searchsorted`) is required, not a `reindex`.

## Why not reuse `tests/sim.py`'s DGP

Its `0.8*(z>1)` threshold term is a real non-linearity, and `rows()` only ever exposes raw `[x0,
z_k]` to any fitted model — so `CoxTimeVaryingFitter` would be misspecified there too, defeating the
point of giving it a correctly-specified home turf (the same reasoning Slice 2's own plan-review
correction already established for the static case).

## Test shape

```python
# bench/tvc_coxtv_truth_check.py
RATE0, BETA = 0.15, 0.8
K, END, HORIZON = 8, 8.0, 6.0
GRID = np.linspace(0.0, HORIZON, 61)
EPSILON = 0.18  # ~2x the observed gap (0.086-0.092), calibrated out-of-band (seeds 10000-10004, 20010-20014)

def true_cumhaz(z, times): ...       # piecewise-constant closed form, verified vs numerical integration
def simulate(n, rng): ...             # same event-time-inversion pattern as tests/sim.py
def rows(z, U, event): ...            # counting-process rows, single covariate
def _rftvc_survival(...): ...         # SurvivalForestTV.predict_cumulative_hazard(..., intervals=)
def _baseline_step_at(cox, times): ... # searchsorted lookup -- NOT reindex (continuous event times)
def _cox_survival(...): ...           # combine baseline increments with predict_partial_hazard
def replicate(seed, n_train=400, n_test=200, n_estimators=200) -> (rftvc_ise, cox_ise)
def run(n_reps=10, seed0=0, **kw) -> (n_reps, 2) array
```

```python
# tests/test_tvc_coxtv_truth.py
def test_true_cumhaz_matches_numerical_integration(): ...   # fast, exact
def test_replicate_smoke(): ...                              # fast, R=1, small scale
def test_rftvc_ise_within_epsilon_of_cox():                 # default tier, R=10
    res = run()
    assert res[:, 0].mean() <= res[:, 1].mean() + EPSILON
```

## Acceptance criteria

- Runtime ~4-5s at R=10, `n_estimators=200` — default tier (`not slow`), matching Slices 5-10's
  discipline, not Slice 2's `slow` marker (that predates the "slow gates never run in CI" lesson).
- `EPSILON` calibrated from seeds distinct from the gate's own (0-9), stated before the gate runs.
- `docs/plans/simulation-validation-findings.md`: new row (8) for this check; closing summary's
  "no landmark-estimator or TVC cross-check against an external tool exists at all" bullet updated
  to reflect TVC now has one (landmark still doesn't — out of scope here).
- One `codex:rescue` diff review before merge, given the hand-rolled Breslow-baseline-lookup code
  is exactly the kind of "looks right, silently wrong" surface this session has repeatedly found
  real bugs in.
