# Design: competing risks vs. `randomForestSRC` on a known-truth DGP

## Executive summary

Closes the last open external-tool-parity gap on the punchlist. Research found the shape that made
Slice 11 possible — a fitted competitor whose per-row output can be chained into a subject-level
path — does **not** hold for `randomForestSRC`'s competing-risks predict, which returns one CIF per
input *row*, not per subject-path, with no path-chaining mechanism in its source or docs. The fix is
the same move Slice 2/11's own plan review already validated once: **don't force a TVC comparison
onto a tool that doesn't cleanly support it — pick a DGP shape the tool actually handles**, here a
static (non-TVC), one-row-per-subject competing-risks DGP, exactly `bench/s14_cr_parity.R`'s
2-variable `Surv(time, status)` call shape (S14 already proved this works on real data). Research also
found R is never invoked live inside a `pytest`-gated test anywhere in this repo — the only two
existing patterns are (a) a manual, reported-only `bench/*.py` driver (S14's own precedent) or (b) a
fixture-regeneration script whose R output is committed as static JSON, read back by a real
`pytest`-gated test (`tests/fixtures/make_*.py`). Recommendation: **(b)**, to actually close the gate
consistently with every other slice in this pass (Slices 5-11 all landed a real default-tier CI gate,
not a manual report).

## Why not the existing TVC competing-risks DGP (`bench/s14_cr_sim.py`)

Its `true_cif` closed form is for `z_k` redrawn per unit interval — genuinely TVC, start-stop rows,
`id` column required. Feeding that into `rfsrc(Surv(id, start, stop, event) ~ ., ...)` runs without
error (confirmed), but its CIF predict is per-row: with `z_k` varying per interval, a subject's rows
give *different* rfsrc CIFs per row with no documented way to combine them into one subject-level
curve comparable to `true_cif`'s subject-level ground truth. Forcing it would mean inventing an
undocumented aggregation and asserting it's correct — exactly the "looks right, silently wrong"
failure mode this codebase has repeatedly caught in review. Not attempted.

## Two approaches considered

**A. Manual reported-only bench script**, mirroring `bench/s14_cr_parity.py` exactly: a `bench/*.py`
driver that shells out to `Rscript` directly, reports ISE-vs-truth for both arms, no pytest gate, no
CI involvement. Cheapest, matches the one piece of prior art that already does an rfsrc comparison.
**Rejected**: every other item on this punchlist (Slices 5-11) was explicitly picked because it
converts a manual/reported-only check into a real default-tier CI gate — reverting to manual-only
here would leave the parity punchlist's last item weaker than all the others for no principled reason
(R's absence from CI is a real constraint, but the fixture pattern below already solves it elsewhere
in this repo).

**B. Fixture-regeneration + CI-gated test** (recommended): a `bench/cr_rfsrc_fixture.R` script (run
once, this session, against the scratch `randomForestSRC` lib found in research) produces rfsrc's CIF
predictions on a **fixed** train/test split of the new static DGP, written to a checked-in
`tests/fixtures/cr_rfsrc_truth.json`. A `bench/cr_rfsrc_truth_check.py` driver (mirrors
`tvc_coxtv_truth_check.py`'s shape) then: (1) regenerates `rftvc`'s own ISE freshly per replication
(no R needed — `rftvc` has no R dependency), and (2) reads the **one** fixed rfsrc CIF fixture and
computes its ISE against the same `true_cif` on the same fixed test set. A `tests/test_cr_rfsrc_truth.py`
default-tier test asserts `mean(rftvc_ISE over R reps) <= rfsrc_ISE_from_fixture + EPSILON`. This
means rfsrc's own R=1 (a single fixed fit, not resampled) is compared against `rftvc`'s R=10 mean —
an asymmetry to flag explicitly in the docstring (rfsrc has no bootstrap/replication in the gate; only
`rftvc`'s own sampling variability is captured). This mirrors the fixture pattern already used for AJ/
concordance/survdiff (`tests/fixtures/make_*.py`), just applied to a competing-risks CIF.

## New DGP: static (non-TVC), one row per subject, closed-form CIF

Two causes, constant-hazard-conditional-on-covariates (piecewise-constant with one piece, i.e.
genuinely exponential given `x`) — the simplest shape both tools handle without misspecification:

```python
# cause-specific hazards, fixed per subject at t=0 (no time-variation at all)
lam1(x) = RATE1 * exp(BETA1 * x)     # cause 1
lam2(x) = RATE2 * exp(BETA2 * x)     # cause 2
lam_tot(x) = lam1(x) + lam2(x)

# closed-form CIF (standard constant-hazard competing-risks result)
F_k(t | x) = (lam_k(x) / lam_tot(x)) * (1 - exp(-lam_tot(x) * t))
```

Single covariate `x ~ N(0,1)`, right-censoring via `Exponential(rate=CENS_RATE)` independent of
`(T, cause)`. Rows: `(id, x, time, status)` with `status in {0, 1, 2}` — exactly S14's 2-variable
`Surv(time, status)` shape, so `rfsrc`'s row = subject with no ambiguity (Q1/Q4's finding doesn't
apply here at all, by construction).

## Test shape

```python
# bench/cr_rfsrc_fixture.R  (run once, manually, this session — RL=<scratch lib path>)
# fixed seed, writes tests/fixtures/cr_rfsrc_truth.json: test-set CIFs at GRID for both causes

# bench/cr_rfsrc_truth_check.py
RATE1, BETA1, RATE2, BETA2, CENS_RATE = ...       # calibrated for a reasonable censoring/event mix
GRID = np.linspace(0.0, HORIZON, N)
EPSILON = ...                                      # additive, calibrated out-of-band, ~2x observed gap

def true_cif(x, t): ...                            # closed form above, verified vs numerical integration
def simulate(n, rng): ...                           # (x, time, status) rows
def replicate(seed, n_train, n_test) -> rftvc_ise   # CompetingRisksForestTV vs true_cif, per-cause -> pooled mean
def run(n_reps=10, seed0=0) -> np.ndarray            # (n_reps,) rftvc ISE
def rfsrc_ise_from_fixture() -> float                # load tests/fixtures/cr_rfsrc_truth.json, compare to true_cif on the SAME fixed test set

# tests/test_cr_rfsrc_truth.py
def test_true_cif_matches_numerical_integration(): ...   # fast, exact, both causes
def test_replicate_smoke(): ...                            # fast, R=1, small scale
def test_rftvc_ise_within_epsilon_of_rfsrc_fixture():      # default tier, R=10
    res = run()
    assert res.mean() <= rfsrc_ise_from_fixture() + EPSILON
```

## Acceptance criteria

- New default-tier test (`not slow`) actually runs under plain `pytest` — verify by running it,
  not by inspecting markers (Slice 5's own mistake, per `[[rftvc-dev-workflow]]`).
- `EPSILON` calibrated from seeds distinct from the gate's own, stated before the gate runs, verified
  across >= 2 independent out-of-band batches — Slice 2/11's discipline.
- `true_cif` verified against numerical integration to `atol=1e-6`, both causes.
- Fixture-generation script (`bench/cr_rfsrc_fixture.R`) and its exact invocation command committed
  alongside the fixture, so it's reproducible if `randomForestSRC` needs to be reinstalled later
  (the scratch lib path is not stable — flagged directly in research).
- Explicit docstring note on the R=1-vs-R=10 asymmetry between the two arms (see Approach B above) —
  this is a real, disclosed limitation, not something to paper over.
- `docs/plans/simulation-validation-findings.md`: closes the remaining "competing-risks vs
  `randomForestSRC` on a known-truth DGP" line in the still-open list; new row for the check itself.
- One `codex:rescue` diff review before merge (closed-form-vs-numerical-integration code and the
  fixture-loading path are exactly the surface this pass has repeatedly found real bugs in).
