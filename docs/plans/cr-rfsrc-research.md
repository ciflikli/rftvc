# Research: competing risks vs. `randomForestSRC` on a known-truth DGP

## Environment check (done first, since Q3 depends on it)

- **R**: `/usr/local/bin/R`, `/usr/local/bin/Rscript`, R 4.5.0 (2025-04-11), aarch64-apple-darwin20.
- **`RL` env var**: unset in this session (`echo $RL` → empty). It does **not** currently resolve to
  anything by default — every invocation must set it explicitly, exactly as `cr_rfsrc_check.R`'s
  header comment says (`RL=<R lib with randomForestSRC> Rscript ...`).
- **Scratch libraries found** (not in this session's own scratchpad — leftovers from two *prior*
  sessions' scratchpads, found by `find` across `/private/tmp/claude-501/...`):
  - `.../cddd4d79-3b34-469c-be69-332855fbf020/scratchpad/rlib/randomForestSRC` (mtime Sep 23 04:51)
  - `.../2d30fca3-8f26-4449-8b67-44b6b98c069d/scratchpad/Rlib/randomForestSRC` (mtime Sep 23 04:51)
  - Both contain identical `randomForestSRC` 3.9.0 (`DESCRIPTION`: `Packaged: 2026-09-22`, `Built: R
    4.5.2; aarch64-apple-darwin20; 2026-09-23`), plus `DiagrammeR`, `data.tree`, `visNetwork` (transitive
    deps of `randomForestSRC`'s plotting functions, unused here).
- **Loads successfully today** (2026-09-29, one week after the mtime) despite the **R version
  mismatch** (lib built under R 4.5.2, this machine's `Rscript` is R 4.5.0):
  ```
  RL=<path> Rscript -e 'library(randomForestSRC, lib.loc=Sys.getenv("RL"))'
  Warning message:
  package 'randomForestSRC' was built under R version 4.5.2
   randomForestSRC 3.9.0
  ```
  Load succeeds with a warning only, not an error. `bench/cr_rfsrc_check.R` was re-run verbatim
  against this scratch lib and completed without error (both the `causes: 0 1 2` and `causes: 0 1`
  branches).
- **"Lost on reboot" claim**: not verified true or false by this session — the libraries plainly
  **were not lost** between whenever the prior sessions installed them (Sep 22-23) and now (Sep 29),
  since `/private/tmp` on this machine evidently survived that whole span without being cleared. The
  memory note's underlying worry (scratch installs under `/private/tmp` can vanish on a reboot,
  since macOS periodically sweeps old `/tmp` contents) is a real macOS behavior in general, but this
  session found no evidence it has happened here — the note should be read as "don't assume a scratch
  lib is there without checking," not as "assume it's always gone." **Does not need reinstalling this
  session** — the existing lib loads and runs correctly right now.
- **Does not require a fresh `install.packages()`** for the design/implementation stage, provided
  whichever of the two lib paths above is still present when that stage runs (re-verify with the same
  `find`/load check, don't assume from this document).

## Q1: what DGP shape does `randomForestSRC`'s competing-risks mode assume/handle well?

**`bench/cr_rfsrc_check.R`'s exact call shape** (re-run against the scratch lib, output captured):

```r
rfsrc(Surv(id, start, stop, event) ~ ., d, ntree=20, splitrule="random")
```

`rfsrc`'s formula parser (`randomForestSRC:::parseFormula`, read via `body()`) does **not** call
`survival::Surv()` — it parses the formula text with `all.vars()`. When the LHS names 4 variables that
are all columns in `data`, the **first** one is pulled out as `subj.names` (a subject-identifier
column) and the remaining 3 are treated as `(start, stop, event)`:

```r
if (fmly == "Surv") {
    if ((sum(is.element(yvar.names, names(data))) != 2) && (sum(is.element(yvar.names, names(data))) != 4)) {
        stop("Survival formula incorrectly specified.")
    } else {
        if (sum(is.element(yvar.names, names(data))) == 4) {
            subj.names <- yvar.names[1]
            yvar.names <- yvar.names[-1]
        }
    }
    family <- "surv"
```

(Confirmed directly: calling `survival::Surv(id, start, stop, event)` outside a formula throws
`Error in match.arg(type) : 'arg' must be NULL or a character vector`, since `Surv`'s real 4th
positional argument is `type` — a character string — not `event`. `rfsrc` never evaluates that call;
it only reads the formula's variable names as text. `randomForestSRC` doesn't even export or import a
function called `Surv` into its namespace — `exists("Surv", where=asNamespace("randomForestSRC"))` is
`FALSE`.)

**Family detection is automatic from the number of distinct nonzero causes**, not from a fixed
argument. Re-running the spike's two branches on the same 200-id, ~600-row start-stop dataset:

| `event` column | `f$family` |
|---|---|
| raw 3-valued codes `{0,1,2}` | `"surv-CR"` |
| collapsed to `{0,1}` (`as.integer(event>0)`) | `"surv"` |

Both ran without error at `splitrule="random"` — no error surfaced for either family with start-stop
(4-variable) rows.

**Splitting rules for competing risks**, from `?rfsrc`'s Details (`tools::Rd_db` extraction of
`rfsrc.Rd`, package version 3.9.0):

```
• Competing risks (see Ishwaran et al., 2014)
   1. "logrankCR" (default): Gray's test-based weighted log-rank splitting.
   2. "logrank": cause-specific weighted log-rank; use 'cause' to target specific events.
```

However, `splitrule="logrankCR"` **failed** when tried directly against a 60-id / 180-row dataset in
this session:
```
Error in if (splitrule.idx != which(splitrule.names == "tdc.gradient")) { :
  argument is of length zero
Calls: rfsrc -> get.grow.splitinfo
```
and the true default (`splitrule=NULL`, letting `rfsrc` pick) failed too, on the same small dataset:
```
Error in if (splitrule == "random") { : argument is of length zero
```
Both errors disappeared at the spike's original scale (`n=200` ids, `splitrule="random"` explicit) —
not independently diagnosed further (out of this research task's scope), but it means **`splitrule`
choice and/or scenario/sample size interact with what actually runs without error** on start-stop CR
data with this build; `"random"` is the one splitrule confirmed to work in both this session's runs
and the pre-existing spike file.

**`predict()`'s return shape — the central finding for this question.** Extending the spike (fit on
200 ids / 606 rows, then `predict(f, newdata=...)` on 5 new ids with 2-4 rows each = 16 rows):

```
f$n = 606   length(f$time.interest) = 124
dim(f$cif) = 606,124,2        # in-bag: one CIF per TRAINING ROW, not per id (200 ids, 606 rows)
dim(f$cif.oob) = 606,124,2
dim(f$chf) = 606,124,2

predict: dim(pr$cif) = 16,124,2   pr$n = 16   # one CIF per NEWDATA ROW (5 ids, 16 rows), not per id
```

A single-row-per-id `newdata` (5 ids, `start=0, stop=6`, one row each, no genuine time-varying
covariate) does give one CIF row per id (`dim(pr2$cif) = 5,124,2`) — but only because it happens to
have exactly one row per subject, not because `rfsrc` aggregated anything.

Digging into why (`body(randomForestSRC:::rfsrc)`, searching for `subj`): `subj.names` is threaded
through to the native call (used for something — not traced further, e.g. possibly grouping bootstrap
draws by subject), but the row-count variable that actually drives sampling and output shape is
computed **before** any subject collapsing:

```r
n <- nrow(xvar)              # = number of ROWS in the data, all 606 of them
...
subj.unique.count <- n       # literally aliases nrow(xvar), NOT length(unique(subj))
```

and the in-bag/OOB hazard ensembles are reshaped using that same `subj.unique.count`:
```r
hazard <- matrix(nativeOutput$allEnsbKHZ, c(subj.unique.count, ...))
```
i.e. `subj.unique.count` is a **row count**, not a subject count, despite the name. Nothing in the
traced code path collapses multiple rows belonging to the same `id` into one prediction. The
`rfsrc.Rd` documentation (full text extracted via `tools::Rd2txt`) contains **no** mention of
"repeated", "cluster", "longitudinal", "counting process", "time-dependent/time-varying", or any
explanation of what the 4-variable `Surv(id, start, stop, event)` form actually does beyond what the
formula parser itself reveals — it is effectively undocumented in the public help.

**Contrast with `CompetingRisksForestTV.predict_cumulative_incidence`** (`src/rftvc/_competing.py:274-293`):
```python
def predict_cumulative_incidence(
    self, X, times=None, *, cause=None, intervals=None, ids=None, origin=None, extrapolate="none"
):
    """... Without ``intervals``, each row of ``X`` is a subject whose covariates
    are fixed from time 0. With ``intervals`` (and ``ids``), rows are a
    covariate path per subject, and the result, one row per subject in order
    of first appearance, is conditional on being event-free at ``origin``...
    Returns shape ``(n, n_causes_, n_times)`` ...
    """
```
`rftvc` explicitly chains a subject's multiple interval-rows (`intervals=`, `ids=`) into **one**
`(n_subjects, n_causes, n_times)` output per subject. `randomForestSRC`'s CR predict, as directly
observed above, returns one CIF **per input row**, with no built-in path-chaining mechanism found in
the traced source. (Slice 11's `CoxTimeVaryingFitter` comparator had an analogous gap — no built-in
`predict_survival_function` for TVC — but it was closable by hand, because a Cox model factors into a
single global non-parametric baseline (`baseline_cumulative_hazard_`) times a per-row scalar partial
hazard, so per-interval increments can be manually cumulative-summed across a subject's path
(`bench/tvc_coxtv_truth_check.py:129-167`, `_baseline_step_at` + `predict_partial_hazard`). A random
forest's leaf-level cause-specific hazard is not decomposable into "one global baseline × per-row
multiplier" — each row's CIF/CHF already bakes in that row's own covariates and its own leaf
assignment, with no exposed separable baseline to reuse across a chain of leaves the way Cox's is.
Whether any such chaining is possible/correct for `randomForestSRC`'s row-level output was not
established in this research pass — no mechanism for it was found in the traced source or the public
docs.)

**Direct answer to Q1's DGP-shape question**: `randomForestSRC`'s CR mode, exercised via the existing
spike's exact call shape, ran without error on `tests/sim.py`/`bench/s14_cr_sim.py`-style start-stop
rows (delayed entry, per-interval redrawn covariates, threshold non-linearity in the DGP was not part
of the spike's synthetic `x1`/`x2`, so non-linearity-tolerance wasn't separately tested here). It does
**not** error on TVC-shaped input. What it does not do — confirmed directly — is return a
subject-level path prediction from that input; it returns one prediction per row. `bench/s14_cr_parity.R`
(S14's real-data comparison) sidesteps this entirely by using **plain right-censored `(time, status)`
rows** (`rfsrc(Surv(time, status) ~ ., data = train, ...)`, `bench/s14_cr_parity.R:12`) — one row per
subject, no start-stop, no TVC, no `subj.names` mechanism invoked at all.

## Q2: is there already a closed-form-CIF competing-risks DGP with known `F_k(t|x)` in this repo?

Yes — `bench/s14_cr_sim.py`'s `true_cif` (lines 120-133):

```python
def true_cif(lam, t):
    """Closed-form Aalen–Johansen of piecewise-constant hazards: ``F`` of shape (n, J, len(t))."""
    n, _, J = lam.shape
    total = lam.sum(axis=2)
    S_k = np.concatenate([np.ones((n, 1)), np.exp(-np.cumsum(total, axis=1))], axis=1)  # S at integers
    F_k = np.zeros((n, K + 1, J))
    for k in range(K):
        F_k[:, k + 1] = F_k[:, k] + lam[:, k] / total[:, k, None] * (S_k[:, k] * (1 - np.exp(-total[:, k])))[:, None]
    k = np.minimum(np.floor(t).astype(int), K - 1)
    dt = t - k
    h = total[:, k]  # (n, len(t))
    frac = lam[:, k, :] / h[:, :, None]
    F = F_k[:, k, :] + frac * (S_k[:, k] * (1 - np.exp(-h * dt)))[:, :, None]
    return np.transpose(F, (0, 2, 1))
```

This is used both by `bench/cr_forest_truth_check.py` (imported directly, line 44: `from
bench.s14_cr_sim import GRID, ise, test_paths, training_rows, true_cif`) and by
`tests/test_cr_sim_truth.py` (per `simulation-validation-findings.md` row 2's description — not read in
full this pass, referenced in the questions file). `bench/cr_forest_truth_check.py`'s existing gate
(`replicate`, lines 51-58) already does exactly the "fit `CompetingRisksForestTV`, compare its
`predict_cumulative_incidence` against `true_cif`" pattern, on **scenario A** of `s14_cr_sim.py`'s DGP
(external covariates only — `training_rows`/`test_paths` build counting-process rows with `z_k`
redrawn per unit interval, same delayed-entry/censoring convention as the TVC DGP discussed in Q1):

```python
def replicate(seed, n_train=500, n_test=200, n_estimators=N_ESTIMATORS):
    rng = np.random.default_rng(seed)
    x, y, ids = training_rows("A", n_train, rng)
    xt, iv, ids_t, f_true = test_paths("A", n_test, rng)
    m = CompetingRisksForestTV(causes=[1, 2], n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(x, y, ids)
    f_hat = m.predict_cumulative_incidence(xt, GRID, intervals=iv, ids=ids_t)
    return ise(f_hat, f_true)
```

This gate's pass rule (`ISE_THRESHOLD = 0.10`, calibrated from an out-of-band pilot, `bench/cr_forest_truth_check.py:29-39`)
verifies `CompetingRisksForestTV`'s own fit against `true_cif`, but has **no external-tool comparator**
(`randomForestSRC` or otherwise) — it is a pure fitted-model-vs-truth check, exactly the gap
`simulation-validation-findings.md` row 2b already closed, distinct from the still-open
external-tool-parity gap this task's parent question set is about.

Note `true_cif`'s DGP (scenario A) is genuinely **TVC** (`z_k` redrawn per unit interval, entering the
hazard as `0.12 * exp(0.8*z + 0.5*x0)` per `hazards()`, `bench/s14_cr_sim.py:44`) — the same shape Q1
found `randomForestSRC`'s CR predict does not cleanly support at the subject-path level.

## Q3: R/`randomForestSRC` scratch-install toolchain

Covered in full in the "Environment check" section above. Summary of the specific sub-questions:
- **R version**: 4.5.0 (system), lib built under 4.5.2 (loads anyway, with a warning, not an error).
- **Package version**: `randomForestSRC` 3.9.0, `Packaged: 2026-09-22`.
- **Scratch lib path**: two candidates found, both under other sessions' now-orphaned scratchpad
  dirs (paths above); neither is under this session's own scratchpad, and neither is `RL`-resolvable
  by default (the env var is unset).
- **`Rscript` invocation**: exactly as `cr_rfsrc_check.R`'s header says —
  `RL=<path> Rscript bench/cr_rfsrc_check.R` — re-run verbatim in this session and it worked.
- **"Lost on reboot"**: not reproduced as an actual loss in this session; the libs were still present
  and loadable a week after being built. Does not need reinstalling **this session**, given one of the
  two found paths — but the path is not stable/predictable (it lives under a prior session's UUID-named
  scratchpad, which is not something a later session, CI run, or even the design/implementation slice
  of this same task can rely on being there without re-checking).

## Q4: start-stop (left-truncated) vs. plain right-censored rows in CR mode

Answered concretely by both the `get.grow.event.info` body dump (already in `cr_rfsrc_check.R`'s own
output, re-run this session) and Q1's `parseFormula` reading:

- **2-variable LHS** (`Surv(time, event) ~ .`, e.g. `bench/s14_cr_parity.R:12`,
  `rfsrc(Surv(time, status) ~ ., data = train, ...)`): `r.dim <- 2`, `start.time <- NULL`. Plain
  right-censored, one row per subject, no `subj.names`.
- **4-variable LHS** (`Surv(id, start, stop, event) ~ .`): `subj.names <- yvar.names[1]`, remaining 3
  vars parsed as `(start, stop, event)` inside `get.grow.event.info`'s `r.dim <- 3` branch — reading
  the dumped function body, this branch is **structurally identical** to the `r.dim <- 2` branch
  except for also carrying `start.time`: same `event.type <- unique(na.omit(cens)[na.omit(cens) > 0])`,
  same `time.interest` construction. No extra requirement (no ordering constraint, no distinct-entry
  check, nothing else) was surfaced in that function body for the start-stop case beyond what
  right-censored data already needs.
- What **is** required, per `parseFormula`'s `stop("Survival formula incorrectly specified.")` check:
  the LHS must have **exactly 2 or exactly 4** variable names matching columns in `data` — 3 is
  rejected outright (so a bare `Surv(start, stop, event)` with no id column, the actual counting-process
  form `survival::Surv(start, stop, event, type="counting")` would use, is **not** how `rfsrc` expects
  this — it wants the `id` column tacked on as a 4th LHS variable, unlike real `survival::Surv`).
- **Does the chosen DGP need start-stop shape at all?** Not established either way as a requirement by
  `rfsrc` itself — both 2-variable and 4-variable forms run without error (confirmed for 4-variable in
  this session's re-run; 2-variable is what S14's existing parity script already uses successfully on
  real right-censored data). The determining factor is Q1's finding: 4-variable/start-stop input **can**
  be fed to `rfsrc`, but its CR predict output is per-row, not per-subject-path — so whether start-stop
  shape is *useful* for a TVC-truth comparison (as opposed to merely accepted without erroring) is the
  open question Q1's finding actually bears on, not a hard API requirement from `rfsrc`'s side.

## Q5: pass/fail rule and epsilon-calibration discipline (Slice 2 / Slice 11) — does it transfer?

**Slice 2** (`bench/lifelines_truth_check.py:40-54`, static DGP, `CoxPHFitter`):
```
Pass rule ... over R=10 replications (seeds 0-9), mean(rftvc_ISE) <= mean(cox_ISE) + 0.08 —
roughly double the pilot's observed gap ... An additive tolerance is used instead of a ratio to the
best-observed ISE, since the reference ISE here is small and a ratio would be unstable near a
near-zero denominator.
```
Calibration: out-of-band pilot, seeds `10_000..10_004`, 5 reps at the module's own defaults; per-seed
`(rftvc_ISE, cox_ISE)` pairs recorded in the docstring; `EPSILON = 0.08` set **before** the gate's own
R=10 seeds (0-9) were run.

**Slice 11** (`bench/tvc_coxtv_truth_check.py:46-55`, TVC DGP, `CoxTimeVaryingFitter`): identical
structure — additive epsilon (`EPSILON = 0.18`, ≈2x observed gap), two independent out-of-band pilot
batches (seeds `10000-10004` and `20010-20014`), R=10 on seeds 0-9 for the gate itself, `not slow`
(default CI tier). Both mirrored in `tests/test_tvc_coxtv_truth.py:61-65`:
```python
def test_rftvc_ise_within_epsilon_of_cox():
    res = run()
    assert np.all(np.isfinite(res))
    gap = res[:, 0].mean() - res[:, 1].mean()
    assert gap <= EPSILON, f"rftvc ISE exceeded cox ISE by more than epsilon: gap={gap}, EPSILON={EPSILON}"
```

**Does the additive-epsilon, single-comparator-ISE-gap rule transfer to competing risks as-is?** Not
established by this research pass either way as a statistical matter (that judgment is out of scope
for research; this only inventories what exists). What **is** established as a structural fact: Slice
2/11's rule is a **single scalar gap** (`mean(rftvc_ISE) - mean(other_ISE)`) because both are
single-event survival problems with one ISE per replication. Competing risks intrinsically produces
**one ISE per cause** — `bench/cr_forest_truth_check.py`'s own pass rule (lines 29-39) already had to
decide this for the truth-only (no external tool) case, and chose **"mean per-cause ISE (averaged over
both causes and all replications) <= 0.10"** — a single pooled scalar, not two separate epsilons per
cause. `bench/s14_cr_sim.py`'s `ise()` function (line 136-138) returns a **per-cause** vector
(`shape (J,)`, via `np.trapezoid(..., axis=2).mean(axis=0)`), so both "one pooled epsilon over the mean
across causes" and "one epsilon per cause" are directly constructible from what already exists in this
codebase — which of the two matches Slice 2/11's discipline more closely, or whether a new discipline is
needed, was not decided by this research task (design-stage judgment, not a fact to report here).

## Q6: is there a `rpy2`-free pattern for shelling out to `Rscript` from an actual pytest test (not just a manual `bench/*.py` driver)?

**No** — confirmed by searching the whole `tests/` tree for `Rscript`/`subprocess` usage. The pattern
that exists is: **fixture regeneration scripts, run manually, whose output is a checked-in static file
that pytest then reads** — R itself never runs inside the pytest process or CI's `pytest -m "not slow
and not network"` invocation (`pyproject.toml:57`):

```
tests/fixtures/make_aj_fixtures.py:1:   """Regenerate tests/fixtures/aj_survfit.json with R survival::survfit (needs Rscript).
tests/fixtures/make_aj_fixtures.py:65:      out = subprocess.run(["Rscript", "-e", r], capture_output=True, text=True, check=True).stdout
tests/fixtures/make_concordance_fixtures.py:1: """Regenerate tests/fixtures/concordance_cp.json with R survival::concordance (needs Rscript).
tests/fixtures/make_survdiff_fixtures.py:1:   """Regenerate tests/fixtures/survdiff.json with R survival::survdiff (needs Rscript).
```
with the corresponding checked-in outputs actually present:
```
tests/fixtures/aj_survfit.json
tests/fixtures/concordance_cp.json
tests/fixtures/cr_brier.json
tests/fixtures/survdiff.json
```
No `test_*.py` file under `tests/` calls `Rscript` or `subprocess` at all (`grep -rln "Rscript"
tests/test_*.py` → no matches).

`bench/s14_cr_parity.py` (S14's real-data `randomForestSRC` comparison — the closest prior art for
"compare rftvc to rfsrc") is itself a **manual driver script**, not a pytest test: it calls
`subprocess.run(["Rscript", str(Path(__file__).with_suffix(".R")), ...])` directly inside `main()`
(`bench/s14_cr_parity.py:140-142`), and per `simulation-validation-findings.md` row 7b's blind-spot
note, "`randomForestSRC` is R-only, so PBC vs rfsrc (S14) is still a manual, reported-only comparison"
— confirming this was a deliberate choice already made once in this codebase, not an oversight.

**Direct answer**: the two established patterns in this repo are (a) `bench/*.py` manual driver +
`bench/*.R` script, R output reported only, never gated in CI (S14's approach), or (b) a fixture
regeneration script whose R-derived output is committed as a static JSON/data file that a real
`pytest`-gated test then compares against (`tests/fixtures/make_*.py` + the checked-in fixtures). There
is **no** existing example of a `pytest`-gated test that shells out to `Rscript` live during the actual
test run — R is not guaranteed present in CI (stated directly in the questions file, consistent with
everything found here: this session's own `RL` had to be manually pointed at a leftover scratch
install that isn't part of any tracked, reproducible environment).

## Codebase references (file:line, for the design stage)

- `bench/cr_rfsrc_check.R:1-8` — the exact `rfsrc()` call shape verified to run without error this session.
- `bench/s14_cr_parity.R:1-25`, `bench/s14_cr_parity.py:1-172` — the existing real-data (no ground truth) rfsrc comparison; 2-variable `Surv(time, status)`, subprocess-driven, manual only.
- `bench/cr_forest_truth_check.py:1-64` — existing truth-only (no external tool) CI gate; `ISE_THRESHOLD = 0.10`, single pooled per-cause-mean epsilon.
- `bench/s14_cr_sim.py:120-138` — `true_cif` (closed-form AJ) and `ise` (per-cause ISE vector), scenario A/B/C DGPs, `training_rows`/`test_paths`.
- `bench/tvc_coxtv_truth_check.py:1-59, 129-167` — Slice 11's exact template: DGP docstring discipline, `_baseline_step_at`/`_cox_survival` path-chaining (the mechanism found NOT to have an obvious rfsrc analog in Q1).
- `tests/test_tvc_coxtv_truth.py:1-65` — Slice 11's test file shape to mirror structurally.
- `src/rftvc/_competing.py:274-293` — `CompetingRisksForestTV.predict_cumulative_incidence`, the subject-path aggregation `randomForestSRC`'s predict does not do.
- `tests/fixtures/make_aj_fixtures.py:1,65`, `make_concordance_fixtures.py:1,50`, `make_survdiff_fixtures.py:1,38` — the only existing R-shell-out pattern, fixture-regeneration-only.
- `pyproject.toml:51-57` — `slow`/`network` markers, CI's default `pytest -m "not slow and not network"`.
