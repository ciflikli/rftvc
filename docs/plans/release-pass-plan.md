# Release-readiness pass: implementation plan (Stage 4)

Parent: `docs/plans/release-pass-research.md` (facts), `docs/plans/release-pass-design.md`
(Approach C, approved). Sequenced by cost-of-fixing-later, not by module: Track 1
(breaking-shaped, before any tag) → Track 2 (packaging/CI, parallel/independent) →
Track 3 (docs, trailing, non-gating). Tracks 1 and 2 can be worked as separate PRs/
sessions concurrently; within a track, slices are ordered by real dependency.

## Status
- [x] T1-slice-1 `random_state` contract unification (`_validation.py` + `_estimator.py` +
      `_competing.py` + `inspection.py`)
- [x] T1-slice-2 `_estimator.py` `TypeError`/`ValueError` split (`_check_int`,
      `_resolve_max_features`, `_resolve_n_draw`)
- [x] T1-slice-3 `ntime` coarsening `UserWarning`
- [x] T1-slice-4 Version-sync mechanism (`__init__.py`, `pyproject.toml`, both `Cargo.toml`s,
      CI version-agreement check)
- [ ] T2-slice-1 `pyproject.toml` packaging metadata + `py.typed`
- [ ] T2-slice-2 CI gap fixes (`ci.yml` Python 3.10 floor + Windows)
- [ ] T2-slice-3 `wheels.yml` TestPyPI → PyPI publish job with dry-run step
- [ ] T3-slice-1 Docstring `Examples` sections across the public API
- [ ] T3-slice-2 `api.rst` `CR_DTYPE`/`SURV_DTYPE` entries + `compatibility.rst` support matrix
- [x] T3-slice-3 `README.md` (new)
- [ ] T3-slice-4 `CHANGELOG.md` (new, Keep a Changelog, fresh start)

---

## Track 1 — breaking-shaped fixes (before any tag)

### T1-slice-1: `random_state` contract unification

**Why first in Track 1**: touches the widest surface (three modules' constructors plus
docstrings); the other Track 1 slices are smaller and independent of this one, but this
one should land first since T1-slice-2's `_check_int` split is unrelated in content and
can be reviewed on its own diff without this one's noise.

**Files:**
- `src/rftvc/_validation.py` — add `check_random_state_or_generator`.
- `src/rftvc/_estimator.py` — `_BaseForestTV._fit`, `random_state` docstring (both
  `SurvivalForestTV` and via `_BaseForestTV` shared text).
- `src/rftvc/_competing.py` — `CompetingRisksForestTV` docstring reference line.
- `src/rftvc/inspection.py` — `_entropy` becomes a thin wrapper over the shared helper (or
  is removed in favor of it); update its three `random_state` docstrings
  (`permutation_importance`, `drop_column_importance`, `hazard_effect`).
- `src/rftvc/landmark.py` — check for any `random_state` docstring text that references
  the estimator contract (`LandmarkSurvivalForest`/`LandmarkCompetingRisksForest` wrap
  `SurvivalForestTV`/`CompetingRisksForestTV`, confirm no separate contract statement
  needs updating).

**Concrete changes:**

```python
# src/rftvc/_validation.py — new, near the top-level helpers
from sklearn.utils import check_random_state


def check_random_state_or_generator(random_state):
    """int, RandomState, Generator, or None -> RandomState or Generator.

    Same contract everywhere ``random_state`` appears in the public API. Unlike
    ``sklearn.utils.check_random_state``, this accepts a ``numpy.random.Generator``
    and returns it unchanged (``Generator`` has no seedable-from-int-inside-sklearn
    path).
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return check_random_state(random_state)
```

`_estimator.py:124`:
```python
# before
rng = check_random_state(self.random_state)
# after
rng = check_random_state_or_generator(self.random_state)
```
Replace the `from sklearn.utils import check_random_state` import in `_estimator.py`
with `from ._validation import check_random_state_or_generator` (keep
`check_random_state` imported only if still used elsewhere in the file — check before
removing; `check_array`/`check_is_fitted` stay).

`inspection.py`'s `_entropy` (`inspection.py:67-77`) already implements this contract;
replace its body with a call to the shared helper plus its existing entropy-seed logic,
or keep `_entropy` as the local seed-derivation step but have it call
`check_random_state_or_generator` instead of duplicating the `isinstance(..., Generator)`
branch inline. Import from `._validation` alongside the module's existing
`_validation` import block (`inspection.py:14-22`).

Update every `random_state` parameter docstring (four estimator classes, three
`inspection` functions — `permutation_importance`, `drop_column_importance`,
`hazard_effect`) to the identical line: `int, RandomState instance, Generator, or
None, default=None`.

**Tests:**
- `tests/test_estimator.py` (or wherever `SurvivalForestTV`/`CompetingRisksForestTV`
  constructor validation is tested): add a case fitting with
  `random_state=np.random.default_rng(0)` and asserting it fits without error and
  gives deterministic results across two fits with the same `Generator` seed value
  (a fresh `default_rng(0)` each time, since a `Generator` instance is stateful and
  consumed by one fit).
- Same for `CompetingRisksForestTV`.
- `tests/test_inspection.py` (or module covering `inspection.py`): confirm
  `permutation_importance`/`hazard_effect` still accept a `Generator` (regression guard
  — this path already worked, must not break).
- Docstring-consistency: no new test required beyond existing docstring lint/build if
  the repo has one; otherwise this is covered by the docs-build CI job already passing.

**Acceptance criteria:**
- `SurvivalForestTV(random_state=np.random.default_rng(0)).fit(...)` and
  `CompetingRisksForestTV(random_state=np.random.default_rng(0)).fit(...)` both work.
- `_estimator.py` no longer calls `sklearn.utils.check_random_state` directly on
  `self.random_state`.
- All four estimator classes' and three `inspection` functions' `random_state`
  docstring lines read identically.
- Fast test suite green.

**T1-slice-1 done — deviations from the plan:**
- Only two `inspection` functions actually declare a `random_state` parameter
  (`permutation_importance`, `drop_column_importance`); `hazard_effect` doesn't take one.
  The plan's "three functions" was inaccurate — updated both that exist.
- `_competing.py` and `landmark.py` have no separate `random_state` contract text of
  their own (both just reference `SurvivalForestTV`'s docstring), so no edit was needed
  there beyond what `_estimator.py`'s docstring update already covers.
- `rng.randint(...)` (RandomState-only) vs `rng.integers(...)` (Generator-only) aren't
  interchangeable, so `_fit`'s engine-seed draw (`_estimator.py`, was line 147) needed a
  small new `rng_seed()` helper in `_validation.py` alongside
  `check_random_state_or_generator`, not called out in the plan's concrete-changes
  snippet. `inspection.py`'s `_entropy` already handled this split inline and needed no
  further change beyond swapping its fallback branch to the shared
  `check_random_state_or_generator`.
- Added regression tests for `Generator` support to `test_forest.py` (`SurvivalForestTV`),
  `test_cr_core.py` (`CompetingRisksForestTV`), `test_inspection_perm.py`
  (`permutation_importance`), and `test_inspection_loco.py` (`drop_column_importance`).
- `test_fit_design.py` monkeypatched `est_mod.check_random_state` by name; updated both
  occurrences to `check_random_state_or_generator`.
- Full fast suite (`pytest -m "not slow and not network"`) green: 654 passed, 4 skipped,
  96 xfailed.

---

### T1-slice-2: `_estimator.py` `TypeError`/`ValueError` split

**Files:**
- `src/rftvc/_estimator.py` — `_check_int` (`_estimator.py:411-414` region, exact line
  numbers may shift after T1-slice-1's import edits — locate by function name),
  `_resolve_max_features` (`_estimator.py:397-409`), `_resolve_n_draw`
  (`_estimator.py:383-395`).

**Concrete changes:**

```python
# _check_int — before
def _check_int(self, name, minimum):
    value = getattr(self, name)
    if not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")

# after
def _check_int(self, name, minimum):
    value = getattr(self, name)
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
```

```python
# _resolve_n_draw — before (relevant branches)
if isinstance(ms, (bool, np.bool_)):
    raise ValueError(f"invalid max_samples={ms!r}: use an int count or a float fraction")
...
raise ValueError(f"invalid max_samples={ms!r}")

# after
if isinstance(ms, (bool, np.bool_)):
    raise TypeError(f"invalid max_samples={ms!r}: use an int count or a float fraction")
if isinstance(ms, numbers.Integral):
    if ms < 1 or (ms > n_ids and not self.bootstrap):
        raise ValueError(f"max_samples={ms} must be in [1, n_ids={n_ids}] without bootstrap")
    return int(ms)
if isinstance(ms, numbers.Real) and 0 < ms <= 1:
    return max(1, int(round(ms * n_ids)))
raise TypeError(f"invalid max_samples={ms!r}")  # not an int, not a fraction-shaped float
```

```python
# _resolve_max_features — before (final branches)
if isinstance(mf, numbers.Integral) and 1 <= mf:
    return min(int(mf), p)
if isinstance(mf, numbers.Real) and 0 < mf <= 1:
    return max(1, int(mf * p))
raise ValueError(f"invalid max_features={mf!r}")

# after — an out-of-range int/float (e.g. mf=0, mf=1.5) is still a value problem;
# only a genuinely wrong type (e.g. a string not in {"sqrt","log2"}, or a list) is
# TypeError. Requires distinguishing "right type, wrong value" from "wrong type"
# explicitly rather than falling through to one raise:
if isinstance(mf, numbers.Integral):
    if mf < 1:
        raise ValueError(f"invalid max_features={mf!r}: int must be >= 1")
    return min(int(mf), p)
if isinstance(mf, numbers.Real):
    if not (0 < mf <= 1):
        raise ValueError(f"invalid max_features={mf!r}: float must be in (0, 1]")
    return max(1, int(mf * p))
raise TypeError(f"invalid max_features={mf!r}")
```

Message text is unchanged throughout (still `"...got {value!r}"` / `"invalid X=...: ..."`)
— only the exception class changes for the type-mismatch branches, per the design's
explicit note that this is the only intended change.

**Tests:**
- `tests/test_estimator.py`: for each of `n_estimators`, `min_ids_leaf`,
  `min_events_leaf`, `max_bins`, `oob_buffer` (whichever go through `_check_int`):
  a case passing a non-integer (e.g. `"5"`, `2.5`, `None`) asserts `pytest.raises(TypeError)`;
  a case passing an in-range-type-but-below-minimum int asserts `pytest.raises(ValueError)`.
- `max_features`: `pytest.raises(TypeError)` for e.g. `max_features=[1, 2]` or
  `max_features="bogus"` (not `"sqrt"`/`"log2"` and not numeric — confirm this still
  falls to the `TypeError` branch, since a bad string is arguably closer to a value
  problem than a type problem; if the design intends bad-string-enum values to stay
  `ValueError`, adjust the implementation above accordingly and note the deviation in
  this slice's PR description). `pytest.raises(ValueError)` for `max_features=0` or
  `max_features=1.5`.
- `max_samples`: `pytest.raises(TypeError)` for `max_samples=True` (currently a
  deliberate `bool` rejection) and for a non-numeric type; `pytest.raises(ValueError)`
  for an out-of-range int/float.
- Update any existing test currently asserting `pytest.raises(ValueError)` on what is
  now a type-mismatch input — grep `tests/` for these three functions' existing error
  tests before writing new ones, to avoid a duplicate/contradictory assertion.

**Acceptance criteria:**
- Every existing error-path test in `tests/` for `_check_int`/`_resolve_max_features`/
  `_resolve_n_draw` passes with the correct exception class (updated where the class
  changed).
- New type-vs-value test cases exist for all three functions and pass.
- Fast test suite green.

**T1-slice-2 done — deviations from the plan:**
- `max_features=True`/`np.True_` is silently accepted as `max_features=1` (bool is a
  `numbers.Integral` subclass, and `_resolve_max_features`'s int branch doesn't reject
  bools the way `_resolve_n_draw` explicitly does for `max_samples`). Left unchanged —
  pre-existing behavior, out of this slice's stated scope (only `max_samples` had an
  explicit bool-rejection branch to preserve). Not tested as an error case in
  `test_invalid_max_features_type`.
- Two existing tests asserted `ValueError` on what is now a type-mismatch input and were
  updated: `test_forest.py::test_invalid_max_samples` split into
  `test_invalid_max_samples_value` (kept `ValueError`) and
  `test_invalid_max_samples_type` (new, `TypeError`); `test_validation.py::test_invalid_params`
  dropped its `{"max_features": "bogus"}` case into a new
  `test_invalid_max_features_string_raises_type_error` (`TypeError`).
- New `ValueError`/`TypeError` parametrized cases added to `test_forest.py` for
  `max_features`, and for every `_check_int`-covered constructor param
  (`n_estimators`, `min_events_leaf`, `min_ids_leaf`, `max_depth`, `ntime`, `oob_buffer`).
- Full fast suite (`pytest -m "not slow and not network"`) green: 668 passed, 4 skipped,
  96 xfailed.
- Codex review of PR #42 caught two real issues, both fixed: (1) the out-of-range
  `ValueError` branches in `_resolve_n_draw`/`_resolve_max_features` had grown a
  `": float must be in (0, 1]"`/`": int must be >= 1"` suffix, contradicting this slice's
  own "message text is unchanged" acceptance criterion — reverted to the original
  `f"invalid max_samples={ms!r}"`/`f"invalid max_features={mf!r}"` text. (2) flagged that
  a non-scalar `max_features` (e.g. a numpy array) hits `mf == "sqrt"`'s ambiguous-truth
  `ValueError` before reaching the new type check — confirmed this is unchanged pre-existing
  behavior (same on `main` before this slice, since arrays were never a supported/documented
  type), left as-is.

---

### T1-slice-3: `ntime` coarsening `UserWarning`

**Files:**
- `src/rftvc/_estimator.py` — `_BaseForestTV._fit` (`_estimator.py:106-117` region,
  right after `n_coarsen_dropped_rows_`/`n_coarsen_lost_events_` are set).

**Concrete change:**

```python
# src/rftvc/_estimator.py, _fit(), right after:
#   self.coarse_grid_ = d.grid
#   self.n_coarsen_dropped_rows_ = d.n_rows - d.kept.size
#   self.n_coarsen_lost_events_ = d.lost
if self.n_coarsen_dropped_rows_ or self.n_coarsen_lost_events_:
    warnings.warn(
        f"ntime coarsening dropped {self.n_coarsen_dropped_rows_} row(s) and "
        f"{self.n_coarsen_lost_events_} event(s); see n_coarsen_dropped_rows_ / "
        "n_coarsen_lost_events_ on the fitted estimator.",
        UserWarning,
        stacklevel=2,
    )
```
`warnings` is already imported in `_estimator.py` (module header). Place this inside
the `if d.kept is not None:` branch (the coarse-mode path), not the `else` branch —
the attributes only exist when coarsening actually ran.

**Tests:**
- `tests/test_estimator.py`: a case fitting with `ntime=` set small enough (relative to
  a synthetic dataset with near-duplicate/adjacent event times) to force a nonzero
  `n_coarsen_dropped_rows_` or `n_coarsen_lost_events_`, asserting
  `pytest.warns(UserWarning, match="ntime coarsening")`.
- A companion case fitting with `ntime=None` or an `ntime` fine enough to drop nothing,
  asserting `warnings.catch_warnings()` records no such warning (no false positive).

**Acceptance criteria:**
- Fitting with lossy `ntime` coarsening reliably emits exactly one `UserWarning`
  matching the design's message text.
- Fitting with lossless (or no) coarsening emits no such warning.
- Fast test suite green (existing `ntime` tests updated if any assert "no warnings" via
  `-W error` or similar strictness that this new warning would now trip).

**T1-slice-3 done — deviations from the plan:**
- Tests live in `tests/test_coarsen.py` (the module that already owns `ntime`/coarsening
  tests), not a `tests/test_estimator.py` (which doesn't exist in this repo).
- The plan's suggested companion "no false positive" case (`ntime` fine enough to drop
  nothing) needed care: `TABLE`'s own rows always drop `B` regardless of grid resolution
  (delayed entry + event inside the same bin as the origin is a structural loss, not a
  granularity one — confirmed via `test_huge_ntime_is_the_exact_event_set`). Used a
  separate minimal two-id, no-delayed-entry fixture instead of reusing `TABLE` for the
  lossless case. Also added a plain `ntime=None` case (`test_no_ntime_does_not_warn`).
- Existing lossy-`ntime` tests (`test_single_node_matches_nelson_aalen_on_hand_coarsened_rows`,
  `test_event_times_are_the_coarse_grid_even_when_a_point_loses_its_events`, others in this
  file) were left unwrapped — no `-W error`/strict-warnings config exists in
  `pyproject.toml`, so the new warning doesn't fail them; it only appears in pytest's
  warnings summary.
- Full fast suite (`pytest -m "not slow and not network"`) green: 675 passed, 4 skipped,
  96 xfailed.
- Codex review of PR #43 caught a real issue, fixed: `stacklevel=2` pointed the warning
  at `fit()`'s own internal `return self._fit(...)` line, not the user's call site,
  since `fit()` is a thin wrapper around `_fit()` (two frames between the `warnings.warn`
  call and the user, not one). Bumped to `stacklevel=3`; verified with `python -W always`
  that a direct `SurvivalForestTV.fit(...)` call now attributes to the caller's own line.
  Landmark-wrapped fits (`LandmarkSurvivalForest.fit` → `_fit_forest` → `forest.fit` →
  `_fit`) have one more frame than this accounts for and will still be one frame off —
  not fixed, since a single fixed `stacklevel` can't be exactly right for both call
  depths, and direct estimator use is the common case.

---

### T1-slice-4: version-sync mechanism

**Why last in Track 1**: independent of the API/error-handling slices above (touches
`__init__.py`, `pyproject.toml`, `Cargo.toml`, CI — no shared files with T1-slice-1/2/3),
but grouped in Track 1 since the design flags it as a before-any-tag item (the
`0.1.0.dev0` vs `0.1.0` divergence becomes a live question for consumers the moment a
tag exists).

**Files:**
- `src/rftvc/__init__.py` — `__version__` (line 42).
- `pyproject.toml` — `version` under `[project]`.
- `rust/rftvc-py/Cargo.toml` — `version` under `[package]`.
- `rust/rftvc-core/Cargo.toml` — `version` under `[package]` (keep in sync with
  `rftvc-py`'s, per current convention of both starting at `0.1.0`).
- `.github/workflows/ci.yml` — new step in the `test` job (or a new lightweight job) to
  check `pyproject.toml` and `rust/rftvc-py/Cargo.toml` versions agree.

**Concrete changes:**

```python
# src/rftvc/__init__.py — before
__version__ = "0.1.0.dev0"

# after
from importlib.metadata import version as _version

__version__ = _version("rftvc")
```

Version bump convention going forward: `pyproject.toml`'s `version` (PEP 440, e.g.
`"0.1.0rc1"`) and `rust/rftvc-py/Cargo.toml`'s `version` (Cargo SemVer, e.g.
`"0.1.0-rc.1"`) are bumped together by hand at release time; `rust/rftvc-core/Cargo.toml`
tracks `rftvc-py`'s version as an internal convention (neither crate is published, so
this is cosmetic consistency, not a publish requirement).

CI version-agreement check (new step, `ci.yml`'s `test` job, after checkout, before the
Rust checks — cheap enough to run on every PR):
```yaml
      - name: Check pyproject/Cargo version agreement
        run: |
          PY_VER=$(grep -m1 '^version = ' pyproject.toml | sed -E 's/version = "([^"]+)"/\1/')
          RS_VER=$(grep -m1 '^version = ' rust/rftvc-py/Cargo.toml | sed -E 's/version = "([^"]+)"/\1/')
          # strip PEP 440 pre-release markers (rc1, .dev0) and Cargo's "-rc.1" to compare
          # only the dotted numeric core; exact prerelease-tag equivalence is a human
          # judgment call at release time, not something worth encoding in a regex here.
          PY_CORE=$(echo "$PY_VER" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')
          RS_CORE=$(echo "$RS_VER" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')
          if [ "$PY_CORE" != "$RS_CORE" ]; then
            echo "pyproject.toml version core ($PY_CORE) != Cargo.toml version core ($RS_CORE)"
            exit 1
          fi
```

**Tests:**
- `tests/test_package.py` (new, or add to an existing top-level test file): assert
  `rftvc.__version__` is a non-empty string matching `pyproject.toml`'s declared
  version core (read `pyproject.toml` at test time, or hardcode-and-update per release
  — prefer reading the file to avoid the test itself becoming another hand-synced
  copy).
- Manual/CI check: the new `ci.yml` step above is itself the test for the Cargo/pyproject
  agreement; no separate pytest needed for that half.

**Acceptance criteria:**
- `import rftvc; rftvc.__version__` reflects the installed package metadata, not a
  hardcoded literal.
- `pip install -e .` (editable, dev workflow) still resolves `rftvc.__version__`
  correctly via `importlib.metadata` (editable installs register metadata — confirm
  with a local `uv pip install -e .` + `python -c "import rftvc; print(rftvc.__version__)"`
  smoke check as part of this slice's manual verification, not necessarily a CI test).
- CI fails if `pyproject.toml` and `rust/rftvc-py/Cargo.toml` version cores diverge.
- Fast test suite green.

**T1-slice-4 done — deviations from the plan:**
- No version bump: `pyproject.toml` (`0.1.0.dev0`, core `0.1.0`) and both `Cargo.toml`s
  (`0.1.0`) already agreed on the core, so the CI check passes as-is; version bumping is
  deferred to actual release time per the plan's own convention note.
- New `tests/test_package.py` reads `pyproject.toml`/`rust/rftvc-py/Cargo.toml` with a
  plain regex, not `tomllib` — `tomllib` is 3.11+ only and T2-slice-2 (not yet landed)
  adds a Python 3.10 CI leg; a `tomllib`-based test would break there.
- Verified with `uv pip install --python .venv -e .` +
  `.venv/bin/python -c "import rftvc; print(rftvc.__version__)"` → `0.1.0.dev0`, matching
  the plan's manual smoke-check note.
- Full fast suite (`pytest -m "not slow and not network"`) green: 674 passed, 4 skipped,
  96 xfailed (this branch is based on `main` post-T1-slice-1/2, pre-T1-slice-3, hence the
  count differing from that PR's).

---

## Track 2 — packaging/CI (parallel, independent of Track 1)

### T2-slice-1: `pyproject.toml` packaging metadata + `py.typed`

**Files:**
- `pyproject.toml` — `[project]` table additions.
- `src/rftvc/py.typed` — new, empty marker file.
- `pyproject.toml`'s `[tool.maturin]` or a `MANIFEST`-equivalent — confirm `py.typed`
  ships in the built wheel (maturin's `python-source = "src"` should include it
  automatically since it's a plain file under the package directory; verify by
  building a wheel locally and checking its contents, not just assuming).

**Concrete changes:**

```toml
[project]
name = "rftvc"
version = "0.1.0.dev0"
description = "Scikit-learn compatible random survival forests with time-varying covariates"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "Gokhan Ciflikli", email = "gokhanciflikli@gmail.com" }]
keywords = ["survival-analysis", "random-forest", "scikit-learn", "time-varying-covariates"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Rust",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]
dependencies = ["numpy>=1.24", "scikit-learn>=1.6", "polars>=1.0", "narwhals>=1.30"]

[project.urls]
Homepage = "https://github.com/ciflikli/rftvc"
Documentation = "https://rftvc.readthedocs.io"
Repository = "https://github.com/ciflikli/rftvc"
Changelog = "https://github.com/ciflikli/rftvc/blob/main/CHANGELOG.md"
```
The `readme = "README.md"` key creates a hard dependency on T3-slice-3 existing before
this metadata is fully truthful (a `readme` key pointing at a nonexistent file breaks
the build) — **land T3-slice-3 first, or in the same PR as this slice**, even though
T3 is nominally "trailing." Note this as the one cross-track ordering exception; every
other Track 3 item is genuinely deferrable past the tag.

`classifiers`' Python-version list matches whatever T2-slice-2 (CI gap fix) actually
tests — if that slice lands first, use its final matrix here; if this slice lands
first, revisit the classifiers list once T2-slice-2 merges.

```
# src/rftvc/py.typed — new file, empty (PEP 561 marker)
```

**Tests:**
- A packaging smoke test (can be manual, documented in the PR description, or a CI step
  in `wheels.yml`'s existing `test` job): build a wheel, install it, and check
  `importlib.resources` / direct file inspection that `rftvc/py.typed` is present in the
  installed package (`python -c "import rftvc, pathlib; assert (pathlib.Path(rftvc.__file__).parent / 'py.typed').exists()"`).
- No new pytest unit test needed for `pyproject.toml` metadata itself (nothing in the
  Python API changes); the packaging smoke check above is the acceptance mechanism.

**Acceptance criteria:**
- `pyproject.toml` has `readme`, `authors`, `keywords`, `classifiers`, `[project.urls]`.
- `src/rftvc/py.typed` exists and ships inside a built wheel (verified once by local
  `maturin build` + wheel content inspection).
- `pip install rftvc` (once published) would show authors/urls/keywords on PyPI's
  project page — not verifiable pre-publish, but `pyproject.toml`'s TOML is valid and
  `python -m build --sdist` (or `maturin build --sdist`) succeeds without error.

---

### T2-slice-2: CI gap fixes (Python 3.10 floor + Windows)

**Files:**
- `.github/workflows/ci.yml` — `test` job's matrix and Python version.

**Concrete change:**

```yaml
# before
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      ...
      - name: Build and install
        run: |
          uv venv --python 3.11
          uv pip install -e . --group dev
      - name: Python tests
        run: .venv/bin/python -m pytest

# after
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: astral-sh/setup-uv@v6
      - name: Rust checks
        if: matrix.os != 'windows-latest' || matrix.python == '3.10'
        working-directory: rust
        run: |
          cargo fmt --all --check
          cargo clippy --all-targets -p rftvc-core -- -D warnings
          cargo test -p rftvc-core
      - name: Build and install
        run: |
          uv venv --python ${{ matrix.python }}
          uv pip install -e . --group dev
      - name: Python tests
        shell: bash
        run: .venv/bin/python -m pytest
```
The Rust-checks dedup (`if:` guard restricting `cargo fmt`/`clippy`/`cargo test` to one
combination) avoids running the identical Rust checks 6 times per push once the matrix
expands to 3 OSes × 2 Python versions; adjust the exact guard condition once this is
implemented and CI timing is observed — the illustrative condition above is a starting
point, not a hard requirement. `.venv/bin/python`/`.venv/bin/pytest` paths need a
Windows-compatible form (`shell: bash` on `windows-latest` runners with Git Bash
generally makes `.venv/bin/...`-style POSIX paths work via `uv venv`'s cross-platform
layout — confirm locally or in a draft PR before merging; if `uv venv` on Windows
produces `.venv/Scripts/` instead, branch the run command per-OS).

**Tests:**
- No new pytest tests — this slice's "test" is the CI matrix itself passing on all 6
  combinations (3 OS × 2 Python versions minus any Rust-check dedup skip).
- Watch for Windows-specific test failures surfaced for the first time (path
  separators, line endings, `RFTVC_DATA` cache directory handling in the network-marked
  fixtures) — fix as found; this slice's scope includes fixing any real Windows-only
  test failure it uncovers, not just adding the matrix and leaving red CI.

**Acceptance criteria:**
- `ci.yml`'s `test` job runs the full pytest suite on `windows-latest` and on Python
  3.10, in addition to the existing macOS/Linux + 3.11 combination.
- All matrix combinations green (any Windows-only failures found are fixed as part of
  this slice, per the design's explicit call-out that this is a real CI gap, not
  cosmetic).

---

### T2-slice-3: `wheels.yml` TestPyPI → PyPI publish job with dry-run

**Depends on**: T1-slice-4 (version-sync) landing first is not strictly required, but
this slice should not be the one that first cuts a real tag — sequence it so the
dry-run is exercised against a `0.1.0rc1`-shaped tag only after Track 1 is merged.

**Files:**
- `.github/workflows/wheels.yml` — new `publish` job(s), `id-token: write` permission,
  header comment update (the "Nothing is published in v1" line is now false and must be
  corrected).

**Concrete changes:**

```yaml
# wheels.yml header comment — before
# abi3 wheels (one per platform, CPython >= 3.10) built with maturin, then
# installed and smoke-tested on each OS. Nothing is published in v1.

# after
# abi3 wheels (one per platform, CPython >= 3.10) built with maturin, then
# installed and smoke-tested on each OS, then published: TestPyPI on every
# tag push (dry run via `--repository testpypi`), PyPI only on a tag matching
# the release pattern after the TestPyPI step succeeds.
```

```yaml
  publish-testpypi:
    name: publish to TestPyPI
    needs: [build, sdist, test]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheel-*
          merge-multiple: true
          path: dist
      - uses: actions/download-artifact@v4
        with:
          name: sdist
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    name: publish to PyPI
    needs: [publish-testpypi]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheel-*
          merge-multiple: true
          path: dist
      - uses: actions/download-artifact@v4
        with:
          name: sdist
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
```
Uses PyPI Trusted Publishing (OIDC via `id-token: write`), not a stored token secret —
requires the `testpypi`/`pypi` GitHub Environments to be configured with Trusted
Publisher entries on TestPyPI/PyPI's project settings first (a one-time manual setup
step outside this repo's files, documented in the PR description as a prerequisite,
not something this slice's diff can itself configure). The "dry-run" the design asks
for is `publish-testpypi` itself — a real upload to TestPyPI (the accepted meaning of
"dry run" for a package that has never been published, since PyPI has no true
no-op/simulate mode for `gh-action-pypi-publish`) gated to run before `publish-pypi`
ever executes.

**Tests:**
- No pytest tests. Acceptance is exercised by actually pushing a `v0.1.0rc1`-shaped tag
  once Track 1 has merged, and confirming: TestPyPI receives the upload, `publish-pypi`
  does not run until `publish-testpypi` succeeds, and (for the real first tag) PyPI
  receives the upload.
- Until that real tag is pushed, this slice's PR can be verified with
  `workflow_dispatch` plus a temporary `if:` override or a fork/test tag pushed to
  TestPyPI only (do not push a real PyPI-publishing tag as part of merging this PR —
  the actual `0.1.0rc1` tag push is a separate, later action after Track 1 and this
  slice are both merged).

**Acceptance criteria:**
- `wheels.yml` has a `publish-testpypi` job gated on `needs: [build, sdist, test]` and a
  tag-ref condition, using Trusted Publishing.
- `publish-pypi` is gated on `needs: [publish-testpypi]`, so PyPI never receives an
  upload that TestPyPI rejected.
- Header comment no longer claims nothing is published.
- No secrets (`TWINE_PASSWORD`/`PYPI_API_TOKEN`) are introduced — Trusted Publishing
  only.

---

## Track 3 — docs (trailing; can land before or after the first tag)

### T3-slice-1: docstring `Examples` sections across the public API

**Files:**
- `src/rftvc/_estimator.py` — `SurvivalForestTV` class docstring.
- `src/rftvc/_competing.py` — `CompetingRisksForestTV` class docstring.
- `src/rftvc/landmark.py` — `LandmarkSurvivalForest`, `LandmarkCompetingRisksForest`.
- `src/rftvc/inspection.py` — `permutation_importance`, `drop_column_importance`,
  `hazard_effect`, `path_effect`.
- `src/rftvc/metrics.py` — at least `piecewise_exponential_score`, `brier_landmark`,
  `concordance_index_cr` (the most-used entry points; full `__all__` coverage is the
  target but these three are the minimum bar for this slice to be worth merging alone).
- `src/rftvc/model_selection.py` — `landmark_cross_validate`.

**Concrete changes:** each docstring gains a numpydoc `Examples` section, e.g.:

```python
    """...existing Parameters/Attributes...

    Examples
    --------
    >>> import numpy as np
    >>> from rftvc import SurvivalForestTV, make_survival_y
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 2))
    >>> t = rng.exponential(1.0, size=50)
    >>> y = make_survival_y(np.minimum(t, 1.0), t <= 1.0)
    >>> model = SurvivalForestTV(n_estimators=20, random_state=0).fit(X, y)
    >>> model.predict_risk(X[:3], 1.0).shape
    (3,)
    """
```
Each example should be small enough to run in well under a second (`n_estimators=20`,
`n<=50`) and produce a deterministic, easily-asserted output (a shape, not exact
floats, unless a fixed `random_state` makes exact values worth pinning).

**Tests:**
- Add doctest collection: either a `pytest --doctest-modules` addition to
  `pyproject.toml`'s `[tool.pytest.ini_options]` (scoped to `src/rftvc/` only, to avoid
  doctest-collecting the test suite itself), or a dedicated
  `tests/test_docstring_examples.py` that runs `doctest.testmod` per module. Prefer the
  `--doctest-modules` route if it doesn't destabilize existing `addopts`; otherwise the
  dedicated test file. Either way, every new `Examples` block must actually execute in
  CI, not just look plausible.

**Acceptance criteria:**
- Every class/function listed in Files has a numpydoc `Examples` section.
- All `Examples` blocks pass as doctests in CI (new failures block the PR, not a later
  one).
- `docs` CI job (`sphinx-build -E -W --keep-going`) still passes with the new sections
  rendered.

---

### T3-slice-2: `api.rst` `CR_DTYPE`/`SURV_DTYPE` entries + `compatibility.rst` support matrix

**Files:**
- `docs/source/api.rst` — `Data` section (currently `api.rst:18-32`).
- `docs/source/compatibility.rst` — new support-matrix block.

**Concrete changes:**

```rst
.. api.rst, Data section — add CR_DTYPE/SURV_DTYPE to the existing autosummary list
   (they are np.dtype instances, not classes/functions; autosummary/numpydoc can still
   generate a stub page for them via :data: role treatment — confirm the generated page
   renders sensibly for a bare np.dtype object, since it has no rftvc-authored
   docstring per the research's finding; if autosummary renders unhelpfully, add a short
   .. data:: CR_DTYPE / .. data:: SURV_DTYPE block with a one-line description instead
   of relying on autosummary alone)

   make_survival_y
   check_survival_y
   make_competing_risks_y
   check_competing_risks_y
   check_counting_process
   make_landmark_data
   landmark_features
   LandmarkData
   CR_DTYPE
   SURV_DTYPE
```

```rst
.. compatibility.rst — new section, after the existing predict/score bullets

Support matrix
---------------

- Python: 3.10 - 3.13 (``requires-python = ">=3.10"``; CI tests 3.10 and 3.13).
- scikit-learn: >= 1.6.
- Platforms: Linux (x86_64, aarch64), macOS (aarch64, x86_64), Windows (x64) —
  prebuilt abi3 wheels for each, built and smoke-tested in ``wheels.yml``.
- polars >= 1.0, numpy >= 1.24, narwhals >= 1.30.
```
The exact Python/OS list here must match whatever T2-slice-2 lands as the real CI
matrix (not just `wheels.yml`'s build targets) — write this slice's text after
T2-slice-2 merges, or reconcile the two if they land out of order.

**Tests:**
- `tests/test_docs_claims.py` already exists (per S20's precedent) — add a case
  reading `docs/source/compatibility.rst` and asserting it contains `"3.10"` and a
  scikit-learn version string, so the support matrix can't silently be deleted by a
  later edit without breaking a test (mirroring S20's own guard pattern for
  `foundation.rst`).
- No new test needed for the `api.rst` autosummary addition beyond the existing `docs`
  CI job building without new Sphinx warnings.

**Acceptance criteria:**
- `CR_DTYPE`/`SURV_DTYPE` appear in `api.rst`'s `Data` autosummary and generate a page
  (even a minimal one) under `docs/_build/html/generated/api/`.
- `compatibility.rst` states an explicit Python/sklearn/platform support matrix that
  matches `pyproject.toml`/CI's actual values at merge time.
- `docs` CI job passes with no new Sphinx warnings.
- New `test_docs_claims.py` case passes.

---

### T3-slice-3: `README.md` (new)

**Depends on**: should land before or alongside T2-slice-1 (see that slice's note) —
`pyproject.toml`'s `readme = "README.md"` key requires this file to exist for the build
to succeed at all.

**Files:**
- `README.md` — new, repo root.

**Concrete content outline** (per design's "point at docs site / case studies rather
than duplicating them"):
- One-line tagline (reuse `docs/source/index.rst`'s: "Random survival forests for
  time-varying covariates, with a Rust engine and a scikit-learn compatible Python
  API").
- Short feature list (3-5 bullets, condensed from `index.rst`'s five, not copy-pasted
  verbatim at full length).
- Install instructions: `pip install rftvc` (once published; note pre-publish status
  honestly if this lands before the first tag — e.g. "not yet published; build from
  source with `maturin develop`" — update once `0.1.0` is on PyPI).
- One minimal quickstart code block (reuse `index.rst`'s three-liner: `make_survival_y`,
  `fit`, `predict_risk`).
- Links: docs site (`https://rftvc.readthedocs.io`), case studies
  (`docs/source/case_studies/`), `compatibility.rst`'s rendered page, `CHANGELOG.md`
  (once T3-slice-4 exists — link it even if added in the same PR).
- License line (MIT, matching `pyproject.toml`).

**Tests:**
- `tests/test_docs_claims.py` (or a new small check): assert `README.md` exists at repo
  root and contains a small set of stable substrings (e.g. the tagline, a `pip install
  rftvc` line, a link to `readthedocs.io`) — a light guard against accidental deletion,
  matching the existing docs-claims-guard convention rather than inventing a new
  mechanism.

**Acceptance criteria:**
- `README.md` exists, is non-duplicative of `index.rst`'s full prose (points at it
  instead), and renders correctly on GitHub (check via `gh repo view` or the PR preview,
  not just local Markdown rendering assumptions).
- `pyproject.toml`'s `readme = "README.md"` key (from T2-slice-1) resolves successfully
  in a local build.

**T3-slice-3 done — deviations from the plan:**
- Landed standalone (not bundled with T2-slice-1), since T2-slice-1 hasn't started yet —
  satisfies the plan's "land T3-slice-3 first" ordering option.
- Install instructions say "Not yet published to PyPI. Build from source with uv and
  maturin" (`uv pip install -e . --group dev`), not `maturin develop` as the plan's
  outline suggested — matches this repo's actual documented dev workflow
  ([[rftvc-dev-workflow]] memory, `ci.yml`), not a generic maturin invocation.
- The quickstart is a new small synthetic example (`numpy.random.default_rng` + 200
  rows), not `index.rst`'s three-liner verbatim — `index.rst`'s snippet references
  undefined variables (`stop`, `event`, `start`, `ids`, `X_path`, `y_path`) and isn't
  standalone-runnable; the README's version had to actually execute. Verified by
  extracting the fenced code block from the rendered `README.md` and running it
  byte-for-byte (`.venv/bin/python` on the extracted block) — ran clean, `risk.shape ==
  (5,)`.
- `CHANGELOG.md` link is present but T3-slice-4 hasn't landed yet, so it currently 404s
  — explicitly allowed by the plan's own "link it even if added in the same PR" note;
  will resolve once T3-slice-4 merges.
- The `compatibility.rst` link points at `https://rftvc.readthedocs.io/en/latest/compatibility.html`
  even though `compatibility.rst` doesn't yet have the T3-slice-2 support-matrix content
  (that slice hasn't landed) — the page itself already exists (pre-dates this pass), so
  the link isn't broken, just less complete than it will be.
- New test: `tests/test_docs_claims.py::test_readme_exists_and_has_stable_substrings`
  (existence + 4 stable substrings — tagline phrase, install command, readthedocs
  link, the estimator class name), following the file's existing guard convention.
- Full fast suite (`pytest -m "not slow and not network"`) green: 678 passed, 4 skipped,
  96 xfailed. GitHub rendering not yet checked (pending push/PR).
- Codex review of PR #46 caught a real gap, fixed: the install instructions named only
  uv and maturin, not the Rust toolchain maturin itself needs to build the PyO3
  extension (CI's own `dtolnay/rust-toolchain@stable` step confirms this is a real
  prerequisite, not implied by "maturin" alone). Reworded to name Rust explicitly.

---

### T3-slice-4: `CHANGELOG.md` (new)

Research Q7 (`release-pass-research.md` §7): no existing `CHANGELOG.md`; 33 merged
PRs use a slice/ticket-tag commit convention (`S20 T1: ...`, `Release pass: ...`), not
Conventional Commits. Per the task's own instruction, start fresh from this
release-pass point rather than reconstructing a 30-PR retrospective.

**Files:**
- `CHANGELOG.md` — new, repo root.

**Concrete content**, Keep a Changelog format (`https://keepachangelog.com/en/1.1.0/`),
starting at `[Unreleased]`:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries begin at the release-readiness pass (2026-09); earlier development is not
retroactively itemized here — see `git log` for the full slice-by-slice history.

## [Unreleased]

### Added
- `py.typed` marker for PEP 561 type-checking support.
- `random_state` now accepts a `numpy.random.Generator` everywhere it appears
  (estimators, `inspection` functions), matching what `inspection.py` already
  supported.
- A `UserWarning` when `ntime` coarsening drops rows or loses events.

### Changed
- `SurvivalForestTV`/`CompetingRisksForestTV` hyperparameter validation now raises
  `TypeError` for wrong-type arguments and `ValueError` for out-of-range values
  (previously both raised `ValueError`).
- `rftvc.__version__` now reflects installed package metadata instead of a
  hand-maintained literal.

### Fixed
- (release-pass fixes land here as they merge)
```
The exact entry list should be updated to match whichever Track 1/2 slices have
actually merged by the time this file is written — the skeleton above is a starting
point per the design's illustrative content, not a final, frozen list; true it up
against `git log` immediately before merging this slice.

**Tests:**
- `tests/test_docs_claims.py` (or similar): assert `CHANGELOG.md` exists and starts
  with `# Changelog` and contains `## [Unreleased]` — same lightweight existence/shape
  guard as `README.md`'s.

**Acceptance criteria:**
- `CHANGELOG.md` exists, follows Keep a Changelog format, and its `[Unreleased]`
  section accurately lists every Track 1/2 change merged so far at the time this slice
  lands (not a placeholder).
- No attempt to retroactively itemize all 33 prior PRs — explicitly scoped to start
  from the release pass, per research Q7's finding and the task instruction.
