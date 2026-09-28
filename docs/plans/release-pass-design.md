# Release-Readiness Pass — Design Discussion (Stage 3)

Parent research: `docs/plans/release-pass-research.md`. This stage proposes sequencing
options and picks one, to feed a Stage 4 vertical-slice implementation plan.

## Executive summary

rftvc has never published a real wheel — `wheels.yml` says so explicitly ("Nothing is
published in v1") — so every "public API" gap the research found (the `random_state`
type mismatch, `_estimator.py` collapsing `TypeError`/`ValueError`, the silent `ntime`
coarsening path) is currently free to fix: no external caller can be depending on the
current, inconsistent contract. That is not true of docs/packaging metadata, which is
cheap to iterate forever via ordinary patch releases after publish. The sequencing
question is therefore not "everything vs. something" but "what has a closing window
before the first tag vs. what doesn't." Recommendation: fix the breaking-change-shaped
items (API/error contracts, silent-failure warning, version-sync mechanism) before
cutting `0.1.0rc1`, do packaging/publish-infra in parallel since it's independent work,
and let docstring/README/CHANGELOG polish trail into the rc period and even past
`0.1.0` final rather than gating the first publish.

## Approaches considered

### A — Big bang: fix everything before any release

Do all eight research areas — naming/signature consistency, error-handling, docstrings,
versioning, packaging, README, changelog, compatibility docs — as one exhaustive pass,
then cut `0.1.0` (no `rc`) once it's all done.

- **Pro**: one release, maximally polished, no "known gaps" caveat needed anywhere.
- **Con**: mixes correctness-shaped fixes (random_state, error types) with pure-cosmetic
  ones (Examples sections, docstring restructuring) in a single unbounded review pass.
  Given the repo's own history — ~30 small S-slice PRs rather than large batches — this
  is a scope-creep risk and defers the actual goal (getting a real wheel on PyPI so
  outside users can install it) behind an open-ended "polish" backlog with no natural
  stopping point. It also wastes the one thing an rc *is for*: exercising the publish
  pipeline (TestPyPI dry run, trusted publishing, wheel smoke tests) before it matters.

### B — Ship packaging/docs first as `0.1.0rc1`, defer API fixes to a later rc

Get `pyproject.toml`, the publish job, README, CHANGELOG, and docs consistency done
first and tag `0.1.0rc1` to validate the TestPyPI → PyPI pipeline end-to-end. Do the
`random_state`/error-handling/silent-failure fixes in a follow-up `rc2` before calling
`0.1.0` final.

- **Pro**: fastest path to *seeing a wheel install from PyPI*, which is the one thing
  that's never been exercised and carries the most unknown risk (trusted publishing
  config, abi3 tag correctness, sdist completeness).
- **Con**: gets the sequencing backwards for the thing that actually has a closing
  window. The moment `rc1` is public, someone can `pip install rftvc==0.1.0rc1` and
  write code that does `except ValueError` around a call that should raise `TypeError`,
  or passes a `numpy.random.Generator` to an estimator's `random_state` expecting it to
  work because `inspection.py` accepts one. Fixing it in `rc2` is now a breaking change
  *against a published release*, even a pre-release — exactly the cost this pass exists
  to avoid paying later. This approach spends the free-fixes window on the wrong thing.

### C (recommended) — API/error/silent-failure fixes first, packaging in parallel, docs trailing

Sequence by "cost of fixing later," not by module or by research-question order:

1. **Now, before any tag**: the items that become expensive the instant something is
   published — `random_state` type contract, `_estimator.py`'s collapsed
   `TypeError`/`ValueError`, the `ntime` coarsening silent-failure warning, and the
   version-sync mechanism (fixing the `0.1.0.dev0` vs `0.1.0` divergence once, before a
   tag makes "which number is canonical" a live question for consumers of the git
   history).
2. **In parallel, independent of (1)**: packaging metadata (`pyproject.toml`
   classifiers/authors/urls/keywords/readme, `py.typed`, the `wheels.yml` publish job
   with a TestPyPI dry run). None of this touches the Python API surface, so it has no
   sequencing dependency on (1) and can be a separate PR track.
3. **Trailing, can continue past first publish**: docstring `Examples` sections,
   `api.rst`'s `CR_DTYPE`/`SURV_DTYPE` entries, `compatibility.rst`'s support-matrix
   statement, README, CHANGELOG. All of these are additive, non-breaking, and normal
   patch-release material — there is no reason to gate `0.1.0rc1` on them being
   complete, only on them being *started* (a README and CHANGELOG existing at all is a
   near-zero-cost bar to clear before tagging; a full Examples-section sweep is not).

Then: `0.1.0rc1` → TestPyPI dry run once (1) and (2) are done → `0.1.0` final to PyPI
once the rc round-trip is verified, with (3) allowed to keep landing as ordinary PRs
before and after.

- **Pro**: spends the one non-renewable resource (the pre-publish window where API
  changes are free) on the things that need it, and treats the renewable resource
  (docs, forever patchable) accordingly. Keeps the existing small-slice PR cadence
  instead of introducing one large "release pass" commit.
- **Con**: requires discipline to actually cut `0.1.0rc1` before docs are "done," which
  may feel premature to a maintainer used to polishing before shipping. Mitigated by
  treating `rc1` as explicitly pre-release and by having README/CHANGELOG minimally
  present (not exhaustive) as the actual gate.

## Recommendation

**Approach C.** The deciding factor is asymmetry of reversal cost, not thoroughness or
speed: API/error-type contracts and the version scheme are cheap to change now and
expensive after even one published `rc`, while docs/packaging-metadata gaps are cheap
to change at any point in the project's life. Sequencing by that asymmetry — not by
"everything" (A) or by "whatever unblocks a wheel fastest" (B) — is what actually
protects the first PyPI publish from becoming the thing that locks in today's
inconsistencies.

## Illustrative snippets

### `random_state` contract — unify on the wider (sklearn + Generator) contract

The inspection layer's contract is strictly more permissive and already implemented
(`inspection.py:67-77`); the fix is to adopt it in the estimator layer rather than
invent a third variant:

```python
# src/rftvc/_validation.py (new shared helper, used by _estimator.py, _competing.py,
# and inspection.py in place of each module's own random-state handling)
def check_random_state_or_generator(random_state):
    """int, RandomState, Generator, or None -> RandomState or Generator.

    Same contract everywhere `random_state` appears in the public API. Unlike
    sklearn.utils.check_random_state, this accepts a numpy.random.Generator and
    returns it unchanged (Generator has no seedable-from-int-inside-sklearn path).
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return check_random_state(random_state)  # sklearn.utils, unchanged for int/RandomState/None
```

Docstrings for every `random_state` parameter (four estimators, three `inspection`
functions) get the same line: `int, RandomState instance, Generator, or None,
default=None`. This is the only breaking-shaped change here — worth doing once, now.

### `_estimator.py` error types — split `_check_int` the way `_validation.py` already does

```python
# src/rftvc/_estimator.py — before
def _check_int(value, name, minimum):
    if not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)

# after — matches _validation.py's TypeError-for-structure / ValueError-for-value split
def _check_int(value, name, minimum):
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)
```

Same split applies to `_resolve_max_features` and `_resolve_n_draw`. Message text is
unchanged (already follows the `"...got {value!r}"` convention) — only the exception
class changes for the type-mismatch branch.

### `ntime` silent-drop path — warn like `piecewise_exponential_score` already does

```python
# src/rftvc/_estimator.py, _fit(), right after n_coarsen_dropped_rows_ /
# n_coarsen_lost_events_ are computed — mirrors metrics.py:787-793's zero_rate pattern
if n_coarsen_dropped_rows_ or n_coarsen_lost_events_:
    warnings.warn(
        f"ntime coarsening dropped {n_coarsen_dropped_rows_} row(s) and "
        f"{n_coarsen_lost_events_} event(s); see n_coarsen_dropped_rows_ / "
        "n_coarsen_lost_events_ on the fitted estimator.",
        UserWarning,
        stacklevel=2,
    )
```

### `pyproject.toml` — minimum `[project]` table for a real PyPI listing

```toml
[project]
name = "rftvc"
version = "0.1.0rc1"
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

(`classifiers`' Python-version list should match whatever `ci.yml`'s matrix actually
tests after the CI-gap fix, not just `wheels.yml`'s smoke-test versions.)

### Version-sync mechanism — single source of truth, no hand-kept duplicate

Simplest fix given maturin already builds this package: let `Cargo.toml` stay the
build-tool version of record and have Python read the *installed* metadata rather than
hardcode a string a second time:

```python
# src/rftvc/__init__.py — before
__version__ = "0.1.0.dev0"

# after
from importlib.metadata import version as _version
__version__ = _version("rftvc")
```

```toml
# rust/rftvc-py/Cargo.toml — adopt the same prerelease scheme pyproject.toml uses,
# so "0.1.0rc1" (PEP 440) and "0.1.0-rc.1" (Cargo SemVer) refer to the same release
# instead of two numbers that happen to collide only at "0.1.0"
version = "0.1.0-rc.1"
```

This removes the hand-sync burden going forward (Python always reflects whatever was
actually installed) but still requires bumping two files per release; a CI check that
fails if `pyproject.toml`'s version and `Cargo.toml`'s version disagree (stripping
PEP 440 vs. SemVer prerelease syntax for comparison) is cheap insurance and can be
added as part of the packaging-track PR rather than the API-fix track.
