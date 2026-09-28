# Release-readiness research findings

Parent: `docs/plans/release-pass-questions.md`. Research only — no recommendations, no
proposed fixes. Facts, file:line references, and direct excerpts for each question, plus
an additional-findings section for things the questions didn't anticipate.

## 1. Public API naming/signature consistency

Compared `src/rftvc/_estimator.py` (`SurvivalForestTV`), `src/rftvc/_competing.py`
(`CompetingRisksForestTV`), `src/rftvc/landmark.py` (`LandmarkSurvivalForest`,
`LandmarkCompetingRisksForest`), `src/rftvc/inspection.py`, `src/rftvc/metrics.py`,
`src/rftvc/model_selection.py` side by side.

**Shared counting-process params are identical between the two forests.** `SurvivalForestTV.__init__`
(`_estimator.py:520-538`) and `CompetingRisksForestTV.__init__` (`_competing.py:101-124`) use the
same names/defaults for `n_estimators=500`, `max_features="sqrt"`, `max_depth=None`,
`min_ids_leaf=15`, `min_events_leaf=3`, `max_bins=255`, `ntime=None`, `resample_unit="id"`,
`block_length=None`, `oob_buffer=1`, `max_samples=None`, `bootstrap=False`, `oob_score=False`,
`n_jobs=None`, `random_state=None`. `aggregate` is the one shared name with a **different**
default and value set by design: `"hazard"` (of `("hazard", "survival")`) for `SurvivalForestTV`
vs `"cif"` (of `("hazard", "cif")`) for `CompetingRisksForestTV` (`_estimator.py:97`,
`_competing.py:99`) — documented, not a bug, but the same string `"hazard"` means "ensemble
tree-average hazard" in both, so the option is genuinely shared, only the second option differs.

**Docstring *structure* differs between the two forests for parameters they do share.**
`SurvivalForestTV`'s docstring (`_estimator.py:430-486`) documents every parameter individually
with its own `name : type, default=...` line. `CompetingRisksForestTV`'s docstring
(`_competing.py:38-76`) instead groups most shared params into one line —
"`n_estimators, max_features, max_depth, min_ids_leaf, max_bins, ntime, resample_unit,
block_length, oob_buffer, max_samples, bootstrap, n_jobs, random_state` \ As in
`SurvivalForestTV`" (`_competing.py:40-43`) — but pulls `min_events_leaf` and `oob_score` back out
into their own fully-restated entries right after (`_competing.py:44-45, 52-54`) even though
their meaning and default (`3`, `False`) are unchanged from `SurvivalForestTV`. So the same
docstring convention ("as in X") is applied inconsistently even within one class's own
docstring — no stated reason why `min_events_leaf`/`oob_score` get pulled out and, e.g.,
`max_bins` doesn't.

**`random_state` accepts a different type set depending on where it's used.** The estimators'
`random_state` docstrings say `int, RandomState or None, default=None` (`_estimator.py:486`,
implied identical in `_competing.py`) and are passed straight to
`sklearn.utils.check_random_state` (`_estimator.py:124`), which does **not** accept a
`numpy.random.Generator`. `inspection.py`'s `_entropy` helper (`inspection.py:67-77`), used by
`permutation_importance`, `drop_column_importance` and `hazard_effect`'s `random_state`, is
documented as `int, RandomState, Generator or None` (`inspection.py:272-274, 867-868`) and
explicitly branches on `isinstance(random_state, np.random.Generator)` (`inspection.py:71-72`).
Same parameter name, same conceptual role ("seed the randomness"), different accepted types
between the estimator layer and the inspection layer.

**`oob_score` (estimator constructor) vs `oob` (inspection function argument) are two different
parameters with similar names and overlapping purpose.** `SurvivalForestTV(oob_score=...)`
(`_estimator.py:535`) toggles computing `oob_prediction_`/`oob_score_` at fit time.
`inspection.permutation_importance(..., oob=False, ...)` (`inspection.py:175`) toggles whether the
*importance* computation itself scores the training rows out-of-bag. They are related concepts
(both mean "out-of-bag") named `oob_score` in one place and `oob` in another, with no shared
naming convention documented.

**`n_jobs` type annotation differs across the surface for the same accepted values.**
`SurvivalForestTV`: `"n_jobs : int or None, default=None"` (`_estimator.py:484-485`).
`inspection.permutation_importance`: `"n_jobs : int, default=None"` (`inspection.py:275-276`,
also `drop_column_importance` at `inspection.py:869-870`) — omits `or None` even though `None` is
the stated default and is accepted (passed straight to `joblib.effective_n_jobs`, which accepts
`None`). Cosmetic, but a literal reading of the second form suggests `None` is not a valid value
even though it's the default.

**Numpydoc `Returns` sections are present in `metrics.py`/`inspection.py` functions but absent
from the four estimator classes' prediction methods.** `permutation_importance`, `hazard_effect`,
`path_effect`, `drop_column_importance` (`inspection.py`) and `piecewise_exponential_score`,
`brier_landmark`, etc. (`metrics.py`) all have an explicit `Returns\n-------` numpydoc section.
`SurvivalForestTV.predict_cumulative_hazard`, `.predict`, `.score`, `.predict_survival_function`,
`.predict_risk` (`_estimator.py:618-689`) and the equivalent `CompetingRisksForestTV` methods
(`_competing.py:260-337`) describe their return shape only in prose inside the summary line/body,
with no `Returns` header at all.

**`min_ids_leaf` vs the CR-only `min_events_leaf_cause` naming pattern.** `min_ids_leaf` (units)
and `min_events_leaf` (events) are both leaf-size floors shared by both forests. CR adds
`min_events_leaf_cause` (`_competing.py:69-71`), which follows the `min_events_leaf` naming
pattern but has no `min_ids_leaf_cause` counterpart — an intentional asymmetry (there is no
per-cause id-count floor), not flagged as such anywhere in the docstring.

**`causes` (vocabulary, plural) vs `cause` (single selection, singular) is used consistently**
across `CompetingRisksForestTV.causes`, `inspection.*`'s `cause=`, `metrics.concordance_index_cr`'s
`cause=`, and `landmark.LandmarkCompetingRisksForest.causes`/`score_cause` — no inconsistency found
here.

**`windows` (plural, `int` or edges) vs a single scoring window is consistent**: `windows=8` in
`permutation_importance`, `drop_column_importance`, `hazard_effect`, `metrics.event_windows` all
share the same "count or explicit edges" convention and default `8` (`inspection.py:183, 249,
535, 776, 849`; `metrics.py:586`).

## 2. Error handling consistency

**A real, largely-followed convention exists in `_validation.py`: `TypeError` for
structural/dtype problems, `ValueError` for value problems.** `_start_stop` raises `TypeError` for
a malformed/missing-field structured array (`_validation.py:118-136`, "expected a 1-d structured
array with fields..."), `_as_bool` raises `TypeError` for a non-boolean/non-{0,1} event column
(`_validation.py:87-95`), `split_frame` raises `ValueError` when `ids` names a column but `X`
isn't a DataFrame (`_validation.py:59-61`, arguably a type mismatch reported as `ValueError`).
Value violations (negative labels, `start >= stop`, no events, gaps, overlaps) are uniformly
`ValueError` throughout `_validation.py` and `_blocks.py`.

**The estimator layer's own parameter validation (`_estimator.py`) does not follow that
convention: it uses `ValueError` for both wrong-type and wrong-value parameter errors.**
`_check_int` (`_estimator.py:411-414`) raises `ValueError` whether `value` fails
`isinstance(value, numbers.Integral)` (a type problem) or fails the minimum bound (a value
problem): `"{name} must be an integer >= {minimum}, got {value!r}"`. Same pattern in
`_resolve_max_features` (`_estimator.py:397-409`, invalid `max_features` type or value both raise
`ValueError`) and `_resolve_n_draw` (`_estimator.py:383-395`, invalid `max_samples` type or bound
both raise `ValueError`). So two different conventions for "wrong kind of input" coexist in the
same module family: one (`_validation.py`, for `y`/`X`/`ids`) type-vs-value-separated, the other
(`_estimator.py`, for scalar hyperparameters) collapsed onto `ValueError`.

**One explicit `TypeError` use in the estimator/landmark layer for a genuinely wrong object
type**, distinct from the scalar-hyperparameter `ValueError` pattern above:
`LandmarkCompetingRisksForest.fit` — `"if not isinstance(forest, CompetingRisksForestTV): raise
TypeError(...)"` (`landmark.py:450-451`) — and `inspection._family`'s estimator-type check
(`inspection.py:46-55`, `raise TypeError` for an unsupported estimator class) and
`inspection.path_effect`'s landmark-estimator rejection (`inspection.py:693-694`, `raise
TypeError("path_effect is not defined for landmark estimators...")`). These three are consistent
with each other (wrong object type → `TypeError`) but this "wrong estimator type" pattern is
never used for the scalar-hyperparameter checks above, which raise `ValueError` even when the
actual problem is a type mismatch (e.g. passing a string where an int is expected in
`_check_int`).

**Message style is fairly uniform** — nearly every raised error interpolates the offending value
with `!r` and states what was expected, e.g. `f"resample_unit must be 'id' or 'block', got
{self.resample_unit!r}"` (`_estimator.py:354`), `f"kind must be 'average' or 'individual', got
{kind!r}"` (`inspection.py:568`), `f"scoring must be 'pe' for counting-process estimators...got
{scoring!r}"` (`inspection.py:323`). No module deviates from this "state the rule, then `got
{value!r}`" phrasing.

**Warnings are used consistently as the "loud but non-fatal" channel** — `UserWarning` is used for:
naive `strata=None` permutation (`inspection.py:353-357`), model-doesn't-beat-null
`share_of_gain=NaN` (`inspection.py:403-405`, `metrics.py`'s analogous
`piecewise_exponential_score` zero_rate warning at `metrics.py:787-793`), unpermuted-row fraction
(`inspection.py:388-394`), missing-cause-in-`causes` (`_validation.py:200-204`), rows with no
out-of-bag tree (`_estimator.py:606-612`, `_competing.py:220-226`), and block-OOB leak risk under
landmarking (`landmark.py:277-284`). All consistently `warnings.warn(..., UserWarning,
stacklevel=...)`.

**A genuinely silent (no warning) path: coarsening (`ntime`) can drop rows and lose events with
no warning emitted, only via after-the-fact attributes.** `_fit` (`_estimator.py:108-117`) sets
`n_coarsen_dropped_rows_` and `n_coarsen_lost_events_` whenever `ntime` coarsening drops rows or
loses events, but no `warnings.warn` call accompanies this — a user must know to inspect these
attributes after `fit` to discover data was silently dropped. Contrast with
`piecewise_exponential_score`'s `zero_rate_share`, which is also an informational fraction but
*does* trigger a `UserWarning` above a 1% threshold (`metrics.py:787-793`). No comparable
threshold/warning exists for `n_coarsen_dropped_rows_`/`n_coarsen_lost_events_`, however large a
fraction of rows/events they represent.

**A second silent-by-design path, documented but unwarned: `hazard_effect`'s `support_mask`.**
"A window with no row at risk gives `NaN` for every grid value" (`inspection.py:514-515`) — this
is stated in the docstring but, unlike the structurally similar `zero_rate_share` mechanism in
`piecewise_exponential_score`, there is no `UserWarning` raised when a caller's grid/window
combination produces an all-NaN or mostly-NaN row of the result; the caller must read
`support_mask` themselves.

**`_score_landmark`'s conversion of `UndefinedMetricError` to `NaN` is documented, not silent**:
`landmark_cross_validate`'s docstring explicitly states "A score that raises
`metrics.UndefinedMetricError` at a landmark is NaN there; any other error propagates"
(`model_selection.py:329-330`), and the `try/except UndefinedMetricError` in `_score_landmark`
(`model_selection.py:271-273`) matches that contract exactly — this is a documented, intentional
degrade-to-NaN, not an undocumented silent failure.

## 3. Docstring completeness and consistency

**Every name in `rftvc.__all__` (`src/rftvc/__init__.py:23-41`) has *some* docstring**, with two
exceptions that cannot have one in the normal sense: `CR_DTYPE` and `SURV_DTYPE`
(`_validation.py:10-11`) are plain `np.dtype(...)` literals with no docstring attached (module-level
comments only: `"""Validation of the counting-process survival target."""` is the module
docstring, not theirs) — `help(rftvc.CR_DTYPE)` shows numpy's generic dtype help, not anything
rftvc-authored. Every class and function has a docstring with at least a summary line; the four
estimator/landmark classes and `inspection`'s four functions have full `Parameters`/`Returns` (or
`Parameters`/`Attributes`) sections. `metrics.py`'s eleven `__all__` names all have docstrings,
including the two "thin" ones: `UndefinedMetricError` (`metrics.py:47-48`, one-line docstring, no
Parameters — appropriate for an exception class) and `PEScore` (`metrics.py:566-583`, a
`NamedTuple` with a docstring documenting its non-obvious fields, no Parameters section — also
appropriate, since it's a return type users don't construct directly).

**`model_selection.__all__`'s three names** (`GroupTimeSplit`, `RollingOriginSplit`,
`landmark_cross_validate`; `model_selection.py:34`) all have docstrings.
`landmark_cross_validate`'s is the most thorough in the codebase (`model_selection.py:291-347`,
full `Parameters`/`Returns`). `RollingOriginSplit`/`GroupTimeSplit` document their constructor
params (`model_selection.py:75-84, 136-143`) but not their `split`/`get_n_splits` methods
individually (sklearn-splitter convention, consistent with how `sklearn.model_selection`'s own
splitters document — not flagged as a gap).

**No docstring in the public surface has a numpydoc `Examples` section.** Checked all four
estimator classes (`_estimator.py:417-518`, `_competing.py:18-97`, `landmark.py:305-335,
392-416`), all four `inspection` functions (`inspection.py:159-303, 507-566, 635-691, 766-899`),
and every `metrics`/`model_selection` `__all__` member — none contains an `Examples` section, only
`Parameters`/`Attributes`/`Returns` prose. The one runnable code sample in the entire docs surface
is the quickstart snippet in `docs/source/index.rst:16-22` (not attached to any specific
docstring).

**`docs/source/api.rst`'s autosummary lists match `__all__` for `model_selection`, `inspection`,
and `metrics` exactly**, but the top-level `rftvc` "Data" section
(`docs/source/api.rst:18-32`) omits two names that *are* in `rftvc.__all__`:
`CR_DTYPE` and `SURV_DTYPE` (`__init__.py:24-25`) are exported but never listed in `api.rst`'s
autosummary anywhere, so they have no generated API reference page even though they are part of
the public surface an import-star or `dir(rftvc)` user would see. Every other `rftvc.__all__`
name (the four estimators, `LandmarkData`, `landmark_features`, `make_landmark_data`, the four
`check_*`/`make_*` validation helpers) is listed in `api.rst`. The three submodules `inspection`,
`metrics`, `model_selection` are themselves in `rftvc.__all__` (`__init__.py:38-40`) but are
documented via `.. currentmodule::` sections rather than an autosummary entry for the submodule
object itself — consistent with how Sphinx/numpydoc projects normally structure this, not flagged
as a gap.

## 4. Versioning mechanics

`__version__` is a **hardcoded literal string**: `src/rftvc/__init__.py:42`,
`__version__ = "0.1.0.dev0"`, matching `pyproject.toml:7`'s `version = "0.1.0.dev0"` by hand, with
no `importlib.metadata.version("rftvc")` call and no maturin build-time substitution mechanism in
use (maturin's `[tool.maturin]` section, `pyproject.toml:29-32`, does not reference a version file
or `dynamic = ["version"]`; `[project]` does not declare `version` as `dynamic`, so maturin does
not manage it — the two copies are independent strings that must be kept in sync by hand).

`rust/rftvc-py/Cargo.toml:3` and `rust/rftvc-core/Cargo.toml:3` both say plain `version = "0.1.0"`
— no `-dev`/prerelease suffix. Cargo's SemVer format supports prerelease identifiers
(`0.1.0-dev.0`) but neither crate uses one. Neither crate has a `publish` key set (checked both
`Cargo.toml` files and the workspace `rust/Cargo.toml`; `grep -rn "publish"` found nothing), so
there is no explicit guard against an accidental `cargo publish` of either internal crate, though
nothing in CI currently runs `cargo publish`.

## 5. PyPI packaging readiness

`pyproject.toml`'s entire `[project]` table (`pyproject.toml:5-11`):
```
name = "rftvc"
version = "0.1.0.dev0"
description = "Scikit-learn compatible random survival forests with time-varying covariates"
requires-python = ">=3.10"
license = "MIT"
dependencies = ["numpy>=1.24", "scikit-learn>=1.6", "polars>=1.0", "narwhals>=1.30"]
```
No `readme`, `authors`/`maintainers`, `classifiers`, `keywords`, or `urls` (homepage/repository/
documentation) keys are present anywhere in the file (confirmed by grepping `pyproject.toml` for
each). There is also no `[project.optional-dependencies]` (the dev/docs groups are
`[dependency-groups]`, a `uv`/PEP 735-style table, not `optional-dependencies`, so a plain `pip
install rftvc[dev]` would not resolve them from a published sdist/wheel the way
`optional-dependencies` would).

`.github/workflows/wheels.yml` builds abi3 wheels on 5 os/target combinations plus an sdist, and
runs an install+smoke-test job, but has **no publish job** of any kind: no `pypa/gh-action-pypi-publish`
step, no `twine upload`, no `id-token: write` permission block (needed for PyPI Trusted
Publishing), and no reference to `TWINE_PASSWORD`/`PYPI_API_TOKEN` secrets anywhere in the file.
There is also no separate TestPyPI job or `repository-url` override anywhere in the repo (checked
`.github/workflows/*.yml` in full). The workflow's own header comment states this directly:
`"# abi3 wheels (...) built with maturin, then installed and smoke-tested on each OS. Nothing is
published in v1."` (`wheels.yml:3-4`).

## 6. README

There is no root `README.md` (confirmed: `ls -a .` at repo root lists no `README*` file).
`docs/source/index.rst` (full contents read) already contains: a one-line tagline ("Random
survival forests for time-varying covariates, with a Rust engine and a scikit-learn compatible
Python API"), a five-bullet feature list (counting-process data with delayed entry, whole-subject
resampling, landmark super-model, time-aware CV/IPCW metrics, log-rank criterion + coarse grid), a
runnable quickstart code block (`index.rst:16-22`, three lines: `make_survival_y`, `fit`, and a
path-based `predict_risk` call), and a toctree linking `user_guide/index`, `case_studies/index`,
`compatibility`, `api`. `docs/source/compatibility.rst` documents the scikit-learn `Pipeline`/
`cross_validate`/`GridSearchCV` compatibility story in more depth than the index page. Five case
studies exist under `docs/source/case_studies/` (`btscs.rst`, `ebmt4.rst`, `pbc2.rst`,
`rossi.rst`, plus `index.rst`), any of which a README could point to rather than restate.

## 7. Changelog scope

There is no `CHANGELOG.md` anywhere in the repo (confirmed by `find`/`ls`, matches the "Known
already" note). `git log --oneline --merges` shows **33 merged pull requests** total (not ~30, a
slightly higher count than the parent doc's estimate), spanning from early slices (S5 onward, per
memory) through the just-completed `release-pass/rc-validation-determinism` PR (`#33`, the most
recent merge, `4984d5f`). Commit-message style on `main` (`git log --oneline`, 151 commits total)
is consistent throughout: short imperative/descriptive subject lines prefixed by a slice or topic
tag, e.g. `"S20 T1: extract metrics._window_exposure_1"`, `"S19: record user approval of the §7.2
history-rule deviation"`, `"RC validation T4: examples/rc_validation.py"`, `"Release pass:
pyarrow-free polars note + routine real-data warnings docs"` — this is closer to a slice/ticket
convention (each commit tagged with its originating plan slice, e.g. `S19`, `S20`, `RC validation`,
`Release pass`) than to Conventional Commits (`feat:`/`fix:`/`docs:` prefixes) or any changelog-
generation-tool convention; no commit in the sampled 40 uses a Conventional Commits prefix.

## 8. Compatibility/deprecation surface

`docs/source/compatibility.rst` (full contents read) documents: `get_params`/`set_params`/`clone`/
pickling support; `n_features_in_`/`feature_names_in_` (any pandas/polars/pyarrow DataFrame via
narwhals, with a pyarrow-free polars-to-pandas conversion snippet); `predict`/`score` semantics for
each estimator (mortality + counting-process C for `SurvivalForestTV`, `F_k` of `score_cause` +
Wolbers' C for `CompetingRisksForestTV`); metadata routing for `ids` via
`set_fit_request(ids=True)`/`set_score_request(ids=True)`; the structured-array `y` and its
implications for `check_estimator` (a generated table of expected `check_estimator` failures and
their survival-adapted replacement tests, from `docs/source/generated/compat.csv`); and a note that
`LandmarkSurvivalForest`/`LandmarkCompetingRisksForest` are outside `check_estimator`'s scope
entirely (tested directly instead). **It does not state a Python version floor/ceiling, a
scikit-learn version floor/ceiling, or a platform/OS support list anywhere in its own text** — no
mention of `3.10`, no mention of `scikit-learn>=1.6`, no mention of the five wheels.yml build
targets. Those numbers live only in `pyproject.toml:9` (`requires-python = ">=3.10"`,
`dependencies = [..., "scikit-learn>=1.6", ...]`, `pyproject.toml:11`) and
`.github/workflows/wheels.yml:16-21` (linux x86_64/aarch64, macOS aarch64/x86_64-15-intel, windows
x64) and `wheels.yml:56-58` (tested against Python 3.10 and 3.13 specifically, not the full
3.10-3.13 range). So there is no cross-check possible from `compatibility.rst`'s own text against
`pyproject.toml`/`wheels.yml`, because the page never restates the promise in the first place —
the only place a user would learn the supported Python/sklearn/platform matrix is `pyproject.toml`
and the CI YAML themselves, not the docs page whose title is "scikit-learn compatibility."

## Additional findings

- **`ci.yml`'s Python test job only runs on `ubuntu-latest`/`macos-latest`, and only tests one
  Python version (whatever `uv venv --python 3.11` picks up, no matrix)**, while `requires-python
  = ">=3.10"` and `wheels.yml` explicitly builds/tests Python 3.10 and 3.13. So the merge-gating CI
  never actually runs the Python test suite on 3.10 (the stated floor) or on Windows, only the
  release-only `wheels.yml` smoke test (a tiny 6-line script, not the real pytest suite) touches
  those. A regression that only affects Python 3.10 or Windows would not be caught by `ci.yml` at
  all, only (partially) by the wheels smoke test on tag push.

- **`[dependency-groups]` (PEP 735 / uv-native) is used for `dev`/`docs` extras instead of the more
  universally-installable `[project.optional-dependencies]`.** This works fine with `uv` (as CI's
  `uv pip install -e . --group dev` shows, `ci.yml:16-17`) but a plain `pip install
  rftvc[docs]` against a published PyPI wheel would not resolve these groups at all — `pip`
  does not read `[dependency-groups]`. Anyone wanting the docs/dev extras from a PyPI install
  (rather than a source checkout with `uv`) currently has no path to get them.

- **`pyproject.toml` has no `[project.scripts]`/console entry points, and none of the example
  scripts (`examples/rc_validation.py`, `examples/rossi_case_study.py`, etc.) are packaged or
  referenced from `pyproject.toml`** — they live purely as a `docs/source/case_studies/`-adjacent
  convenience for building the docs, with no `MANIFEST.in`/`package_data` question yet resolved for
  whether they should ship in the sdist.

- **The module docstring convention itself is inconsistent in scope**: some files open with a
  one-line docstring (`_estimator.py:1`, `"""Scikit-learn compatible survival forest
  estimator."""`), others open with a multi-paragraph design explainer
  (`landmark.py:1-8`, `metrics.py:1-20`) that duplicates content also present in the corresponding
  `docs/source/user_guide/*.rst` page (e.g. `metrics.py`'s censoring-weight explanation vs.
  `evaluation.rst`) — not checked for divergence between the two copies, but a second source of the
  same explanation that could drift.

- **`inspection.py`'s `_HazardEffectResult.values` warning workaround is itself undiscoverable from
  `api.rst`**: the class `_HazardEffectResult` (`inspection.py:28-43`) is not in `__all__` and not
  documented on its own page; a user only learns about the `.values`-shadows-`dict.values` footgun
  by reading `hazard_effect`'s docstring (`inspection.py:551-556`) closely, not from any type
  reference (the `Returns` section says "Bunch with `values`, ...", not `_HazardEffectResult`).

- **No `py.typed` marker file was found under `src/rftvc/`** (not checked directly against a
  question above, but relevant to packaging completeness): a `pip install`ed rftvc would not
  advertise itself as type-checked to `mypy`/`pyright` consumers even if the annotations are
  otherwise reasonable, since PEP 561 requires an explicit `py.typed` marker to opt in.
