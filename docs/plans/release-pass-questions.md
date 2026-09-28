# Research Questions: Release-Readiness Pass

Parent: user request 2026-09-27, last item on the roadmap before rftvc has outside users ([[rftvc-roadmap]] memory). Scope per that memory: API consistency, docs, naming, error handling/defaults, packaging/PyPI setup, version bump off `0.1.0.dev0`, changelog. This covers the *whole public surface* (estimators, `inspection`, `metrics`, `model_selection`, `landmark`, `_validation`'s public helpers) as it stands after S16-S20 + the rc-validation slice, not any single module.

## Known already (no need to re-research)
- No `README.md` at repo root, no `CHANGELOG.md` anywhere.
- `pyproject.toml` version is `0.1.0.dev0`; `src/rftvc/__init__.py` hardcodes a matching `__version__ = "0.1.0.dev0"` string (not derived from package metadata).
- `.github/workflows/wheels.yml` builds abi3 wheels on 5 targets on tag push, but its own header comment says "Nothing is published in v1" — no PyPI publish step exists yet.
- `.github/workflows/ci.yml` runs Rust checks (fmt/clippy/test) + Python pytest + a docs build job.
- LICENSE file exists (MIT, matches `pyproject.toml`'s `license = "MIT"`).
- Public API surface is `src/rftvc/__init__.py`'s `__all__` (estimators, landmark, validation helpers, and the `inspection`/`metrics`/`model_selection` submodules) plus whatever each of those submodules exports.
- Error handling already uses `ValueError`/`TypeError` throughout `_validation.py` and `_blocks.py` with descriptive messages — no custom exception hierarchy.
- Docs site exists at `docs/source/` (Sphinx, pydata theme) with `api.rst`, `user_guide/`, `case_studies/`, `compatibility.rst` — this is not a from-scratch docs job, it's a consistency/completeness review of what's there.

## Questions

1. **Public API naming/signature consistency.** Across the four estimator classes (`SurvivalForestTV`, `CompetingRisksForestTV`, `LandmarkSurvivalForest`, `LandmarkCompetingRisksForest`) and the `inspection`/`metrics`/`model_selection` functions, are parameter names, defaults, and return shapes consistent for the same concept (e.g. `n_estimators`, `min_ids_leaf`, `random_state`, `oob`, `cause`/`causes`, `windows`)? List every inconsistency found, file:line.

2. **Error handling consistency.** Do all four estimators and the public `inspection`/`model_selection` functions validate inputs the same way (same exception types for the same class of mistake, same message style/verbosity), or did different S-slices land different conventions? Are there any silent failure paths (wrong result with no warning) versus loud ones?

3. **Docstring completeness and consistency.** Do all public classes/functions in `__all__` (and `inspection.__all__`, `metrics.__all__`, `model_selection.__all__`) have numpydoc-style docstrings with Parameters/Returns/Examples sections? Which are missing or thin? Does `docs/source/api.rst`'s autosummary list match what's actually exported?

4. **Versioning mechanics.** How is `__version__` currently wired (hardcoded string vs. derived from `importlib.metadata` vs. build-time substitution via maturin)? What do comparable maturin+PyO3 scikit-learn-compatible packages (e.g. check `pyproject.toml`/`__init__.py` patterns other Rust-Python hybrid sklearn-adjacent projects use) do to keep the Python string and `Cargo.toml`/`pyproject.toml` versions in sync? Note `rust/rftvc-py/Cargo.toml`'s version (`0.1.0`, no `.dev0`) differs in format from `pyproject.toml`'s (`0.1.0.dev0`) — is that divergence intentional/harmless or a landmine?

5. **PyPI packaging readiness.** What's missing from `pyproject.toml` for a real PyPI publish beyond version (classifiers, project URLs, readme field, keywords)? Does `wheels.yml` need a publish job (trusted publishing / API token), and is there a TestPyPI dry-run step anywhere? What does an abi3 wheel + maturin project typically need in `[project]` that this one currently lacks?

6. **README.** What do the docs site's `index.rst` and existing case studies already say that a root `README.md` should summarize/link to, so the README doesn't duplicate content that will drift out of sync? What do comparable scikit-learn-compatible PyPI packages put in their README (install, quickstart snippet, links to docs)?

7. **Changelog scope.** Given there's no `CHANGELOG.md` yet and ~30 merged PRs of history (S5 through the rc-validation/release-pass slices), should the first changelog entry be a single "0.1.0" retrospective summary or start fresh from this release-pass point forward? What convention (Keep a Changelog, conventional commits, etc.) fits the existing commit-message style (`git log --oneline` conventions already in use)?

8. **Compatibility/deprecation surface.** `docs/source/compatibility.rst` already exists — what does it currently promise (Python versions, sklearn versions, platform support), and does it match what `pyproject.toml`'s `requires-python`/`dependencies` and `wheels.yml`'s build matrix actually deliver? Any gaps between promised and actual support?

## Codebase references
- `src/rftvc/__init__.py` — public API surface, hardcoded `__version__`
- `pyproject.toml`, `rust/rftvc-py/Cargo.toml` — packaging/version config
- `.github/workflows/{ci,wheels}.yml` — build/publish automation
- `src/rftvc/_validation.py`, `_blocks.py` — existing error-handling conventions
- `docs/source/{index,api,compatibility}.rst`, `docs/source/user_guide/`, `docs/source/case_studies/` — existing docs to audit for consistency, not rewrite from scratch
- `src/rftvc/inspection/__init__.py` (or wherever `inspection.__all__` lives), `metrics.py`, `model_selection.py` — submodule public surfaces to check against `api.rst`
