# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries begin at the release-readiness pass (2026-09); earlier development is not
retroactively itemized here — see `git log` for the full slice-by-slice history.

## [Unreleased]

### Added
- Native NaN handling for survival and competing-risks forests. Split search
  considers both missing-value directions and missing-versus-observed splits;
  prediction and pickling preserve each node's learned route. Tree exports
  include `missing_goes_right` and use a NaN threshold for missingness splits.
- `rftvc.impute.impute_locf` for per-subject carry-forward imputation of
  time-varying covariates.
- String and categorical columns in `X`: a fitted one-hot encoding shared by fit
  and prediction, missing-value routing preserved through it, unseen levels
  rejected, and category vocabularies included in fit-design fingerprints.
  `inspection.permutation_importance`, `drop_column_importance`, `hazard_effect`
  and landmark stacking treat each categorical column as one unit
  (`docs/bench/categorical-width.md`).

### Changed
- Missing-value split search now uses bin-prefix histograms instead of
  rescanning every resampling unit per candidate threshold, matching the
  no-missing path's technique since 0.1.0. A 25-tree, 8,000-subject fit with
  1% missing entries fell from 0.53s to 0.20s; split oracles and held-out
  concordance are unchanged (`docs/bench/fit-matrix.md`).

## [0.2.0] - 2026-09-29

### Added
- `SurvivalForestTV.export_tree` / `CompetingRisksForestTV.export_tree`: one tree's split
  structure (`children_left`, `children_right`, `feature`, `threshold`, `leaf`), in
  scikit-learn's `Tree` attribute convention, for building custom tree diagrams. A leaf's
  Nelson-Aalen cumulative hazard curve is `forest_.leaf_profile(tree, leaf)`.
- `rftvc.viz`, an optional plotting module (`pip install rftvc[viz]`):
  `plot_survival_curve`, `plot_cumulative_incidence`, `plot_importance`,
  `plot_hazard_effect` and `plot_calibration` (Altair charts over existing rftvc return
  values), and `plot_tree` (a dependency-free SVG tree diagram built on `export_tree`).

## [0.1.0] - 2026-09-28

### Added
- `random_state` now accepts a `numpy.random.Generator` everywhere it appears
  (`SurvivalForestTV`, `CompetingRisksForestTV`, `inspection.permutation_importance`,
  `inspection.drop_column_importance`), matching what `inspection.py` already
  supported internally.
- A `UserWarning` when `ntime` coarsening drops rows or loses events, pointing at the
  fitted estimator's `n_coarsen_dropped_rows_`/`n_coarsen_lost_events_` attributes.
- CI now tests 6 combinations (Linux, macOS, Windows × Python 3.10, 3.13), replacing
  the previous Linux/macOS-only, Python-3.11-only coverage.
- CI now fails if `pyproject.toml` and `rust/rftvc-py/Cargo.toml` disagree on their
  version core.
- `pyproject.toml` now declares `readme`, `authors`, `keywords`, `classifiers` and
  `[project.urls]`; the package ships a `py.typed` marker (PEP 561) in both the wheel
  and sdist.
- Tag pushes now publish to TestPyPI, then to PyPI once the TestPyPI upload succeeds,
  via Trusted Publishing (no stored token).

### Changed
- Missing-value split search now counts subject membership with bin-prefix
  histograms instead of scanning every subject at each threshold. A 25-tree,
  8,000-subject synthetic fit with 1% missing entries fell from 0.53 to 0.20
  seconds on the benchmark machine; split oracles and held-out concordance
  remain unchanged.
- `SurvivalForestTV`/`CompetingRisksForestTV` hyperparameter validation now raises
  `TypeError` for a wrong-type argument and `ValueError` only for an out-of-range value
  of the correct type (previously both raised `ValueError`).
- `rftvc.__version__` now reflects installed package metadata instead of a
  hand-maintained literal.
