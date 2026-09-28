# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries begin at the release-readiness pass (2026-09); earlier development is not
retroactively itemized here — see `git log` for the full slice-by-slice history.

## [Unreleased]

### Added
- `random_state` now accepts a `numpy.random.Generator` everywhere it appears
  (`SurvivalForestTV`, `CompetingRisksForestTV`, `inspection.permutation_importance`,
  `inspection.drop_column_importance`), matching what `inspection.py` already
  supported internally.
- A `UserWarning` when `ntime` coarsening drops rows or loses events, pointing at the
  fitted estimator's `n_coarsen_dropped_rows_`/`n_coarsen_lost_events_` attributes.
- CI now tests Python 3.10 (the declared floor) and Windows, in addition to the
  existing Linux/macOS + Python 3.13 coverage.

### Changed
- `SurvivalForestTV`/`CompetingRisksForestTV` hyperparameter validation now raises
  `TypeError` for a wrong-type argument and `ValueError` only for an out-of-range value
  of the correct type (previously both raised `ValueError`).
- `rftvc.__version__` now reflects installed package metadata instead of a
  hand-maintained literal.

### Fixed
- (release-pass fixes land here as they merge)
