# S7 slice plan: sklearn compatibility, DataFrame input, wheels, docs and case studies

Branch `feat/s7-compat`. Parent: `plan.md` S7; design.md "Data contract" (compatibility matrix), D5, D7.

## Decisions (defaults; revisit in review)
1. **DataFrame input (D7) via `narwhals`** (new runtime dependency; `narwhals>=1.30`).
   - `X` may be a pandas, polars or pyarrow DataFrame. It is converted with `nw.from_native(X, eager_only=True).to_numpy()`, which sets `feature_names_in_`. Predicting with different or reordered column names raises; plain arrays give no names (sklearn semantics).
   - `y` may also be a DataFrame with `start` / `stop` / `event` columns. It is converted to the structured array.
   - `ids` may be an array **or a column name of `X`** (risk table: "ids also accepted as a DataFrame column name"). That column is removed from the features.
2. **sklearn tags** (sklearn ≥ 1.6 `__sklearn_tags__`):
   - `target_tags.required = True`;
   - `input_tags.allow_nan = False`;
   - `estimator_type = None` (not a classifier or regressor);
   - `non_deterministic = False`.
   - Compatible with sklearn 1.5? No. The minimum becomes `scikit-learn>=1.6` (tags API and `expected_failed_checks`), a deviation from `>=1.5`.
3. **Metadata routing:** `fit(X, y, ids=None, ...)` makes sklearn generate `set_fit_request(ids=...)`. A test runs `cross_validate(..., params={"ids": ids})` with routing enabled and a `GroupKFold`, and checks that `ids` reaches `fit` sliced to each training fold.
4. **Compatibility matrix:** `tests/test_sklearn_compat.py` uses `parametrize_with_checks([SurvivalForestTV(n_estimators=5, ...)], expected_failed_checks=...)`. Every exclusion has a one-line reason, mostly "y must be a structured survival target". The same table appears in the docs as the compatibility matrix. `LandmarkSurvivalForest` takes a DataFrame and no `y`, so it is out of `check_estimator` scope (documented); `clone` / `get_params` / pickling are tested for it directly.
5. **Wheels** (`.github/workflows/wheels.yml`):
   - Build with **`PyO3/maturin-action`**, not cibuildwheel (deviation): it is the standard for maturin projects and builds abi3 wheels natively. One `cp310-abi3` wheel per platform: Linux x86_64 and aarch64 (manylinux2014), macOS arm64 and x86_64, Windows x64; plus an sdist.
   - A test job installs each wheel on Linux, macOS and Windows with Python 3.10 and 3.13, then runs an import-and-fit smoke test.
   - Triggers: tags `v*`, manual dispatch, and PRs touching packaging files. No publishing to PyPI in v1.
6. **Docs:** Sphinx + numpydoc + pydata-sphinx-theme (user decision). `docs/source/` holds:
   - **User guide:**
     - the three data views: one row per subject (right-censored), counting-process rows (TVCs, delayed entry), stacked landmark data;
     - the time grid (exact vs `ntime`);
     - prediction along covariate paths (`origin`, `extrapolate`);
     - choosing the error estimate: new subjects (OOB, `GroupKFold`) vs future periods (`RollingOriginSplit`) vs both (`GroupTimeSplit`), with IPCW notes.
   - **API reference** (autodoc + numpydoc) and the **compatibility matrix**.
   - **Case studies** (item 7).
   - CI gate: `sphinx-build -W --keep-going` in a new `docs` job. Case-study figures and tables are precomputed by scripts (no network at build time).
7. **Case studies** (scripts under `examples/`; results committed as tables/text included in the pages):
   - **PBC2** (`pbcseq`, fixture already committed):
     - a counting-process forest along biomarker paths vs a landmark super-model;
     - evaluation with nested landmark CV (`GroupKFold`, new patients);
     - landmark Brier score / C-D AUC and a calibration table.
   - **BTSCS: war duration**, Cunningham & Lemke (2013) replication data (user decision; basis of the author's thesis, Ciflikli 2018):
     - `examples/data/cunningham_lemke.py` downloads the archive from the authors' page into a cache (`~/.cache/rftvc`) and **verifies the SHA-256** (`f156a1d3…c142`). Nothing is redistributed; the page cites the source and says no licence is stated.
     - Preprocessing mirrors the authors' `stset clenddate, id(CLID) origin(clstartdate) failure(clend==1)` + `stcox`: time = days since the war's first start; each war is truncated at its first `clend == 1`; rows with missing covariates are dropped (listwise, as `stcox` does); the resulting gaps are delayed re-entry (`gap_policy="split_id"`). Reported: 382 wars, 341 terminations, the rows dropped.
     - Models: the counting-process forest with the paper's covariates (civil, territory, recurringwar, logtroopratio, democ, logtotaltroops, logtotalpop) vs the paper's Cox model (lifelines, time-varying).
     - Evaluation: new-conflict CV (`GroupKFold` by `CLID`) with the counting-process C (`concordance_index_cp`), and a landmark model at war-years 1, 2, 5 with horizon 2 years.
     - Also shown: path predictions for two example wars.
8. **Out of scope:** publishing to PyPI; a `person_period.py` view (design.md lists it; not needed for v1, deferred).

## Tasks
- [ ] narwhals input + `feature_names_in_` + `ids` column name + y as DataFrame; tests
- [ ] tags, sklearn ≥ 1.6, metadata-routing test, `check_estimator` matrix with documented exclusions
- [ ] wheels.yml (+ smoke test job) and a green run on the PR
- [ ] Sphinx skeleton, API reference, user guide, compatibility matrix; `docs` CI job
- [ ] C&L loader (checksum) + BTSCS case study; PBC2 case study
- [ ] plan.md tick + "S7 done" notes
