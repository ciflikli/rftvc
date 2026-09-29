# Plan: `rftvc.viz` (optional plotting extra)

## Status
- [x] Decisions locked with user (chat, 2026-09-29): Altair for tabular charts, hand-rolled
  SVG for the tree diagram, optional extra (`pip install rftvc[viz]`), v1 = all six functions.
- [x] Slice 1: packaging + `plot_importance` + `plot_survival_curve`
- [x] Slice 2: `plot_cumulative_incidence` + `plot_hazard_effect` + `plot_calibration`
- [x] Slice 3: `plot_tree` (SVG, zero-dependency) — built all six in one pass in
  `src/rftvc/viz.py` rather than three separate commits; caught a real bug in its own
  test suite (`max_depth` truncation left unvisited nodes at a stale default depth/x of
  0, drawing them overlapping the root — fixed by tracking a `visited` mask in
  `_tree_layout` and skipping unvisited nodes when rendering).
- [x] Slice 4: docs (api.rst, README, CHANGELOG) + full-suite verification
- [x] `codex:rescue` diff review (separate Codex thread from the `export_tree` review), 2
  real findings, both fixed:
  - `plot_cumulative_incidence`'s default `cause_labels` was `1, 2, ...` — coincidentally
    matched rftvc's own default cause labels but silently wrong for a model with
    non-contiguous/custom cause labels (`causes_` need not start at 1). Changed the
    default to an explicitly uninterpreted axis position (`"cause[0]"`, `"cause[1]"`, ...)
    and documented passing `cause_labels=forest.causes_` for the real ones.
  - The module-level `pytest.importorskip("altair")` skipped the *dependency-free*
    `plot_tree` tests too when Altair was absent, which is exactly the case those tests
    exist to cover. Moved the skip to only the Altair-consuming tests (verified by
    actually uninstalling altair: 8 tree/guard tests ran and passed, the other 11
    skipped cleanly — then reinstalled altair and confirmed the accidental `narwhals`
    uninstall along the way didn't leave anything broken).
  - Also added: a real split-root `max_depth=0` truncation test (distinct from the
    existing genuine-root-leaf test — this is the shape that triggered the original
    `_tree_layout` bug), and a nonzero-cause-index `plot_hazard_effect` test that checks
    actual values against a hand-picked axis slice (not just `isinstance`), per Codex's
    "tests are shallow" finding.

## Design decisions

- **Module**: `src/rftvc/viz.py`, one flat file (matches `metrics.py`/`inspection.py`
  convention — a module per concern, not a package).
- **Dependency**: `altair` is optional. `pyproject.toml` gets
  `[project.optional-dependencies] viz = ["altair>=5"]`, and `altair` is added to the
  `dev` dependency group so it's actually exercised in tests/CI, not just guarded.
- **Import guarding**: no top-level `import altair` in `viz.py` (that would make even
  `plot_tree`, which needs no altair, fail to import without the extra). Each of the
  five Altair-based functions calls a shared `_require_altair()` at its own top, which
  does the import and raises a friendly `ImportError` naming `pip install rftvc[viz]` if
  missing. `plot_tree` has no such guard: it is genuinely zero-dependency.
- **Input shape**: every function takes the *return value* of an existing rftvc call
  directly (arrays from `predict_survival_function`/`predict_cumulative_incidence`, the
  `Bunch` from `permutation_importance`/`drop_column_importance`/`hazard_effect`/
  `export_tree`, the polars `DataFrame` from `calibration_table`) — no new statistics computed
  in `viz.py`, purely rendering. `plot_importance` duck-types on `importances_mean`,
  `importances_se`, `feature_names` so it works unmodified for both importance functions.
- **Return type**: the five Altair functions return an `altair.Chart` (or `LayerChart`)
  the caller can further customize/save/display; `plot_tree` returns an SVG string.

## Signatures (v1)

```python
def plot_survival_curve(times, curve, *, labels=None, kind="survival"): -> alt.Chart
    """curve: predict_survival_function/predict_cumulative_hazard output, shape (n, n_times).
    kind: "survival" or "hazard" (y-axis label only). labels: per-row name, default row index."""

def plot_cumulative_incidence(times, cif, *, cause_labels=None, subject_labels=None): -> alt.Chart
    """cif: predict_cumulative_incidence output, shape (n_subjects, n_causes, n_times).
    One line per (subject, cause); color=cause, facet or dash=subject when n_subjects > 1."""

def plot_importance(result, *, top_n=None): -> alt.Chart
    """result: permutation_importance / drop_column_importance Bunch. Horizontal bar,
    sorted descending by |importances_mean|, error bar from importances_se."""

def plot_hazard_effect(result, *, cause=None): -> alt.Chart
    """result: hazard_effect Bunch. One line per grid value across window midpoints;
    cause required (int) if result.hazard is 3-D (competing risks, cause=None was passed)."""

def plot_calibration(table): -> alt.Chart
    """table: calibration_table output (polars DataFrame). Scatter mean_risk vs
    observed_risk, point size ~ n, plus a y=x reference line."""

def plot_tree(forest_or_tree, tree=0, *, max_depth=None, feature_names=None): -> str
    """forest_or_tree: a fitted SurvivalForestTV/CompetingRisksForestTV (calls
    export_tree(tree) internally) or an export_tree() Bunch directly. Returns an SVG
    string (Reingold-Tilford-style even layout; not literally the D3 algorithm, a
    simpler per-depth-row layout is enough for typical rftvc leaf counts)."""
```

## Slice 1: packaging + `plot_importance` + `plot_survival_curve`
**Files:** `pyproject.toml` (modify), `src/rftvc/viz.py` (new), `tests/test_viz.py` (new)
**Acceptance:** both functions return `alt.Chart`; `plot_importance` works on both
`permutation_importance` and `drop_column_importance` output; missing-altair path raises
the friendly `ImportError` (tested by monkeypatching `sys.modules["altair"] = None`, not
by actually uninstalling).

## Slice 2: `plot_cumulative_incidence` + `plot_hazard_effect` + `plot_calibration`
**Files:** `src/rftvc/viz.py`, `tests/test_viz.py`
**Acceptance:** CIF plot handles both a single subject and multiple; hazard_effect plot
raises a clear error when `result.hazard.ndim == 3` and no `cause` given; calibration
plot consumes `calibration_table`'s actual polars schema unmodified.

## Slice 3: `plot_tree`
**Files:** `src/rftvc/viz.py`, `tests/test_viz.py`
**Acceptance:** valid SVG (parses with `xml.etree.ElementTree`) for a multi-leaf tree and
for a root-leaf (`max_depth=0`) tree; accepts both a fitted estimator and a bare
`export_tree()` Bunch; uses `-2`/`-1` sentinels from `export_tree` correctly (leaf
detection must use `leaf >= 0`, not `feature`, since `feature == -2` at every leaf).

## Slice 4: docs + verification
**Files:** `docs/source/api.rst`, `README.md`, `CHANGELOG.md`
**Acceptance:** `Visualization` autosummary section added (mirrors `Inspection`/`Metrics`
sections); full default pytest tier green; `uv pip install -e . --group dev` (with
`altair` now in `dev`) succeeds from clean.
