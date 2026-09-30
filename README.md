# rftvc

[![CI](https://github.com/ciflikli/rftvc/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ciflikli/rftvc/actions/workflows/ci.yml)
[![Coverage gate](https://img.shields.io/badge/coverage%20gate-%E2%89%A595%25-brightgreen)](https://github.com/ciflikli/rftvc/blob/main/.github/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rftvc)](https://pypi.org/project/rftvc/)
[![Python versions](https://img.shields.io/pypi/pyversions/rftvc)](https://pypi.org/project/rftvc/)
[![Documentation](https://app.readthedocs.org/projects/rftvc/badge/?version=latest)](https://rftvc.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ciflikli/rftvc)](LICENSE)

**Random survival forests for covariates that change over time.** A Rust engine
with a scikit-learn compatible Python API.

<p align="center">
  <img src="assets/readme/tvc-survival.png" width="48%" alt="Predicted survival for two changing covariate paths with the same baseline">
  <img src="assets/readme/importance.png" width="48%" alt="Cause 1 hazard importance: the time-varying signal ranks above noise, with standard-error bars">
</p>

- **Follow the full path:** counting-process rows `(id, start, stop, event, X)`,
  delayed entry, native missing-value splits, and categorical covariates.
- **Predict what matters next:** survival, hazard, competing-risk cumulative
  incidence, and rolling landmark predictions from observed history.
- **Inspect the fit:** time-stratified permutation importance, drop-column
  importance, partial-effect curves, calibration, and [plotting helpers](https://rftvc.readthedocs.io/en/latest/api.html#visualization).
- **Respect subjects:** subject-level resampling, out-of-bag estimates, leaf
  sizes, and time-aware validation.

## Install

```console
pip install rftvc
```

Use `pip install rftvc[viz]` for Altair charts. The SVG tree diagram needs no
extra dependency.

## Quickstart

One row per subject works like a conventional survival forest:

```python
import numpy as np
from rftvc import SurvivalForestTV, make_survival_y

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
t = rng.exponential(np.exp(-0.5 * X[:, 0]))
y = make_survival_y(np.minimum(t, 2.0), t <= 2.0)

forest = SurvivalForestTV(n_estimators=200, random_state=0).fit(X, y)
risk = forest.predict_risk(X[:5], horizon=1.0)
```

For repeated measurements, give each subject one row per interval. Covariates
on a row must be known at its `start`:

```python
# panel columns: id, start, stop, event, age, biomarker
X = panel[["age", "biomarker"]]
y = make_survival_y(panel["stop"], panel["event"], start=panel["start"])
forest = SurvivalForestTV(n_estimators=200, random_state=0).fit(X, y, ids=panel["id"])

# A subject's future covariate path, with one row per forecast interval:
survival = forest.predict_survival_function(
    future[["age", "biomarker"]], times=[1, 2, 3],
    intervals=make_survival_y(future["stop"], np.zeros(len(future), dtype=bool), start=future["start"]),
    ids=future["id"],
)
```

Competing events use the same rows, with `0` for censoring and positive cause
labels for events:

```python
from rftvc import CompetingRisksForestTV, make_competing_risks_y, inspection
from rftvc.viz import plot_cumulative_incidence, plot_importance

y_cr = make_competing_risks_y(panel["stop"], panel["cause"], start=panel["start"])
cr = CompetingRisksForestTV(n_estimators=200, random_state=0).fit(X, y_cr, ids=panel["id"])
cif = cr.predict_cumulative_incidence(X.iloc[:1], times=[1, 2, 3])
plot_cumulative_incidence([1, 2, 3], cif, cause_labels=cr.causes_)

# Use held-out counting-process rows for feature importance.
importance = inspection.permutation_importance(cr, X_test, y_test, ids=test_ids, cause=1)
plot_importance(importance)
```

See the [user guide](https://rftvc.readthedocs.io/en/latest/user_guide/index.html)
for path prediction and landmark models, and the [case studies](https://rftvc.readthedocs.io/en/latest/case_studies/index.html)
for worked examples on real data. The figures above can be rebuilt with
[`examples/readme_plots.py`](examples/readme_plots.py).

## From source

Install a [Rust toolchain](https://rustup.rs/) and [uv](https://docs.astral.sh/uv/), then:

```console
git clone https://github.com/ciflikli/rftvc.git
cd rftvc
uv venv
uv pip install -e . --group dev
```

[Compatibility](https://rftvc.readthedocs.io/en/latest/compatibility.html) ·
[Changelog](CHANGELOG.md) · MIT license
