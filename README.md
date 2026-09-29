# rftvc

Random survival forests for **time-varying covariates**, with a Rust engine and a
scikit-learn compatible Python API.

- Counting-process data `(id, start, stop, event, X)` with delayed entry
  (left truncation) and covariates that change over follow-up.
- Whole-subject resampling, id-level out-of-bag estimates, and leaf sizes
  counted in subjects, not rows.
- Landmark data building and a landmark super-model for dynamic prediction.
- Time-aware cross-validation and IPCW landmark metrics.
- An exact log-rank split criterion, and an opt-in coarse time grid for large data.
- Optional plotting (`rftvc.viz`, `pip install rftvc[viz]`): survival/hazard/CIF curves,
  importance and calibration charts (Altair), and a dependency-free SVG tree diagram.

## Install

Not yet published to PyPI. Build from source: a [Rust toolchain](https://rustup.rs/)
(via [maturin](https://www.maturin.rs/)) and [uv](https://docs.astral.sh/uv/) are
required.

```console
git clone https://github.com/ciflikli/rftvc.git
cd rftvc
uv venv
uv pip install -e . --group dev
```

Add `--extra viz` (or `pip install rftvc[viz]` once published) for `rftvc.viz`'s
Altair-based charts; its SVG tree diagram (`rftvc.viz.plot_tree`) needs no extra install.

## Quickstart

```python
import numpy as np
from rftvc import SurvivalForestTV, make_survival_y

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
t = rng.exponential(np.exp(-0.5 * X[:, 0]))
event = t <= 2.0
y = make_survival_y(np.minimum(t, 2.0), event)

forest = SurvivalForestTV(n_estimators=200, random_state=0).fit(X, y)
risk = forest.predict_risk(X[:5], horizon=1.0)
```

## Documentation

- [Docs site](https://rftvc.readthedocs.io) — user guide, API reference, case studies.
- [Case studies](docs/source/case_studies/) — worked examples on real datasets.
- [Compatibility](https://rftvc.readthedocs.io/en/latest/compatibility.html) — supported
  Python, scikit-learn and platform versions.
- [Changelog](CHANGELOG.md)

## License

MIT
