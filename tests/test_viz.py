"""rftvc.viz: five Altair charts (rendering existing rftvc results) + a zero-dependency SVG tree."""

import builtins
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rftvc import CompetingRisksForestTV, SurvivalForestTV, inspection, make_competing_risks_y, make_survival_y, viz
from rftvc.metrics import calibration_table, event_windows


def _data(n=150, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    t = rng.exponential(np.exp(-0.5 * X[:, 0]))
    y = make_survival_y(np.minimum(t, 2.0), t <= 2.0)
    return X, y, t


def _forest(**kw):
    X, y, t = _data()
    return SurvivalForestTV(n_estimators=15, random_state=0, **kw).fit(X, y), X, y, t


def test_plot_survival_curve():
    alt = pytest.importorskip("altair")
    forest, X, y, t = _forest()
    times = np.array([0.5, 1.0, 1.5, 2.0])
    S = forest.predict_survival_function(X[:3], times)
    chart = viz.plot_survival_curve(times, S)
    assert isinstance(chart, alt.Chart)
    chart.to_dict()  # validates the spec


def test_plot_survival_curve_shape_mismatch_raises():
    pytest.importorskip("altair")
    with pytest.raises(ValueError, match="curve must have shape"):
        viz.plot_survival_curve([1.0, 2.0], [[0.9]])


def test_plot_survival_curve_invalid_kind_raises():
    pytest.importorskip("altair")
    with pytest.raises(ValueError, match="kind must be"):
        viz.plot_survival_curve([1.0], [[0.9]], kind="bogus")


def test_plot_cumulative_incidence_single_and_multi_subject():
    alt = pytest.importorskip("altair")
    X, y, t = _data()
    rng = np.random.default_rng(0)
    cause = np.where(y["event"], rng.integers(1, 3, size=len(y)), 0)
    ycr = make_competing_risks_y(y["stop"], cause, start=y["start"])
    forest = CompetingRisksForestTV(n_estimators=10, random_state=0).fit(X, ycr)
    times = np.array([0.5, 1.0, 1.5])

    cif_one = forest.predict_cumulative_incidence(X[:1], times)
    chart_one = viz.plot_cumulative_incidence(times, cif_one)
    assert isinstance(chart_one, alt.Chart)
    chart_one.to_dict()
    # default cause labels are an uninterpreted axis position, not the model's real labels
    # (which need not be contiguous/start at 1) -- see forest.causes_ for the real ones.
    assert set(chart_one.data["cause"]) == {"cause[0]", "cause[1]"}

    cif_many = forest.predict_cumulative_incidence(X[:3], times)
    chart_many = viz.plot_cumulative_incidence(times, cif_many)
    assert isinstance(chart_many, alt.FacetChart)
    chart_many.to_dict()

    # explicit cause_labels (the model's real ones) and subject_labels pass through unmodified
    labeled = viz.plot_cumulative_incidence(times, cif_one, cause_labels=forest.causes_, subject_labels=["s0"])
    assert set(labeled.data["cause"]) == set(str(c) for c in forest.causes_)
    assert set(labeled.data["subject"]) == {"s0"}

    # a 2-D, single-selected-cause result plots as one cause
    cif_2d = forest.predict_cumulative_incidence(X[:2], times, cause=int(forest.causes_[0]))
    chart_2d = viz.plot_cumulative_incidence(times, cif_2d, cause_labels=[forest.causes_[0]])
    assert isinstance(chart_2d, alt.FacetChart)  # 2 subjects
    assert set(chart_2d.data["cause"]) == {str(forest.causes_[0])}
    subject0 = chart_2d.data.filter(chart_2d.data["subject"] == "0")
    np.testing.assert_allclose(sorted(subject0["value"]), sorted(cif_2d[0]))


def test_plot_cumulative_incidence_bad_shape_raises():
    pytest.importorskip("altair")
    with pytest.raises(ValueError, match="cif must have shape"):
        viz.plot_cumulative_incidence([1.0, 2.0], np.zeros((2, 2, 3)))


@pytest.mark.parametrize("fn", ["permutation_importance", "drop_column_importance"])
def test_plot_importance_both_functions(fn):
    alt = pytest.importorskip("altair")
    forest, X, y, t = _forest()
    ids = np.arange(len(y))
    kw = dict(n_repeats=3, random_state=0) if fn == "permutation_importance" else dict(cv=3, random_state=0)
    result = getattr(inspection, fn)(forest, X, y, ids=ids, **kw)
    chart = viz.plot_importance(result)
    assert isinstance(chart, alt.LayerChart)
    chart.to_dict()

    top1 = viz.plot_importance(result, top_n=1)
    assert len(top1.data) == 1


def test_plot_importance_requires_the_right_shape():
    pytest.importorskip("altair")
    from sklearn.utils import Bunch

    with pytest.raises(TypeError, match="missing"):
        viz.plot_importance(Bunch(importances_mean=np.array([1.0])))
    with pytest.raises(ValueError, match="same shape"):
        viz.plot_importance(
            Bunch(importances_mean=np.array([1.0, 2.0]), importances_se=np.array([0.1]), feature_names=np.array(["a", "b"]))
        )


def test_plot_hazard_effect_survival_and_competing_risks():
    alt = pytest.importorskip("altair")
    forest, X, y, t = _forest()
    result = inspection.hazard_effect(forest, X, y, feature=0, windows=4)
    chart = viz.plot_hazard_effect(result)
    assert isinstance(chart, alt.Chart)
    chart.to_dict()

    with pytest.raises(ValueError, match="only for a competing-risks result"):
        viz.plot_hazard_effect(result, cause=0)

    X2, y2, t2 = _data()
    rng = np.random.default_rng(0)
    cause = np.where(y2["event"], rng.integers(1, 3, size=len(y2)), 0)
    ycr = make_competing_risks_y(y2["stop"], cause, start=y2["start"])
    crf = CompetingRisksForestTV(n_estimators=10, random_state=0).fit(X2, ycr)
    cr_result = inspection.hazard_effect(crf, X2, ycr, feature=0, windows=4)
    assert cr_result.hazard.shape[1] >= 2  # two causes, so cause=1 exercises a real non-zero axis index
    with pytest.raises(ValueError, match="cause .* is required"):
        viz.plot_hazard_effect(cr_result)

    cr_chart = viz.plot_hazard_effect(cr_result, cause=1)
    assert isinstance(cr_chart, alt.Chart)
    # the rendered data must be cause 1's own slice, not cause 0's (an axis off-by-one would show that)
    first_value = str(cr_result["values"][0])
    row = cr_chart.data.filter(cr_chart.data["value"] == first_value).sort("window_mid")
    np.testing.assert_allclose(row["hazard"].to_numpy(), cr_result.hazard[0, 1, :])
    assert not np.allclose(row["hazard"].to_numpy(), cr_result.hazard[0, 0, :])

    with pytest.raises(ValueError, match="cause must be in"):
        viz.plot_hazard_effect(cr_result, cause=99)


def test_plot_calibration():
    alt = pytest.importorskip("altair")
    forest, X, y, t = _forest()
    windows = event_windows(forest, n_windows=5)
    risk = 1.0 - forest.predict_survival_function(X, windows)
    table = calibration_table(y, risk[:, -1], windows[-1], n_bins=4)
    chart = viz.plot_calibration(table)
    assert isinstance(chart, alt.LayerChart)
    chart.to_dict()


def test_plot_calibration_requires_the_right_columns():
    pytest.importorskip("altair")
    import polars as pl

    with pytest.raises(TypeError, match="missing columns"):
        viz.plot_calibration(pl.DataFrame({"bin": [0]}))


def test_plot_tree_from_estimator_and_from_bunch():
    forest, X, y, t = _forest()
    svg_a = viz.plot_tree(forest, 0)
    svg_b = viz.plot_tree(forest.export_tree(0))
    assert svg_a == svg_b
    ET.fromstring(svg_a)  # valid XML


def test_plot_tree_root_leaf():
    X, y, t = _data()
    forest = SurvivalForestTV(n_estimators=1, max_depth=0, random_state=0).fit(X, y)
    svg = viz.plot_tree(forest, 0)
    root = ET.fromstring(svg)
    assert sum(1 for _ in root.iter("{http://www.w3.org/2000/svg}rect")) == 1


def test_plot_tree_max_depth_zero_on_a_real_split_root():
    """Distinct from test_plot_tree_root_leaf: here the tree DOES split at the root, and
    max_depth=0 forces the *display* cut right there -- exercises the "root itself is a cut
    marker" path, not "root is genuinely a leaf" (this is exactly the shape of node that
    triggered the max_depth staleness bug: an unvisited subtree hanging off a cut node)."""
    forest, X, y, t = _forest()
    assert forest.export_tree(0).node_count > 1  # a real split tree, not accidentally a root leaf
    svg = viz.plot_tree(forest, 0, max_depth=0)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert sum(1 for _ in root.iter(f"{ns}rect")) == 1
    assert sum(1 for _ in root.iter(f"{ns}line")) == 0
    assert next(root.iter(f"{ns}text")).text == "..."


def test_plot_tree_max_depth_collapses_deeper_nodes():
    forest, X, y, t = _forest()
    full = ET.fromstring(viz.plot_tree(forest, 0))
    shallow = ET.fromstring(viz.plot_tree(forest, 0, max_depth=1))
    ns = "{http://www.w3.org/2000/svg}"
    assert sum(1 for _ in shallow.iter(f"{ns}rect")) < sum(1 for _ in full.iter(f"{ns}rect"))
    assert any(t.text == "..." for t in shallow.iter(f"{ns}text"))


def test_plot_tree_competing_risks():
    X, y, t = _data()
    rng = np.random.default_rng(0)
    cause = np.where(y["event"], rng.integers(1, 3, size=len(y)), 0)
    ycr = make_competing_risks_y(y["stop"], cause, start=y["start"])
    crf = CompetingRisksForestTV(n_estimators=5, random_state=0).fit(X, ycr)
    ET.fromstring(viz.plot_tree(crf, 0))


def test_plot_tree_feature_names():
    pd = pytest.importorskip("pandas")
    X, y, t = _data(p=2)
    Xdf = pd.DataFrame(X, columns=["alpha", "beta"])
    forest = SurvivalForestTV(n_estimators=3, random_state=0).fit(Xdf, y)
    svg = viz.plot_tree(forest, 0)
    assert "alpha" in svg or "beta" in svg


def test_plot_tree_invalid_max_depth_raises():
    forest, X, y, t = _forest()
    with pytest.raises(ValueError, match="max_depth"):
        viz.plot_tree(forest, 0, max_depth=-1)


def test_missing_altair_raises_a_friendly_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "altair":
            raise ImportError("simulated: altair not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"pip install rftvc\[viz\]"):
        viz.plot_survival_curve([1.0], [[0.5]])
