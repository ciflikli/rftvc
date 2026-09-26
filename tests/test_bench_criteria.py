"""S8 bake-off harness: metric helpers, the decision rule and a smoke run."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from bench.criteria import run as bench_run
from bench.criteria.common import SMOKE, align, calibration_summary, cluster_bootstrap, row_losses
from bench.criteria.summarise import decide
from rftvc import LandmarkSurvivalForest, SurvivalForestTV
from rftvc.model_selection import RollingOriginSplit, landmark_cross_validate
from tests.sim_panel import simulate_panel


@pytest.fixture(scope="module")
def cv_run():
    panel = simulate_panel(n_units=150, n_periods=48, seed=1)
    model = LandmarkSurvivalForest(horizon=6.0, history_features=["x", "z"], step=6.0,
                                   forest=SurvivalForestTV(n_estimators=10, min_ids_leaf=5, random_state=0))
    return landmark_cross_validate(model, panel, RollingOriginSplit(2, test_size=6, gap=6),
                                   scoring=["brier", "integrated_brier"], n_times=4, return_predictions=True)


def test_row_losses_average_to_each_fold_landmark_score(cv_run):
    scores, preds = cv_run
    rows = row_losses(preds, 6.0, 4)
    per = rows.group_by(["fold", "landmark"]).agg(pl.col("brier_row").mean(), pl.col("ibs_row").mean())
    both = scores.join(per, on=["fold", "landmark"])
    assert both.height == scores.height
    np.testing.assert_allclose(both["brier_row"], both["brier"], rtol=1e-12)
    np.testing.assert_allclose(both["ibs_row"], both["integrated_brier"], rtol=1e-12)


def test_calibration_summary_on_exact_and_constant_predictions():
    # No censoring; two risk groups whose observed event shares equal their predictions.
    time = np.r_[np.full(2, 1.0), np.full(8, 9.0), np.full(6, 1.0), np.full(4, 9.0)]
    event = np.ones(20, bool)
    risk = np.r_[np.full(10, 0.2), np.full(10, 0.6)]
    ici, slope = calibration_summary(time, event, risk, 5.0, n_bins=2)
    assert ici == pytest.approx(0.0, abs=1e-12) and slope == pytest.approx(1.0)
    ici, slope = calibration_summary(time, event, np.full(20, 0.3), 5.0, n_bins=2)
    assert ici == pytest.approx(0.1) and np.isnan(slope)  # one bin: observed 0.4 vs 0.3


def test_bootstrap_is_paired_and_zero_for_identical_arms(cv_run):
    _, preds = cv_run
    rows = row_losses(preds, 6.0, 4)
    worse = rows.with_columns(pl.col("risk") * 0.5, pl.col("ibs_row") + 0.01)
    arms = align({("a", "hazard"): rows, ("b", "hazard"): rows, ("c", "hazard"): worse})
    pairs = [(("b", "hazard"), ("a", "hazard")), (("c", "hazard"), ("a", "hazard"))]
    cmp, point = cluster_bootstrap(arms, pairs, 6.0, 5, n_boot=30, seed=0)
    same = cmp.filter(pl.col("arm") == "b/hazard")
    assert (same["diff"].abs().max(), same["lo"].abs().max(), same["hi"].abs().max()) == (0.0, 0.0, 0.0)
    ibs = cmp.filter((pl.col("arm") == "c/hazard") & (pl.col("metric") == "ibs")).row(0, named=True)
    # A constant shift survives every paired resample exactly.
    assert ibs["diff"] == pytest.approx(0.01) and ibs["lo"] == pytest.approx(0.01) and ibs["hi"] == pytest.approx(0.01)
    assert point[("c", "hazard")]["ibs"] - point[("a", "hazard")]["ibs"] == pytest.approx(0.01)


def test_align_rejects_arms_with_different_rows(cv_run):
    _, preds = cv_run
    with pytest.raises(ValueError, match="different rows"):
        align({("a", "hazard"): preds, ("b", "hazard"): preds[1:]})


def _write_rule_inputs(out, *, ibs_hi=-0.001, ici_lo=-0.01, ratio=1.5, sim_bound=-0.001):
    cmp = []
    for d in ("pbc2", "panel"):
        for agg in ("hazard", "survival"):
            cmp.append({"dataset": d, "arm": f"poisson/{agg}", "reference": f"logrank/{agg}", "metric": "ibs",
                        "diff": -0.002, "lo": -0.004, "hi": ibs_hi if d == "pbc2" else 0.001})
            cmp.append({"dataset": d, "arm": f"poisson/{agg}", "reference": f"logrank/{agg}", "metric": "ici",
                        "diff": 0.0, "lo": ici_lo, "hi": 0.01})
        for m, lo, hi in (("ibs", -0.001, 0.001), ("ici", 0.001, 0.002)):
            cmp.append({"dataset": d, "arm": "logrank/survival", "reference": "logrank/hazard", "metric": m,
                        "diff": 0.0, "lo": lo, "hi": hi})
    pl.DataFrame(cmp).write_csv(out / "a_compare.csv")
    summ = [{"dataset": d, "arm": f"{c}/{a}", "fit_seconds": ratio if c == "poisson" else 1.0}
            for d in ("pbc2", "panel") for c in ("logrank", "poisson") for a in ("hazard", "survival")]
    pl.DataFrame(summ).write_csv(out / "a_summary.csv")
    sim = [{"dataset": "sim", "arm": f"poisson/{a}", "reference": f"logrank/{a}", "metric": "ise",
            "diff": sim_bound - 0.002, "se": 0.001} for a in ("hazard", "survival")]
    pl.DataFrame(sim).write_csv(out / "b_sim_compare.csv")


@pytest.mark.parametrize("change,adopt", [
    ({}, True),
    ({"ibs_hi": 0.001}, False),  # no dataset where the IBS interval favours the challenger
    ({"ici_lo": 0.001}, False),  # an ICI interval favours log-rank
    ({"ratio": 2.5}, False),  # too slow
    ({"sim_bound": 0.0005}, False),  # sim ISE gain within 2 SE
])
def test_decision_rule(tmp_path, change, adopt):
    _write_rule_inputs(tmp_path, **change)
    checks, verdict = decide(tmp_path)
    got = dict(zip(verdict["decision"], verdict["adopt"]))
    assert got == {"default -> poisson": adopt, "D11 -> survival": False}
    assert (tmp_path / "decisions.csv").exists()


def test_smoke_run_writes_every_output(tmp_path):
    cfg = replace(SMOKE, n_boot=5, n_jobs=1)
    bench_run.run(cfg, tmp_path, log=lambda *_: None)
    for name in ("a_scores", "a_summary", "a_compare", "b_sim", "b_sim_compare"):
        assert pl.read_csv(tmp_path / f"{name}.csv").height > 0, name
    summary = pl.read_csv(tmp_path / "a_summary.csv")
    assert summary.height == 8 and summary["ibs"].is_finite().all()
    assert len(list((tmp_path / "raw").glob("*.parquet"))) == 8
    _, verdict = decide(tmp_path)
    assert verdict.height == 4
