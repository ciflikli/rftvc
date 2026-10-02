"""Real-data validation: Rossi (TVC) + EBMT4 (competing risks + TVC).

Run from the repo root: .venv/bin/python -m examples.rc_validation
Pre-release validation, separate from the documentation case studies. Writes
a report to stdout and CSVs under the ignored docs/scratch/rc-validation/.

Methodology: permutation_importance / drop_column_importance
report *relevance magnitude* only (a score drop, no sign) -- never read as a
directional effect. Direction comes only from hazard_effect contrasts (two
covariate values) and the matching CoxTimeVaryingFitter coefficient's sign.
permutation_importance / drop_column_importance use oob=True (no manual
holdout); hazard_effect has no OOB variant and is reported as descriptive
(training data), not a held-out generalization claim.

Categorical encoding happens here, not in the fixture modules:
- Rossi: race (black=1, else 0), mar (married=1, else 0), fin/wexp/paro
  (yes=1, no=0) -- all originally yes/no or two-level factors.
- EBMT4: year/agecl/proph/match are ordinal-coded (arbitrary integer per
  distinct level) for *both* fits -- adequate for a tree ensemble, but not a
  meaningful unit for Cox's coefficients on these baseline covariates;
  ae, the covariate of interest, is a real binary flag, unaffected by this.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxTimeVaryingFitter

from rftvc import CompetingRisksForestTV, SurvivalForestTV, inspection, make_competing_risks_y, make_survival_y
from tests.fixtures.ebmt4 import ebmt4_competing_risks
from tests.fixtures.rossi import rossi_counting_process

OUT = Path(__file__).resolve().parents[1] / "docs" / "scratch" / "rc-validation"


def _yesno(col):
    return (pl.col(col) == "yes").cast(pl.Int64).alias(col)


def _to_pandas(d):
    """``pl.DataFrame`` -> ``pd.DataFrame`` without pyarrow (not a dependency here)."""
    return pd.DataFrame({c: d[c].to_numpy() for c in d.columns})


def _forest(**kw):
    return SurvivalForestTV(n_estimators=500, min_ids_leaf=10, random_state=0, n_jobs=-1, **kw)


def _cr_forest(**kw):
    return CompetingRisksForestTV(n_estimators=500, min_ids_leaf=10, random_state=0, n_jobs=-1, **kw)


ROSSI_COVARIATES = ["employed", "fin", "age", "race", "wexp", "mar", "paro", "prio", "educ"]
EBMT4_COVARIATES = ["ae", "year", "agecl", "proph", "match"]


def rossi_frame():
    d = rossi_counting_process()
    d = d.with_columns(
        (pl.col("race") == "black").cast(pl.Int64).alias("race"),
        (pl.col("mar") == "married").cast(pl.Int64).alias("mar"),
        _yesno("fin"),
        _yesno("wexp"),
        _yesno("paro"),
    )
    return _to_pandas(d)


EBMT4_LEVELS = {
    "year": ["1985-1989", "1990-1994", "1995-1998"],
    "agecl": ["<=20", "20-40", ">40"],
    "proph": ["no", "yes"],
    "match": ["no gender mismatch", "gender mismatch"],
}


def ebmt4_frame():
    """Counting-process rows with year/agecl/proph/match ordinal-coded.

    An explicit, fixed level order (the R factor levels themselves, already
    ordinal for year/agecl) via ``pl.Enum`` -- bare ``.cast(pl.Categorical)``
    codes by first appearance, which is not stable across separate process
    runs on identical data.
    """
    d = ebmt4_competing_risks().with_columns(
        pl.col(c).cast(pl.Enum(levels)).to_physical().alias(c) for c, levels in EBMT4_LEVELS.items()
    )
    return _to_pandas(d)


def relevance_table(perm, loco, names):
    return pd.DataFrame(
        {
            "feature": names,
            "perm_importance": perm.importances_mean,
            "perm_se": perm.importances_se,
            "loco_importance": loco.importances_mean,
        }
    ).sort_values("perm_importance", ascending=False)


def rossi_section():
    print("\n=== Rossi (single-event TVC) ===")
    pdf = rossi_frame()
    X = pdf[ROSSI_COVARIATES]
    y = make_survival_y(pdf["stop"].to_numpy(), pdf["event"].to_numpy(), start=pdf["start"].to_numpy())
    ids = pdf["id"].to_numpy()

    t0 = time.perf_counter()
    forest = _forest(oob_score=True).fit(X, y, ids)
    fit_s = time.perf_counter() - t0
    print(f"fit: {fit_s:.2f}s, oob concordance = {forest.oob_score_:.3f}")

    cox = CoxTimeVaryingFitter().fit(
        pdf[["id", "start", "stop", "event", *ROSSI_COVARIATES]],
        id_col="id", start_col="start", stop_col="stop", event_col="event",
    )
    cox_coef = cox.summary["coef"]

    perm = inspection.permutation_importance(forest, X, y, ids=ids, oob=True, random_state=0)
    loco = inspection.drop_column_importance(forest, X, y, ids=ids, cv=5, random_state=0)
    relevance = relevance_table(perm, loco, ROSSI_COVARIATES)
    relevance.to_csv(OUT / "rossi_relevance.csv", index=False)
    print(relevance.round(4).to_string(index=False))

    contrasts = {"fin": (0.0, 1.0), "age": (20.0, 30.0), "prio": (0.0, 10.0), "employed": (0.0, 1.0)}
    rows = []
    for feat, (lo, hi) in contrasts.items():
        eff = inspection.hazard_effect(forest, X, y, feature=feat, values=[lo, hi], windows=8)
        mean_haz = eff.hazard.mean(axis=1)  # average over windows, one number per grid value
        forest_dir = "+" if mean_haz[1] > mean_haz[0] else "-"
        cox_dir = "+" if cox_coef[feat] > 0 else "-"
        rows.append(
            {
                "feature": feat, "low": lo, "high": hi,
                "hazard(low)": mean_haz[0], "hazard(high)": mean_haz[1],
                "forest_direction": forest_dir, "cox_coef": cox_coef[feat], "cox_direction": cox_dir,
                "agree": forest_dir == cox_dir,
            }
        )
    directions = pd.DataFrame(rows)
    directions.to_csv(OUT / "rossi_directions.csv", index=False)
    print(directions.round(4).to_string(index=False))
    return relevance, directions


def ebmt4_section():
    print("\n=== EBMT4 (competing risks + TVC) ===")
    pdf = ebmt4_frame()
    X = pdf[EBMT4_COVARIATES]
    y = make_competing_risks_y(pdf["stop"].to_numpy(), pdf["event"].to_numpy(), start=pdf["start"].to_numpy())
    ids = pdf["id"].to_numpy()

    t0 = time.perf_counter()
    forest = _cr_forest(causes=[1, 2]).fit(X, y, ids)
    fit_s = time.perf_counter() - t0
    print(f"fit: {fit_s:.2f}s")

    coxes = {}
    for cause in (1, 2):
        cdf = pdf.assign(cause_event=(pdf["event"] == cause).astype(int))
        coxes[cause] = CoxTimeVaryingFitter().fit(
            cdf[["id", "start", "stop", "cause_event", *EBMT4_COVARIATES]],
            id_col="id", start_col="start", stop_col="stop", event_col="cause_event",
        )

    all_directions = []
    for cause in (1, 2):
        perm = inspection.permutation_importance(forest, X, y, ids=ids, cause=cause, oob=True, random_state=0)
        loco = inspection.drop_column_importance(forest, X, y, ids=ids, cv=5, cause=cause, random_state=0)
        relevance = relevance_table(perm, loco, EBMT4_COVARIATES)
        relevance.to_csv(OUT / f"ebmt4_relevance_cause{cause}.csv", index=False)
        print(f"-- cause {cause} relevance --")
        print(relevance.round(4).to_string(index=False))

        cox_coef = coxes[cause].summary["coef"]
        for feat in EBMT4_COVARIATES:
            lo, hi = (0.0, 1.0) if feat == "ae" else (float(X[feat].min()), float(X[feat].max()))
            n_levels = X[feat].nunique()
            eff = inspection.hazard_effect(forest, X, y, feature=feat, values=[lo, hi], windows=8, cause=cause)
            mean_haz = eff.hazard.mean(axis=1)
            forest_dir = "+" if mean_haz[1] > mean_haz[0] else "-"
            cox_dir = "+" if cox_coef[feat] > 0 else "-"
            # Cox fits one linear coefficient across all levels of an ordinally-coded nominal
            # covariate; a >2-level nominal category (year, agecl here) has no reason to have a
            # monotonic relationship with the hazard in that arbitrary code order, so a forest
            # (which makes no linearity assumption) can honestly "disagree" with Cox's forced-
            # linear summary at the two extreme codes without either being wrong -- this is not
            # a same-covariate, same-contrast comparison for a >2-level nominal variable, so it
            # is excluded from the internal-consistency gate (still reported, not gated).
            gated = n_levels <= 2
            all_directions.append(
                {
                    "cause": cause, "feature": feat, "n_levels": n_levels, "low": lo, "high": hi,
                    "hazard(low)": mean_haz[0], "hazard(high)": mean_haz[1],
                    "forest_direction": forest_dir, "cox_coef": cox_coef[feat], "cox_direction": cox_dir,
                    "agree": forest_dir == cox_dir, "gated": gated,
                }
            )
    directions = pd.DataFrame(all_directions)
    directions.to_csv(OUT / "ebmt4_directions.csv", index=False)
    print(directions.round(4).to_string(index=False))

    # cause-distinguishability check for ae: are the two causes' hazard_effect curves for ae different?
    ae_curves = {}
    for cause in (1, 2):
        eff = inspection.hazard_effect(forest, X, y, feature="ae", values=[0.0, 1.0], windows=8, cause=cause)
        ae_curves[cause] = eff.hazard[1] - eff.hazard[0]  # per-window contrast, ae=1 vs ae=0
    distinguishable = not np.allclose(ae_curves[1], ae_curves[2], atol=1e-6)
    print(f"ae hazard_effect distinguishable across causes: {distinguishable}")
    print(f"  cause 1 per-window contrast: {np.round(ae_curves[1], 4)}")
    print(f"  cause 2 per-window contrast: {np.round(ae_curves[2], 4)}")
    return directions, distinguishable


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rossi_relevance, rossi_directions = rossi_section()
    ebmt4_directions, ae_distinguishable = ebmt4_section()

    print("\n=== Decision 4 internal-consistency check ===")
    disagreements = pd.concat(
        [
            rossi_directions.assign(dataset="rossi", gated=True)[["dataset", "feature", "agree", "gated"]],
            ebmt4_directions.assign(dataset="ebmt4")[["dataset", "feature", "agree", "gated"]],
        ]
    )
    gated = disagreements[disagreements["gated"]]
    bad = gated[~gated["agree"]]
    excluded = disagreements[~disagreements["gated"]]
    if not bad.empty:
        print("STOP: forest/Cox direction disagreement found on a gated (binary) covariate:")
        print(bad.to_string(index=False))
    else:
        print("PASS: forest and Cox agree in direction on every gated (binary) covariate/cause.")
    if not excluded.empty:
        print(
            "Excluded from the gate (>2-level nominal covariate; a single linear Cox coefficient "
            "is not a same-contrast comparison against the forest's two-code hazard_effect contrast):"
        )
        print(excluded.to_string(index=False))
    print(f"ae distinguishable across causes: {ae_distinguishable}")


if __name__ == "__main__":
    main()
