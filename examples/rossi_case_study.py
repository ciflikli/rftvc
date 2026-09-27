"""Rossi case study: does financial aid lower the hazard of rearrest?

Run from the repo root: .venv/bin/python -m examples.rossi_case_study
Downloads Rossi (carData::Rossi) on first use, checksum-verified and cached
under RFTVC_DATA -- never redistributed with rftvc (tests/fixtures/rossi.py;
the licence discussion is in docs/plans/rc-validation-findings.md). Writes
the tables docs/source/case_studies/rossi.rst includes into
docs/source/case_studies/generated/.
"""

import json
from pathlib import Path

import pandas as pd
import polars as pl
from lifelines import CoxTimeVaryingFitter

from rftvc import SurvivalForestTV, inspection, make_survival_y
from tests.fixtures.rossi import rossi_counting_process

OUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "case_studies" / "generated"
COVARIATES = ["employed", "fin", "age", "race", "wexp", "mar", "paro", "prio", "educ"]
# The plan's expected directions (Rossi, Berk & Lenihan 1980): financial aid,
# employment and age lower the hazard of rearrest; prior convictions raise it.
CONTRASTS = {"fin": (0.0, 1.0), "age": (20.0, 30.0), "prio": (0.0, 10.0), "employed": (0.0, 1.0)}


def _to_pandas(d):
    """``pl.DataFrame`` -> ``pd.DataFrame`` without pyarrow (docs/source/compatibility.rst)."""
    return pd.DataFrame({c: d[c].to_numpy() for c in d.columns})


def frame():
    """Weekly counting-process rows with the binary covariates encoded 0/1."""
    d = rossi_counting_process().with_columns(
        (pl.col("race") == "black").cast(pl.Int64).alias("race"),
        (pl.col("mar") == "married").cast(pl.Int64).alias("mar"),
        (pl.col("fin") == "yes").cast(pl.Int64).alias("fin"),
        (pl.col("wexp") == "yes").cast(pl.Int64).alias("wexp"),
        (pl.col("paro") == "yes").cast(pl.Int64).alias("paro"),
    )
    return _to_pandas(d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = frame()
    X = pdf[COVARIATES]
    y = make_survival_y(pdf["stop"].to_numpy(), pdf["event"].to_numpy(), start=pdf["start"].to_numpy())
    ids = pdf["id"].to_numpy()

    forest = SurvivalForestTV(
        n_estimators=500, min_ids_leaf=10, random_state=0, n_jobs=-1, oob_score=True
    ).fit(X, y, ids)

    cox = CoxTimeVaryingFitter().fit(
        pdf[["id", "start", "stop", "event", *COVARIATES]],
        id_col="id", start_col="start", stop_col="stop", event_col="event",
    )
    cox.summary[["coef", "exp(coef)", "p"]].round(3).rename_axis("covariate").to_csv(OUT / "rossi_cox.csv")

    # Relevance ranking: a score-drop magnitude, not a signed effect (docs/source/user_guide/importance.rst).
    perm = inspection.permutation_importance(forest, X, y, ids=ids, oob=True, random_state=0)
    relevance = pd.DataFrame(
        {"feature": COVARIATES, "perm_importance": perm.importances_mean}
    ).sort_values("perm_importance", ascending=False)
    relevance.round(4).to_csv(OUT / "rossi_relevance.csv", index=False)

    # Direction: hazard_effect two-point contrasts vs the Cox coefficient's sign.
    cox_coef = cox.summary["coef"]
    rows = []
    for feat, (lo, hi) in CONTRASTS.items():
        eff = inspection.hazard_effect(forest, X, y, feature=feat, values=[lo, hi], windows=8)
        mean_haz = eff.hazard.mean(axis=1)
        forest_dir = "+" if mean_haz[1] > mean_haz[0] else "-"
        cox_dir = "+" if cox_coef[feat] > 0 else "-"
        rows.append(
            {
                "feature": feat, "low": lo, "high": hi,
                "hazard(low)": round(float(mean_haz[0]), 4), "hazard(high)": round(float(mean_haz[1]), 4),
                "forest_direction": forest_dir, "cox_coef": round(float(cox_coef[feat]), 4),
                "cox_direction": cox_dir, "agree": forest_dir == cox_dir,
            }
        )
    directions = pd.DataFrame(rows)
    directions.to_csv(OUT / "rossi_directions.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "subjects": int(pdf["id"].nunique()),
                "rows": len(pdf),
                "events": int(pdf["event"].sum()),
                "oob_cindex": round(float(forest.oob_score_), 3),
            }
        ]
    )
    summary.to_csv(OUT / "rossi_summary.csv", index=False)

    print(json.dumps(summary.iloc[0].to_dict(), indent=1))
    print(relevance.to_string(index=False))
    print(directions.to_string(index=False))


if __name__ == "__main__":
    main()
