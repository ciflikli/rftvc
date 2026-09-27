"""EBMT4 case study: a competing-risks fit with a time-varying covariate.

Run from the repo root: .venv/bin/python -m examples.ebmt4_case_study
Downloads mstate::ebmt4 on first use, checksum-verified and cached under
RFTVC_DATA -- never redistributed with rftvc (tests/fixtures/ebmt4.py; the
licence discussion is in docs/plans/rc-validation-findings.md). Writes the
tables docs/source/case_studies/ebmt4.rst includes into
docs/source/case_studies/generated/.

``ae`` is a generic adverse event, not acute GvHD (mstate's own
documentation: the data are simplified for illustration, not clinical
conclusions -- docs/plans/rc-validation-plan.md Decision 3). No expected
sign is claimed for it; the check here is whether the forest and a
cause-specific Cox fit agree in direction on the same data, and whether the
two causes' hazard_effect curves for ae are distinguishable at all.
"""

import json
from pathlib import Path

import pandas as pd
import polars as pl
from lifelines import CoxTimeVaryingFitter

from rftvc import CompetingRisksForestTV, inspection, make_competing_risks_y
from tests.fixtures.ebmt4 import ebmt4_competing_risks

OUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "case_studies" / "generated"
COVARIATES = ["ae", "year", "agecl", "proph", "match"]
# Explicit, deterministic level order (the R factor levels themselves, which
# are already ordinal for year/agecl) -- plain `.cast(pl.Categorical)` codes
# by first appearance, which is not stable across process runs (see
# docs/plans/rc-validation-findings.md's finding on examples/rc_validation.py).
LEVELS = {
    "year": ["1985-1989", "1990-1994", "1995-1998"],
    "agecl": ["<=20", "20-40", ">40"],
    "proph": ["no", "yes"],
    "match": ["no gender mismatch", "gender mismatch"],
}


def frame():
    """Counting-process rows with year/agecl/proph/match ordinal-coded, ae already 0/1."""
    d = ebmt4_competing_risks().with_columns(
        pl.col(c).cast(pl.Enum(levels)).to_physical().alias(c) for c, levels in LEVELS.items()
    )
    return pd.DataFrame({c: d[c].to_numpy() for c in d.columns})  # pyarrow-free (compatibility.rst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = frame()
    X = pdf[COVARIATES]
    y = make_competing_risks_y(pdf["stop"].to_numpy(), pdf["event"].to_numpy(), start=pdf["start"].to_numpy())
    ids = pdf["id"].to_numpy()

    forest = CompetingRisksForestTV(
        n_estimators=500, min_ids_leaf=10, random_state=0, n_jobs=-1, causes=[1, 2]
    ).fit(X, y, ids)

    coxes = {}
    for cause in (1, 2):
        cdf = pdf.assign(cause_event=(pdf["event"] == cause).astype(int))
        coxes[cause] = CoxTimeVaryingFitter().fit(
            cdf[["id", "start", "stop", "cause_event", *COVARIATES]],
            id_col="id", start_col="start", stop_col="stop", event_col="cause_event",
        )

    relevance_rows, direction_rows = [], []
    for cause in (1, 2):
        perm = inspection.permutation_importance(forest, X, y, ids=ids, cause=cause, oob=True, random_state=0)
        loco = inspection.drop_column_importance(forest, X, y, ids=ids, cv=5, cause=cause, random_state=0)
        for feat, p, lc in zip(COVARIATES, perm.importances_mean, loco.importances_mean):
            relevance_rows.append({"cause": cause, "feature": feat, "perm_importance": p, "loco_importance": lc})

        cox_coef = coxes[cause].summary["coef"]
        for feat in COVARIATES:
            lo, hi = (0.0, 1.0) if feat == "ae" else (float(X[feat].min()), float(X[feat].max()))
            n_levels = X[feat].nunique()
            eff = inspection.hazard_effect(forest, X, y, feature=feat, values=[lo, hi], windows=8, cause=cause)
            mean_haz = eff.hazard.mean(axis=1)
            forest_dir = "+" if mean_haz[1] > mean_haz[0] else "-"
            cox_dir = "+" if cox_coef[feat] > 0 else "-"
            # A >2-level nominal covariate's two-point contrast is not a same-contrast
            # comparison against Cox's single forced-linear coefficient (docs/source/
            # user_guide/importance.rst); the gate below applies only to binary covariates.
            direction_rows.append(
                {
                    "cause": cause, "feature": feat, "n_levels": n_levels, "low": lo, "high": hi,
                    "hazard(low)": round(float(mean_haz[0]), 4), "hazard(high)": round(float(mean_haz[1]), 4),
                    "forest_direction": forest_dir, "cox_coef": round(float(cox_coef[feat]), 4),
                    "cox_direction": cox_dir, "agree": forest_dir == cox_dir, "gated": n_levels <= 2,
                }
            )

    relevance = pd.DataFrame(relevance_rows).round(4)
    relevance.to_csv(OUT / "ebmt4_relevance.csv", index=False)
    directions = pd.DataFrame(direction_rows)
    directions.to_csv(OUT / "ebmt4_directions.csv", index=False)

    # Cause-distinguishability: ae's hazard_effect contrast (1 vs 0), per window, per cause.
    contrast_rows = []
    for cause in (1, 2):
        eff = inspection.hazard_effect(forest, X, y, feature="ae", values=[0.0, 1.0], windows=8, cause=cause)
        contrast = eff.hazard[1] - eff.hazard[0]
        for window, c in enumerate(contrast, start=1):
            contrast_rows.append({"cause": cause, "window": window, "ae_contrast": round(float(c), 4)})
    pd.DataFrame(contrast_rows).to_csv(OUT / "ebmt4_ae_contrast.csv", index=False)

    gated = directions[directions["gated"]]
    summary = pd.DataFrame(
        [
            {
                "patients": int(pdf["id"].nunique()),
                "rows": len(pdf),
                "relapse_events": int((pdf.groupby("id")["event"].max() == 1).sum()),
                "death_without_relapse_events": int((pdf.groupby("id")["event"].max() == 2).sum()),
                "binary_covariates_gated": int(len(gated)),
                "binary_covariates_agree": int(gated["agree"].sum()),
            }
        ]
    )
    summary.to_csv(OUT / "ebmt4_summary.csv", index=False)

    print(json.dumps(summary.iloc[0].to_dict(), indent=1))
    print(relevance.to_string(index=False))
    print(directions.to_string(index=False))


if __name__ == "__main__":
    main()
