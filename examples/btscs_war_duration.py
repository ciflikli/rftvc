"""BTSCS case study: how long do wars last? (Cunningham & Lemke 2013 data).

Run from the repo root: .venv/bin/python -m examples.btscs_war_duration
Downloads the replication archive on first use (checksum-verified; see
examples/data/cunningham_lemke.py) and writes the tables included by
docs/source/case_studies/btscs.rst into docs/source/case_studies/generated/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from sklearn.model_selection import GroupKFold

from examples.data.cunningham_lemke import COVARIATES, load_counting_process
from rftvc import SurvivalForestTV, check_counting_process, make_survival_y
from rftvc.metrics import concordance_index_cp

OUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "case_studies" / "generated"
YEAR = 365.25


def forest(**kw):
    return SurvivalForestTV(n_estimators=500, min_ids_leaf=10, random_state=0, n_jobs=-1, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, report = load_counting_process()
    X = pdf[COVARIATES].to_numpy()
    y = make_survival_y(pdf["stop"], pdf["event"], start=pdf["start"])
    ids = pdf["CLID"].to_numpy()

    # 1. The paper's model: Cox with time-varying covariates (lifelines), all data.
    ctv = CoxTimeVaryingFitter().fit(
        pdf[["CLID", "start", "stop", "event", *COVARIATES]], id_col="CLID", start_col="start",
        stop_col="stop", event_col="event",
    )
    ctv.summary[["coef", "exp(coef)", "p"]].round(3).rename_axis("covariate").to_csv(OUT / "btscs_cox.csv")

    # 2. New-war cross-validation: GroupKFold by war, counting-process C on held-out war-years.
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, groups=ids)):
        f = forest().fit(X[tr], y[tr], ids[tr], gap_policy="split_id")
        cox = CoxTimeVaryingFitter().fit(
            pdf.iloc[tr][["CLID", "start", "stop", "event", *COVARIATES]], id_col="CLID",
            start_col="start", stop_col="stop", event_col="event",
        )
        cox_risk = cox.predict_partial_hazard(pdf.iloc[te][COVARIATES]).to_numpy()
        rows.append({
            "fold": fold,
            "wars": int(np.unique(ids[te]).size),
            "terminations": int(y["event"][te].sum()),
            "forest": concordance_index_cp(y[te], f.predict(X[te]), ids=ids[te]),
            "cox": concordance_index_cp(y[te], cox_risk, ids=ids[te]),
        })
    cv = pd.DataFrame(rows)
    total = {"fold": "all", "wars": cv["wars"].sum(), "terminations": cv["terminations"].sum(),
             "forest": cv["forest"].mean(), "cox": cv["cox"].mean()}
    cv = pd.concat([cv.astype({"fold": str}), pd.DataFrame([total])], ignore_index=True)
    cv.round(3).to_csv(OUT / "btscs_cv.csv", index=False)
    mean = cv.iloc[-1]

    # 3. Full fit with id-level OOB (a new-war estimate from the same fit).
    full = forest(oob_score=True).fit(X, y, ids, gap_policy="split_id")

    # 4. Path predictions: P(war still ongoing at t | observed covariate path), two example wars.
    examples = []
    # The longest civil and interstate wars observed from onset without a gap.
    first_seg = []
    for clid, w in pdf.groupby("CLID"):
        seg_end = np.flatnonzero(np.r_[w["start"].to_numpy()[1:] != w["stop"].to_numpy()[:-1], True])[0] + 1
        if w["start"].iloc[0] == 0:
            first_seg.append((clid, int(w["civil"].iloc[0]), w["stop"].iloc[seg_end - 1], w.iloc[:seg_end]))
    picks = [max((c for c in first_seg if c[1] == k), key=lambda c: c[2]) for k in (1, 0)]
    for clid, _, _, w in picks:
        times = np.array([1, 2, 3, 5]) * YEAR
        times = times[times <= w["stop"].max()]
        S = full.predict_survival_function(
            w[COVARIATES].to_numpy(), times, intervals=make_survival_y(w["stop"], np.zeros(len(w), bool), start=w["start"]),
            ids=w["CLID"].to_numpy(),
        )[0]
        raw = w.iloc[0]
        examples.append({
            "CLID": int(clid), "civil": int(raw["civil"]), "first_year": int(raw["year"]),
            "years_observed": round(w["stop"].max() / YEAR, 1),
            **{f"S({t / YEAR:.0f}y)": round(float(s), 3) for t, s in zip(times, S)},
        })
    pd.DataFrame(examples).to_csv(OUT / "btscs_paths.csv", index=False)

    summary = {
        **report,
        "chains": check_counting_process(y["start"].copy(), y["stop"].copy(), y["event"].copy(), ids, gap_policy="split_id").n_groups,
        "oob_cindex": round(float(full.oob_score_), 3),
        "cv_forest_cindex": round(float(mean["forest"]), 3),
        "cv_cox_cindex": round(float(mean["cox"]), 3),
    }
    (OUT / "btscs_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(cv.round(3).to_string())
    print(pd.DataFrame(examples).to_string())


if __name__ == "__main__":
    main()
