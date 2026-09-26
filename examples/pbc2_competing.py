"""PBC2 competing risks: transplant vs death, landmark cumulative incidence from repeated biomarkers.

Run from the repo root: .venv/bin/python -m examples.pbc2_competing
Data: R survival's ``pbcseq`` (committed fixture), with the outcome as a cause
label on each patient's last row: transplant (1), death (2), else 0. Writes the
tables included by docs/source/case_studies/pbc2.rst.

At landmarks s = 1..4 years, the landmark competing-risks super-model predicts
F_k(s + 2y | event-free at s, history up to s) for each cause, in held-out
patients (5-fold GroupKFold by patient, ``landmark_cross_validate``). The
reference is the same model with single-leaf trees: the Aalen–Johansen
estimate of the training landmark rows, without covariates. Metrics: the
cause-specific IPCW Brier score and integrated Brier over (0, 2y] and Wolbers'
cause-specific C at 2y, with censoring weights from each test landmark's risk set.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from examples.pbc2_landmark import HISTORY, LANDMARKS, W, YEAR
from rftvc import CompetingRisksForestTV, LandmarkCompetingRisksForest
from rftvc.model_selection import landmark_cross_validate
from tests.fixtures.pbcseq import pbcseq_competing_risks

OUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "case_studies" / "generated"
CAUSES = {1: "transplant", 2: "death"}


def main():
    df = pbcseq_competing_risks()
    rows = []
    for cause, cause_name in CAUSES.items():
        for model_name, depth in (("landmark competing-risks forest", None), ("Aalen–Johansen (no covariates)", 0)):
            model = LandmarkCompetingRisksForest(
                horizon=W, landmarks=LANDMARKS, history_features=HISTORY, causes=[1, 2], score_cause=cause,
                forest=CompetingRisksForestTV(n_estimators=300, max_depth=depth, random_state=0),
            )
            res = landmark_cross_validate(model, df, GroupKFold(5), ("brier", "integrated_brier", "cindex_incident"),
                                          n_times=8)
            res = pd.DataFrame(res.to_dicts())
            res["model"], res["cause"] = model_name, cause_name
            rows.append(res)
    res = pd.concat(rows)
    res["landmark_years"] = res["landmark"] / YEAR
    by_lm = res.groupby(["cause", "landmark_years", "model"], sort=False).agg(
        n=("n", "sum"), cases=("n_cases", "sum"), brier=("brier", "mean"), ibs=("integrated_brier", "mean"),
        c=("cindex_incident", "mean")).reset_index()
    overall = res.groupby(["cause", "model"], sort=False)[["brier", "integrated_brier", "cindex_incident"]].mean()
    OUT.mkdir(parents=True, exist_ok=True)
    by_lm.round(3).to_csv(OUT / "pbc2_cr_landmarks.csv", index=False)
    overall.round(3).reset_index().to_csv(OUT / "pbc2_cr_overall.csv", index=False)
    print(by_lm.round(3).to_string())
    print(overall.round(3).to_string())


if __name__ == "__main__":
    main()
