"""PBC2 case study: dynamic prediction of death from repeated liver biomarkers.

Run from the repo root: .venv/bin/python -m examples.pbc2_landmark
Data: R survival's ``pbcseq`` (committed fixture, 312 patients). Writes the tables
included by docs/source/case_studies/pbc2.rst.

At landmarks s = 1..4 years, each model predicts P(death by s + 2y | alive at s,
history up to s) for the patients still under observation, in held-out patients
(5-fold GroupKFold by patient):

- counting-process forest: fitted on biomarker paths; at s it predicts along
  the observed path up to s, with covariates carried forward after s (the named
  "locf" scenario: future biomarkers are unknown at s);
- landmark super-model: LandmarkSurvivalForest on history summaries at s;
- reference: Kaplan–Meier of the training patients' landmark outcomes (no covariates).

IPCW censoring weights come from the test risk set at each landmark.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold

from rftvc import LandmarkSurvivalForest, SurvivalForestTV, make_landmark_data, make_survival_y
from rftvc.metrics import KaplanMeierCensoring, brier_landmark, calibration_table, cindex_dynamic
from tests.fixtures.pbcseq import pbcseq_counting_process

OUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "case_studies" / "generated"
YEAR = 365.25
W = 2 * YEAR
LANDMARKS = np.arange(1, 5) * YEAR
BASE = ["age", "sex", "trt", "edema", "ascites"]
MARKERS = ["log_bili", "albumin", "protime"]
HISTORY = BASE + MARKERS + [("log_bili", "max"), ("log_bili", "first")]


def _forest():
    return SurvivalForestTV(n_estimators=300, random_state=0, n_jobs=-1)


def _km_risk(y, w):
    """Kaplan–Meier P(T <= w) from right-censored outcomes (no covariates)."""
    times = np.unique(y["stop"][y["event"] & (y["stop"] <= w)])
    s = 1.0
    for t in times:
        s *= 1 - np.sum(y["event"] & (y["stop"] == t)) / np.sum(y["stop"] >= t)
    return 1 - s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pbcseq_counting_process()
    ids_all = df["id"].to_numpy()
    patients = np.unique(ids_all)
    rows, pooled = [], []
    for fold, (tr_p, te_p) in enumerate(GroupKFold(5).split(patients, groups=patients)):
        train = df.filter(pl.col("id").is_in(patients[tr_p]))
        test = df.filter(pl.col("id").is_in(patients[te_p]))
        cp = _forest().fit(
            train.select(BASE + MARKERS).to_numpy(),
            make_survival_y(train["stop"], train["event"], start=train["start"]),
            train["id"].to_numpy(),
        )
        lm = LandmarkSurvivalForest(horizon=W, landmarks=np.r_[0.0, LANDMARKS], history_features=HISTORY,
                                    forest=_forest()).fit(train)
        lm_train = make_landmark_data(train, horizon=W, landmarks=LANDMARKS, history_features=HISTORY)
        for s in LANDMARKS:
            data = make_landmark_data(test, horizon=W, landmarks=[s], history_features=HISTORY)
            # Landmark super-model
            r_lm = 1 - lm.forest_.predict_survival_function(data.X, [W])[:, 0]
            # Counting-process forest along each test patient's path up to s (then LOCF).
            # A visit exactly at s is known at s: it becomes a tiny row (s, s + 1e-6] whose
            # covariates are then carried forward; earlier rows are cut at s.
            path = test.filter(pl.col("id").is_in(data.ids) & (pl.col("start") <= s)).with_columns(
                pl.when(pl.col("start") == s).then(pl.lit(s + 1e-6))
                .otherwise(pl.min_horizontal("stop", pl.lit(s))).alias("stop")
            )
            r_cp = cp.predict_risk(
                path.select(BASE + MARKERS).to_numpy(), s + W,
                intervals=make_survival_y(path["stop"], np.zeros(path.height, bool), start=path["start"]),
                ids=path["id"].to_numpy(), origin=s, extrapolate="locf",
            )
            order = pd.Index(pd.unique(path["id"].to_numpy())).get_indexer(data.ids)
            assert (order >= 0).all(), "every landmark subject needs a path"
            r_cp = r_cp[order]
            # Reference: training-fold KM at this landmark
            r_km = np.full(data.ids.size, _km_risk(lm_train.y[lm_train.s == s], W))
            cens = KaplanMeierCensoring().fit(data.y)
            for name, r in (("counting-process forest", r_cp), ("landmark super-model", r_lm), ("Kaplan–Meier", r_km)):
                rows.append({
                    "fold": fold, "landmark_years": s / YEAR, "model": name, "n": data.ids.size,
                    "deaths": int((data.y["event"] & (data.y["stop"] <= W)).sum()),
                    "brier": brier_landmark(data.y, r, W, censoring_estimator=cens),
                    "auc": np.nan if name == "Kaplan–Meier" else cindex_dynamic(data.y, r, W, censoring_estimator=cens),
                })
            pooled.append((data.y, r_lm))
    res = pd.DataFrame(rows)
    table = res.groupby(["landmark_years", "model"], sort=False).agg(
        n=("n", "sum"), deaths=("deaths", "sum"), brier=("brier", "mean"), auc=("auc", "mean")
    ).reset_index()
    table.round(3).to_csv(OUT / "pbc2_landmarks.csv", index=False)
    overall = res.groupby("model", sort=False)[["brier", "auc"]].mean().round(3).reset_index()
    overall.to_csv(OUT / "pbc2_overall.csv", index=False)
    y = np.concatenate([p[0] for p in pooled])
    r = np.concatenate([p[1] for p in pooled])
    cal = pd.DataFrame(calibration_table(y, r, W, n_bins=5).to_dicts())
    cal.round(3).to_csv(OUT / "pbc2_calibration.csv", index=False)
    print(table.round(3).to_string())
    print(overall.to_string())
    print(cal.round(3).to_string())


if __name__ == "__main__":
    main()
