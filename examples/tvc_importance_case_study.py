"""Importance/effects case study: cause-specific hazard vs cumulative-incidence importance.

Run from the repo root: .venv/bin/python -m examples.tvc_importance_case_study
Data: the competing-risks generator, scenario A (``bench/s14_cr_sim.py``):
``z`` (redrawn each unit interval) drives cause 1's hazard; ``x0`` and ``x1``
drive cause 2's; three columns are pure noise. Writes the tables included by
docs/source/case_studies/importance.rst.

Fits two models on the same simulated data:
- a ``CompetingRisksForestTV`` on the raw counting-process rows, scored for
  cause-specific-hazard importance (``permutation_importance``,
  ``drop_column_importance``, ``scoring="pe"``) and ``hazard_effect``;
- a ``LandmarkCompetingRisksForest`` (stacked, ``history_features=["z", ("z",
  "mean"), "x0", "x1"]``) on the same rows, scored for cumulative-incidence
  (``F_k``) importance (``scoring="brier"``) and the level-vs-history
  diagnostic on ``z``.

This shows the design's required point: ``z`` matters for cause 1's hazard
and, through it, for cause 1's own cumulative incidence. The competition
mechanism (a covariate can matter for a competing cause's ``F_k`` purely by
changing how long subjects stay at risk for it) is a real effect in general,
but at this sample size ``z``'s measured ``F_2`` importance comes out
indistinguishable from zero -- reported as measured, not forced to
demonstrate a large competition effect (see the case study's own text).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from bench.s14_cr_sim import training_rows
from rftvc import CompetingRisksForestTV, LandmarkCompetingRisksForest, inspection

OUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "case_studies" / "generated"
SCENARIO = "A"
FEATURES = ["z", "x0", "x1", "noise0", "noise1", "noise2"]


def _as_frame(X, y, ids):
    return pl.DataFrame(
        {
            "id": ids,
            "start": y["start"],
            "stop": y["stop"],
            "event": y["event"],
            **{name: X[:, i] for i, name in enumerate(FEATURES)},
        }
    )


def main():
    rng = np.random.default_rng(0)
    Xa, y, ids = training_rows(SCENARIO, 800, rng)
    Xta, yt, idt = training_rows(SCENARIO, 400, np.random.default_rng(1))
    X, Xt = pd.DataFrame(Xa, columns=FEATURES), pd.DataFrame(Xta, columns=FEATURES)

    t0 = time.perf_counter()
    cp = CompetingRisksForestTV(n_estimators=300, causes=[1, 2], random_state=0).fit(X, y, ids)
    fit_cp_s = time.perf_counter() - t0

    # cause-specific-hazard importance, and its per-window decomposition for z (the true cause-1 signal)
    perm_s = loco_s = 0.0
    rows = []
    for k in (1, 2):
        t0 = time.perf_counter()
        perm_k = inspection.permutation_importance(cp, Xt, yt, ids=idt, cause=k, random_state=0, n_bootstrap=50)
        perm_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        loco_k = inspection.drop_column_importance(cp, X, y, ids=ids, cv=5, cause=k, random_state=0)
        loco_s += time.perf_counter() - t0
        for j, name in enumerate(perm_k.feature_names):
            rows.append(
                {
                    "cause": k,
                    "feature": name,
                    "perm_importance": perm_k.importances_mean[j],
                    "perm_se": perm_k.importances_se[j],
                    "loco_importance": loco_k.importances_mean[j],
                }
            )
    cause_hazard = pd.DataFrame(rows)
    cause_hazard.to_csv(OUT / "tvc_importance_cause_hazard.csv", index=False)

    # feature names now resolve directly ("z") since cp is fit on named DataFrames
    window_rows = [
        {"cause": k, "window_hi": edge, "importance": val}
        for k in (1, 2)
        for edge, val in zip(
            inspection.permutation_importance(cp, Xt, yt, ids=idt, cause=k, features=["z"], random_state=0, n_bootstrap=0)
            .window_edges[1:],
            inspection.permutation_importance(
                cp, Xt, yt, ids=idt, cause=k, features=["z"], random_state=0, n_bootstrap=0
            ).importances_window[0],
        )
    ]
    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(OUT / "tvc_importance_windowed_z.csv", index=False)

    effect = inspection.hazard_effect(cp, Xt, yt, feature="z", cause=1, windows=6)
    effect_df = pd.DataFrame(
        {"z": effect["values"], **{f"window_{i + 1}": effect.hazard[:, i] for i in range(effect.hazard.shape[1])}}
    )
    effect_df.to_csv(OUT / "tvc_importance_hazard_effect.csv", index=False)

    # landmark model: F_k importance (Brier) and the level-vs-history diagnostic
    df = _as_frame(Xa, y, ids)
    dft = _as_frame(Xta, yt, idt)
    landmark_model = LandmarkCompetingRisksForest(
        horizon=2.0, landmarks=[1.0, 2.0, 3.0, 4.0, 5.0],
        history_features=["z", ("z", "mean"), "x0", "x1"], causes=[1, 2],
        forest=CompetingRisksForestTV(n_estimators=300, random_state=0),
    )
    t0 = time.perf_counter()
    landmark_model = landmark_model.fit(df)
    fit_lm_s = time.perf_counter() - t0

    cif_rows = []
    for k in (1, 2):
        res = inspection.permutation_importance(
            landmark_model, dft, features=["z"], scoring="brier", cause=k, random_state=0, n_bootstrap=50
        )
        cif_rows.append({"cause": k, "feature": "z", "cif_importance_brier": res.importances_mean[0]})
    cif_df = pd.DataFrame(cif_rows)
    cif_df.to_csv(OUT / "tvc_importance_cif_brier.csv", index=False)

    diagnostic = {
        "history_given_level": inspection.permutation_importance(
            landmark_model, dft, features=["z_mean"], conditional_on=["z"], random_state=0, n_bootstrap=50
        ).importances_mean[0],
        "level_given_history": inspection.permutation_importance(
            landmark_model, dft, features=["z"], conditional_on=["z_mean"], random_state=0, n_bootstrap=50
        ).importances_mean[0],
    }
    pd.DataFrame([diagnostic]).to_csv(OUT / "tvc_importance_level_history.csv", index=False)

    timing = pd.DataFrame(
        [
            {"step": "fit CompetingRisksForestTV (800 ids, 300 trees)", "seconds": fit_cp_s},
            {"step": "permutation_importance (held-out, n_bootstrap=50, both causes)", "seconds": perm_s},
            {"step": "drop_column_importance (cv=5, both causes)", "seconds": loco_s},
            {"step": "fit LandmarkCompetingRisksForest", "seconds": fit_lm_s},
        ]
    )
    timing.to_csv(OUT / "tvc_importance_timing.csv", index=False)

    print(cause_hazard.to_string(index=False))
    print(window_df.to_string(index=False))
    print(cif_df.to_string(index=False))
    print(diagnostic)
    print(timing.to_string(index=False))


if __name__ == "__main__":
    main()
