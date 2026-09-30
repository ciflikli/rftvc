"""Rebuild the README figures from a seeded simulation and case-study results.

Run from the repository root: .venv/bin/python -m examples.readme_plots
The importance CSV is produced by examples.tvc_importance_case_study and is
committed with the case study.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from rftvc import SurvivalForestTV, make_survival_y


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "readme"
INK, MUTED, GRID = "#17233B", "#63738A", "#E5EAF0"
BLUE, PLUM, DARK = "#5D95C5", "#906388", "#46353A"


def style():
    # Match rftvc.viz's Inter → Arial stack and colors without a global theme.
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Inter", "Arial") if name in installed), "sans-serif")
    plt.rcParams.update({
        "font.family": font,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def survival_plot():
    rng = np.random.default_rng(24)
    features, starts, stops, events, ids = [], [], [], [], []
    for subject in range(420):
        baseline = rng.normal()
        for year in range(6):
            z = rng.normal(loc=0.15 * year)
            rate = 0.08 * np.exp(0.7 * baseline + 0.95 * z)
            event_time = rng.exponential(1 / rate)
            features.append([baseline, z])
            starts.append(float(year))
            stops.append(year + min(event_time, 1.0))
            events.append(event_time < 1.0)
            ids.append(subject)
            if event_time < 1.0:
                break

    forest = SurvivalForestTV(n_estimators=100, min_ids_leaf=8, random_state=24).fit(
        np.asarray(features), make_survival_y(stops, events, start=starts), ids=np.asarray(ids)
    )
    path_starts = np.tile(np.arange(6, dtype=float), 2)
    path_stops = path_starts + 1
    z = np.r_[np.linspace(1.2, -1.2, 6), np.linspace(-1.2, 1.2, 6)]
    paths = np.c_[np.zeros(12), z]
    intervals = make_survival_y(path_stops, np.zeros(12, dtype=bool), start=path_starts)
    times = np.linspace(0, 6, 121)
    curves = forest.predict_survival_function(
        paths, times, intervals=intervals, ids=np.repeat([0, 1], 6)
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    for curve, label, color in zip(curves, ["falling z", "rising z"], [BLUE, PLUM]):
        ax.step(times, curve, where="post", lw=3, label=label, color=color)
    ax.set(xlim=(0, 6), ylim=(0, 1.02), xlabel="Follow-up time", ylabel="Predicted survival")
    ax.set_title("Same baseline, different covariate paths", loc="left", fontsize=19, weight="bold", pad=20)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=12)
    ax.tick_params(labelsize=11)
    fig.tight_layout(pad=2)
    fig.savefig(OUT / "tvc-survival.png", dpi=200)
    plt.close(fig)


def importance_plot():
    data = pd.read_csv(ROOT / "docs/source/case_studies/generated/tvc_importance_cause_hazard.csv")
    data = data.loc[data.cause == 1].sort_values("perm_importance")
    colors = [BLUE if name == "z" else PLUM if name in {"x0", "x1"} else GRID for name in data.feature]
    names = ["z · time varying" if name == "z" else name for name in data.feature]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=200)
    y = np.arange(len(data))
    ax.barh(y, data.perm_importance, color=colors, height=0.64)
    ax.errorbar(data.perm_importance, y, xerr=data.perm_se, fmt="none", ecolor=DARK, capsize=4, lw=1.6)
    ax.set(yticks=y, yticklabels=names, xlabel="Change in piecewise-exponential score per event")
    ax.set_title("The changing covariate drives cause 1 hazard", loc="left", fontsize=19, weight="bold", pad=20)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=12, axis="y", length=0, pad=10)
    ax.tick_params(labelsize=11, axis="x")
    ax.set_xlim(-0.075, 0.49)
    fig.tight_layout(pad=2)
    fig.savefig(OUT / "importance.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    survival_plot()
    importance_plot()
