"""Measure one-hot category width, fit cost and held-out concordance.

Run from the repository root::

    python bench/categorical_width.py --output /tmp/category-width.jsonl

The paired ``max_features`` settings expose indicator-level sampling effects;
this is a decision aid, not a cross-library speed comparison.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rftvc import SurvivalForestTV, make_survival_y
from rftvc.metrics import concordance_index_cp


def sample(n, levels, seed):
    rng = np.random.default_rng(seed)
    category = rng.integers(levels, size=n)
    numeric = rng.normal(size=(n, 8))
    # Noncontiguous high-risk levels require several one-hot threshold splits.
    group = ((category * 7) % levels) < levels // 2
    log_hazard = 0.8 * numeric[:, 0] + 0.7 * group
    event_time = rng.exponential(np.exp(-log_hazard))
    censor_time = rng.exponential(2.5, size=n)
    y = make_survival_y(np.minimum(event_time, censor_time), event_time <= censor_time)
    X = pd.DataFrame(numeric, columns=[f"z{j}" for j in range(8)])
    X["group"] = pd.Categorical([f"level-{j:03d}" for j in category])
    return X, y


def run(n_train, n_test, trees, levels, seed, max_features):
    X_train, y_train = sample(n_train, levels, seed)
    X_test, y_test = sample(n_test, levels, seed + 10000)
    model = SurvivalForestTV(n_estimators=trees, max_features=max_features,
                             min_ids_leaf=5, min_events_leaf=2,
                             n_jobs=1, random_state=seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    risk = model.predict(X_test)
    return {
        "n_train": n_train, "n_test": n_test, "n_trees": trees,
        "levels": levels, "seed": seed, "max_features": max_features,
        "encoded_features": model.n_encoded_features_,
        "fit_seconds": round(fit_seconds, 4),
        "model_bytes": len(pickle.dumps(model)),
        "heldout_c": round(concordance_index_cp(y_test, risk), 5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 8, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [run(args.n_train, args.n_test, args.trees, k, seed, mf)
            for k in args.levels for seed in args.seeds for mf in ("sqrt", None)]
    output = "\n".join(json.dumps(row) for row in rows) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
