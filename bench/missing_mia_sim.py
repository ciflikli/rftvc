"""Closed-form missingness-signal check for native MIA splits.

The latent binary risk class sets both the exponential event rate and whether
the sole feature is missing. This is deliberately a favorable MNAR case for
MIA, not a claim about arbitrary missingness mechanisms. At horizon 2 the
true survival is exp(-rate * 2), so predictions can be compared to truth
without a noisy observed-event score.
"""

import argparse

import numpy as np

from rftvc import SurvivalForestTV, make_survival_y


HORIZON = 2.0
RATES = np.array([0.08, 0.8])


def _sample(rng, n):
    risk_class = rng.integers(0, 2, size=n)
    x = np.zeros((n, 1))
    x[risk_class == 1, 0] = np.nan
    event_time = rng.exponential(1.0 / RATES[risk_class])
    y = make_survival_y(np.minimum(event_time, HORIZON), event_time <= HORIZON)
    truth = np.exp(-RATES[risk_class] * HORIZON)
    return x, y, truth


def run_seed(seed, *, n_train=400, n_test=400):
    rng = np.random.default_rng(seed)
    x_train, y_train, _ = _sample(rng, n_train)
    x_test, _, truth = _sample(rng, n_test)
    options = dict(n_estimators=1, max_features=1, max_depth=1,
                   min_ids_leaf=10, min_events_leaf=2, max_samples=1.0,
                   random_state=seed)
    mia = SurvivalForestTV(**options).fit(x_train, y_train)
    keep = ~np.isnan(x_train[:, 0])
    complete_case = SurvivalForestTV(**options).fit(x_train[keep], y_train[keep])
    mia_pred = mia.predict_survival_function(x_test, [HORIZON])[:, 0]
    cc_pred = complete_case.predict_survival_function(x_test, [HORIZON])[:, 0]
    mia_mse = float(np.mean((mia_pred - truth) ** 2))
    cc_mse = float(np.mean((cc_pred - truth) ** 2))
    return mia_mse, cc_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="*", type=int, default=list(range(10)))
    args = parser.parse_args()
    for seed in args.seeds:
        mia, cc = run_seed(seed)
        print(f"seed={seed} mia_mse={mia:.6f} complete_case_mse={cc:.6f} ratio={mia / cc:.4f}")
