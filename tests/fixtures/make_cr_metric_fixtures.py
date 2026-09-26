"""Regenerate tests/fixtures/cr_brier.json with comprisk 0.8.0 (IPCW Brier for competing risks).

comprisk is not a dependency: install it anywhere importable, e.g.
``uv pip install --target /tmp/cr_lib comprisk==0.8.0`` and run from the repo root
``PYTHONPATH=/tmp/cr_lib python tests/fixtures/make_cr_metric_fixtures.py``.

comprisk's Brier (Gerds–Schumacher, extended to competing risks) weights cases
and competing events by 1/G(T-) and subjects event-free past t by 1/G(t), with a
Kaplan–Meier censoring model (events before censoring at ties). This equals
rftvc's convention (controls: stop >= t, weight 1/G(t-)) when no observed time
equals an evaluation time, which the continuous data below guarantee.
"""

import json
from pathlib import Path

import numpy as np
from comprisk import evaluation

OUT = Path(__file__).with_name("cr_brier.json")


def main():
    rng = np.random.default_rng(20260926)
    n = 150
    t1, t2, c = rng.exponential(2.0, n), rng.exponential(3.0, n), rng.exponential(3.0, n)
    time = np.minimum(np.minimum(t1, t2), c)
    event = np.where(time == c, 0, np.where(t1 < t2, 1, 2))
    eval_times = np.array([0.5, 1.0, 1.7, 2.5])
    assert not np.isin(time, eval_times).any()
    # Per-subject predicted CIFs, non-decreasing in t, one set per cause.
    probs = {k: np.sort(rng.uniform(0, 0.6, (n, eval_times.size)), axis=1) for k in (1, 2)}
    t_unique, G = evaluation._km_censoring_cr(time, event)
    out = {"time": time.tolist(), "event": event.tolist(), "eval_times": eval_times.tolist(), "causes": {}}
    for k, p in probs.items():
        _, brier = evaluation._per_time_auc_brier(p, time, event, eval_times, t_unique, G, k)
        out["causes"][str(k)] = {"probs": p.tolist(), "brier": brier.tolist()}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
