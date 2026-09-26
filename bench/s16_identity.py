"""S16 bit-identity check: S9 cases plus competing-risks and block-mode fits.

    python -m bench.s16_identity OUT.npz            # on a main build, then on the S16 build
    python -m bench.s16_identity --compare A.npz B.npz
"""

import sys

import numpy as np

from bench import s9_identity
from bench.perf_fit import synth
from rftvc import CompetingRisksForestTV, SurvivalForestTV, make_competing_risks_y


def _cr_case(seed=3):
    X, y, ids = synth(3000, rows_per_id=4, seed=seed)
    rng = np.random.default_rng(seed)
    labels = np.where(y["event"], rng.integers(1, 3, size=len(y)), 0)
    return X, make_competing_risks_y(y["stop"], labels, start=y["start"]), ids


def extra():
    res = {}
    X, y, ids = synth(3000, rows_per_id=5, seed=4)
    for buf in (0, 1):
        for ntime in (None, 40):
            m = SurvivalForestTV(n_estimators=40, resample_unit="block", block_length=2.0, oob_buffer=buf,
                                 ntime=ntime, oob_score=True, random_state=2).fit(X, y, ids)
            key = f"block_{buf}_{ntime}"
            res[key + "_chf"] = m.predict_cumulative_hazard(X[:300])
            res[key + "_oob"] = m.oob_prediction_
            res[key + "_oob_n"] = m.oob_n_trees_
    X, y, ids = _cr_case()
    for agg in ("hazard", "cif"):
        for ntime in (None, 40):
            m = CompetingRisksForestTV(n_estimators=40, aggregate=agg, ntime=ntime, oob_score=True,
                                       random_state=5).fit(X, y, ids)
            key = f"cr_{agg}_{ntime}"
            res[key + "_cif"] = m.predict_cumulative_incidence(X[:300])
            res[key + "_cumhaz"] = m.predict_cumulative_hazard(X[:300])
            res[key + "_oob"] = m.oob_prediction_
            res[key + "_apply"] = m.apply(X[:300])
    return res


def main(out):
    tmp = out + ".s9.npz"
    s9_identity.main(tmp)
    res = dict(np.load(tmp))
    res.update(extra())
    np.savez(out, **res)
    print("wrote", out, len(res), "arrays")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(s9_identity.compare(sys.argv[2], sys.argv[3]) > 0)
    main(sys.argv[1])
