"""S9 bit-identity check: dump predictions and pickle sizes of the current build.

    python -m bench.s9_identity OUT.npz            # on a main build, then on the S9 build
    python -m bench.s9_identity --compare A.npz B.npz
"""

import pickle
import sys

import numpy as np
from sksurv.column import encode_categorical
from sksurv.datasets import load_gbsg2

from bench.perf_fit import synth
from rftvc import SurvivalForestTV, make_landmark_data, make_survival_y
from tests.fixtures.pbcseq import pbcseq_counting_process


def _gbsg2():
    X, y = load_gbsg2()
    return encode_categorical(X).to_numpy(float), make_survival_y(y["time"], y["cens"]), None


def _landmark():
    df = pbcseq_counting_process()
    d = make_landmark_data(df, horizon=730.5, landmarks=[365.25, 730.5, 1095.75, 1461.0],
                           history_features=["log_bili", "albumin", "protime", ("log_bili", "max")])
    return d.X, d.y, d.ids, "stacked"


def main(out):
    res = {}
    cases = {
        "gbsg2": _gbsg2() + ("counting_process",),
        "tvc": synth(4000, rows_per_id=5, seed=3) + ("counting_process",),
        "landmark": _landmark(),
    }
    for name, (X, y, ids, layout) in cases.items():
        for agg in ["hazard", "survival"]:
            for ntime in [None, 50]:
                key = f"{name}_{agg}_{ntime}"
                m = SurvivalForestTV(n_estimators=40, aggregate=agg, ntime=ntime, oob_score=True,
                                     random_state=1).fit(X, y, ids, layout=layout)
                Xq = X[:300]
                res[key + "_chf"] = m.predict_cumulative_hazard(Xq)
                res[key + "_oob"] = m.oob_prediction_
                res[key + "_apply"] = m.apply(Xq)
                if ids is not None and layout == "counting_process":
                    sel = ids < ids[0] + 50
                    iv = np.array(list(zip(y["start"][sel], y["stop"][sel])),
                                  dtype=[("start", float), ("stop", float)])
                    res[key + "_path"] = m.predict_cumulative_hazard(X[sel], m.event_times_[::7], intervals=iv,
                                                                     ids=ids[sel], extrapolate="locf")
                blob = pickle.dumps(m)
                res[key + "_pickle_bytes"] = np.array(len(blob))
                m2 = pickle.loads(blob)
                assert np.array_equal(m2.predict_cumulative_hazard(Xq), res[key + "_chf"], equal_nan=True)
    np.savez(out, **res)
    print("wrote", out, len(res), "arrays")


def compare(a, b):
    A, B = np.load(a), np.load(b)
    assert set(A.files) == set(B.files), set(A.files) ^ set(B.files)
    bad = 0
    for k in sorted(A.files):
        if k.endswith("_pickle_bytes"):
            print(f"{k}: {int(A[k]):,} -> {int(B[k]):,} ({1 - B[k] / A[k]:.0%} smaller)")
        elif not np.array_equal(A[k], B[k], equal_nan=True):
            bad += 1
            print("DIFFERENT:", k)
    print("bit-identical" if bad == 0 else f"{bad} arrays differ")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]) > 0)
    main(sys.argv[1])
