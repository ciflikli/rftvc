"""Regenerate tests/fixtures/cr_rfsrc_truth.json with randomForestSRC 3.9.0 (needs Rscript + a
scratch ``randomForestSRC`` install -- not a dependency; see docs/plans/cr-rfsrc-research.md for
how to find/rebuild a scratch lib).

Generates a fixed train/test split of ``bench.cr_rfsrc_truth_check``'s static competing-risks DGP
in Python, writes it to CSV, fits ``randomForestSRC::rfsrc(Surv(time, status) ~ x, ...)`` in R
(splitrule "random" -- see cr_rfsrc_truth_check.py's module docstring for why), predicts on the
test set, and writes the test covariates + CIF predictions (interpolated onto the shared ``GRID``)
to JSON. The fixture stores the test set's own ``x`` values alongside the predictions so the
pytest-side truth check recomputes ``true_cif`` from the *same* covariates used to generate the
predictions, never from an independently-regenerated test set (Codex plan review finding, applied).

Run from the repo root, with ``RL`` pointing at a scratch lib containing randomForestSRC:
``RL=<path> python tests/fixtures/make_cr_rfsrc_fixture.py``

Uses a fixed seed (777) distinct from cr_rfsrc_truth_check.py's own gate seeds (0-9) and its
out-of-band calibration seeds (10000+), so the fixture's train/test split is independent of both.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bench.cr_rfsrc_truth_check import GRID, N_ESTIMATORS, simulate

OUT = Path(__file__).with_name("cr_rfsrc_truth.json")
SEED = 777
N_TRAIN, N_TEST = 400, 200

R_SCRIPT = """
suppressMessages(library(randomForestSRC, lib.loc = Sys.getenv("RL")))
train <- read.csv("{train_csv}")
test <- read.csv("{test_csv}")
fit <- rfsrc(Surv(time, status) ~ x, data = train, ntree = {ntree}, splitrule = "random", seed = -1)
pr <- predict(fit, newdata = test)
times <- pr$time.interest
grid <- c({grid})
idx <- findInterval(grid, times)
out <- list()
for (k in 1:2) {{
  cif <- pr$cif[, , k]
  vals <- sapply(idx, function(i) if (i == 0) rep(0, nrow(cif)) else cif[, i])
  out[[k]] <- vals
}}
write.csv(out[[1]], "{cif1_csv}", row.names = FALSE)
write.csv(out[[2]], "{cif2_csv}", row.names = FALSE)
"""


def main():
    rng = np.random.default_rng(SEED)
    x_tr, t_tr, s_tr = simulate(N_TRAIN, rng)
    x_te, _, _ = simulate(N_TEST, rng)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_csv, test_csv = tmp / "train.csv", tmp / "test.csv"
        cif1_csv, cif2_csv = tmp / "cif1.csv", tmp / "cif2.csv"
        pd.DataFrame({"x": x_tr, "time": t_tr, "status": s_tr}).to_csv(train_csv, index=False)
        pd.DataFrame({"x": x_te}).to_csv(test_csv, index=False)
        r = R_SCRIPT.format(
            train_csv=train_csv,
            test_csv=test_csv,
            ntree=N_ESTIMATORS,
            grid=",".join(f"{g:.10g}" for g in GRID),
            cif1_csv=cif1_csv,
            cif2_csv=cif2_csv,
        )
        env = {**os.environ}
        assert env.get("RL"), "RL must point at a scratch lib containing randomForestSRC"
        subprocess.run(["Rscript", "-e", r], check=True, env=env)
        cif1 = pd.read_csv(cif1_csv).to_numpy()
        cif2 = pd.read_csv(cif2_csv).to_numpy()

    cif = np.stack([cif1, cif2], axis=1)  # (n_test, 2, len(GRID))
    assert cif.shape == (N_TEST, 2, GRID.size)
    out = {"x_test": x_te.tolist(), "grid": GRID.tolist(), "cif": cif.tolist(), "seed": SEED, "ntree": N_ESTIMATORS}
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({cif.shape})")


if __name__ == "__main__":
    main()
