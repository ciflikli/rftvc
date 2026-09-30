# Direct mortality prediction at 250 and 1,000 trees

Scikit-survival's [`low_memory=True`](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html)
stores a scalar mortality prediction in each tree node instead of a full
time-indexed hazard and survival curve. Its
forest can then predict scalar risk but cannot predict cumulative hazard or
survival functions. `rftvc` already stores sparse event indices and cumulative
hazards per leaf, so it can keep its curve APIs and compute mortality directly
from those leaf profiles. Each hazard jump is weighted by the number of
requested evaluation times at or after that event, then averaged across trees.
This avoids allocating a subjects-by-times matrix for scalar prediction.

The existing survival aggregation rule is nonlinear; that branch computes a
curve per subject and sums it without allocating a batch-sized curve matrix.
Only the hazard aggregation branch uses the sparse weighted-jump shortcut.

On the fixed static workload of 1,600 training and 400 held-out subjects,
250 or 1,000 trees and four threads, each cell has one discarded warm-up and five
measured runs in fresh processes. All runs use the same generated data and
model seed. The tables compare the pre-change benchmarks at
[250 trees](fit-repeated-static-250t-2026-09-30.md) and
[1,000 trees](fit-repeated-static-1000t-2026-09-30.md) with the post-change
benchmarks at [250 trees](mortality-fastpath-250t-2026-09-30.md) and
[1,000 trees](mortality-fastpath-1000t-2026-09-30.md), all on the same machine.

| Trees | Arm | Median prediction time for 400 subjects | Median fit time | Peak process RSS | Held-out Harrell C |
|---:|---|---:|---:|---:|---:|
| 250 | `rftvc` before | 0.0493 s | 0.1330 s | 0.199 GB | 0.671 |
| 250 | `rftvc` after | 0.0029 s | 0.1314 s | 0.196 GB | 0.671 |
| 250 | scikit-survival, low memory | 0.0133 s | 3.1631 s | 0.209 GB | 0.670 |
| 1,000 | `rftvc` before | 0.1975 s | 0.5065 s | 0.206 GB | 0.670 |
| 1,000 | `rftvc` after | 0.0114 s | 0.5116 s | 0.204 GB | 0.670 |
| 1,000 | scikit-survival, low memory | 0.0518 s | 12.5574 s | 0.217 GB | 0.669 |
| 1,000 | scikit-survival, curve capable | 0.2717 s | 12.7013 s | 3.435 GB | 0.669 |

The observed `rftvc` prediction reduction is about 17× at each tree count;
its post-change scalar prediction is about 4.5× faster than scikit-survival
low memory in these workloads. These comparisons are specific to this dataset, machine, thread
count and package settings. Peak RSS includes the entire process, not just
the model. The libraries have different split searches and leaf rules, and the
five runs measure runtime variation rather than accuracy uncertainty.

The direct result matches the sum of the full cumulative hazard curve within
`1e-12` in tests covering the complete grid, sparse time subsets, reversed
time subsets, coarsened time grids and both aggregation modes. The Python
suite and Rust suite pass with the new path.
