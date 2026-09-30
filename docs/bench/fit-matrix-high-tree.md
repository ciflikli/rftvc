# Forest scaling at 250 and 1,000 trees

Runner: `bench/fit_matrix.py`, 2026-09-30. Each cell has one discarded
fresh-process warm-up and five measured fresh-process fits. Dataset, held-out
subjects, model seed, feature count, and four threads are fixed between the two
tree counts **within each workload**. The static workload has 1,600 training
and 400 test subjects; each competing-risk workload has 480 training and 120
test subjects. The smaller competing-risk size keeps exact-grid
`randomForestSRC` repeatable in a practical runtime. Do not compare fit times
across these workload sizes.

To reproduce, use the commands in the [protocol](fit-matrix-repeated.md),
setting `--n-ids 2000` for static or `--n-ids 600` for competing risks, and
run each with `--trees 250` and `--trees 1000`. Set the matching output and
report filenames shown below. The JSONL metadata retains all run settings.

Detailed reports and raw repetitions:

| Workload | 250 trees | 1,000 trees |
|---|---|---|
| Static | [report](fit-repeated-static-250t-2026-09-30.md) | [report](fit-repeated-static-1000t-2026-09-30.md) |
| Competing and lower event rate | [report](fit-repeated-competing-250t-2026-09-30.md) | [report](fit-repeated-competing-1000t-2026-09-30.md) |

Median fit seconds (five measured runs):

| Workload | Package | 250 trees | 1,000 trees | 1,000 / 250 |
|---|---|---:|---:|---:|
| Static | rftvc | 0.133 | 0.507 | 3.8× |
| Static | scikit-survival, low memory | 3.163 | 12.557 | 4.0× |
| Static | scikit-survival, curve capable | 3.263 | 12.701 | 3.9× |
| Static | ranger | 2.144 | 9.102 | 4.2× |
| Static | randomForestSRC | 4.513 | 16.994 | 3.8× |
| Competing | rftvc | 0.033 | 0.113 | 3.5× |
| Competing | comprisk | 0.279 | 0.707 | 2.5× |
| Competing | randomForestSRC | 5.767 | 22.435 | 3.9× |
| Lower event rate | rftvc | 0.023 | 0.077 | 3.4× |
| Lower event rate | comprisk | 0.264 | 0.688 | 2.6× |
| Lower event rate | randomForestSRC | 1.802 | 7.090 | 3.9× |

The static arms' held-out Harrell C values at 1,000 trees all lie between
0.669 and 0.670 on this one test set. At 1,000 trees, `rftvc` mortality
prediction takes 0.198 s for 400 subjects versus 0.052 s for scikit-survival's
low-memory mortality path. The scikit-survival curve-capable arm takes 0.272 s
and reaches 3.44 GB peak process RSS; its low-memory arm reaches 0.217 GB.
This identifies a prediction bottleneck and a memory-mode tradeoff for the
next optimization pass. The R static prediction paths may compute richer
native objects than the scalar output used for scoring.

The subsequent [direct mortality optimization](mortality-fastpath.md) addresses
that `rftvc` prediction bottleneck. The table above records the pre-change
measurement; it has not been rewritten with post-change times.

For competing risks, the three packages predict cause-1 CIF at the training
median follow-up. `randomForestSRC` computes a full native CIF grid before
selecting that horizon, so its prediction time represents more work than the
Python single-horizon calls. Cause-1 Wolbers C is reported in each detailed
table, but 120 held-out subjects and one simulated dataset are insufficient
for a quality ranking. `competing_rare` lowers the overall observed event
rate, not just one cause's rate.

These are practical package configurations, not identical algorithms: split
rules, depth limits, and leaf-size definitions differ. The five repetitions
measure timing variation on fixed data; they do not create five independent
accuracy estimates. See the [protocol](fit-matrix-repeated.md) for setup and
further limits.
