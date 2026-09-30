# Repeated forest benchmark protocol

The [static results](fit-repeated-static-2026-09-30.md) compare rftvc 0.3.0,
scikit-survival 0.28.0 in both memory modes, ranger 0.18.0, and
randomForestSRC 3.9.0. The [competing-risk results](fit-repeated-competing-2026-09-30.md)
compare rftvc, comprisk 0.8.0, and randomForestSRC on two cause-specific
hazard simulations. The raw JSONL files are linked from each result report.
These are development benchmarks, not evidence that one algorithm is
universally faster or more accurate.

The [250/1,000-tree comparison](fit-matrix-high-tree.md) extends the suite to
larger forests, with five measurements per cell and separate static and
competing-risk cohort sizes.

Run from the repository root after installing the optional comparison packages
(`scikit-survival`, `comprisk`, and the R packages `ranger`, `randomForestSRC`):

```console
.venv/bin/python -m bench.fit_matrix --cases static --n-ids 3000 \
  --trees 25 --threads 4 --repeats 5 --warmups 1 \
  --arms rftvc sksurv sksurv_curve ranger rfsrc \
  --output docs/bench/fit-repeated-static-2026-09-30.jsonl \
  --report docs/bench/fit-repeated-static-2026-09-30.md

.venv/bin/python -m bench.fit_matrix --cases competing competing_rare \
  --n-ids 1000 --trees 25 --threads 4 --repeats 5 --warmups 1 \
  --arms rftvc comprisk rfsrc \
  --output docs/bench/fit-repeated-competing-2026-09-30.jsonl \
  --report docs/bench/fit-repeated-competing-2026-09-30.md
```

The runner creates one deterministic dataset per case, with training subjects
selected by `id % 5 != 0` and the others held out. Every warm-up and measured
repetition fits in a new process. The warm-ups warm OS caches but do **not**
remove first-fit compilation cost within a process. Five measured repetitions
give the runtime median and range; C is calculated from the same predictions
and held-out subjects each time, so its repeated value is **not** a sampling
interval. Data generation, CSV parsing, imports, and scoring are outside fit
timing. Peak RSS is whole-process memory, available for Python arms only.

The static comparison uses 25 trees, four threads, `sqrt(p)` candidate
features, replacement sampling of 63.2% of training subjects, and nominal
leaf size 15. Scikit-survival's `low_memory=True` and `False` arms both time
the same mortality prediction; the latter retains the ability to generate
full survival curves. The R packages' split rules, threshold search, and
leaf-size semantics are different, so their matching settings are practical
approximations. Native `predict` outputs are used for static Harrell C.

The competing-risk generator uses two independent exponential event times
with different feature-dependent cause-specific hazards. `competing_rare`
lowers the **overall** observed event rate; it does not make just one cause
rare. All three arms predict cause-1 CIF at the training median follow-up,
scored with rftvc's Wolbers C on the same held-out subjects. The R package
builds its full native CIF grid before selecting that point, while the Python
arms request one point directly; its prediction time is therefore a different
workload. The Python packages also use different splitting and depth defaults.

At 2,400 training subjects, median static fit time was 0.031 s for rftvc,
0.892 s for scikit-survival low-memory, 0.913 s for its curve-capable mode,
0.990 s for ranger, and 1.262 s for randomForestSRC. Held-out C was
0.666, 0.650, 0.650, 0.647, and 0.651 respectively. At 800 training subjects
in the standard competing case, fit medians were 0.011 s for rftvc,
0.155 s for comprisk, and 2.982 s for randomForestSRC; cause-1 C was
0.693, 0.707, and 0.668. These differences in C come from one synthetic
dataset and 200 or 600 test subjects. Use the existing real-data parity
suite and larger, repeated dataset seeds before drawing quality conclusions.

The benchmark remains separate from the package's CI gate. A future release
benchmark should expand to repeated subject splits and known-truth TVC/CR
quality checks. The current TVC case times counting-process fitting but its
event times depend on baseline covariates, so it is not a TVC quality test.
