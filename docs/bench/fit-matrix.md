# Fit-time matrix, 2026-09-29

Runner: `bench/fit_matrix.py` and `bench/fit_matrix.R`. Raw results:
[`fit-matrix-2026-09-29.jsonl`](fit-matrix-2026-09-29.jsonl) for Python
and [`fit-matrix-r-2026-09-29.jsonl`](fit-matrix-r-2026-09-29.jsonl) for R.

Run from the repository root:

```console
.venv/bin/python -m bench.fit_matrix --trees 25 --threads 4 --output docs/bench/fit-matrix-2026-09-29.jsonl
```

The runner writes one CSV per case and fits each arm in a fresh process using
the same rows. Fit timing excludes data generation and CSV loading. The
deterministic held-out set is every fifth subject, and Harrell's C is computed
on its first 1,000 rows using each library's standard mortality-style `predict`
output. The other subjects are used for training. Prediction timing is for
those 1,000 held-out rows. Peak RSS is the whole process, including runtime,
imports and input arrays, so it is not the model's incremental memory.

Settings: 25 trees, four threads, `sqrt(p)` candidate features, replacement
sampling of 63.2% of training subjects, and nominal leaf size 15. `rftvc` uses
`min_events_leaf=1`, exact event-time grid, and `max_bins=255` for numerical
features. scikit-survival uses `low_memory=True`, which retains `predict` but
does not retain survival curves. Leaf minimums and candidate thresholds are
not semantically identical across implementations; the benchmark measures
practical configurations, not an identical algorithm. One timed fit per cell
does not support claims about small differences. Use the existing
`parity-suite.md` for repeated-fold accuracy checks on real datasets.

Machine: macOS 26.6.2 arm64, Python 3.11.5, `rftvc` 0.2.0,
scikit-survival 0.28.0. Numbers are seconds; C is held-out concordance.
This first matrix was measured before the missing-value split optimization
below, so its missing-value fit time is the baseline rather than current code.

| Case | Rows / features | Event rate | `rftvc` fit / predict / C | scikit-survival fit / predict / C |
|---|---:|---:|---:|---:|
| Static | 10,000 / 10 | 0.403 | 0.138 / 0.058 / 0.669 | 13.900 / 0.014 / 0.667 |
| Rare event | 10,000 / 10 | 0.095 | 0.053 / 0.008 / 0.623 | 14.964 / 0.014 / 0.643 |
| Wide | 5,000 / 100 | 0.406 | 0.209 / 0.030 / 0.631 | 11.011 / 0.012 / 0.656 |
| Tied times | 10,000 / 10 | 0.403 | 0.044 / 0.001 / 0.680 | 1.778 / 0.014 / 0.679 |
| 20% feature NaN | 10,000 / 10 | 0.403 | 0.482 / 0.059 / 0.661 | 21.912 / 0.014 / 0.650 |
| TVC, 5 rows/id | 10,000 / 10 | 0.410 | 0.032 / 0.011 / — | — |

After installing `ranger` 0.18.0 and `randomForestSRC` 3.9.0, the R arms ran
on the same generated cases with the same training and held-out subject ids.
Each R fit used the exact event-time grid; `randomForestSRC` used `nsplit=0`
and disabled its internal performance calculation. As above, split-search and
leaf-size semantics still differ between libraries.

| Case | `ranger` fit / predict / C | `randomForestSRC` fit / predict / C |
|---|---:|---:|
| Static | 23.371 / 0.197 / 0.655 | 22.314 / 0.764 / 0.648 |
| Rare event | 8.835 / 0.039 / 0.603 | 6.587 / 0.187 / 0.621 |
| Wide | 14.338 / 0.122 / 0.635 | 12.851 / 0.418 / 0.651 |
| Tied times | 2.564 / 0.006 / 0.671 | 0.570 / 0.017 / 0.665 |

The TVC case is `rftvc` only because it fits counting-process rows; these
comparison arms fit one outcome per subject. The R arms skip the missing-value
case until an identical missingness protocol is chosen.

## Profile priorities

1. The 20% NaN case took 3.5 times as long as the complete static case in
   `rftvc`. A controlled [missingness sweep before the change](fit-missing-sweep-4t.jsonl)
   showed the penalty already at 1% NaN: 0.532 seconds versus 0.142 seconds
   with no missing values. The old missing-value split path rescanned every
   resampling unit for every threshold. Four bin-prefix histograms now replace
   those scans while preserving mixed-unit membership. On the same cases,
   [after the change](fit-missing-sweep-4t-after.jsonl), 1% NaN took 0.202
   seconds and 20% took 0.193 seconds (from 0.492). The one-thread runs
   ([before](fit-missing-sweep-1t.jsonl), [after](fit-missing-sweep-1t-after.jsonl))
   show the same improvement. Held-out C was unchanged to the reported digits
   on all four cases; Rust missing-split oracles and Python missing-data tests
   pass. These are single fits, so treat the exact ratios as indicative.
2. Exact-grid `predict()` over 1,000 held-out rows is slower than
   scikit-survival's reduced-memory `predict()` on the static and wide cases.
   Horizon-specific `predict_risk()` is much cheaper in the previous S6
   benchmark. A direct mortality accumulation path is worth profiling before
   changing prediction behavior.
3. The rare-event and wide cases have C differences of about 0.02 in opposite
   directions. One held-out set is insufficient to assess a quality tradeoff;
   use repeated folds or known-truth simulations before changing defaults.

The earlier [S6 performance benchmark](s6-perf.md) covers 100 trees up to one
million rows; the [S6 profile](../scratch/perf.md) identifies score and
bin-by-bin split-search costs on complete data. This matrix expands workload
coverage and led to the missing-value split optimization above.
