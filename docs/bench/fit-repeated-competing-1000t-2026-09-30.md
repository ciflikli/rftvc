# Repeated survival-forest fit matrix

Python 3.11.5; macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs; 1000 trees; 4 threads; 5 measured runs and 1 discarded warm-up run per cell.

All repetitions use the same generated rows, held-out subjects, and model seed; spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. Peak RSS, where available, covers the whole process. Leaf rules and split search differ.

Static arms predict mortality; competing-risk arms score cause-1 CIF at the training median follow-up. The R competing-risk prediction computes its native full CIF grid before extracting that horizon, while Python arms request one horizon. Interpret those prediction times separately. Harrell C (static) and Wolbers C (competing) are computed on the held-out subjects.

| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |
|---|---:|---|---:|---:|---:|---:|---:|
| competing | 480/120 | comprisk 0.8.0 | 0.7066 [0.7020, 0.7165] | 0.0192 [0.0188, 0.0200] | 0.383 | 0.605 | 5/5 |
| competing | 480/120 | rfsrc 3.9.0 | 22.4350 [22.3320, 22.5760] | 0.0790 [0.0780, 0.0810] | — | 0.632 | 5/5 |
| competing | 480/120 | rftvc 0.3.0 | 0.1133 [0.1112, 0.1154] | 0.0102 [0.0099, 0.0106] | 0.195 | 0.625 | 5/5 |
| competing_rare | 480/120 | comprisk 0.8.0 | 0.6883 [0.6749, 0.7319] | 0.0152 [0.0147, 0.0158] | 0.335 | 0.549 | 5/5 |
| competing_rare | 480/120 | rfsrc 3.9.0 | 7.0900 [7.0160, 7.2200] | 0.0520 [0.0510, 0.0540] | — | 0.616 | 5/5 |
| competing_rare | 480/120 | rftvc 0.3.0 | 0.0770 [0.0761, 0.1683] | 0.0088 [0.0080, 0.0094] | 0.193 | 0.623 | 5/5 |

[Raw JSONL](fit-repeated-competing-1000t-2026-09-30.jsonl) retains each repetition and full configuration.
