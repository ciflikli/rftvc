# Repeated survival-forest fit matrix

Python 3.11.5; macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs; 250 trees; 4 threads; 5 measured runs and 1 discarded warm-up run per cell.

All repetitions use the same generated rows, held-out subjects, and model seed; spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. Peak RSS, where available, covers the whole process. Leaf rules and split search differ.

Static arms predict mortality; competing-risk arms score cause-1 CIF at the training median follow-up. The R competing-risk prediction computes its native full CIF grid before extracting that horizon, while Python arms request one horizon. Interpret those prediction times separately. Harrell C (static) and Wolbers C (competing) are computed on the held-out subjects.

| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |
|---|---:|---|---:|---:|---:|---:|---:|
| static | 1600/400 | rftvc 0.3.0 | 0.1314 [0.1304, 0.1317] | 0.0029 [0.0029, 0.0029] | 0.196 | 0.671 | 5/5 |

[Raw JSONL](mortality-fastpath-250t-2026-09-30.jsonl) retains each repetition and full configuration.
