# Repeated survival-forest fit matrix

Python 3.11.5; macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs; 250 trees; 4 threads; 5 measured runs and 1 discarded warm-up run per cell.

All repetitions use the same generated rows, held-out subjects, and model seed; spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. Peak RSS, where available, covers the whole process. Leaf rules and split search differ.

Static arms predict mortality; competing-risk arms score cause-1 CIF at the training median follow-up. The R competing-risk prediction computes its native full CIF grid before extracting that horizon, while Python arms request one horizon. Interpret those prediction times separately. Harrell C (static) and Wolbers C (competing) are computed on the held-out subjects.

| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |
|---|---:|---|---:|---:|---:|---:|---:|
| static | 1600/400 | ranger 0.18.0 | 2.1440 [2.0190, 2.5950] | 0.1560 [0.1550, 0.1720] | — | 0.666 | 5/5 |
| static | 1600/400 | rfsrc 3.9.0 | 4.5130 [4.4640, 4.5640] | 0.2060 [0.2030, 0.2090] | — | 0.669 | 5/5 |
| static | 1600/400 | rftvc 0.3.0 | 0.1330 [0.1307, 0.1371] | 0.0493 [0.0477, 0.0503] | 0.199 | 0.671 | 5/5 |
| static | 1600/400 | sksurv 0.28.0 | 3.1631 [3.1242, 3.1931] | 0.0133 [0.0128, 0.0266] | 0.209 | 0.670 | 5/5 |
| static | 1600/400 | sksurv_curve 0.28.0 | 3.2625 [3.1827, 3.3996] | 0.0867 [0.0713, 0.1003] | 1.089 | 0.670 | 5/5 |

[Raw JSONL](fit-repeated-static-250t-2026-09-30.jsonl) retains each repetition and full configuration.
