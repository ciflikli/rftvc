# Repeated survival-forest fit matrix

Python 3.11.5; macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs; 1000 trees; 4 threads; 5 measured runs and 1 discarded warm-up run per cell.

All repetitions use the same generated rows, held-out subjects, and model seed; spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. Peak RSS, where available, covers the whole process. Leaf rules and split search differ.

Static arms predict mortality; competing-risk arms score cause-1 CIF at the training median follow-up. The R competing-risk prediction computes its native full CIF grid before extracting that horizon, while Python arms request one horizon. Interpret those prediction times separately. Harrell C (static) and Wolbers C (competing) are computed on the held-out subjects.

| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |
|---|---:|---|---:|---:|---:|---:|---:|
| static | 1600/400 | ranger 0.18.0 | 9.1020 [8.7770, 10.5090] | 0.7250 [0.7150, 0.8030] | — | 0.670 | 5/5 |
| static | 1600/400 | rfsrc 3.9.0 | 16.9940 [16.4640, 17.3990] | 0.7860 [0.7790, 0.8070] | — | 0.670 | 5/5 |
| static | 1600/400 | rftvc 0.3.0 | 0.5065 [0.4991, 0.5340] | 0.1975 [0.1916, 0.1993] | 0.206 | 0.670 | 5/5 |
| static | 1600/400 | sksurv 0.28.0 | 12.5574 [12.4386, 12.7681] | 0.0518 [0.0504, 0.0584] | 0.217 | 0.669 | 5/5 |
| static | 1600/400 | sksurv_curve 0.28.0 | 12.7013 [12.6280, 12.9765] | 0.2717 [0.2576, 0.3545] | 3.435 | 0.669 | 5/5 |

[Raw JSONL](fit-repeated-static-1000t-2026-09-30.jsonl) retains each repetition and full configuration.
