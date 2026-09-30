# Repeated survival-forest fit matrix

Python 3.11.5; macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs; 25 trees; 4 threads; 5 measured runs and 1 discarded warm-up run per cell.

All repetitions use the same generated rows, held-out subjects, and model seed; spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. Peak RSS, where available, covers the whole process. Leaf rules and split search differ.

Static arms predict mortality; competing-risk arms score cause-1 CIF at the training median follow-up. The R competing-risk prediction computes its native full CIF grid before extracting that horizon, while Python arms request one horizon. Interpret those prediction times separately. Harrell C (static) and Wolbers C (competing) are computed on the held-out subjects.

| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |
|---|---:|---|---:|---:|---:|---:|---:|
| static | 2400/600 | ranger 0.18.0 | 0.9900 [0.9750, 1.0700] | 0.0330 [0.0310, 0.0340] | — | 0.647 | 5/5 |
| static | 2400/600 | rfsrc 3.9.0 | 1.2620 [1.2380, 1.2970] | 0.1080 [0.1070, 0.1170] | — | 0.651 | 5/5 |
| static | 2400/600 | rftvc 0.3.0 | 0.0309 [0.0299, 0.0318] | 0.0117 [0.0115, 0.0122] | 0.202 | 0.666 | 5/5 |
| static | 2400/600 | sksurv 0.28.0 | 0.8919 [0.8692, 0.9100] | 0.0139 [0.0117, 0.0139] | 0.209 | 0.650 | 5/5 |
| static | 2400/600 | sksurv_curve 0.28.0 | 0.9134 [0.8917, 0.9380] | 0.0258 [0.0236, 0.0382] | 0.423 | 0.650 | 5/5 |

[Raw JSONL](fit-repeated-static-2026-09-30.jsonl) retains each repetition and full configuration.
