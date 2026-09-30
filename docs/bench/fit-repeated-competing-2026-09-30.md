# Repeated survival-forest fit matrix

Python 3.11.5; macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs; 25 trees; 4 threads; 5 measured runs and 1 discarded warm-up run per cell.

All repetitions use the same generated rows, held-out subjects, and model seed; spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. Peak RSS, where available, covers the whole process. Leaf rules and split search differ.

Static arms predict mortality; competing-risk arms score cause-1 CIF at the training median follow-up. The R competing-risk prediction computes its native full CIF grid before extracting that horizon, while Python arms request one horizon. Interpret those prediction times separately. Harrell C (static) and Wolbers C (competing) are computed on the held-out subjects.

| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |
|---|---:|---|---:|---:|---:|---:|---:|
| competing | 800/200 | comprisk 0.8.0 | 0.1552 [0.1500, 0.1560] | 0.0029 [0.0027, 0.0031] | 0.288 | 0.707 | 5/5 |
| competing | 800/200 | rfsrc 3.9.0 | 2.9820 [2.8960, 3.0290] | 0.0250 [0.0230, 0.0260] | — | 0.668 | 5/5 |
| competing | 800/200 | rftvc 0.3.0 | 0.0111 [0.0109, 0.0112] | 0.0013 [0.0011, 0.0015] | 0.192 | 0.693 | 5/5 |
| competing_rare | 800/200 | comprisk 0.8.0 | 0.1540 [0.1508, 0.1554] | 0.0028 [0.0026, 0.0028] | 0.286 | 0.726 | 5/5 |
| competing_rare | 800/200 | rfsrc 3.9.0 | 0.6120 [0.6070, 0.6150] | 0.0130 [0.0120, 0.0140] | — | 0.642 | 5/5 |
| competing_rare | 800/200 | rftvc 0.3.0 | 0.0079 [0.0078, 0.0080] | 0.0012 [0.0008, 0.0017] | 0.192 | 0.662 | 5/5 |

[Raw JSONL](fit-repeated-competing-2026-09-30.jsonl) retains each repetition and full configuration.
