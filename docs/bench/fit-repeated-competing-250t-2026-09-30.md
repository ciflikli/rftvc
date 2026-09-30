# Repeated survival-forest fit matrix

Python 3.11.5; macOS-26.6.2-arm64-arm-64bit; 10 logical CPUs; 250 trees; 4 threads; 5 measured runs and 1 discarded warm-up run per cell.

All repetitions use the same generated rows, held-out subjects, and model seed; spread measures runtime variation, not accuracy uncertainty. Each run is a fresh process. Fit timing excludes imports and CSV loading; prediction timing covers up to 1,000 subjects. Peak RSS, where available, covers the whole process. Leaf rules and split search differ.

Static arms predict mortality; competing-risk arms score cause-1 CIF at the training median follow-up. The R competing-risk prediction computes its native full CIF grid before extracting that horizon, while Python arms request one horizon. Interpret those prediction times separately. Harrell C (static) and Wolbers C (competing) are computed on the held-out subjects.

| Case | Subjects (train/test) | Arm (version) | Fit s median [min, max] | Predict s median [min, max] | Peak RSS GB | C | Runs |
|---|---:|---|---:|---:|---:|---:|---:|
| competing | 480/120 | comprisk 0.8.0 | 0.2787 [0.2604, 0.2804] | 0.0065 [0.0063, 0.0067] | 0.307 | 0.606 | 5/5 |
| competing | 480/120 | rfsrc 3.9.0 | 5.7670 [5.6980, 5.8600] | 0.0220 [0.0220, 0.0230] | — | 0.635 | 5/5 |
| competing | 480/120 | rftvc 0.3.0 | 0.0326 [0.0322, 0.0336] | 0.0034 [0.0029, 0.0040] | 0.192 | 0.622 | 5/5 |
| competing_rare | 480/120 | comprisk 0.8.0 | 0.2638 [0.2520, 0.2835] | 0.0053 [0.0053, 0.0054] | 0.295 | 0.535 | 5/5 |
| competing_rare | 480/120 | rfsrc 3.9.0 | 1.8020 [1.7680, 1.8890] | 0.0150 [0.0140, 0.0160] | — | 0.655 | 5/5 |
| competing_rare | 480/120 | rftvc 0.3.0 | 0.0226 [0.0225, 0.0232] | 0.0028 [0.0027, 0.0045] | 0.191 | 0.623 | 5/5 |

[Raw JSONL](fit-repeated-competing-250t-2026-09-30.jsonl) retains each repetition and full configuration.
