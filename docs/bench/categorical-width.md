# Categorical width check for 0.3.0

Run `bench/categorical_width.py` with its defaults (2,000 training subjects,
1,000 held-out subjects, 50 trees, seeds 1–3, one thread). The right-censored
synthetic outcome depends on one numeric predictor and a noncontiguous half of
the category levels; seven numeric features are noise. The benchmark includes
encoding time in `fit_seconds` and measures held-out counting-process C.

| Levels | Encoded features | `max_features` | Fit seconds, mean | Held-out C, mean |
|---:|---:|:---|---:|---:|
| 3 | 11 | `sqrt` | 0.296 | 0.684 |
| 3 | 11 | all | 0.831 | 0.683 |
| 8 | 16 | `sqrt` | 0.313 | 0.687 |
| 8 | 16 | all | 0.975 | 0.680 |
| 32 | 40 | `sqrt` | 0.327 | 0.676 |
| 32 | 40 | all | 1.372 | 0.670 |

Raw measurements: [categorical-width.jsonl](categorical-width.jsonl). This
single synthetic setting does not establish superiority over native partitions
or grouped feature sampling. It shows no performance or held-out quality
failure that warrants either change for 0.3.0. Revisit on real mixed-type
datasets, especially when a category needs a multi-level split near the root.
