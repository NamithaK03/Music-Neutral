# Controlled synthetic follow-up results

Mean (sample SD) across ten seeds; 1,000 interactions per seed. All variants share scenarios and all four genre actions.

| Noise | Preference SD | Fusion | Policy | NDCG@10 | Skip rate | Fallback |
|---|---|---|---|---|---|---|
| 0.5 | 0.0 | fixed | deterministic | 0.666 (0.009) | 0.568 (0.016) | 0.907 (0.008) |
| 0.5 | 0.0 | fixed | LinUCB | 0.638 (0.010) | 0.581 (0.015) | 0.907 (0.008) |
| 0.5 | 0.0 | rescaled | deterministic | 0.975 (0.004) | 0.347 (0.022) | 0.067 (0.008) |
| 0.5 | 0.0 | rescaled | LinUCB | 0.814 (0.065) | 0.430 (0.046) | 0.067 (0.008) |
| 0.5 | 0.0 | text | deterministic | 0.911 (0.007) | 0.390 (0.016) | 0.240 (0.020) |
| 0.5 | 0.0 | text | LinUCB | 0.775 (0.068) | 0.458 (0.036) | 0.240 (0.020) |
| 0.5 | 0.5 | fixed | deterministic | 0.711 (0.156) | 0.529 (0.145) | 0.907 (0.008) |
| 0.5 | 0.5 | fixed | LinUCB | 0.792 (0.137) | 0.454 (0.174) | 0.907 (0.008) |
| 0.5 | 0.5 | rescaled | deterministic | 0.929 (0.058) | 0.372 (0.132) | 0.067 (0.008) |
| 0.5 | 0.5 | rescaled | LinUCB | 0.913 (0.075) | 0.351 (0.157) | 0.067 (0.008) |
| 0.5 | 0.5 | text | deterministic | 0.885 (0.071) | 0.400 (0.126) | 0.240 (0.020) |
| 0.5 | 0.5 | text | LinUCB | 0.902 (0.078) | 0.365 (0.162) | 0.240 (0.020) |
| 1.5 | 0.0 | fixed | deterministic | 0.720 (0.011) | 0.526 (0.012) | 0.740 (0.017) |
| 1.5 | 0.0 | fixed | LinUCB | 0.677 (0.019) | 0.545 (0.013) | 0.740 (0.017) |
| 1.5 | 0.0 | rescaled | deterministic | 0.851 (0.008) | 0.436 (0.015) | 0.220 (0.008) |
| 1.5 | 0.0 | rescaled | LinUCB | 0.735 (0.039) | 0.485 (0.031) | 0.220 (0.008) |
| 1.5 | 0.0 | text | deterministic | 0.789 (0.012) | 0.480 (0.012) | 0.377 (0.015) |
| 1.5 | 0.0 | text | LinUCB | 0.686 (0.029) | 0.531 (0.022) | 0.377 (0.015) |
| 1.5 | 0.5 | fixed | deterministic | 0.749 (0.132) | 0.500 (0.140) | 0.740 (0.017) |
| 1.5 | 0.5 | fixed | LinUCB | 0.849 (0.122) | 0.412 (0.185) | 0.740 (0.017) |
| 1.5 | 0.5 | rescaled | deterministic | 0.833 (0.064) | 0.439 (0.121) | 0.220 (0.008) |
| 1.5 | 0.5 | rescaled | LinUCB | 0.892 (0.080) | 0.374 (0.163) | 0.220 (0.008) |
| 1.5 | 0.5 | text | deterministic | 0.790 (0.080) | 0.468 (0.125) | 0.377 (0.015) |
| 1.5 | 0.5 | text | LinUCB | 0.822 (0.155) | 0.423 (0.211) | 0.377 (0.015) |

These are new simulator measurements, not replications of legacy values. Shared scenarios enable within-generator comparisons; neither human benefit nor algorithmic novelty follows from these results. Graded NDCG uses completion probabilities and is not comparable with the legacy binary-relevance NDCG.

All seed-level metrics and paired differences are retained in `results/controlled_study.json`.
