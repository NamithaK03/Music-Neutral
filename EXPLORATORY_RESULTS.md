# Exploratory comparisons after inspection of v2 results

These comparisons were not prespecified. Saved runs only; no experimental rerun. All 24 cells, including LinUCB, are reported. Intervals use the existing seed-paired bootstrap (10,000 resamples, RNG 0), pointwise and unadjusted. Full metrics are in `results/controlled_v2_exploratory_contrasts.json`.

|Noise|Preference SD|Fusion|LinUCB|Contrast|Mean NDCG difference|95% interval|Positive seeds|
|---|---|---|---|---|---|---|---|
|low|0.0|text|False|ungated minus relabel|0.085506|[0.081595, 0.089682]|10/10|
|low|0.0|text|False|downweight minus ungated|-0.000398|[-0.000518, -0.000285]|0/10|
|low|0.0|text|True|ungated minus relabel|0.034486|[-0.022544, 0.092233]|6/10|
|low|0.0|text|True|downweight minus ungated|0.014244|[-0.050723, 0.091940]|4/10|
|low|0.0|rescaled|False|ungated minus relabel|0.024961|[0.022732, 0.027232]|10/10|
|low|0.0|rescaled|False|downweight minus ungated|0.000005|[-0.000185, 0.000194]|3/10|
|low|0.0|rescaled|True|ungated minus relabel|0.021309|[-0.028204, 0.065005]|8/10|
|low|0.0|rescaled|True|downweight minus ungated|-0.038660|[-0.092498, 0.021634]|3/10|
|low|0.5|text|False|ungated minus relabel|0.060160|[0.038767, 0.081258]|10/10|
|low|0.5|text|False|downweight minus ungated|-0.000337|[-0.000482, -0.000202]|0/10|
|low|0.5|text|True|ungated minus relabel|0.003805|[-0.032744, 0.031334]|8/10|
|low|0.5|text|True|downweight minus ungated|-0.001732|[-0.016330, 0.015666]|4/10|
|low|0.5|rescaled|False|ungated minus relabel|0.018465|[0.011495, 0.026129]|10/10|
|low|0.5|rescaled|False|downweight minus ungated|0.000047|[-0.000086, 0.000207]|3/10|
|low|0.5|rescaled|True|ungated minus relabel|0.006668|[-0.010090, 0.022302]|8/10|
|low|0.5|rescaled|True|downweight minus ungated|-0.002309|[-0.024371, 0.019275]|5/10|
|high|0.0|text|False|ungated minus relabel|0.038748|[0.033597, 0.043335]|10/10|
|high|0.0|text|False|downweight minus ungated|0.000314|[-0.000193, 0.000765]|7/10|
|high|0.0|text|True|ungated minus relabel|0.041134|[0.012559, 0.070264]|8/10|
|high|0.0|text|True|downweight minus ungated|-0.012121|[-0.034023, 0.012812]|4/10|
|high|0.0|rescaled|False|ungated minus relabel|0.029083|[0.024557, 0.033409]|10/10|
|high|0.0|rescaled|False|downweight minus ungated|0.010173|[0.009068, 0.011350]|10/10|
|high|0.0|rescaled|True|ungated minus relabel|0.034151|[0.001395, 0.068355]|7/10|
|high|0.0|rescaled|True|downweight minus ungated|-0.006218|[-0.026130, 0.014080]|3/10|
|high|0.5|text|False|ungated minus relabel|0.017679|[-0.014838, 0.049220]|7/10|
|high|0.5|text|False|downweight minus ungated|-0.000149|[-0.000939, 0.000534]|5/10|
|high|0.5|text|True|ungated minus relabel|-0.002183|[-0.072272, 0.081303]|4/10|
|high|0.5|text|True|downweight minus ungated|0.051053|[0.001026, 0.108017]|7/10|
|high|0.5|rescaled|False|ungated minus relabel|0.015807|[-0.005653, 0.035893]|7/10|
|high|0.5|rescaled|False|downweight minus ungated|0.008372|[0.007351, 0.009344]|10/10|
|high|0.5|rescaled|True|ungated minus relabel|-0.038550|[-0.075917, -0.007727]|2/10|
|high|0.5|rescaled|True|downweight minus ungated|0.040506|[-0.000145, 0.079667]|9/10|
|asymmetric|0.0|text|False|ungated minus relabel|0.044610|[0.039552, 0.049407]|10/10|
|asymmetric|0.0|text|False|downweight minus ungated|0.000776|[0.000247, 0.001271]|7/10|
|asymmetric|0.0|text|True|ungated minus relabel|0.026530|[0.001353, 0.052928]|7/10|
|asymmetric|0.0|text|True|downweight minus ungated|-0.018366|[-0.042562, 0.006441]|3/10|
|asymmetric|0.0|rescaled|False|ungated minus relabel|0.023764|[0.019740, 0.027236]|10/10|
|asymmetric|0.0|rescaled|False|downweight minus ungated|0.009022|[0.007584, 0.010318]|10/10|
|asymmetric|0.0|rescaled|True|ungated minus relabel|0.020546|[-0.015804, 0.058113]|8/10|
|asymmetric|0.0|rescaled|True|downweight minus ungated|0.013939|[-0.012140, 0.040808]|6/10|
|asymmetric|0.5|text|False|ungated minus relabel|0.022104|[-0.011022, 0.054283]|7/10|
|asymmetric|0.5|text|False|downweight minus ungated|0.000058|[-0.000823, 0.000873]|5/10|
|asymmetric|0.5|text|True|ungated minus relabel|-0.004616|[-0.031294, 0.022955]|5/10|
|asymmetric|0.5|text|True|downweight minus ungated|0.045736|[0.012226, 0.079517]|7/10|
|asymmetric|0.5|rescaled|False|ungated minus relabel|0.013220|[-0.006326, 0.031735]|7/10|
|asymmetric|0.5|rescaled|False|downweight minus ungated|0.007275|[0.005527, 0.009161]|10/10|
|asymmetric|0.5|rescaled|True|ungated minus relabel|0.011638|[-0.006361, 0.029077]|6/10|
|asymmetric|0.5|rescaled|True|downweight minus ungated|-0.006866|[-0.020417, 0.007049]|4/10|
