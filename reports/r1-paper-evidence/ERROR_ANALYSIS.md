# Paired error analysis
Image counts enumerate all comparable anomaly–normal pairs within class (half credit for ties). Class-macro increments reproduce the exact paired AUROC change. The counts sum across classes, so dividing the net count by all pairs can give a different number.
|Dataset|Comparable image pairs|Corrected|Broken|Image AUROC change (pp)|
|---|---:|---:|---:|---:|
|visa|96,200|1,322.0|794.0|+0.581|
|mvtec|39,607|315.0|302.0|-0.050|
|mpdd|8,204|134.0|77.0|+1.045|
|btad|23,429|193.0|161.0|+0.478|
|ksdd2|98,340|992.0|462.0|+0.539|
|dagm|826,384|1,747.0|269.0|+0.278|
|dtd|28,169|247.0|77.0|+0.557|
|oct17|187,500|1,908.0|1,928.0|-0.011|
|brainmri|15,190|24.0|6.0|+0.118|
|brain_ad|1,968,000|30,390.0|20,775.0|+0.489|
|his|996,982|17,386.0|14,371.0|+0.302|

We sampled 100,000 independently drawn foreground/background pairs with replacement per category at the full evaluation resolution (518 × 518), including normal-image background. Methods use identical coordinates. Intervals in this table measure Monte Carlo uncertainty only (delta ± 1.96 SE). The exact official point estimate is supplied wherever the archived ensemble intervention matches. All class-level counts are in the CSV. Standalone H-off retains its native calibration. Final-off retains fusion and color.
|Dataset|Transition|Sample corrected|Sample broken|Mean Δ AUROC ± 1.96 MC SE (pp)|Exact Δ (pp)|
|---|---|---:|---:|---:|---:|
|mvtec|Hoff → H|52,242.0|53,942.0|-0.113 ± 0.042|—|
|mvtec|FinalOff → Final|32,564.0|24,667.0|+0.526 ± 0.031|+0.531|
|mvtec|A → H|50,165.0|55,275.0|-0.341 ± 0.042|—|
|mvtec|H → AH|29,818.0|17,336.0|+0.832 ± 0.028|—|
|mvtec|A → AH|32,829.0|25,457.0|+0.491 ± 0.031|+0.481|
|mvtec|AH → Final|7,404.0|6,654.0|+0.050 ± 0.015|+0.057|
|visa|Hoff → H|19,416.0|41,579.0|-1.847 ± 0.040|—|
|visa|FinalOff → Final|10,775.0|12,700.0|-0.160 ± 0.025|-0.138|
|visa|A → H|18,489.0|41,104.0|-1.885 ± 0.039|—|
|visa|H → AH|28,747.0|7,654.0|+1.758 ± 0.030|—|
|visa|A → AH|10,835.0|12,357.0|-0.127 ± 0.025|-0.108|
|visa|AH → Final|4,892.0|3,847.0|+0.087 ± 0.015|+0.077|
|dagm|Hoff → H|8,641.0|23,499.0|-1.486 ± 0.035|—|
|dagm|FinalOff → Final|3,340.0|7,394.0|-0.405 ± 0.020|-0.404|
|dagm|A → H|14,617.0|21,840.0|-0.722 ± 0.037|—|
|dagm|H → AH|14,924.0|9,681.0|+0.524 ± 0.031|—|
|dagm|A → AH|4,936.0|6,916.0|-0.198 ± 0.021|-0.196|
|dagm|AH → Final|3,545.0|2,344.0|+0.120 ± 0.015|+0.124|

## Does fusion restore the ordering that H broke?

The denominator is the loss of pairwise credit when changing A to H. The numerator credits only increases from H to plain A/H *on those same pairs*, capped at the original loss. The fraction is computed within class and macro-averaged. Raw sums in the next columns weight classes by the number of affected pairs.

|Dataset|A→H credit lost in sample|Recovered by A/H|Class-mean recovery ± 1.96 MC SE|
|---|---:|---:|---:|
|mvtec|55,275.0|29,818.0|57.03% ± 0.82 pp|
|visa|41,104.0|28,747.0|60.26% ± 0.77 pp|
|dagm|21,840.0|14,924.0|68.19% ± 1.13 pp|

Residual evidence supports foreground/background reordering in some domains. The DAGM counterexample is visible in both the retained heatmaps and the formal Pixel AP comparison. Partial recovery gives no guarantee of improving on static A. These ranks do not estimate a detector operating-point false-positive rate. No threshold was fitted.

The signed image effect describes Dynamic on the same A/H midpoint. Corrected and broken pairs coexist. Using a donor and retaining only the orthogonal component are interventions on the image readout. Neither produces a localization map.
