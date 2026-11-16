# Frozen inference and historical training cost

NVIDIA GeForce RTX 3090, driver 570.133.07, PyTorch 2.11.0+cu128, CUDA 12.8. The eight fixed images and raw samples are recorded in PROFILE.json. One visual encoding per batch, one source package. Synchronized warm-cache medians in milliseconds per **batch**, 3 rounds × 50 observations, after 5 warm-ups per cell/round. Forward covers encoder and score computation, including score transfer and host arithmetic required by the release. End-to-end starts at image-file open, includes full heatmap recovery, and ends at CPU outputs. It excludes model loading. Peak is the maximum allocated GPU memory over these observations. Median stage times need not sum to the median end-to-end time.

|Batch|Detector|Decode, resize, H2D|Encoder|Readout|Heatmap + output|Forward|End-to-end|Peak GiB|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1|static|42.17|114.38|0.70|2.04|115.14|159.31|2.03|
|1|single|41.24|114.38|3.41|8.95|117.82|168.29|2.04|
|1|ensemble|42.05|114.44|9.56|15.44|124.12|182.40|2.06|
|8|static|331.32|871.12|0.90|11.87|872.04|1215.79|4.71|
|8|single|329.48|872.51|14.84|54.61|887.47|1273.12|4.72|
|8|ensemble|330.95|871.53|41.99|81.55|913.66|1326.42|4.74|

Absolute per-stage allocation peaks (GiB, same GPU, separate eight-image pass). Resident model weights are included in each stage.

|Batch|Detector|Input|Encoder|Readout|Postprocess|
|---:|---|---:|---:|---:|---:|
|1|static|1.66|2.03|1.65|1.66|
|1|single|1.67|2.04|1.68|1.69|
|1|ensemble|1.69|2.06|1.71|1.72|
|8|static|1.67|4.65|1.70|1.72|
|8|single|1.68|4.66|1.84|1.86|
|8|ensemble|1.70|4.68|1.86|1.90|


At weight load, the shared CLIP contains 427,944,193 parameters (304,293,888 visual). On the first 518-pixel input, the pinned visual encoder interpolates its 24×24 position table to 37×37 and stores the expanded table. The resident frozen CLIP then contains 428,756,225 values, without learning additional parameters. The source-trained prompt has 43,008 effective parameter slots, or 5,555,200 including the unused projections stored by PromptLearner. H has 6,155 and Dynamic 2,956,800. Per member this is 3,005,963 distinct effective learned parameters. A and H trained separate matching prompt copies, so the sum of active slots over three sequentially costed fits is 3,048,971. Deployment bakes the text into 1,536 values. One complete deployment contains 431,720,716 model values, three members 437,649,698, including the shared encoder exactly once. Relative to static A, the single full readout adds 1,664,209,280 estimated MACs per image. Two further complete readouts add 3,350,915,504 MACs to a single full member. Each three-member source package occupies 35,584,630 bytes for mvtec; exact sizes for both sources and all training checkpoints are itemized in COST.csv. The small fixed scalar calibration metadata is excluded. Inference updates zero parameters.

|Source|Stage|Mean min per seed|Observed peak GiB (max)|Three seeds GPU-hours|
|---|---|---:|---:|---:|
|visa|A|9.60|6.62|0.480|
|visa|H|10.87|6.65|0.544|
|visa|Dynamic|1.52|5.05|0.076|
|visa|A+H+Dynamic|21.99|6.65|1.099|
|mvtec|A|7.10|5.65|0.355|
|mvtec|H|9.15|5.69|0.458|
|mvtec|Dynamic|1.22|4.08|0.061|
|mvtec|A+H+Dynamic|17.48|5.69|0.874|

GPU-hours sum one allocated GPU per logged fit. The complete two-source fit budget is 1.973 GPU-hours. One-time source visual cache construction is itemized in COST.csv. Setup, queueing, independent calibration and complete from-scratch pipeline wall-clock are unavailable. Peak allocations across stages are not additive.

MACs combine an archived operator count of 588,990,935,040 for the shared visual forward with analytic counts for dense readouts, interpolation, neighborhood distance and smoothing. Exponentials, sorting, scalar operations and transfers are excluded. Parameter and disk counts are exact.
