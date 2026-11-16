# Validation scope

The benchmark tables contain the frozen R1 ensemble, scored after averaging the three predictions. The archived evaluator checked 19 distinct datasets and 71 category entries. Its largest difference from the unrounded table values was 2.85 × 10⁻¹⁴ percentage points (rounded upward). The JSON files in this directory retain that benchmark's verification and latency measurements.

The distribution uses six readout parameter sets exported from 18 epoch-15 checkpoints. The export was checked tensor by tensor. Two RGB inputs for each training source were passed through both the reference implementation and the compact reader implementation on an RTX 3090. Image probabilities and all heatmap pixels agreed exactly on these inputs.

The portable evaluator was separately tested on the original HIS and CVC-300 member-score exports using verified images and masks. HIS reproduced Image AUROC/AP of 61.582757/59.049005. CVC-300 reproduced Pixel AUROC/AP/AUPRO of 96.683925/61.412821/94.105191. A two-image CVC-300 subset also exercised the raw-image inference and evaluation path. These checks used a configured Python 3.11 CUDA environment and freshly compiled kernels.
