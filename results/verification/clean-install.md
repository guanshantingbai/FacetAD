# Clean installation and full benchmark reproduction

Tested source commit: `d406cfdebac42c7f679ec38524bd70c84d2f505a`. A fresh Python 3.11.15 venv had no preinstalled torch or NumPy and no access to system or user site-packages. Both pip installations in the README completed and `pip check` passed. The run used PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, a single RTX 3090, the pinned AnomalyCLIP checkout and two locally built CUDA 12.8 kernels (sm_86). The upstream CLIP checkpoint was downloaded and hash-checked.

The example returned an image probability of 0.4373083124283301 and a 1×518×518 heatmap. Both matched the reference example element for element.

| Evaluation path | Datasets / category entries | Original Image AUROC / AP | Original Pixel AUROC / AP | Expanded Image AUROC / AP | Expanded Pixel AUROC / AP |
|---|---:|---:|---:|---:|---:|
| Frozen member replay | 19 / 71 | 86.300 / 89.062 | 93.781 / 46.709 | 83.776 / 85.410 | 93.908 / 43.423 |
| RGB images and released reader weights | 19 / 71 | 86.299 / 89.062 | 93.781 / 46.709 | 83.776 / 85.410 | 93.908 / 43.423 |

The two runs used the same 28,733 official query images. Reported values are percentages and aggregate by eligible category and dataset. The original means cover 11 eligible domains per task. The extended means cover 14 classification and 16 localization domains. The RGB run encoded each image with the released model and did not read saved member predictions or feature caches.

The replay regenerated the two Markdown comparison tables byte for byte. Its maximum absolute difference across dataset-level metrics was 2.8422e-14 percentage points. At the published one-decimal precision, the RGB run changed 0 image-table rows and 0 pixel-table rows. Its largest unrounded dataset-level difference was 0.002893519 percentage points for dtd image_auc. External reference columns were copied from the published tables.

At three decimal places, the original-domain mean Image AUROC is 86.299 from raw images and 86.300 from member replay (unrounded difference −0.000300 percentage points). This is the only displayed aggregate that changed. The raw-image pixel comparison Markdown is byte-identical to the published file. The complete clean installation, sample and both benchmark runs took about 102 minutes.
