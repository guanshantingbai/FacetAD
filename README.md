# R1 for zero-shot anomaly detection

R1 scores an image and localizes defects with one frozen CLIP ViT-L/14@336px visual forward pass. Three readouts share the same CLS and layer-24 patch tokens. Each member combines static and residual patch scores with an image-conditioned abnormal text prototype. The final image score averages the three probabilities. The final heatmap averages the members' color-guided, full-resolution probability maps.

| Evaluation | Image AUROC / AP (%) | Pixel AUROC / AP (%) |
|---|---:|---:|
| MVTec AD | **93.897 / 97.216** | **91.513 / 40.118** |
| VisA | **83.747 / 86.714** | **95.381 / 25.079** |
| Original benchmark, 11 eligible domains per task | **86.300 / 89.062** | **93.781 / 46.709** |

[Image-level comparisons](results/image_comparison.md) · [Pixel-level comparisons](results/pixel_comparison.md) · [Per-dataset results](results/r1_metrics.csv) · [Per-class results](results/r1_per_class.csv)

## Installation

Use Linux, Python 3.11, an NVIDIA GPU and a CUDA toolkit with `nvcc`. The reference environment uses an RTX 3090, CUDA 12.8 and PyTorch 2.11.0. Run these commands from the extracted repository root.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt

git clone https://github.com/zqhang/AnomalyCLIP.git third_party/AnomalyCLIP
git -C third_party/AnomalyCLIP checkout --detach 3911738c0867544f545a076ad78f3f11d9ecbfdf
make kernels CUDA_ARCH=sm_86
```

For another GPU, set `CUDA_ARCH` to its compute capability. `NVCC=/path/to/nvcc` selects the compiler. Both the Gaussian filter and the exact pixel-metric kernel are built locally.

## Weights and first prediction

Download the [R1 reader weights (61 MiB)](https://github.com/guanshantingbai/readout-transfer-pilot/releases/download/v1.0.0/r1-readers.tar.gz). The combined distribution already contains the two files under `weights/`. For a source-only checkout, extract the companion archive at the repository root.

```sh
echo 'c97308b2af55e41736a0ab167a206d709cd04df96e5760e25af4cf66770eab2e  r1-readers.tar.gz' | sha256sum -c -
tar -xzf r1-readers.tar.gz
```

Download the original CLIP visual checkpoint from the URL used by AnomalyCLIP. R1 verifies the SHA-256 of the visual checkpoint and of the selected reader file before loading.

```sh
mkdir -p weights
curl --fail --location --output weights/ViT-L-14-336px.pt \
  https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt
export R1_CLIP_WEIGHTS="$PWD/weights/ViT-L-14-336px.pt"
python release/infer.py --source visa --output prediction.npz examples/texture.png
```

`--source` names the **training** dataset. For a VisA target, select `mvtec`. For every other benchmark target, select `visa`. The example is a synthetic texture for checking the installation. The output NPZ has `files` in input order, `image` with shape `[N]`, and `pixel` with shape `[N,518,518]`. Higher probabilities indicate greater anomaly evidence. The heatmap uses the resized input coordinates.

## Evaluation

Create a UTF-8 JSON Lines manifest with one record per image. Paths are relative to the manifest, or may be absolute. An empty mask list represents a known empty foreground. For image-only targets (OCT17, BrainMR1 [V] and HIS), the mask field can be omitted.

```json
{"image":"images/001.png","category":"bottle","label":0,"masks":[]}
{"image":"images/002.png","category":"bottle","label":1,"masks":["masks/002.png"]}
```

```sh
python release/evaluate.py mvtec --manifest data/mvtec.jsonl --output mvtec.json
```

The evaluator computes Image AUROC, Image AP, TPR at 5% FPR, Pixel AUROC, Pixel AP and AUPRO. JSON records per-category fractions and the mean across eligible categories in percent. Undefined metrics are JSON `null`. Dense evaluation stores one category's full-resolution maps in a temporary file; set `TMPDIR` to choose its location.

For a bit-level comparison with the saved benchmark, an optional replay accepts the original three member-score files (`111.npz`, `222.npz`, `333.npz`). It verifies the files, manifest order, image bytes, labels and effective masks before computing metrics. The public code and reader archive support raw-image evaluation without these exports. Raw encoding and historical cached features can differ slightly due to batching and floating-point rounding.

```sh
python release/evaluate.py mvtec --manifest data/mvtec.jsonl \
  --member-scores data/mvtec_scores --output mvtec-replay.json
python release/verify_tables.py mvtec-replay.json
```

## Protocol and coverage

Members use seeds 111, 222 and 333 and the fixed epoch-15 checkpoints. Supervision comes from normal and abnormal images with source masks in the official test split of the **other** dataset (2,162 VisA images or 1,725 MVTec AD images). Target labels are used for evaluation. Images are resized to 518×518. The frozen encoder uses DPAM layer 20 and a single 37×37 local grid at layer 24. Fusion and Lab bandwidth are fixed on the training source.

The original 14-target protocol has 11 eligible classification datasets and 11 eligible localization datasets. Adding RSDD, Liver CT, Retina OCT, CVC-ColonDB and CVC-300 brings coverage to 14 classification and 16 localization datasets. Brain MRI [M] and Brain_AD share all 3,715 test images, labels and evaluation masks and count once. Published cross-method comparisons use heterogeneous protocols; the R1 CSV files contain the unrounded measurements.

For non-default installations, `R1_CLIP_CODE` points to the upstream checkout, `R1_CLIP_WEIGHTS` to the CLIP checkpoint, `R1_WEIGHTS` to the directory containing `visa.pt` and `mvtec.pt`, and `R1_KERNELS` to the directory containing the compiled libraries. See [third-party software and assets](THIRD_PARTY.md).
