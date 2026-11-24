# FacetAD: reproducing the released model

FacetAD (called **R1** in the original release) detects and localizes anomalies with a frozen CLIP visual encoder, complementary local readout, and image-conditioned prototype correction. Three members share **one visual forward pass**. This repository reproduces **inference and evaluation of the released checkpoints**; it does not currently provide an end-to-end training recipe.

| Evaluation | Image AUROC / AP (%) | Pixel AUROC / AP (%) |
|---|---:|---:|
| MVTec AD | 93.897 / 97.216 | 91.513 / 40.118 |
| VisA | 83.747 / 86.714 | 95.381 / 25.079 |
| Original benchmark, 11 eligible domains per task | 86.300 / 89.062 | 93.781 / 46.709 |

[Image comparisons](results/image_comparison.md) · [Pixel comparisons](results/pixel_comparison.md) · [Exact dataset results](results/r1_metrics.csv) · [Per-class results](results/r1_per_class.csv)

## 1. Install

Use Linux, Python 3.11, an NVIDIA GPU and a CUDA toolkit containing `nvcc`. The reference hardware is an RTX 3090 (24 GB), with CUDA 12.8 and PyTorch 2.11.0. Download and extract the code archive from the anonymous review page, then open a terminal in the extracted repository root. Run all commands below from that directory.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
python -m pip install -r requirements.txt

git clone https://github.com/zqhang/AnomalyCLIP.git third_party/AnomalyCLIP
git -C third_party/AnomalyCLIP checkout --detach 3911738c0867544f545a076ad78f3f11d9ecbfdf
make kernels CUDA_ARCH=sm_86
```

For another GPU, change `CUDA_ARCH` to its compute capability. If needed, pass `NVCC=/path/to/nvcc` to `make`. The two evaluation/inference operators are built from `kernels/`; no server-specific binaries are required.

## 2. Prepare weights and set paths

The reader weights are required and are not included in the source-only archive. Place `r1-readers.tar.gz` in the repository root before running the commands below. For anonymous review, this archive must be distributed through a separate anonymous attachment or download; no author-identifying hosting link is embedded here. A code-only mirror is not sufficient to run inference without these weights.

```bash
echo 'c97308b2af55e41736a0ab167a206d709cd04df96e5760e25af4cf66770eab2e  r1-readers.tar.gz' | sha256sum -c -
tar -xzf r1-readers.tar.gz  # creates weights/visa.pt and weights/mvtec.pt
curl -fL -o weights/ViT-L-14-336px.pt \
  https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt

cp .env.example .env
# Edit the three paths if data/weights/outputs are stored elsewhere.
source .env
python scripts/check_setup.py
```

The three path settings are `FACETAD_DATA_ROOT`, `FACETAD_WEIGHTS_ROOT`, and `FACETAD_OUTPUT_ROOT`. The defaults are `data/`, `weights/`, and `outputs/`. `.env` is a shell file: explicitly source it; Python does not load it automatically. Keep it out of Git. Scientific settings and checkpoint hashes remain fixed in [configs/config.json](configs/config.json).

## 3. Run a first prediction

```bash
python scripts/infer.py --source visa examples/texture.png
# Real input:
python scripts/infer.py --source visa --output "$FACETAD_OUTPUT_ROOT/example.npz" /path/to/image.png
```

The bundled texture is synthetic and checks installation only. Output contains `files`, image probabilities `[N]`, and pixel probability maps `[N,518,518]` in resized input coordinates. Larger values indicate more anomaly evidence.

**`--source` is the training source:** use `mvtec` for a VisA target and `visa` for every other benchmark target. It is not the name of the input dataset.

## 4. Evaluate and check results

Obtain benchmark images/masks from their providers. For the two primary datasets, generate manifests automatically:

```bash
python scripts/prepare_manifest.py mvtec --root "$FACETAD_DATA_ROOT/mvtec"
python scripts/prepare_manifest.py visa --root "$FACETAD_DATA_ROOT/visa"
```

The VisA command reads the official `split_csv/1cls.csv` test split. The commands check the expected 1,725/2,162 images and 15/12 categories and stop on a different split. They do not change or resplit data. Generated paths are absolute; these manifests are for raw-image evaluation, not historical replay ordering.

For other targets, create one JSONL manifest under `$FACETAD_DATA_ROOT/manifests/`, with one line per image; see [the manifest example](configs/manifest.example.jsonl).

```json
{"image":"mvtec/bottle/test/good/000.png","category":"bottle","label":0,"masks":[]}
{"image":"mvtec/bottle/test/broken_large/000.png","category":"bottle","label":1,"masks":["mvtec/bottle/ground_truth/broken_large/000_mask.png"]}
```

With `FACETAD_DATA_ROOT` set, relative image and mask paths are relative to **that root**, not the manifest directory. Absolute paths also work. Without this variable, the older manifest-relative convention is preserved. An empty mask list means known empty foreground; for image-only datasets, omit `masks`. Do not remove anomalous images merely because their masks are empty.

```bash
# Raw-image evaluation: no prediction caches required.
python scripts/evaluate.py mvtec
python scripts/evaluate.py visa
# A custom manifest/output may be supplied explicitly:
python scripts/evaluate.py mvtec --manifest /path/to/mvtec.jsonl --output "$FACETAD_OUTPUT_ROOT/mvtec.json"
# All 19 targets, once their manifests are prepared:
python scripts/evaluate_all.py
```

Outputs contain per-class fractions and category-macro metrics in percent: Image AUROC/AP, TPR@5% FPR, Pixel AUROC/AP and AUPRO. Inapplicable metrics are `null`. Temporary full-resolution maps are stored one category at a time; set `TMPDIR` to a disk with sufficient space. `evaluate_all.py` also creates separate original/extension summaries using the appropriate valid denominators.

Compare raw-image evaluation with the published values above and the CSVs. Batching and historical feature quantization can cause small differences; an exact-match claim requires the same inputs and numerical path. **Optional historical replay is a different mode**, described in [evaluation details](docs/evaluation.md); it is not required for inference from raw images.

## Protocol and scope

The fixed members use seeds 111/222/333 and epoch 15. VisA official test-only (2,162 labeled normal/abnormal images and masks) supervises all targets except VisA; VisA evaluation uses MVTec AD official test-only supervision (1,725 images). Target labels are used only for evaluation. Inputs are 518×518; CLIP ViT-L/14@336px uses DPAM layer 20 and layer-24 local tokens only. Calibration, color bandwidth and deployment coefficients are frozen. Probabilities/maps are averaged **before** computing ensemble metrics.

The original 14 datasets provide 11 valid image and 11 valid pixel tasks. Five extensions yield totals of 14 image and 16 pixel tasks. Brain MRI [M] and Brain_AD duplicate the same 3,715 evaluation images and count once. Cross-paper baselines use their respective protocols; missing metrics are not zeros. See [evaluation details](docs/evaluation.md) and [third-party assets](THIRD_PARTY.md).

## Repository layout

- `facetad/`: frozen model, preprocessing and exact evaluator.
- `scripts/`: setup check, inference and evaluation commands.
- `configs/`: unchanged scientific configuration, hashes and manifest example.
- `kernels/`: CUDA operator sources; `results/`: published tables and verification records.
- `release/`: compatibility commands for the original release; old `R1_*` path variables remain supported. Do not merge private experiment archives into a public checkout.

Advanced path overrides are documented in [evaluation details](docs/evaluation.md). Weights, datasets and outputs are not tracked in Git.

[Packaging regression checks and their scope](docs/reorganization-validation.md).
