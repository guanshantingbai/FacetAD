# Evaluation details

## Two distinct modes

The default `scripts/evaluate.py` runs the frozen model on RGB images and computes exact full-resolution metrics. No historical feature or prediction cache is needed. The source is automatically selected: MVTec for a VisA target; VisA for other targets. Readers and coefficients are not fitted during evaluation.

The optional `--member-scores` mode replays saved member image probabilities and 37×37 logits. It verifies byte hashes, record order, categories, image labels, and effective masks before evaluating. These exports are optional external assets, not included in the reader download. Never describe this mode as a fresh visual-encoder reproduction.

```bash
python scripts/evaluate.py mvtec --manifest /path/to/mvtec.jsonl \
  --member-scores /path/to/mvtec_scores --output "$FACETAD_OUTPUT_ROOT/mvtec-replay.json"
python scripts/verify_tables.py "$FACETAD_OUTPUT_ROOT/mvtec-replay.json"
```

`verify_tables.py` applies the historical exact-replay tolerance of 1e-7 percentage points. It is not an acceptance threshold for a different raw-encoding numerical path. Unchanged historical verification is in `results/verification/`; new packaging checks are reported separately.

## Manifest and masks

See `configs/manifest.example.jsonl`. Each image has a category and image label (0 normal, 1 abnormal). Localization records require a mask list; `[]` indicates a known empty foreground. Multiple masks are unioned. Mask/image dimensions must agree. The evaluator preserves the released mask resizing and thresholding; do not substitute another resizing rule.

OCT17 (`oct17`), BrainMR1 [V] (`brainmri`) and HIS (`his`) are image-only. CVC-ClinicDB, Endo, Kvasir, CVC-ColonDB and CVC-300 contain no normal evaluation images under this protocol: image metrics are NA. Do not synthesize negatives or average NA as zero. Keep normal images and anomalous-label images with empty masks. A manifest should include the entire published target split, not a selected subset.

When `FACETAD_DATA_ROOT` is set, relative paths in every manifest are rooted there. When unset, relative paths retain the original convention (manifest directory). Absolute paths take precedence. No fallback search silently chooses an alternative image.

## Paths

| Setting | Default | Purpose |
|---|---|---|
| `FACETAD_DATA_ROOT` | repository `data/` | manifest lookup and explicit root for relative data paths |
| `FACETAD_WEIGHTS_ROOT` | repository `weights/` | `visa.pt`, `mvtec.pt`, CLIP weights |
| `FACETAD_OUTPUT_ROOT` | repository `outputs/` | default prediction and metric outputs |
| `FACETAD_CLIP_WEIGHTS` | weights directory, then legacy CLIP cache | optional explicit visual checkpoint |
| `FACETAD_CLIP_CODE` | `third_party/AnomalyCLIP` | optional pinned upstream source location |
| `FACETAD_KERNELS` | `build/` | optional compiled CUDA operator directory |

New path variables take precedence over `R1_WEIGHTS`, `R1_CLIP_WEIGHTS`, `R1_CLIP_CODE`, and `R1_KERNELS`, respectively. The old variables remain supported. `CUDA_VISIBLE_DEVICES` selects the GPU; `TMPDIR` controls temporary dense-map storage. No environment variable changes rho, temperature, ensemble members or fusion weights.

## Compatibility and scope

`python release/infer.py`, `release/evaluate.py` and `release/verify_tables.py` forward to the new implementation. Their explicit input/output arguments remain valid. Configuration files now live under `configs/`. Use the new `facetad` package for Python imports.

This release covers evaluation of published weights. From-scratch training and private experiment recovery are not part of this installation procedure. Dataset licensing and upstream licensing are described in `THIRD_PARTY.md`.
