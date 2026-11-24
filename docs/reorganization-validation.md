# Reproduction-entry reorganization validation (2026-11-24)

Baseline: public release commit `bd38e7b2d3d9768318f0b5dc99f5f350bfefa0b4`. The private experimental Git history was not merged.

## What changed

- Python implementation moved from `release/` to `facetad/`; scientific JSON moved to `configs/`.
- `scripts/` provides commands, setup diagnostics, primary-dataset manifest generation and a complete-target evaluation driver.
- Three primary `FACETAD_*` filesystem variables replace machine-specific instructions. Explicit old `R1_*` overrides and original release commands remain supported.
- README describes inference/evaluation of the published weights; it does not promise a from-scratch training pipeline.

## Checks performed

- AST comparison: all scoring/preprocessing/reader/metric functions and model classes match the original code. Only the upstream code path lookup in `clip()` changed. Scientific JSON files are byte-identical, including checkpoint hashes and color bandwidths. `kernels/` and published result CSVs are unchanged.
- Setup checks passed for both packaged reader files and the original CLIP checkpoint SHA. The pinned upstream commit is `3911738c0867544f545a076ad78f3f11d9ecbfdf`.
- GPU: Linux RTX 3090 physical 0; PyTorch 2.11.0+cu128 / torchvision 0.26.0. Operators rebuilt from the unchanged CUDA sources with the available CUDA 11.7 compiler. Runtime uses the same GPU/precision for old/new comparison.
- An isolated venv inherited the existing verified scientific packages via `--system-site-packages`; missing dependencies were installed from `requirements.txt`. This is **not a second completely fresh torch installation**. The historical clean-install verification remains separate under `results/verification/`.
- Each of the two sources: 8 cached images and 1 real raw image had exactly zero probability and full-resolution heatmap differences between old/new implementations.
- Each target (MVTec/VisA): a paired 4-image subset with normal and abnormal images had exactly equal six-metric evaluation output. This is a packaging regression, **not a rerun of the full 19-dataset benchmark**.
- Generated official MVTec/VisA manifests contain 1,725/2,162 images and exactly match the historical image/label/mask sets. Generated ordering is for raw evaluation and is not advertised as the historical replay ordering. An augmented local VisA split is rejected by the sample-count guard; it is not used for formal evaluation.
- The documented synthetic-image command and the legacy `release/infer.py` wrapper produce identical NPZ arrays, with image shape `[1]` and pixel shape `[1,518,518]`.
- Path tests cover manifest-relative versus explicit-root resolution, absolute paths, and new/legacy variable precedence.

Numerical receipts: [regression](../results/verification/reorganization/regression.json), [manifest check](../results/verification/reorganization/manifest-validation.json). Input images, masks and large prediction arrays are not committed.

## Running the checks

```bash
python -m unittest discover -s tests -v
python scripts/check_setup.py
```

`tests/regression.py --help` documents the optional old/new numerical comparison. It requires private benchmark caches and a preserved public-release checkout; ordinary raw-image reproduction does not require them. No checkpoint or deployment coefficient was changed, and no model was trained.
