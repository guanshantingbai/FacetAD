# Reproducing the evidence package

Start from the release README installation and the original registered evaluation inputs. Use a separate writable worktree. The official CLIP and source packages, source-specific epoch-15 fits and all original caches are read-only. This supplemental workflow calls their frozen readers and evaluators.

|Variable|Expected input|
|---|---|
|R1_WEIGHTS, R1_CLIP_WEIGHTS, R1_CLIP_CODE, R1_KERNELS|Release packages, pinned CLIP checkpoint and dependency, compiled release kernels|
|R1_INPUTS|Official per-dataset manifest.jsonl and three verified member score archives|
|R1_CACHE|Original per-dataset rows.json, packed masks and source cache timing receipts|
|R1_ABLATION|Independent frozen intervention outputs, including ensemble image.npz per domain|
|R1_AB_REPORT|Frozen r1-ablation-final tables, reconstruction data and weight audit|
|R1_VERIFIED_REPLAY|Clean release evaluator JSON for all 14 original domains|
|R1_RECOVERY|Original archive with the exact member-level color-guided class metrics|
|R1_RUNS|Protected-host and fused-Dynamic fit directories with done.json and final.pt|

Initialize a private work directory and private output subdirectory. Use the release Python environment with NumPy, SciPy, Matplotlib, Pillow, scikit-learn, OpenCV and PyTorch. Set OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=1. The following are relative to the worktree root.

```sh
mkdir -p work reports/r1-paper-evidence/private
chmod 700 work reports/r1-paper-evidence/private
python evidence/seeds.py
python evidence/dynamic.py
python evidence/figures.py dynamic
CUDA_VISIBLE_DEVICES=6 python evidence/local.py mvtec visa dagm
python evidence/errors.py
CUDA_VISIBLE_DEVICES=6 python evidence/sparse.py
CUDA_VISIBLE_DEVICES=6 python evidence/figures.py local
CUDA_VISIBLE_DEVICES=0 python evidence/profile_cost.py
CUDA_VISIBLE_DEVICES=0 python evidence/phase_peaks.py
python evidence/costs.py
python evidence/assemble.py
```

Profile on an idle dedicated GPU. Localization and profiling can use separate GPUs. The 3,700,000 sampled pixel pairs are used for the focused correction analysis only. The exact release metrics remain the paper point estimates. CASE_SELECTION.md specifies the figure rules. The protected output contains real samples and a full list of candidate image pairs. The repository includes source, aggregate tables, industrial relative IDs and hashes. Rights clearance for redistribution of the query images has not been established.
