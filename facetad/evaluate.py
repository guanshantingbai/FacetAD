"""Evaluate R1 from RGB images or replay SHA-verified official member outputs."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from .r1_metrics import evaluate_maps, images
from .runtime import (
    CONFIG,
    TRANS,
    Ensemble,
    color_base,
    configure,
    deterministic,
    geometry,
    lab_batch,
    sha,
    source_for,
)

from .paths import DATA, OUTPUT, manifest_path

KEYS = ("image_auc", "image_ap", "tpr", "pixel_auc", "pixel_ap", "aupro")
IMAGE_ONLY = {"oct17", "brainmri", "his"}
REPLAY_HASHES = json.loads((Path(__file__).resolve().parents[1] / "configs/replay-sha256.json").read_text())


def load_manifest(file, pixels):
    with file.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError("Empty JSONL manifest")
    seen = set()
    for row in records:
        path = manifest_path(file, row["image"])
        if path in seen or row["label"] not in (0, 1) or not row["category"]:
            raise ValueError(f"Duplicate image or invalid class/label: {path}")
        seen.add(path)
        row["image"] = path
        if pixels:
            if not isinstance(row.get("masks"), list):
                raise ValueError(f"Expected a masks list for {path}")
            row["masks"] = [manifest_path(file, mask) for mask in row["masks"]]
    return records


def read_mask(row, original_size):
    mask = np.zeros((518, 518), dtype=bool)
    for path in row["masks"]:
        with Image.open(path) as src:
            if src.size != original_size:
                raise ValueError(f"Mask and image have different dimensions: {path}")
            binary = Image.fromarray(
                (np.asarray(src.convert("L")) > 0).astype("uint8") * 255
            )
            mask |= (
                np.asarray(binary.resize((518, 518), Image.Resampling.BILINEAR)) > 127
            )
    return np.packbits(mask)


def official_members(target, folder, records, pixels):
    """Check row alignment before joining anonymous cached scores to a manifest."""
    import hashlib

    digest = hashlib.sha256()
    mask_hash = hashlib.sha256()
    for row in records:
        item = [sha(row["image"]), row["category"], row["label"]]
        digest.update(
            (
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode()
        )
        if pixels:
            with Image.open(row["image"]) as image:
                mask_hash.update(read_mask(row, image.size).tobytes())
    cfg = CONFIG["datasets"][target]
    if digest.hexdigest() != cfg["ordered_rows_sha256"] or (
        pixels and mask_hash.hexdigest() != cfg["packed_masks_sha256"]
    ):
        raise ValueError(
            "Manifest order, image contents, categories, labels or masks differ from the official replay"
        )
    probabilities = []
    grids = []
    for seed in (111, 222, 333):
        file = folder / f"{seed}.npz"
        if sha(file) != REPLAY_HASHES[target][str(seed)]:
            raise ValueError(f"Member score SHA mismatch: {file}")
        with np.load(file, allow_pickle=False) as f:
            probabilities.append(f["image"])
            if pixels:
                grids.append(f["grids"])
    return np.mean(probabilities, axis=0, dtype=np.float64), grids


@torch.inference_mode()
def evaluate(target, manifest, replay=None):
    configure()
    deterministic()
    pixels = target not in IMAGE_ONLY
    records = load_manifest(manifest, pixels)
    source = source_for(target)
    if replay:
        probabilities, grids = official_members(target, replay, records, pixels)
        geo = geometry() if pixels else None
        band = CONFIG["sources"][source]["band"]
    else:
        model = Ensemble(source)
        probabilities = np.empty(len(records), dtype=np.float64)
    labels = np.array([r["label"] for r in records])
    groups = np.array([r["category"] for r in records])
    stats = {}
    for category in np.unique(groups):
        ix = np.flatnonzero(groups == category)
        px = [None] * 3
        with tempfile.TemporaryDirectory() as directory:
            if pixels:
                maps = np.memmap(
                    Path(directory) / "heatmaps",
                    dtype="f4",
                    mode="w+",
                    shape=(len(ix), 518**2),
                )
                masks = np.empty((len(ix), (518**2 + 7) // 8), dtype="uint8")
            if pixels or not replay:
                for start in range(0, len(ix), 8):
                    ids = ix[start : start + 8]
                    batch = []
                    for j, i in enumerate(ids):
                        with Image.open(records[i]["image"]) as image:
                            batch.append(image.convert("RGB"))
                            if pixels:
                                masks[start + j] = read_mask(records[i], image.size)
                    if replay:
                        lab = lab_batch([TRANS.transforms[0](im) for im in batch])
                        heat = torch.stack(
                            [color_base(g[ids], lab, band, geo) for g in grids]
                        ).mean(0)
                    else:
                        probabilities[ids], heat = model.predict(batch)
                    if pixels:
                        maps[start : start + len(ids)] = heat.cpu().numpy()
            if pixels:
                px = evaluate_maps(maps, masks)
                del maps
        stats[str(category)] = dict(
            zip(KEYS, images(labels[ix], probabilities[ix]) + px)
        )
    macro = {}
    for k in KEYS:
        valid = [r[k] for r in stats.values() if r[k] is not None]
        macro[k] = float(np.mean(valid) * 100) if valid else None
    return {
        "dataset": target,
        "source": source,
        "mode": "official_replay" if replay else "raw_images",
        "category": None,
        "per_class_fraction": stats,
        "macro_percent": macro,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=sorted(CONFIG["datasets"]))
    p.add_argument("--manifest", type=Path, help="Default: FACETAD_DATA_ROOT/manifests/<target>.jsonl")
    p.add_argument(
        "--member-scores",
        type=Path,
        help="Directory of official 111/222/333.npz member exports",
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    a.manifest = a.manifest or DATA / "manifests" / (a.target + ".jsonl")
    a.output = a.output or OUTPUT / (a.target + ".json")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(a.target, a.manifest, a.member_scores)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["macro_percent"])
