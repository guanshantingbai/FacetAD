import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release"))
import runtime as rt
import evaluate as ev
from PIL import Image

OUT = ROOT / "reports/r1-paper-evidence"
WORK = ROOT / "work"
SEEDS = (111, 222, 333)
TARGETS = ("mvtec", "visa", "dagm")


def root(key):
    return Path(os.environ[key])


def meta(n):
    return json.loads((root("R1_CACHE") / n / "rows.json").read_text())


def manifest(n):
    return ev.load_manifest(
        root("R1_INPUTS") / n / "manifest.jsonl", n not in ev.IMAGE_ONLY
    )


def digest(*keys):
    return hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()


def sha(p):
    return rt.sha(p)


def write(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def jsonout(path, x):
    path.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def safe_id(r):
    i = r["id"]
    assert not Path(i).is_absolute() and ".." not in Path(i).parts
    return i


def data(n):
    records = meta(n)
    doc = manifest(n)
    cfg = rt.CONFIG["datasets"][n]
    assert len(doc) == len(records) == cfg["count"]
    mask = np.load(root("R1_CACHE") / n / "maskbits.npy", mmap_mode="r")
    h = hashlib.sha256()
    mh = hashlib.sha256()
    for r, m in zip(records, doc):
        safe_id(r)
        assert (r["category"], r["y"]) == (m["category"], m["label"])
        h.update(
            (
                json.dumps(
                    [r["sha"], r["category"], r["y"]],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
    assert h.hexdigest() == cfg["ordered_rows_sha256"]
    if n not in ev.IMAGE_ONLY:
        for a in mask:
            mh.update(a.tobytes())
        assert mh.hexdigest() == cfg["packed_masks_sha256"]
    return records, doc, mask


def rgb(row, m):
    assert sha(m["image"]) == row["sha"]
    with Image.open(m["image"]) as im:
        return rt.TRANS.transforms[0](im.convert("RGB"))
