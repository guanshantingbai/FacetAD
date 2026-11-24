"""Compare an exact replay (percent) with the frozen table values."""

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("files", nargs="+", type=Path)
args = p.parse_args()
table = Path(__file__).resolve().parents[1] / "results/r1_metrics.csv"
with table.open() as f:
    reference = {r["dataset"]: r for r in csv.DictReader(f)}
for file in args.files:
    r = json.loads(file.read_text())
    assert r["category"] is None, file
    expected = reference[r["dataset"]]
    for key, value in r["macro_percent"].items():
        if expected[key] == "NA":
            assert value is None, (file, key, value)
        else:
            assert value is not None and abs(value - float(expected[key])) < 1e-7, (
                file,
                key,
                value,
                expected[key],
            )
    print(r["dataset"], "verified")
