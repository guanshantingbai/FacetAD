from common import (
    csv,
    json,
    np,
    ROOT,
    ev,
    OUT,
    WORK,
    SEEDS,
    root,
    sha,
    write,
    jsonout,
    data,
)
from sklearn.metrics import roc_auc_score, average_precision_score

metrics = ["image_auc", "image_ap", "pixel_auc", "pixel_ap", "aupro"]


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def summary():
    refs = read(ROOT / "results/r1_metrics.csv")
    cells = {}
    static = {}
    archive = root("R1_RECOVERY") / "top24-confidence/pixel/guide"
    ab = read(root("R1_AB_REPORT") / "per_domain.csv")
    errors = []
    hashes = []
    for ref in refs:
        if ref["protocol"] != "original":
            continue
        n = ref["dataset"]
        rs, _, _ = data(n)
        groups = np.array([r["category"] for r in rs])
        y = np.array([r["y"] for r in rs])
        b = json.loads((root("R1_VERIFIED_REPLAY") / (n + ".json")).read_text())
        assert b["dataset"] == n and b["mode"] == "official_replay"
        table = [
            r for r in read(ROOT / "results/r1_per_class.csv") if r["dataset"] == n
        ]
        for r in table:
            for k in metrics:
                v = b["per_class_fraction"][r["category"]][k]
                if v is None:
                    assert r[k] == "NA"
                else:
                    errors.append(abs(float(r[k]) - 100 * v))
        for r in ab:
            if (r["dataset"], r["method"]) == (n, "A0"):
                static[n] = np.array(
                    [float(v) if v != "NA" else np.nan for v in [r[k] for k in metrics]]
                )
        for seed in SEEDS:
            p = root("R1_INPUTS") / n / "scores" / f"{seed}.npz"
            assert sha(p) == ev.REPLAY_HASHES[n][str(seed)]
            with np.load(p) as f:
                prob = f["image"]
            per = []
            for cat in sorted(set(groups)):
                ix = groups == cat
                im = (
                    [
                        roc_auc_score(y[ix], prob[ix]),
                        average_precision_score(y[ix], prob[ix]),
                    ]
                    if len(set(y[ix])) == 2
                    else [np.nan] * 2
                )
                pix = [np.nan] * 3
                if n not in ev.IMAGE_ONLY:
                    path = archive / n / str(seed) / cat / "metrics.json"
                    a = json.loads(path.read_text())
                    assert a["main_identity_error"] == a["off_identity_error"] == 0
                    pix = a["metrics"]["guided"]
                    hashes.append(
                        dict(dataset=n, seed=seed, category=cat, sha256=sha(path))
                    )
                per.append(np.asarray(im + pix, float) * 100)
            cells[n, seed] = np.nanmean(per, axis=0)
        cells[n, "R1 ensemble"] = np.array(
            [float(ref[k]) if ref[k] != "NA" else np.nan for k in metrics]
        )
    names = list(static)
    assert len(names) == 14 and max(errors) < 1e-7
    for member in (*SEEDS, "R1 ensemble"):
        cells["core", member] = np.nanmean([cells[n, member] for n in names], axis=0)
    static["core"] = np.nanmean(list(static.values()), axis=0)
    for n, expected, idx in [
        ("mvtec", 93.897, 0),
        ("mvtec", 91.513, 2),
        ("visa", 83.747, 0),
        ("visa", 95.381, 2),
        ("core", 86.300, 0),
        ("core", 93.781, 2),
        ("core", 46.709, 3),
    ]:
        assert round(cells[n, "R1 ensemble"][idx], 3) == expected
    result = []
    tex = []
    md = []
    for n in ("mvtec", "visa", "core"):
        stack = [cells[n, s] for s in SEEDS]
        mean = np.mean(stack, axis=0)
        sd = np.std(stack, axis=0, ddof=1)
        for member, v in [
            *zip(SEEDS, stack),
            ("member mean", mean),
            ("member sample SD", sd),
            ("static A ensemble", static[n]),
            ("R1 ensemble", cells[n, "R1 ensemble"]),
            ("R1 ensemble minus member mean", cells[n, "R1 ensemble"] - mean),
        ]:
            result.append(dict(dataset=n, model=member, **dict(zip(metrics, v))))
            display = [f"{x:.3f}" for x in v]
            if member == "member mean":
                display = [f"{x:.3f} ± {t:.3f}" for x, t in zip(v, sd)]
            if member != "member sample SD":
                md.append("| " + " | ".join([n, str(member), *display]) + " |")
                tex.append(
                    " & ".join([n, str(member), *display]).replace("±", r"$\pm$")
                    + r" \\"
                )
    write(OUT / "SEED_RESULTS.csv", result)
    write(OUT / "seed_pixel_provenance.csv", hashes)
    (OUT / "SEED_RESULTS.md").write_text(
        "# Frozen members and probability ensembles\n\nClass-macro percentages. Core averages 11 eligible datasets per task. Member dispersion uses sample SD (n=3). The ensemble is scored once after averaging predictions.\n\n|Dataset|Model|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|\n|---|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(md)
        + "\n"
    )
    (OUT / "SEED_RESULTS.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{llrrrrr}",
                r"\toprule",
                r"Dataset & Readout & I-AUC & I-AP & P-AUC & P-AP & AUPRO \\",
                r"\midrule",
                *tex,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
    )
    jsonout(
        WORK / "gate.json",
        dict(
            max_core_class_error_pp=max(errors),
            core=list(cells["core", "R1 ensemble"]),
            pixel_sources=len(hashes),
        ),
    )


if __name__ == "__main__":
    summary()
