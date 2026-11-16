"""Render registered real queries with a shared probability scale."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import (
    csv,
    sys,
    Path,
    np,
    rt,
    ev,
    OUT,
    WORK,
    TARGETS,
    SEEDS,
    sha,
    root,
    meta,
    manifest,
    write,
    data,
    rgb,
)
from local import rank, packed_mask
import torch


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def photo(ax, im):
    ax.imshow(im, interpolation="nearest")
    ax.axis("off")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(
            OUT / (name + "." + ext), dpi=180, bbox_inches="tight", pad_inches=0.08
        )
    plt.close(fig)


EVENTS = {
    "correct": "Incorrect pair corrected",
    "breaks": "Correct pair broken",
    "orth": "Orthogonal correction recovers the order",
}


def dynamic():
    rows = read(OUT / "fig_dynamic_pairs.csv")
    available = {}
    used = set()
    for typ, preferred in [("correct", "mvtec"), ("breaks", "dagm"), ("orth", "visa")]:
        opts = [r for r in rows if r["event"] == typ]
        r = next((r for r in opts if r["dataset"] == preferred), opts[0])
        available[typ] = r
        used.add((typ, r["dataset"], r["anomaly_id"], r["normal_id"]))
    for r in rows:
        r["displayed"] = int(
            (r["event"], r["dataset"], r["anomaly_id"], r["normal_id"]) in used
        )
    write(OUT / "fig_dynamic_pairs.csv", rows)
    fig = plt.figure(figsize=(9.5, 7.7), layout="constrained")
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.85])
    for j, (typ, r) in enumerate(available.items()):
        rs = meta(r["dataset"])
        doc = manifest(r["dataset"])
        indices = {v["id"]: i for i, v in enumerate(rs)}
        for k, s in enumerate(("anomaly", "normal")):
            idx = indices[r[s + "_id"]]
            ax = fig.add_subplot(grid[j, k])
            photo(ax, rgb(rs[idx], doc[idx]))
            ax.set_title(
                ("Anomalous" if k == 0 else "Normal")
                + "  |  "
                + Path(r[s + "_id"]).name,
                fontsize=9,
            )
        ax = fig.add_subplot(grid[j, 2])
        ax.axis("off")
        ax.text(0, 0.95, EVENTS[typ], fontsize=12, weight="bold")
        ax.text(0, 0.82, r["dataset"].upper() + "  /  " + r["category"], fontsize=10)
        cells = []
        for key, name in [
            ("off", "Off"),
            ("on", "Current image"),
            ("orth", "Orthogonal"),
            ("donor", "Donor"),
        ]:
            a, b = map(float, (r[key + "_anomaly"], r[key + "_normal"]))
            cells.append([name, f"{a:.4f}", f"{b:.4f}", f"{a - b:+.4f}"])
        t = ax.table(
            cellText=cells,
            colLabels=["Dynamic", "Anomaly", "Normal", "Gap"],
            cellLoc="right",
            bbox=[0, 0.10, 1, 0.57],
            colWidths=[0.34, 0.22, 0.22, 0.22],
        )
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        for (y, x), cell in t.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#e9eff2" if y == 0 else "#f5f7f8" if y % 2 else "white")
    save(fig, "FIG_DYNAMIC")
    (OUT / "fig_dynamic_caption.md").write_text(
        "**Image-level ranking under a frozen Dynamic correction.** Each row pairs an anomalous and a normal query from the same class. The gap is the anomalous probability minus the normal probability after averaging three members. The recipient image and static A/H midpoint are fixed for all four columns. The donor changes only the abnormal-prototype displacement. Positive gaps give the correct ordering. Dynamic leaves the deployed heatmap unchanged. Relative IDs, labels and per-image logit changes are in fig_dynamic_pairs.csv. The three examples are diagnostic selections.\n"
    )


EVENT_NAMES = {
    "excluded": "Native miss, recovery criterion failed",
    "miss": "Native miss recovered",
    "repair": "Fusion partly repairs H",
    "color": "Boundary improved",
    "damage": "Negative transfer",
}


def priority(r):
    a, h, f, edge, guide = [
        float(r[k]) for k in ("A", "H", "AH", "boundary_AH", "boundary_Final")
    ]
    return {
        "miss": f - a if a <= 0.5 and h > a and f > a else None,
        "repair": f - h if a > f > h else None,
        "color": guide - edge if guide > edge else None,
        "damage": a - h if h < a and f < a else None,
    }


@torch.inference_mode()
def panels(n, rows):
    rs, doc, mask = data(n)
    model = rt.Ensemble(rt.source_for(n))
    # local.py verified the full manifests and RGB/masks before producing the
    # ranking pool. Recheck the member files and the selected RGBs at rendering.
    grids = []
    for seed in SEEDS:
        file = root("R1_INPUTS") / n / "scores" / f"{seed}.npz"
        assert sha(file) == ev.REPLAY_HASHES[n][str(seed)]
        with np.load(file, allow_pickle=False) as pack:
            grids.append(pack["grids"])
    output = {}
    for r in rows:
        idx = int(r["index"])
        cat = r["category"]
        ix = np.array([i for i, x in enumerate(rs) if x["category"] == cat])
        offset = int(np.where(ix == idx)[0][0])
        ids = ix[offset // 8 * 8 : offset // 8 * 8 + 8]
        i = offset % 8
        ims = [rgb(rs[k], doc[k]) for k in ids]
        lab = rt.lab_batch(ims)
        stack = []
        for arr, m in zip(grids, model.members):
            v = arr[ids]
            a = rt.maps(v[:, 0])
            h = rt.maps(v[:, 1])
            beta, _, b = m.coef.cpu().numpy()
            ho = rt.maps(beta * v[:, 0] + b)
            stack.append(
                [a, h, ho, 0.5 * (a + h), rt.color_base(v, lab, model.band, model.geo)]
            )
        scores = [
            torch.stack(s).mean(0)[i].cpu().numpy().reshape(518, 518)
            for s in zip(*stack)
        ]
        gt = packed_mask(mask[idx]).reshape(518, 518)
        for name, v in zip(("A", "H", "Hoff", "AH", "Final"), scores):
            if name in r:
                assert abs(rank(v, gt, ~gt) - float(r[name])) < 1e-12
        output[idx] = [ims[i], gt, *scores]
    return output


def example_pairs(rows, images):
    """Find one exact inversion witness per illustration in row-major order."""
    from scipy.ndimage import distance_transform_edt

    witnesses = []
    for r in rows:
        _, gt, *maps = images[r["dataset"], int(r["index"])]
        pos, neg = gt, ~gt
        if r["event"] == "color":
            pos = pos & (distance_transform_edt(pos) <= 14)
            neg = neg & (distance_transform_edt(neg) <= 14)
        bg = np.flatnonzero(neg)
        a, h, _, ah, final = [m.ravel() for m in maps]
        for i in np.flatnonzero(pos):
            if r["event"] == "color":
                valid = (ah[i] < ah[bg]) & (final[i] > final[bg])
            elif r["event"] == "repair":
                valid = (a[i] > a[bg]) & (h[i] < h[bg]) & (ah[i] > ah[bg])
            elif r["event"] == "miss":
                valid = (a[i] < a[bg]) & (h[i] > h[bg]) & (ah[i] > ah[bg])
            else:
                valid = (a[i] > a[bg]) & (h[i] < h[bg]) & (ah[i] < ah[bg])
            if np.any(valid):
                j = bg[np.argmax(valid)]
                row = dict(
                    event=r["event"],
                    dataset=r["dataset"],
                    id=r["id"],
                    fg_y=i // 518,
                    fg_x=i % 518,
                    bg_y=j // 518,
                    bg_x=j % 518,
                )
                for key, m in zip(("A", "H", "Hoff", "AH", "Final"), maps):
                    row[key + "_fg"] = float(m.ravel()[i])
                    row[key + "_bg"] = float(m.ravel()[j])
                witnesses.append(row)
                break
    write(OUT / "localization_rank_witnesses.csv", witnesses)


def page(rows, images):
    fig, axs = plt.subplots(
        len(rows), 7, figsize=(12.5, len(rows) * 2.28 + 0.45), squeeze=False
    )
    fig.subplots_adjust(
        left=0.015, right=0.93, top=0.90, bottom=0.025, wspace=0.04, hspace=0.38
    )
    for i, r in enumerate(rows):
        for j, (ax, img) in enumerate(
            zip(axs[i], images[r["dataset"], int(r["index"])])
        ):
            if j == 0:
                photo(ax, img)
            elif j == 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.axis("off")
            else:
                heat = ax.imshow(
                    img, cmap="inferno", vmin=0, vmax=1, interpolation="nearest"
                )
                ax.axis("off")
            if i == 0:
                ax.set_title(
                    [
                        "Input",
                        "GT",
                        "Native A",
                        "H alone",
                        "H residual off",
                        "A/H fusion",
                        "Final R1 + Color",
                    ][j],
                    fontsize=10,
                )
        left = axs[i, 0].get_position()
        comparison = {
            "excluded": "A/H minus A",
            "damage": "H minus A",
            "color": "Final minus A/H, boundary",
            "miss": "A/H minus A",
            "repair": "A/H minus H",
        }[r["event"]]
        signed = (-1 if r["event"] == "damage" else 1) * float(r["priority"])
        fig.text(
            left.x0,
            left.y0 - 0.012,
            f"{EVENT_NAMES[r['event']]}  |  {r['dataset'].upper()} / {r['category']}  |  {r['id']}  |  {comparison} {100 * signed:+.2f} pp",
            fontsize=8.3,
            va="top",
        )
    cax = fig.add_axes([0.947, 0.2, 0.009, 0.52])
    fig.colorbar(heat, cax=cax, label="Anomaly probability")
    return fig


def local():
    pool = [r for n in TARGETS for r in read(WORK / n / "case_pool.csv")]
    write(OUT / "localization_screened.csv", pool)
    candidates = []
    main = []
    by = {}
    for typ, preferred in [
        ("miss", "mvtec"),
        ("repair", "visa"),
        ("color", "mvtec"),
        ("damage", "dagm"),
    ]:
        opts = []
        for n in TARGETS:
            valid = [
                dict(r, event=typ, priority=priority(r)[typ])
                for r in pool
                if r["dataset"] == n and priority(r)[typ] is not None
            ]
            valid.sort(key=lambda r: (-r["priority"], r["id"]))
            if valid:
                opts.append(valid[0])
        if opts:
            chosen = next(
                (r for r in opts if r["dataset"] == preferred),
                max(opts, key=lambda r: r["priority"]),
            )
            main.append(chosen)
        candidates.extend(opts)
    for n in TARGETS:
        if not any(r["dataset"] == n for r in main):
            opts = [r for r in candidates if r["dataset"] == n]
            if opts:
                main.append(opts[0])
    for r in candidates:
        r["screen"] = "primary"
    if (WORK / "sparse_selected.csv").is_file():
        for r in read(WORK / "sparse_selected.csv"):
            candidates.append(
                dict(
                    {k: r.get(k, "NA") for k in pool[0]},
                    event="miss",
                    priority=float(r["priority"]),
                    screen="secondary",
                )
            )
    for r in candidates:
        r["displayed"] = int(r in main)
    write(OUT / "fig_localization_cases.csv", candidates)
    excluded = []
    extra = OUT / "localization_miss_supplement.csv"
    if extra.is_file():
        for r in pool + read(extra):
            if float(r["A"]) > 0.5:
                continue
            a, h, f = [float(r[k]) for k in ("A", "H", "AH")]
            if h > a and f > a:
                continue
            excluded.append(
                dict(
                    dataset=r["dataset"],
                    category=r["category"],
                    id=r["id"],
                    index=r["index"],
                    A=a,
                    H=h,
                    AH=f,
                    event="excluded",
                    priority=f - a,
                    reason="; ".join(
                        name
                        for name, value in (
                            ("H fails to improve", h),
                            ("fusion fails to improve", f),
                        )
                        if value <= a
                    ),
                )
            )
        if excluded:
            write(OUT / "fig_localization_excluded.csv", excluded)
    for n in TARGETS:
        unique = {r["index"]: r for r in candidates + excluded if r["dataset"] == n}
        by.update({(n, k): v for k, v in panels(n, list(unique.values())).items()})
    example_pairs(candidates, by)
    save(page(main, by), "FIG_LOCALIZATION")
    with PdfPages(OUT / "FIG_LOCALIZATION_CANDIDATES.pdf") as pdf:
        for typ in ("miss", "repair", "color", "damage", "excluded"):
            rs = [r for r in candidates + excluded if r["event"] == typ]
            if rs:
                fig = page(rs, by)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.08)
                plt.close(fig)
    (OUT / "fig_localization_caption.md").write_text(
        "**Local residual evidence in real test queries.** White pixels denote ground-truth defects. All panels share the same registered coordinates and the heatmaps use one absolute [0,1] color scale. H residual off retains the trained native calibration in H alone. A/H fusion averages the two plain full-resolution maps. Final R1 also uses color-guided interpolation. Changes under each row are within-image foreground/background AUROC differences. For native miss, partial repair, boundary and damage the displayed differences are fusion minus A, fusion minus H, final minus fusion (14-pixel boundary band), and H minus A, respectively. Examples were selected by the predeclared ranks, not by heatmap appearance. The last candidate page shows the remaining three below-chance A examples that failed the recovery criterion. The displays do not estimate event prevalence outside this query set.\n"
    )


if __name__ == "__main__":
    if sys.argv[1] == "dynamic":
        dynamic()
    else:
        local()
