"""Complete a sparse qualitative panel without changing the primary screen."""

from common import np, rt, ev, OUT, WORK, TARGETS, root, digest, write, data
from local import rank, packed_mask
from figures import read, priority, panels
from scipy.ndimage import distance_transform_edt
import torch


@torch.inference_mode()
def run():
    rt.configure()
    rt.deterministic()
    primary = [r for n in TARGETS for r in read(WORK / n / "case_pool.csv")]
    eligible = [r for r in primary if priority(r)["miss"] is not None]
    if len({r["dataset"] for r in eligible}) >= 2:
        return
    excluded = {(r["dataset"], r["id"]) for r in primary}
    extra = []
    log = []
    for n in TARGETS:
        rs, doc, mask = data(n)
        _, grids = ev.official_members(n, root("R1_INPUTS") / n / "scores", doc, True)
        for cat in sorted({r["category"] for r in rs}):
            ix = np.array([i for i, r in enumerate(rs) if r["category"] == cat])
            for j in range(0, len(ix), 8):
                ids = ix[j : j + 8]
                sel = [
                    k
                    for k, i in enumerate(ids)
                    if mask[i].any() and (n, rs[i]["id"]) not in excluded
                ]
                if not sel:
                    continue
                member_batches = [g[ids] for g in grids]
                maps = [rt.maps(v[:, 0]) for v in member_batches]
                a = torch.stack(maps).mean(0).cpu().numpy()
                survivors = []
                for k in sel:
                    i = ids[k]
                    m = packed_mask(mask[i])
                    auc = rank(a[k], m, ~m)
                    row = dict(
                        dataset=n,
                        category=cat,
                        id=rs[i]["id"],
                        index=int(i),
                        foreground=int(m.sum()),
                        A=auc,
                        H="NA",
                        AH="NA",
                    )
                    log.append(row)
                    if auc <= 0.5:
                        survivors.append((k, row, m))
                if survivors:
                    hs = [rt.maps(v[:, 1]) for v in member_batches]
                    h = torch.stack(hs).mean(0).cpu().numpy()
                    ah = (
                        torch.stack([0.5 * (v + w) for v, w in zip(maps, hs)])
                        .mean(0)
                        .cpu()
                        .numpy()
                    )
                    for k, r, m in survivors:
                        r["H"] = rank(h[k], m, ~m)
                        r["AH"] = rank(ah[k], m, ~m)
                        if min(r["H"], r["AH"]) > r["A"]:
                            extra.append(dict(r, priority=r["AH"] - r["A"]))
    write(
        OUT / "localization_miss_supplement.csv",
        sorted(
            log,
            key=lambda r: (r["dataset"], r["category"], digest(r["dataset"], r["id"])),
        ),
    )
    chosen = []
    used = {r["dataset"] for r in eligible}
    while extra and len(chosen) < 2:
        extra.sort(key=lambda r: (r["dataset"] in used, -r["priority"], r["id"]))
        r = extra.pop(0)
        chosen.append(r)
        used.add(r["dataset"])
    for n in TARGETS:
        subset = [r for r in chosen if r["dataset"] == n]
        if not subset:
            continue
        out = panels(n, subset)
        for r in subset:
            _, gt, a, h, ho, f, g = out[r["index"]]
            r["Hoff"] = rank(ho, gt, ~gt)
            pos = gt & (distance_transform_edt(gt) <= 14)
            neg = ~gt & (distance_transform_edt(~gt) <= 14)
            r["boundary_AH"] = rank(f, pos, neg)
            r["boundary_Final"] = rank(g, pos, neg)
            r["Final"] = rank(g, gt, ~gt)
    if chosen:
        write(WORK / "sparse_selected.csv", chosen)


if __name__ == "__main__":
    run()
