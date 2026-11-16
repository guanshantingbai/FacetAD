"""Exact case ranks and paired, classwide pixel sampling on real registered maps."""

from common import sys, np, rt, ev, WORK, root, digest, write, data, rgb
from scipy.ndimage import distance_transform_edt
import torch
from concurrent.futures import ThreadPoolExecutor

SECTIONS = ("A", "H", "Hoff", "AH", "Final", "FinalOff")
N = 100000


def ordering(pos, neg):
    return (pos > neg).astype(float) + 0.5 * (pos == neg)


def rank(v, fg, bg):
    a = v[fg]
    b = np.sort(v[bg])
    if not len(a) or not len(b):
        return np.nan
    return (
        np.searchsorted(b, a, "left").mean() + np.searchsorted(b, a, "right").mean()
    ) / (2 * len(b))


def packed_mask(m):
    return np.unpackbits(m, count=518**2).view(bool)


def sample(mask, ix, rng):
    counts = np.array([packed_mask(mask[i]).sum() for i in ix])
    result = []
    for positive, weights in [(True, counts), (False, 518**2 - counts)]:
        loc = rng.choice(len(ix), N, p=weights / weights.sum())
        order = np.argsort(loc, kind="stable")
        bounds = np.searchsorted(loc[order], np.arange(len(ix) + 1))
        queries = {}
        for k, i in enumerate(ix):
            slots = order[bounds[k] : bounds[k + 1]]
            if len(slots):
                valid = np.flatnonzero(packed_mask(mask[i]) == positive)
                queries[i] = (slots, rng.choice(valid, len(slots)))
        result.append(queries)
    return result


@torch.inference_mode()
def evaluate(n):
    rt.configure()
    rt.deterministic()
    rs, doc, masks = data(n)
    model = rt.Ensemble(rt.source_for(n))
    folder = root("R1_INPUTS") / n / "scores"
    prob, grids = ev.official_members(n, folder, doc, True)
    archived = np.load(root("R1_ABLATION") / n / "image.npz")
    assert np.array_equal(archived["ids"], [r["id"] for r in rs])
    np.testing.assert_array_equal(prob, archived["A3"])
    work = WORK / n
    work.mkdir(exist_ok=True)
    candidates = []
    stats = []
    for cat in sorted({r["category"] for r in rs}):
        ix = np.array([i for i, r in enumerate(rs) if r["category"] == cat])
        eligible = sorted(
            (i for i in ix if masks[i].any()), key=lambda i: digest(n, rs[i]["id"])
        )[:12]
        queries = sample(
            masks,
            ix,
            np.random.default_rng(int(digest("paper-pixels-v1", n, cat)[:16], 16)),
        )
        vals = np.empty((2, len(SECTIONS), N), np.float32)
        for j in range(0, len(ix), 8):
            ids = ix[j : j + 8]
            with ThreadPoolExecutor(4) as pool:
                ims = list(pool.map(lambda i: rgb(rs[i], doc[i]), ids))
            lab = rt.lab_batch(ims)
            member = []
            for grid, reader in zip(grids, model.members):
                v = grid[ids]
                a = rt.maps(v[:, 0])
                h = rt.maps(v[:, 1])
                beta, _, b = reader.coef.cpu().numpy()
                off = beta * v[:, 0] + b
                ho = rt.maps(off)
                base = 0.5 * (a + h)
                final = 0.5 * (base + rt.guided(v, lab, model.band, model.geo))
                vo = np.stack([v[:, 0], off], axis=1)
                finaloff = 0.5 * (
                    0.5 * (a + ho) + rt.guided(vo, lab, model.band, model.geo)
                )
                member.append([a, h, ho, base, final, finaloff])
                if j == 0:
                    assert torch.equal(
                        final, rt.color_base(v, lab, model.band, model.geo)
                    )
            maps = np.stack(
                [torch.stack(m).mean(0).cpu().numpy() for m in zip(*member)], axis=1
            )
            for k, i in enumerate(ids):
                for side, q in enumerate(queries):
                    if i in q:
                        slots, index = q[i]
                        vals[side][:, slots] = maps[k][:, index]
                if i not in eligible:
                    continue
                gt = packed_mask(masks[i]).reshape(518, 518)
                edgepos = (gt & (distance_transform_edt(gt) <= 14)).ravel()
                edgeneg = (~gt & (distance_transform_edt(~gt) <= 14)).ravel()
                r = dict(
                    dataset=n,
                    category=cat,
                    id=rs[i]["id"],
                    index=int(i),
                    foreground=int(gt.sum()),
                )
                for name, v in zip(SECTIONS, maps[k]):
                    r[name] = rank(v, gt.ravel(), ~gt.ravel())
                    if name in ("AH", "Final"):
                        r["boundary_" + name] = rank(v, edgepos, edgeneg)
                candidates.append(r)
        scores = {
            name: ordering(vals[0, k], vals[1, k]) for k, name in enumerate(SECTIONS)
        }
        for before, after in [
            ("Hoff", "H"),
            ("FinalOff", "Final"),
            ("A", "H"),
            ("H", "AH"),
            ("A", "AH"),
            ("AH", "Final"),
        ]:
            change = scores[after] - scores[before]
            lost = np.maximum(scores["A"] - scores["H"], 0)
            recovered = np.minimum(lost, np.maximum(scores["AH"] - scores["H"], 0))
            joint = recovered.sum() / lost.sum() if lost.sum() else np.nan
            # Delta method for the paired ratio of mean restored mass to mean lost mass.
            ratio_se = (
                (recovered - joint * lost).std(ddof=1) / (np.sqrt(N) * lost.mean())
                if lost.sum()
                else np.nan
            )
            stats.append(
                dict(
                    task="pixel",
                    dataset=n,
                    category=cat,
                    before=before,
                    after=after,
                    pairs=N,
                    corrected=np.maximum(change, 0).sum(),
                    broken=np.maximum(-change, 0).sum(),
                    delta_auc_pp=100 * change.mean(),
                    mc_se_pp=100 * change.std(ddof=1) / np.sqrt(N),
                    A_to_H_lost=lost.sum(),
                    AH_restored=recovered.sum(),
                    restored_fraction=joint,
                    ratio_mc_se=ratio_se,
                )
            )
        write(work / "case_pool.csv", candidates)
        write(work / "pair_stats.csv", stats)
        print(n, cat, "done", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        evaluate(n)
