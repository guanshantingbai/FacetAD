"""Exact tie-aware within-class ranking changes. Export patient-bearing candidates privately."""

import gzip
from scipy.special import logit
from common import csv, np, ROOT, ev, OUT, WORK, SEEDS, TARGETS, root, sha, write, data


def run():
    private = OUT / "private"
    private.mkdir(exist_ok=True, mode=0o700)
    selected = {}
    stats = []
    eligible_counts = {}
    baseline = {}
    with gzip.open(private / "all_dynamic_candidates.csv.gz", "wt", newline="") as fd:
        writer = None
        for ds in csv.DictReader((ROOT / "results/r1_metrics.csv").open()):
            n = ds["dataset"]
            if ds["protocol"] != "original" or ds["image_auc"] == "NA":
                continue
            records, doc, _ = data(n)
            f = np.load(root("R1_ABLATION") / n / "image.npz")
            ids = [r["id"] for r in records]
            np.testing.assert_array_equal(f["ids"], ids)
            methods = {"off": "A2", "on": "A3", "orth": "D4", "donor": "D2"}
            probs = {k: f[v] for k, v in methods.items()}
            labels = np.array([r["y"] for r in records])
            original = []
            for s in SEEDS:
                p = root("R1_INPUTS") / n / "scores" / f"{s}.npz"
                assert sha(p) == ev.REPLAY_HASHES[n][str(s)]
                with np.load(p) as pack:
                    original.append(pack["image"])
            np.testing.assert_array_equal(
                np.mean(original, axis=0, dtype=np.float64), probs["on"]
            )
            auc = []
            for cat in sorted({r["category"] for r in records}):
                ix = np.array(
                    [i for i, r in enumerate(records) if r["category"] == cat]
                )
                pos = ix[labels[ix] == 1]
                neg = ix[labels[ix] == 0]
                if not len(pos) * len(neg):
                    continue
                correct = broken = 0.0
                aucsum = 0.0
                for pi in pos:
                    gaps = {k: v[pi] - v[neg] for k, v in probs.items()}
                    val = {
                        k: (v > 0).astype(float) + 0.5 * (v == 0)
                        for k, v in gaps.items()
                    }
                    change = val["on"] - val["off"]
                    correct += np.maximum(change, 0).sum()
                    broken += np.maximum(-change, 0).sum()
                    aucsum += val["on"].sum()
                    corrected = (gaps["off"] < 0) & (gaps["on"] > 0)
                    flags = dict(
                        correct=corrected,
                        breaks=(gaps["off"] > 0) & (gaps["on"] < 0),
                        orth=corrected & (gaps["orth"] > 0),
                    )
                    for typ, mask in flags.items():
                        for j in np.flatnonzero(mask):
                            ni = neg[j]
                            priority = (
                                abs(gaps["on"][j] - gaps["off"][j])
                                if typ != "orth"
                                else -abs(gaps["on"][j] - gaps["orth"][j])
                            )
                            row = dict(
                                event=typ,
                                dataset=n,
                                category=cat,
                                anomaly_id=ids[pi],
                                normal_id=ids[ni],
                                anomaly_label=1,
                                normal_label=0,
                                priority=priority,
                            )
                            for name, v in probs.items():
                                for which, i in [("anomaly", pi), ("normal", ni)]:
                                    row[name + "_" + which] = v[i]
                                    row[name + "_dmargin_" + which] = logit(
                                        v[i]
                                    ) - logit(probs["off"][i])
                            if writer is None:
                                writer = csv.DictWriter(fd, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            eligible_counts[n, typ] = (
                                eligible_counts.get((n, typ), 0) + 1
                            )
                            if n in TARGETS:
                                entries = selected.setdefault((n, typ), [])
                                entries.append(row)
                                entries.sort(
                                    key=lambda r: (
                                        -r["priority"],
                                        r["category"],
                                        r["anomaly_id"],
                                        r["normal_id"],
                                    )
                                )
                                del entries[4:]
                pairs = len(pos) * len(neg)
                auc.append(100 * aucsum / pairs)
                stats.append(
                    dict(
                        task="image",
                        dataset=n,
                        category=cat,
                        before="Dynamic-off",
                        after="Dynamic-on",
                        pairs=pairs,
                        corrected=correct,
                        broken=broken,
                        delta_auc_pp=100 * (correct - broken) / pairs,
                        mc_se_pp=0,
                    )
                )
            baseline[n] = np.mean(auc)
            assert abs(baseline[n] - float(ds["image_auc"])) < 1e-9
    write(WORK / "image_stats.csv", stats)
    write(
        OUT / "fig_dynamic_pairs.csv",
        [r for k in sorted(selected) for r in selected[k]],
    )
    write(
        OUT / "dynamic_candidate_counts.csv",
        [
            dict(dataset=k[0], event=k[1], count=v)
            for k, v in sorted(eligible_counts.items())
        ],
    )


if __name__ == "__main__":
    run()
