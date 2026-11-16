"""Preserve correction and damage separately, including their dependence on H."""

from common import csv, np, OUT, WORK, TARGETS, root, write


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def run():
    values = read(WORK / "image_stats.csv")
    expected = {"mvtec": 15, "visa": 12, "dagm": 10}
    for n in TARGETS:
        rows = read(WORK / n / "pair_stats.csv")
        assert (
            len(rows) == 6 * expected[n]
            and len({v["category"] for v in rows}) == expected[n]
        )
        values.extend(rows)
    metrics = read(root("R1_AB_REPORT") / "per_domain.csv")
    lookup = {
        (r["dataset"], r["method"]): r for r in metrics if r["scope"] == "original"
    }
    groups = {}
    for r in values:
        key = r["task"], r["dataset"], r["before"], r["after"]
        groups.setdefault(key, []).append(r)

    def numeric(rows, k):
        return np.array([float(r[k]) for r in rows])

    cols = [
        "task",
        "dataset",
        "category",
        "before",
        "after",
        "pairs",
        "corrected",
        "broken",
        "delta_auc_pp",
        "mc_se_pp",
        "A_to_H_lost",
        "AH_restored",
        "restored_fraction",
        "ratio_mc_se",
        "exact_delta_auc_pp",
    ]
    data = [{k: r.get(k, "NA") for k in cols} for r in values]
    pack = {}
    for (task, n, b, a), rs in groups.items():
        summary = dict(task=task, dataset=n, category="MACRO", before=b, after=a)
        for k in ("pairs", "corrected", "broken"):
            summary[k] = sum(numeric(rs, k))
        summary["delta_auc_pp"] = numeric(rs, "delta_auc_pp").mean()
        summary["mc_se_pp"] = np.linalg.norm(numeric(rs, "mc_se_pp")) / len(rs)
        if task == "pixel":
            for k in ("A_to_H_lost", "AH_restored"):
                summary[k] = sum(numeric(rs, k))
            summary["restored_fraction"] = numeric(rs, "restored_fraction").mean()
            summary["ratio_mc_se"] = np.linalg.norm(numeric(rs, "ratio_mc_se")) / len(
                rs
            )
        comparisons = {
            ("FinalOff", "Final"): ("A5", "A4"),
            ("A", "AH"): ("A0", "A2"),
            ("AH", "Final"): ("A2", "A4"),
            ("Dynamic-off", "Dynamic-on"): ("A2", "A3"),
        }
        if (b, a) in comparisons:
            p, q = comparisons[b, a]
            k = "pixel_auc" if task == "pixel" else "image_auc"
            summary["exact_delta_auc_pp"] = float(lookup[n, q][k]) - float(
                lookup[n, p][k]
            )
            if task == "image":
                assert (
                    abs(summary["exact_delta_auc_pp"] - summary["delta_auc_pp"]) < 1e-10
                )
        data.append({k: summary.get(k, "NA") for k in cols})
        pack[task, n, b, a] = summary
    write(OUT / "ERROR_ANALYSIS.csv", data)
    txt = [
        "# Paired error analysis",
        "Image counts enumerate all comparable anomaly–normal pairs within class (half credit for ties). Class-macro increments reproduce the exact paired AUROC change. The counts sum across classes, so dividing the net count by all pairs can give a different number.",
        "|Dataset|Comparable image pairs|Corrected|Broken|Image AUROC change (pp)|",
        "|---|---:|---:|---:|---:|",
    ]
    for (task, n, b, a), r in pack.items():
        if task == "image":
            txt.append(
                f"|{n}|{r['pairs']:,.0f}|{r['corrected']:,.1f}|{r['broken']:,.1f}|{r['delta_auc_pp']:+.3f}|"
            )
    txt += [
        "",
        "We sampled 100,000 independently drawn foreground/background pairs with replacement per category at the full evaluation resolution (518 × 518), including normal-image background. Methods use identical coordinates. Intervals in this table measure Monte Carlo uncertainty only (delta ± 1.96 SE). The exact official point estimate is supplied wherever the archived ensemble intervention matches. All class-level counts are in the CSV. Standalone H-off retains its native calibration. Final-off retains fusion and color.",
        "|Dataset|Transition|Sample corrected|Sample broken|Mean Δ AUROC ± 1.96 MC SE (pp)|Exact Δ (pp)|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (task, n, b, a), r in pack.items():
        if task == "pixel":
            ex = r.get("exact_delta_auc_pp")
            ex = "—" if ex is None else f"{ex:+.3f}"
            txt.append(
                f"|{n}|{b} → {a}|{r['corrected']:,.1f}|{r['broken']:,.1f}|{r['delta_auc_pp']:+.3f} ± {1.96 * r['mc_se_pp']:.3f}|{ex}|"
            )
    txt += [
        "",
        "## Does fusion restore the ordering that H broke?",
        "",
        "The denominator is the loss of pairwise credit when changing A to H. The numerator credits only increases from H to plain A/H *on those same pairs*, capped at the original loss. The fraction is computed within class and macro-averaged. Raw sums in the next columns weight classes by the number of affected pairs.",
        "",
        "|Dataset|A→H credit lost in sample|Recovered by A/H|Class-mean recovery ± 1.96 MC SE|",
        "|---|---:|---:|---:|",
    ]
    for n in TARGETS:
        r = pack["pixel", n, "A", "H"]
        txt.append(
            f"|{n}|{r['A_to_H_lost']:,.1f}|{r['AH_restored']:,.1f}|{100 * r['restored_fraction']:.2f}% ± {196 * r['ratio_mc_se']:.2f} pp|"
        )
    txt += [
        "",
        "Residual evidence supports foreground/background reordering in some domains. The DAGM counterexample is visible in both the retained heatmaps and the formal Pixel AP comparison. Partial recovery gives no guarantee of improving on static A. These ranks do not estimate a detector operating-point false-positive rate. No threshold was fitted.",
        "",
        "The signed image effect describes Dynamic on the same A/H midpoint. Corrected and broken pairs coexist. Using a donor and retaining only the orthogonal component are interventions on the image readout. Neither produces a localization map.",
    ]
    (OUT / "ERROR_ANALYSIS.md").write_text("\n".join(txt) + "\n")
    extra = []
    for (n, m), r in lookup.items():
        if m != "A4":
            continue
        for p, q in [("A0", "A2"), ("A5", "A4"), ("A2", "A3")]:
            z = {"dataset": n, "before": p, "after": q}
            for metric in (
                "image_auc",
                "image_ap",
                "tpr",
                "pixel_auc",
                "pixel_ap",
                "aupro",
            ):
                v = lookup[n, p][metric]
                u = lookup[n, q][metric]
                z[metric + "_pp"] = (
                    float(u) - float(v) if u != "NA" and v != "NA" else "NA"
                )
            extra.append(z)
    write(OUT / "exact_domain_differences.csv", extra)


if __name__ == "__main__":
    run()
