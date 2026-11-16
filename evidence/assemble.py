"""Build evidence receipt, quantitative conclusions and a content inventory."""

from common import csv, json, ROOT, rt, ev, OUT, WORK, SEEDS, TARGETS, root, sha, write
from collections import Counter


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def run():
    if not (OUT / "FIG_LOCALIZATION_CANDIDATES.pdf").is_file():
        raise RuntimeError("Incomplete candidate figures")
    for name in ("FIG_LOCALIZATION", "FIG_DYNAMIC"):
        for ext in ("png", "pdf"):
            assert (OUT / (name + "." + ext)).stat().st_size > 50000
    a = read(OUT / "fig_localization_cases.csv")
    b = read(OUT / "fig_dynamic_pairs.csv")
    ids = [
        dict(
            panel="localization",
            event=r["event"],
            dataset=r["dataset"],
            category=r["category"],
            id=r["id"],
            paired_normal_id="",
            displayed=r["displayed"],
            screen=r["screen"],
        )
        for r in a
    ]
    ids.extend(
        dict(
            panel="dynamic",
            event=r["event"],
            dataset=r["dataset"],
            category=r["category"],
            id=r["anomaly_id"],
            paired_normal_id=r["normal_id"],
            displayed=r["displayed"],
            screen="within-category",
        )
        for r in b
    )
    if (OUT / "fig_localization_excluded.csv").is_file():
        ids.extend(
            dict(
                panel="localization",
                event="excluded",
                dataset=r["dataset"],
                category=r["category"],
                id=r["id"],
                paired_normal_id="",
                displayed=0,
                screen="failed native-miss recovery",
            )
            for r in read(OUT / "fig_localization_excluded.csv")
        )
    write(OUT / "CASE_SELECTION_IDS.csv", ids)
    ab = root("R1_AB_REPORT")
    gate = json.loads((WORK / "gate.json").read_text())
    cks = read(OUT / "checkpoint_receipts.csv")
    old = {
        (r["source"], r["seed"], r["name"]): r
        for r in read(ab / "checkpoint_audit.csv")
    }
    assert len(cks) == 18
    for r in cks:
        assert (
            old[
                r["source"],
                r["seed"],
                "main" if r["stage"] == "Dynamic" else r["stage"],
            ]["sha256"]
            == r["checkpoint_sha"]
        )
    profile = json.loads((OUT / "PROFILE.json").read_text())
    stages = json.loads((OUT / "PROFILE_STAGES.json").read_text())
    e = profile["environment"]
    timing = read(OUT / "COST.csv")
    err = read(OUT / "ERROR_ANALYSIS.csv")

    def pixel(n, b, a):
        return next(
            r
            for r in err
            if (r["dataset"], r["category"], r["before"], r["after"])
            == (n, "MACRO", b, a)
        )

    def ms(mode, b, metric):
        return float(
            next(
                r["value"]
                for r in timing
                if r["metric"] == metric
                and r["method"] == mode
                and r["batch"] == str(b)
            )
        )

    def total(stage):
        return (
            sum(
                float(r["fit_seconds"])
                for r in cks
                if stage == "all" or r["stage"] == stage
            )
            / 3600
        )

    diffs = read(OUT / "exact_domain_differences.csv")

    def exact(n, m):
        return float(
            next(
                r[m + "_pp"]
                for r in diffs
                if r["dataset"] == n and r["before"] == "A5" and r["after"] == "A4"
            )
        )

    counts = Counter(r["event"] for r in a)
    lines = [
        "# R1 最终补充材料",
        "",
        "主表采用三个固定成员的概率集成。MVTec Image/Pixel AUROC 为 93.897/91.513，VisA 为 83.747/95.381。核心域 Image AUROC/AP 为 86.300/89.062，Pixel AUROC/AP 为 93.781/46.709。各任务均按其 11 个有效域等权。",
        "",
        "|冻结干预或观察|MVTec AD|VisA|DAGM|",
        "|---|---:|---:|---:|",
    ]
    for title, vals in [
        (
            "完整部署中恢复 H 残余，正式 Pixel AP Δ（pp）",
            [exact(n, "pixel_ap") for n in TARGETS],
        ),
        (
            "完整部署中恢复 H 残余，正式 Pixel AUROC Δ（pp）",
            [exact(n, "pixel_auc") for n in TARGETS],
        ),
        (
            "A/H 挽回 A→H 损失的排序比例（类等权，抽样 %）",
            [100 * float(pixel(n, "A", "H")["restored_fraction"]) for n in TARGETS],
        ),
        (
            "开启 Dynamic，正式 Image AUROC Δ（pp）",
            [
                float(pixel(n, "Dynamic-off", "Dynamic-on")["delta_auc_pp"])
                for n in TARGETS
            ],
        ),
    ]:
        lines.append("|" + title + "|" + "|".join(f"{x:+.3f}" for x in vals) + "|")
    lines += [
        "",
        "1. H 能将原生读出低估的缺陷排到背景之前。Figure A 对所有方法使用同一概率色标。独立冻结消融中，核心 11 域恢复 H 后 Pixel AP 增加 3.685 pp。VisA 的 AP 和 AUROC 变化方向不同，DAGM 的 AP 大幅下降。单独启用 H 和在融合部署中启用 H 也有不同结果。",
        "2. 融合能部分挽回 H 损失的排序，上表给出同一批样本对的条件恢复率。该指标和原图同图背景的局部可视化都不能证明跨域误报率普遍下降。置信范围和明确分母见 ERROR_ANALYSIS。",
        "3. Dynamic 改变同类异常图与正常图的顺序。在 MVTec 纠正 315 对、破坏 302 对，类别等权后 AUROC 下降 0.050 pp。在 VisA 纠正 1,322 对、破坏 794 对。核心域净增 0.393 pp，正交项单独回放增加 0.389 pp。此前三个扩展分类域的平均 AUROC 下降 0.234 pp。图像分数变化不会改变最终定位图。",
        "4. 单成员核心 Image AUROC 为 86.013 ± 0.201，Pixel AUROC 为 93.752 ± 0.031，Pixel AP 为 46.550 ± 0.360（样本标准差）。集成相对三成员均值依次增加 0.287、0.029、0.159 pp。MVTec 和 VisA 的单成员 Image AUROC 标准差分别为 0.735 和 0.732 pp。",
    ]
    lines.append(
        f"5. 两源、三 seed 的 A、H 和 Dynamic 共 18 次历史 fit，合计 {total('all'):.3f} GPU-hours；其中 H 和 Dynamic 相对已有 A 的额外 fit 合计 {total('H') + total('Dynamic'):.3f} GPU-hours。单张 RTX 3090 的 batch 1 端到端中位数分别为静态 A {ms('static', 1, 'e2e median'):.2f}、完整单成员 {ms('single', 1, 'e2e median'):.2f}、三成员 {ms('ensemble', 1, 'e2e median'):.2f} ms。三个成员共享一次视觉编码。"
    )
    lines += [
        "",
        "源特征首次建立的时间另计；安装、排队和单独的校准耗时未取得完整记录。两张主图、候选 PDF 和含医疗数据标识的完整图像对只在受控工作目录交付。",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    inputs = []
    formal = [
        r for r in read(ROOT / "results/r1_metrics.csv") if r["protocol"] == "original"
    ]
    for r in formal:
        n = r["dataset"]
        cfg = rt.CONFIG["datasets"][n]
        for seed in SEEDS:
            verified = ev.REPLAY_HASHES[n][str(seed)]
            assert sha(root("R1_INPUTS") / n / "scores" / f"{seed}.npz") == verified
            inputs.append(
                dict(
                    dataset=n,
                    items=cfg["count"],
                    source=rt.source_for(n),
                    seed=seed,
                    official_score_sha256=verified,
                    ordered_ids_sha256=cfg["ordered_rows_sha256"],
                    packed_masks_sha256=cfg.get("packed_masks_sha256", "NA"),
                    ablation_ensemble_sha256=sha(root("R1_ABLATION") / n / "image.npz"),
                )
            )
    write(OUT / "inputs_sha256.csv", inputs)
    core = [n for n in formal if n["image_auc"] != "NA"]
    assert len(core) == 11
    sourcefiles = [
        ROOT / "README.md",
        ROOT / "release/runtime.py",
        ROOT / "release/evaluate.py",
        ROOT / "release/config.json",
        ROOT / "results/r1_metrics.csv",
        ROOT / "results/r1_per_class.csv",
        ab / "per_domain.csv",
        ab / "checkpoint_audit.csv",
        ab / "reconstruction.csv",
        ab / "donors.csv",
    ]
    write(
        OUT / "source_files_sha256.csv",
        [
            dict(
                file=str(p.relative_to(ROOT))
                if ROOT in p.parents
                else "prior ablation/" + p.name,
                sha256=sha(p),
            )
            for p in sourcefiles
        ],
    )
    dcount = sum(int(r["count"]) for r in read(OUT / "dynamic_candidate_counts.csv"))
    import gzip

    with gzip.open(OUT / "private/all_dynamic_candidates.csv.gz", "rt") as f:
        assert sum(1 for _ in csv.DictReader(f)) == dcount
    maxdelta = max(
        r["max_live_vs_official_image_abs"]
        for r in stages["observations"]
        if r["max_live_vs_official_image_abs"] is not None
    )
    audit = f"""# Evidence audit

Base: origin/main bd38e7b2d3d9768318f0b5dc99f5f350bfefa0b4 (R1 v1.0.0). Independent branch codex/r1-paper-evidence. Release code, weights, inputs and predictions were read-only. No fits, hyperparameter selection or new thresholds. Matched recovered ablation evidence by relative query ID and source weight provenance.

* The official replay of 14 core datasets agrees with the published class-level scores for all five requested seed metrics. Maximum difference {gate["max_core_class_error_pp"]:.12g} pp. Three-member probabilities and full-resolution maps are combined before official classwise scoring. Image and pixel tasks each have 11 valid domains. Release MVTec 93.897/91.513, VisA 83.747/95.381 and core 86.300/93.781 (Pixel AP 46.709) pass the rounded gates. Independent clean-environment replay outputs were reused for this check.
* All 18 final epoch-15 training checkpoints were SHA-checked against their receipts and the earlier ablation audit. Packaged H/gate tensors and text match the corresponding trained state. A and H prompt tensors and A/H/D effective text are identical within each source and seed. CLIP SHA and both package SHAs are verified. Only VisA targets use MVTec-trained weights; other targets use VisA-trained weights.
* 42 official member score files, original row-order digests, packed mask hashes and frozen intervention files are inventoried. Individual image metrics were recomputed from official scores. Individual pixel values use {gate["pixel_sources"]} archived class/member exact evaluator outputs for the same color-guided rule. Sampled pixel rankings and qualitative heatmaps were freshly recomputed from frozen member grids on MVTec/VisA/DAGM. For each retained primary/secondary example fresh heatmap AUC matched its saved ranking within 1e-12. In the secondary screen the first draft materialized a contiguous one-channel array before scoring. A rejected example differed by 3.13e-11 in AUROC. Replaying with the release two-channel batch and channel slice recovered the exact displayed result. Anomaly scores in fig_dynamic_pairs.csv are ensemble probabilities. Delta margins there are differences between ensemble logits; a member-level raw margin is a different quantity.
* Sampling units: exactly 100,000 pixel pairs/class × 37 classes; exact enumeration of all same-class image pairs on 11 datasets. ERROR_ANALYSIS gives both changes of ranking and exact domain-level AUROC differences where a matching frozen intervention exists. Pixel Monte Carlo errors quantify sampling of pairs on fixed queries. They do not cover new patients, images or training seeds.
* Figure A qualified counts: {dict(counts)}. The native-miss recovery criterion was met by just one of 3,507 nonempty-mask images. All three other below-chance A cases failed at least one recovery condition and are shown on the exclusions page. The 2–4 positive-example goal for this phenomenon cannot be met with these data. The primary screen is stored in localization_screened.csv. A secondary native-miss search, if used, preserves the original candidates and limits itself to fulfilling the requested number of illustrations. The archive contains {dcount:,} qualified Dynamic event/pair records, including orthogonal recoveries which can overlap the corrected category. Relative IDs and industrial pair scores are supplied in the compact public CSVs. Real-image figures and the complete pair list are withheld from Git pending redistribution clearance.
* Profiling: {e["gpu"]}, driver {e["driver"]}, PyTorch {e["torch"]}, CUDA {e["cuda"]}. 900 timed batches. TF32 disabled, deterministic, frozen visual tensors and official patch half round-trip. PROFILE_STAGES.json separately contains absolute stage peaks and the max live-vs-cached image difference on the eight timing queries ({maxdelta:.3g}); these checks include live encoding and batch-size effects, while reported paper scores use the official replay. Parameter and disk accounting uses loaded tensors and existing artifacts. Historical training times are logged fit wall-times. Initial source cache construction is recorded separately. System setup, queueing and standalone calibration times are unavailable.
"""
    (OUT / "AUDIT.md").write_text(audit)
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in sorted(paths))
    )


if __name__ == "__main__":
    run()
