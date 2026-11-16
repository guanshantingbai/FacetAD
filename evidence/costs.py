"""Combine actual frozen-model profiling with source-fit receipts."""

from common import json, np, rt, OUT, SEEDS, root, sha, write
import torch


def run():
    prof = json.loads((OUT / "PROFILE.json").read_text())
    obs = prof["observations"]
    env = prof["environment"]
    loaded_clip = env["counts"]["clip_after_dpam"]
    phase = json.loads((OUT / "PROFILE_STAGES.json").read_text())
    clip = sum(phase["precision"].values())
    assert clip - loaded_clip == (37**2 - 24**2) * 1024
    costs = []
    fits = []
    lookup = {}

    def row(section, method, metric, value, unit, source="", seed="", batch=""):
        costs.append(
            dict(
                section=section,
                method=method,
                source=source,
                seed=seed,
                batch=batch,
                metric=metric,
                value=value,
                unit=unit,
            )
        )

    row(
        "parameters",
        "shared CLIP",
        "weight-loaded frozen incl text",
        loaded_clip,
        "parameters",
    )
    row(
        "parameters",
        "shared CLIP",
        "visual only",
        env["counts"]["visual_after_dpam"],
        "parameters",
    )
    row(
        "disk", "shared CLIP", "release archive size", rt.VISUAL.stat().st_size, "bytes"
    )
    for src in ("visa", "mvtec"):
        cache = json.loads((root("R1_CACHE") / src / "ready.json").read_text())
        for k, u in [("wall_seconds", "s"), ("cuda_peak_allocated_bytes", "bytes")]:
            row("source cache", "one-time visual encoding", k, cache[k], u, src)
        pack = rt.WEIGHTS / (src + ".pt")
        assert sha(pack) == rt.CONFIG["sources"][src]["weights_sha256"]
        states = torch.load(pack, map_location="cpu", weights_only=True)
        row(
            "disk",
            "three-member release",
            "source package",
            pack.stat().st_size,
            "bytes",
            src,
        )
        for seed in SEEDS:
            state = states[str(seed)]
            weights = {}
            for method, project, arm in [
                ("A", "top24-protected-host", "A"),
                ("H", "top24-protected-host", "H"),
                ("Dynamic", "top24-fused-dynamic", "main"),
            ]:
                folder = root("R1_RUNS") / project / "fit" / src / str(seed) / arm
                p = folder / "final.pt"
                receipt = json.loads((folder / "done.json").read_text())
                c = torch.load(p, map_location="cpu", weights_only=False)
                assert sha(p) == receipt["sha256"] and (
                    c["epoch"],
                    c["seed"],
                    c["source"],
                    c["arm"],
                ) == (15, seed, src, arm)
                weights[method] = c
                fits.append(
                    dict(
                        source=src,
                        seed=seed,
                        stage=method,
                        checkpoint_sha=receipt["sha256"],
                        epoch=15,
                        bytes=p.stat().st_size,
                        fit_seconds=receipt["seconds"],
                        peak_allocated_bytes=receipt["peak"],
                    )
                )
                for key, val, unit in [
                    ("fit time", receipt["seconds"], "s"),
                    ("one GPU allocation", receipt["seconds"] / 3600, "GPU h"),
                    ("peak allocated", receipt["peak"], "bytes"),
                    ("archived checkpoint size", p.stat().st_size, "bytes"),
                ]:
                    row("fit", method, key, val, unit, src, seed)
            for m in weights:
                assert torch.equal(weights[m]["text"], state["text"])
            for k, v in state["gate"].items():
                assert torch.equal(weights["Dynamic"]["gate"][k], v)
            for k, v in weights["H"]["head"].items():
                assert torch.equal(v, state[k])
            for k, v in weights["A"]["prompt"].items():
                assert torch.equal(v, weights["H"]["prompt"][k])
            assert weights["Dynamic"]["H_sha"] == fits[-2]["checkpoint_sha"]
            for s in ("A", "H"):
                assert (
                    state["alpha_" + s.lower()] == weights[s]["calibration"]["fusion"]
                )
            p = weights["A"]["prompt"]
            effective = sum(
                t.numel()
                for k, t in p.items()
                if k.startswith(("ctx_", "compound_prompts_text"))
            )
            nominal = effective + sum(
                t.numel()
                for k, t in p.items()
                if k.startswith("compound_prompt_projections")
            )
            h = sum(t.numel() for t in weights["H"]["head"].values())
            d = sum(t.numel() for t in state["gate"].values())
            t = state["text"].numel()
    for name, val in [
        ("prompt effective", effective),
        ("prompt incl unused projections", nominal),
        ("H", h),
        ("Dynamic", d),
        ("one member distinct effective learned parameters", effective + h + d),
        ("one member staged effective parameter slots", 2 * effective + h + d),
        ("one member deployed readout incl baked text", t + h + d),
        ("one member deployed CLIP plus readout", clip + t + h + d),
        ("three members deployed CLIP plus readouts", clip + 3 * (t + h + d)),
        ("three members shared backbone", clip),
        ("deployment updated parameters", 0),
    ]:
        row("parameters", name, "count", val, "parameters")
    n = 37**2
    p = 518**2
    dim = 768
    native = (n + 1) * dim
    hidden = native + 2 * (n + 8) * dim + 8 * n * dim + 8 * n
    attn = (2 * n + 3) * dim**2 + 2 * n * dim + dim
    plain = (4 + 2 * 17) * p
    guided = 4 * plain + 12 * p
    for mode, m in [
        ("static", native + plain),
        ("single", native + hidden + attn + guided),
        ("ensemble", 3 * (native + hidden + attn + guided)),
    ]:
        row(
            "compute",
            mode,
            "counted MACs incl shared encoder",
            588990935040 + m,
            "MAC/image",
        )
        row(
            "compute",
            mode,
            "additional vs single static",
            m - (native + plain),
            "MAC/image",
        )
    for b in (1, 8):
        for mode in ("static", "single", "ensemble"):
            samples = [o for o in obs if o["batch"] == b and o["method"] == mode]
            assert all(sum(o["round"] == r for o in samples) == 50 for r in (1, 2, 3))
            v = [
                np.median([o[k] for o in samples])
                for k in ("pre", "encoder", "reader", "post", "forward", "e2e")
            ]
            peak = max(o["peak_bytes"] for o in samples)
            lookup[b, mode] = [*v, peak / 2**30]
            for k, x in zip(("pre", "encoder", "reader", "post", "forward", "e2e"), v):
                row("inference", mode, k + " median", x, "ms/batch", batch=b)
            row("inference", mode, "peak allocated", peak, "bytes", batch=b)
    phase = json.loads((OUT / "PROFILE_STAGES.json").read_text())
    for measurement in phase["observations"]:
        for stage, size in measurement["absolute_stage_peak_bytes"].items():
            row(
                "stage memory",
                measurement["method"],
                stage + " absolute peak allocated",
                size,
                "bytes",
                batch=measurement["batch"],
            )
    write(OUT / "checkpoint_receipts.csv", fits)
    write(OUT / "COST.csv", costs)
    header = "|Batch|Detector|Decode, resize, H2D|Encoder|Readout|Heatmap + output|Forward|End-to-end|Peak GiB|"
    body = [
        "|" + str(b) + "|" + m + "|" + "|".join(f"{x:.2f}" for x in v) + "|"
        for (b, m), v in lookup.items()
    ]
    note = f"""# Frozen inference and historical training cost

{env["gpu"]}, driver {env["driver"]}, PyTorch {env["torch"]}, CUDA {env["cuda"]}. The eight fixed images and raw samples are recorded in PROFILE.json. One visual encoding per batch, one source package. Synchronized warm-cache medians in milliseconds per **batch**, 3 rounds × 50 observations, after 5 warm-ups per cell/round. Forward covers encoder and score computation, including score transfer and host arithmetic required by the release. End-to-end starts at image-file open, includes full heatmap recovery, and ends at CPU outputs. It excludes model loading. Peak is the maximum allocated GPU memory over these observations. Median stage times need not sum to the median end-to-end time.

{header}
|---:|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(body)
    note += "\n\nAbsolute per-stage allocation peaks (GiB, same GPU, separate eight-image pass). Resident model weights are included in each stage.\n\n|Batch|Detector|Input|Encoder|Readout|Postprocess|\n|---:|---|---:|---:|---:|---:|\n"
    for rec in phase["observations"]:
        note += (
            "|"
            + str(rec["batch"])
            + "|"
            + rec["method"]
            + "|"
            + "|".join(
                f"{v / 2**30:.2f}" for v in rec["absolute_stage_peak_bytes"].values()
            )
            + "|\n"
        )
    note += f"""

At weight load, the shared CLIP contains {loaded_clip:,} parameters ({env["counts"]["visual_after_dpam"]:,} visual). On the first 518-pixel input, the pinned visual encoder interpolates its 24×24 position table to 37×37 and stores the expanded table. The resident frozen CLIP then contains {clip:,} values, without learning additional parameters. The source-trained prompt has {effective:,} effective parameter slots, or {nominal:,} including the unused projections stored by PromptLearner. H has {h:,} and Dynamic {d:,}. Per member this is {effective + h + d:,} distinct effective learned parameters. A and H trained separate matching prompt copies, so the sum of active slots over three sequentially costed fits is {2 * effective + h + d:,}. Deployment bakes the text into {t:,} values. One complete deployment contains {clip + t + h + d:,} model values, three members {clip + 3 * (t + h + d):,}, including the shared encoder exactly once. Relative to static A, the single full readout adds {hidden + attn + guided - plain:,} estimated MACs per image. Two further complete readouts add {2 * (native + hidden + attn + guided):,} MACs to a single full member. Each three-member source package occupies {pack.stat().st_size:,} bytes for {src}; exact sizes for both sources and all training checkpoints are itemized in COST.csv. The small fixed scalar calibration metadata is excluded. Inference updates zero parameters.

|Source|Stage|Mean min per seed|Observed peak GiB (max)|Three seeds GPU-hours|
|---|---|---:|---:|---:|
"""
    for src in ("visa", "mvtec"):
        for stage in ("A", "H", "Dynamic", "A+H+Dynamic"):
            fs = [
                f
                for f in fits
                if f["source"] == src
                and (f["stage"] == stage or stage == "A+H+Dynamic")
            ]
            total = sum(f["fit_seconds"] for f in fs)
            note += f"|{src}|{stage}|{total / 3 / 60:.2f}|{max(f['peak_allocated_bytes'] for f in fs) / 2**30:.2f}|{total / 3600:.3f}|\n"
    note += (
        "\nGPU-hours sum one allocated GPU per logged fit. The complete two-source fit budget is "
        + f"{sum(f['fit_seconds'] for f in fits) / 3600:.3f} GPU-hours. "
        + "One-time source visual cache construction is itemized in COST.csv. Setup, queueing, independent calibration and complete from-scratch pipeline wall-clock are unavailable. Peak allocations across stages are not additive.\n\nMACs combine an archived operator count of 588,990,935,040 for the shared visual forward with analytic counts for dense readouts, interpolation, neighborhood distance and smoothing. Exponentials, sorting, scalar operations and transfers are excluded. Parameter and disk counts are exact.\n"
    )
    (OUT / "COST.md").write_text(note)
    tex = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Batch & Method & Encoder & Forward & End-to-end & Peak GiB \\",
        r"\midrule",
    ]
    for (b, m), v in lookup.items():
        tex.append(
            f"{b} & {m} & {v[1]:.2f} & {v[4]:.2f} & {v[5]:.2f} & {v[6]:.2f} " + r"\\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Source & Stage & Min/seed & Peak GiB & GPU-hours (3 fits) \\",
        r"\midrule",
    ]
    for src in ("visa", "mvtec"):
        for stage in ("A", "H", "Dynamic", "A+H+Dynamic"):
            fs = [
                f
                for f in fits
                if f["source"] == src
                and (f["stage"] == stage or stage == "A+H+Dynamic")
            ]
            total = sum(f["fit_seconds"] for f in fs)
            tex.append(
                f"{src} & {stage} & {total / 180:.2f} & {max(f['peak_allocated_bytes'] for f in fs) / 2**30:.2f} & {total / 3600:.3f} "
                + r"\\"
            )
    (OUT / "COST.tex").write_text(
        "\n".join(tex + [r"\bottomrule", r"\end{tabular}"]) + "\n"
    )


if __name__ == "__main__":
    run()
