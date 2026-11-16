"""One additional synchronized eight-ID pass for per-stage allocation peaks and output checks."""

import gc
from common import np, rt, OUT, SEEDS, root, jsonout, data, rgb
import torch


@torch.inference_mode()
def run():
    rt.configure()
    rt.deterministic()
    model = rt.clip()
    model.visual.DAPM_replace(DPAM_layer=20)
    model.cuda().eval()
    cfg = rt.CONFIG["sources"]["visa"]
    state = torch.load(rt.WEIGHTS / "visa.pt", map_location="cpu", weights_only=True)
    rs, doc, _ = data("mvtec")
    geo = rt.geometry()
    exact = np.load(root("R1_ABLATION") / "mvtec/image.npz")
    refs = [
        np.load(root("R1_INPUTS") / "mvtec/scores" / f"{s}.npz")["image"][:8]
        for s in SEEDS
    ]
    results = []

    def begin():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def peak():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    for b in (1, 8):
        for mode in ("static", "single", "ensemble"):

            def once():
                members = (
                    [
                        rt.Member(state[str(s)])
                        for s in (SEEDS if mode == "ensemble" else SEEDS[:1])
                    ]
                    if mode != "static"
                    else []
                )
                text = state["111"]["text"].cuda()
                peaks = []
                errs = []
                finite = True
                for j in range(0, 8, b):
                    begin()
                    ids = list(range(j, j + b))
                    ims = [rgb(rs[i], doc[i]) for i in ids]
                    x = torch.stack(
                        [
                            rt.TRANS.transforms[2](rt.TRANS.transforms[1](im))
                            for im in ims
                        ]
                    ).cuda()
                    pre = peak()
                    begin()
                    g, z = rt.encode(model, x)
                    enc = peak()
                    begin()
                    if mode == "static":
                        axis = (text[1] - text[0]) / 0.07
                        v = (z @ axis).cpu().numpy()
                        p = rt.ph((g @ axis).cpu().numpy(), v, state["111"]["alpha_a"])
                        ref = None
                    else:
                        allscores = [m.score_tokens(g, z) for m in members]
                        p = np.mean([d[0] for d in allscores], axis=0, dtype=np.float64)
                        ref = exact["A3"][ids] if mode == "ensemble" else refs[0][ids]
                    rd = peak()
                    begin()
                    if mode == "static":
                        out = rt.maps(v)
                    else:
                        lab = rt.lab_batch(ims)
                        out = torch.stack(
                            [
                                rt.color_base(d[1], lab, cfg["band"], geo)
                                for d in allscores
                            ]
                        ).mean(0)
                    pix = out.cpu().numpy()
                    pt = peak()
                    peaks.append([pre, enc, rd, pt])
                    finite = finite and np.isfinite(pix).all() and np.isfinite(p).all()
                    if ref is not None:
                        errs.append(float(np.abs(p - ref).max()))
                assert finite
                return dict(
                    batch=b,
                    method=mode,
                    absolute_stage_peak_bytes=dict(
                        zip(
                            ["pre", "encoder", "reader", "post"],
                            map(int, np.max(peaks, axis=0)),
                        )
                    ),
                    max_live_vs_official_image_abs=max(errs) if errs else None,
                )

            results.append(once())
            gc.collect()
            torch.cuda.empty_cache()
    jsonout(
        OUT / "PROFILE_STAGES.json",
        dict(
            gpu=torch.cuda.get_device_name(),
            precision={
                str(x.dtype): sum(
                    p.numel() for p in model.parameters() if p.dtype == x.dtype
                )
                for x in model.parameters()
            },
            observations=results,
        ),
    )


if __name__ == "__main__":
    run()
