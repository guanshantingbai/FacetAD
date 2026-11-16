import gc
import time
import subprocess
from common import np, rt, Image, OUT, WORK, SEEDS, sha, jsonout, data
import torch


def stamp():
    torch.cuda.synchronize()
    return time.perf_counter_ns()


@torch.inference_mode()
def main():
    rt.configure()
    rt.deterministic()
    cfg = rt.CONFIG["sources"]["visa"]
    path = rt.WEIGHTS / "visa.pt"
    assert (
        sha(rt.VISUAL) == rt.CONFIG["clip_sha"] and sha(path) == cfg["weights_sha256"]
    )
    states = torch.load(path, map_location="cpu", weights_only=True)
    model = rt.clip()
    count = lambda m: sum(p.numel() for p in m.parameters())
    counts = {
        "clip_before_dpam": count(model),
        "visual_before_dpam": count(model.visual),
    }
    model.visual.DAPM_replace(DPAM_layer=20)
    model.cuda().eval()
    counts.update(clip_after_dpam=count(model), visual_after_dpam=count(model.visual))
    rs, docs, _ = data("mvtec")
    geo = rt.geometry()
    seeds = list(SEEDS)
    env = dict(
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        .decode()
        .splitlines()[0],
        model_dtype=str(next(model.visual.parameters()).dtype),
        tf32=False,
        deterministic=True,
        ids=[r["id"] for r in rs[:8]],
        weights=cfg["weights_sha256"],
        counts=counts,
        host_to_device=True,
        thermal="warm",
        warmup=5,
        rounds=3,
        samples_per_round=50,
    )
    obs = []
    for rnd in range(3):
        for batch in (1, 8):
            modes = ["static", "single", "ensemble"]
            for mode in modes[rnd:] + modes[:rnd]:
                members = (
                    []
                    if mode == "static"
                    else [
                        rt.Member(states[str(s)])
                        for s in (seeds if mode == "ensemble" else seeds[:1])
                    ]
                )
                text = states["111"]["text"].cuda()
                alpha = states["111"]["alpha_a"]
                for j in range(-5, 50):
                    ids = [(max(j, 0) * batch + i) % 8 for i in range(batch)]
                    if j == 0:
                        torch.cuda.reset_peak_memory_stats()
                    t0 = stamp()
                    res = []
                    for i in ids:
                        with Image.open(docs[i]["image"]) as im:
                            res.append(rt.TRANS.transforms[0](im.convert("RGB")))
                    x = torch.stack(
                        [rt.TRANS.transforms[2](rt.TRANS.transforms[1](v)) for v in res]
                    ).cuda()
                    t1 = stamp()
                    g, z = rt.encode(model, x)
                    t2 = stamp()
                    if mode == "static":
                        axis = (text[1] - text[0]) / 0.07
                        native = (z @ axis).cpu().numpy()
                        prob = rt.ph((g @ axis).cpu().numpy(), native, alpha)
                    else:
                        pairs = [m.score_tokens(g, z) for m in members]
                        probs, grids = zip(*pairs)
                        prob = np.mean(probs, axis=0, dtype=np.float64)
                    t3 = stamp()
                    if mode == "static":
                        p = rt.maps(native)
                    else:
                        lab = rt.lab_batch(res)
                        p = torch.stack(
                            [
                                rt.color_base(grid, lab, cfg["band"], geo)
                                for grid in grids
                            ]
                        ).mean(0)
                    px = p.cpu().numpy()
                    t4 = stamp()
                    assert len(prob) == len(px) == batch
                    if j >= 0:
                        obs.append(
                            dict(
                                round=rnd + 1,
                                batch=batch,
                                method=mode,
                                iteration=j,
                                pre=(t1 - t0) / 1e6,
                                encoder=(t2 - t1) / 1e6,
                                reader=(t3 - t2) / 1e6,
                                post=(t4 - t3) / 1e6,
                                forward=(t3 - t1) / 1e6,
                                e2e=(t4 - t0) / 1e6,
                                peak_bytes=torch.cuda.max_memory_allocated(),
                            )
                        )
                print(rnd + 1, batch, mode, flush=True)
                del members, x, g, z, p, px, text
                gc.collect()
                torch.cuda.empty_cache()
                jsonout(
                    WORK / "profile_progress.json",
                    dict(environment=env, observations=obs),
                )
    jsonout(OUT / "PROFILE.json", dict(environment=env, observations=obs))


if __name__ == "__main__":
    main()
