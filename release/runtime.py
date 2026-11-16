"""Frozen three-member R1 inference and color-guided probability maps."""

import ctypes
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import cv2
import numpy as np
import torch
from reader import ReaderGate
from scipy.ndimage._filters import _gaussian_kernel1d
from scipy.special import expit, logit
from torch.nn import functional as F
from torchvision import transforms as T

PROJECT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("R1_WEIGHTS", PROJECT / "weights")).expanduser()
KERNELS = Path(os.environ.get("R1_KERNELS", PROJECT / "build")).expanduser()
VISUAL = Path(
    os.environ.get("R1_CLIP_WEIGHTS", Path.home() / ".cache/clip/ViT-L-14-336px.pt")
).expanduser()
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())

PARAMS = {
    "Prompt_length": 12,
    "learnabel_text_embedding_depth": 9,
    "learnabel_text_embedding_length": 4,
}

TRANS = T.Compose(
    [
        T.Resize((518, 518), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 21), b""):
            h.update(b)
    return h.hexdigest()


def clip():
    upstream = Path(
        os.getenv("R1_CLIP_CODE", PROJECT / "third_party/AnomalyCLIP")
    ).expanduser()
    if not (upstream / "AnomalyCLIP_lib").is_dir():
        raise FileNotFoundError(
            f"Install the pinned AnomalyCLIP source at {upstream} or set R1_CLIP_CODE"
        )
    sys.path.insert(0, str(upstream))
    import AnomalyCLIP_lib

    return (
        AnomalyCLIP_lib.load(str(VISUAL), device="cpu", design_details=PARAMS)[0]
        .eval()
        .requires_grad_(False)
    )


def configure():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def deterministic():
    # CUDA bilinear backward uses atomic additions by default. Its deterministic
    # implementation keeps the loss and dtype but fixes the reduction order.
    torch.use_deterministic_algorithms(True)


def encode(model, x):
    g, l = model.encode_image(x, [24], DPAM_layer=20)
    assert len(l) == 1 and l[0].shape[1:] == (1370, 768)
    g = F.normalize(g.float(), dim=-1)
    z = F.normalize(l[0][:, 1:].float(), dim=-1).half().float()
    return F.normalize(g, dim=-1), F.normalize(z, dim=-1)


def ph(cls, L, a):
    return (1 - a) * expit(cls) + a * expit(np.partition(L, -14, axis=1)[:, -14:]).mean(
        1
    )


def margin(prob):
    # Endpoints are finite precision equivalents only; report any observed endpoints.
    return logit(
        np.clip(prob.astype(np.float64), np.finfo(float).eps, 1 - np.finfo(float).eps)
    )


def score(p, d):
    return np.where(d == 0, p, expit(margin(p) + d))


def dynamic_margin(g, z, t, gate):
    shift = gate(z, t[1]) - F.normalize(t[1], dim=0)
    return (g * shift).sum(1) / 0.07


@lru_cache(maxsize=1)
def blur():
    lib = ctypes.CDLL(str(KERNELS / "blur.so"))
    lib.filter.argtypes = [ctypes.c_void_p] * 3 + [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib, torch.tensor(_gaussian_kernel1d(4, 0, 16)[16:].copy(), device="cuda")


def maps(logit):
    lib, w = blur()
    p = (
        F.interpolate(
            torch.as_tensor(logit).sigmoid().reshape(-1, 1, 37, 37),
            (518, 518),
            mode="bilinear",
            align_corners=False,
        )
        .cuda()
        .contiguous()
    )
    q = torch.empty_like(p)
    t = torch.empty_like(p)
    lib.filter(
        p.data_ptr(),
        t.data_ptr(),
        q.data_ptr(),
        p.numel(),
        w.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    return q.reshape(-1, 518 * 518)


def mix_maps(x):
    return 0.5 * (maps(x[:, 0]) + maps(x[:, 1]))


def colors(x):
    return F.avg_pool2d(x, 14, 14)


def geometry():
    # Read the actual float32 half-pixel kernel from the same CPU evaluator.
    # Vertically/horizontally replicated border positions carry a single weight.
    eye = torch.eye(37, dtype=torch.float32).reshape(1, 37, 1, 37)
    basis = F.interpolate(eye, (1, 518), mode="bilinear", align_corners=False)[0, :, 0]
    weights, index = basis.topk(2, dim=0)
    assert torch.allclose(weights.sum(0), torch.ones(518), atol=1e-7, rtol=0)
    weights = weights.cuda()
    index = index.cuda()
    dest = []
    for iy, wy in zip(index, weights):
        for ix, wx in zip(index, weights):
            dest.append(
                ((37 * iy[:, None] + ix).flatten(), (wy[:, None] * wx).flatten())
            )
    return dest


def smooth(y):
    lib, w = blur()
    x = y.contiguous().reshape(-1, 1, 518, 518)
    tmp = torch.empty_like(x)
    q = torch.empty_like(x)
    lib.filter(
        x.data_ptr(),
        tmp.data_ptr(),
        q.data_ptr(),
        x.numel(),
        w.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    return q.reshape(-1, 2, 518 * 518)


def guided(grids, x, band, geo):
    mean = colors(x).flatten(2)
    lab = x.flatten(2)
    L = torch.as_tensor(grids, device="cuda").sigmoid()
    logs = []
    for ix, weight in geo:
        dist = (mean[:, :, ix] - lab).square().sum(1)
        logs.append(weight.log() - dist / (2 * band**2))
    w = torch.stack(logs, 1).softmax(1)
    out = sum(w[:, j, None, :] * L[:, :, ix] for j, (ix, _) in enumerate(geo))
    return smooth(out).mean(1)


def color_base(grids, lab, band, geometry):
    return 0.5 * (mix_maps(grids) + guided(grids, lab, band, geometry))


def lab_batch(images):
    lab = np.stack(
        [
            cv2.cvtColor(np.asarray(im, np.float32) / 255, cv2.COLOR_RGB2LAB)
            for im in images
        ]
    )
    return torch.from_numpy(lab).permute(0, 3, 1, 2).cuda()


def source_for(target):
    return "mvtec" if target == "visa" else "visa"


class Member:
    def __init__(self, state):
        self.text = state["text"].cuda()
        self.alpha_a, self.alpha_h = state["alpha_a"], state["alpha_h"]
        self.proto, self.mix, self.coef = (
            state[k].cuda() for k in ("proto", "mix", "coef")
        )
        self.gate = ReaderGate().cuda().eval()
        self.gate.load_state_dict(state["gate"])

    def score_tokens(self, cls, patches):
        direction = self.text[1] - self.text[0]
        native = patches @ (direction / 0.07)
        image = cls @ (direction / 0.07)
        axis = F.normalize(direction, dim=0)

        def orth(x):
            return x - (x @ axis)[..., None] * axis

        proto = F.normalize(orth(self.proto), dim=-1)
        sim = (orth(patches) @ proto.flatten(0, 1).T / 0.07).reshape(
            *patches.shape[:2], 2, 4
        )
        w = self.mix.expand(len(cls), -1, -1).log_softmax(-1)
        residual = (sim * w.exp()[:, None]).sum(-1).diff(dim=-1).squeeze(-1)
        beta, gamma, b = self.coef.unbind()
        local = (beta * native + gamma * residual + b).cpu().numpy()
        native = native.cpu().numpy()
        image = image.cpu().numpy()
        delta = dynamic_margin(cls, patches, self.text, self.gate).cpu().numpy()
        middle = 0.5 * ph(image, native, self.alpha_a) + 0.5 * ph(
            image, local, self.alpha_h
        )
        return score(middle, delta), np.stack((native, local), 1)


class Ensemble:
    def __init__(self, source):
        configure()
        deterministic()
        path = WEIGHTS / f"{source}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Extract the R1 reader archive to {WEIGHTS} or set R1_WEIGHTS"
            )
        if sha(path) != CONFIG["sources"][source]["weights_sha256"]:
            raise ValueError(f"Reader checkpoint SHA mismatch: {path}")
        states = torch.load(path, map_location="cpu", weights_only=True)
        self.members = [Member(states[str(s)]) for s in (111, 222, 333)]
        self.band = CONFIG["sources"][source]["band"]
        self.geo = geometry()
        self.backbone = None

    @torch.inference_mode()
    def predict(self, images):
        if self.backbone is None:
            visual = VISUAL
            if sha(visual) != CONFIG["clip_sha"]:
                raise ValueError(f"Visual checkpoint SHA mismatch: {visual}")
            self.backbone = clip()
            self.backbone.visual.DAPM_replace(DPAM_layer=20)
            self.backbone.cuda()
        resized = [TRANS.transforms[0](im.convert("RGB")) for im in images]
        tensor = torch.stack(
            [TRANS.transforms[2](TRANS.transforms[1](im)) for im in resized]
        ).cuda()
        g, z = encode(self.backbone, tensor)
        return self.predict_tokens(g, z, resized)

    @torch.inference_mode()
    def predict_tokens(self, g, z, resized):
        pairs = [member.score_tokens(g, z) for member in self.members]
        probs, grids = zip(*pairs)
        lab = lab_batch(resized)
        maps = [color_base(grid, lab, self.band, self.geo) for grid in grids]
        return np.mean(probs, axis=0, dtype=np.float64), torch.stack(maps).mean(0)
