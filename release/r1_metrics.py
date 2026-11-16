"""Exact class-level pixel statistics from the frozen benchmark."""

import ctypes
from functools import lru_cache

import numpy as np
import torch
from runtime import KERNELS
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PAGE = 1 << 20


def images(y, s):
    if len(np.unique(y)) < 2:
        return [None] * 3
    f, t, _ = roc_curve(y, s, drop_intermediate=False)
    return [
        float(roc_auc_score(y, s)),
        float(average_precision_score(y, s)),
        float(max(t[f <= 0.05])),
    ]


@lru_cache(maxsize=1)
def library():
    lib = ctypes.CDLL(str(KERNELS / "histogram.so"))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.add_pixels.argtypes = [ptr, ptr, ptr, integer, ptr, integer, integer, ptr, ptr]
    return lib


def reference(packed):
    truth = np.unpackbits(packed, axis=1, count=518**2).view(bool)
    weight = []
    counts = []
    structure = np.ones((3, 3), bool)
    for mask in truth:
        if mask.any():
            lab, k = label(mask.reshape(518, 518), structure=structure)
            values = lab.ravel()[mask]
            weight.append(1.0 / np.bincount(values)[values])
        else:
            k = 0
            weight.append(np.empty(0))
        counts.append(k)
    return truth, weight, np.array(counts)


def exact_grid(logit, packed, map_fn):
    truth, weight, regions = reference(packed)
    n = len(logit)
    assert n == len(truth)
    total_pos = sum(map(len, weight))
    total_neg = truth.size - total_pos
    # Interpolation and positive normalized blur cannot exceed input extrema.
    # Include a whole adjacent float page on each side for rounding at page edges.
    bounds = (
        torch.as_tensor([float(logit.min()), float(logit.max())])
        .sigmoid()
        .numpy()
        .view(np.uint32)
        // PAGE
    )
    low = max(0, int(bounds[0]) - 1)
    high = min(1016, int(bounds[1]) + 1)
    # Up to 3 GiB of exact float-key counters, independent of image count.
    # For constrained devices retain identical results with additional passes.
    free = (
        torch.cuda.mem_get_info()[0]
        + torch.cuda.memory_reserved()
        - torch.cuda.memory_allocated()
    )
    chunks = max(8, min(128, int((free - 1.75 * 2**30) / (PAGE * 16))))
    fg_below = bg_below = auc = ap = pro = 0.0
    for first in range(low, high + 1, chunks):
        pages = min(chunks, high - first + 1)
        bins = torch.zeros((pages * PAGE, 2), device="cuda", dtype=torch.double)
        for k in range(0, n, 16):
            q = map_fn(logit[k : k + 16])
            length = len(q)
            m = truth[k : k + length]
            w = np.zeros(m.shape, dtype=np.float64)
            for v in range(length):
                w[v, m[v]] = weight[k + v]
            m, w = (torch.as_tensor(a, device="cuda") for a in (m, w))
            library().add_pixels(
                q.data_ptr(),
                m.data_ptr(),
                w.data_ptr(),
                q.numel(),
                bins.data_ptr(),
                first,
                pages,
                0,
                torch.cuda.current_stream().cuda_stream,
            )
        # Only score bins with foreground mass enter AUROC, AP and PRO.
        for page in bins.split(4 * PAGE):
            ints = page.view(torch.int32).reshape(-1, 4)
            fg = ints[:, 0]
            bg = ints[:, 1]
            incl = torch.cumsum(bg, dim=0, dtype=torch.int64)
            select = fg.nonzero().flatten()
            if len(select):
                positive = fg[select].double()
                left = incl[select].double() + bg_below - bg[select]
                right = incl[select].double() + bg_below
                tp = total_pos - fg_below - positive.cumsum(0) + positive
                auc += float((positive * (left + right)).sum())
                ap += float((positive * tp / (tp + total_neg - left)).sum())
                if total_neg:
                    a = (total_neg - right) / total_neg
                    b = (total_neg - left) / total_neg
                    dx = (0.3 - a).clamp_min(0).minimum(b - a)
                    integ = (0.3 - b).clamp_min(0) + torch.where(
                        b > a, dx.square() / (2 * (b - a)), 0.0
                    )
                    pro += float((page[select, 1] * integ).sum())
                fg_below += int(positive.sum())
            bg_below += int(incl[-1])
        del bins
    assert (fg_below, bg_below) == (total_pos, total_neg), (
        fg_below,
        bg_below,
        total_pos,
        total_neg,
    )
    if not total_pos or not total_neg:
        return [None] * 3
    return [
        auc / (2 * total_pos * total_neg),
        ap / total_pos,
        pro / (0.3 * regions.sum()),
    ]


def evaluate_maps(m, packed):
    # Exact monotone recoding of zero only for the float-page counter. Does not
    # change any ranks/ties, connected-region weights, or 4096-bin memberships.
    lo, hi = float(m.min()), float(m.max())
    zero = 0.0
    if lo == 0:
        pos = min(
            float(np.min(b, where=b > 0, initial=np.float32(1)))
            for b in np.array_split(m, max(1, len(m) // 8))
        )
        candidate = float(np.float32(min(pos * 0.5, 1 / 8192)))
        if 0 < candidate < pos:
            zero = lo = candidate
            assert int(zero * 4096) == 0
    logit = lambda p: (
        float(np.log(p / (1 - p))) if 0 < p < 1 else (-120.0 if p == 0 else 120.0)
    )
    bounds = np.tile(np.array([logit(lo), logit(max(hi, lo))], np.float32), (len(m), 1))

    def loader(batch):
        ix = (batch.ctypes.data - bounds.ctypes.data) // bounds.strides[0]
        assert np.shares_memory(batch, bounds)
        q = torch.from_numpy(np.asarray(m[ix : ix + len(batch)])).cuda()
        return torch.where(q == 0, zero, q) if zero else q

    return exact_grid(bounds, packed, loader)
