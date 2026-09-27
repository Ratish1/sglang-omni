"""Reproduce flashinfer 0.6.18's CuTe rmsnorm tree in Triton: lane t holds columns 8t+256b+j (b in 0..3, j in 0..7); per-lane fold (variants); butterfly offsets 1,2,4,8,16."""

import torch
import triton
import triton.language as tl
from sgl_kernel import rmsnorm
from triton.language.extra import libdevice

dev = "cuda"


@triton.jit
def butterfly_up(v):  # v: [32] lane partials; pair i with i^1, then i^2, i^4, i^8, i^16
    v = tl.sum(tl.reshape(v, [16, 2]), 1)
    v = tl.sum(tl.reshape(v, [8, 2]), 1)
    v = tl.sum(tl.reshape(v, [4, 2]), 1)
    v = tl.sum(tl.reshape(v, [2, 2]), 1)
    return tl.sum(v, 0)


@triton.jit
def kern(
    X, R, W, OUT, ROUT, eps, D: tl.constexpr, FOLD: tl.constexpr, ROUND: tl.constexpr
):
    row = tl.program_id(0)
    t = tl.arange(0, 32)
    acc = tl.zeros([32], dtype=tl.float32)
    if FOLD == 0:  # sequential, b outer, j inner
        for b in tl.static_range(4):
            for j in tl.static_range(8):
                col = 8 * t + 256 * b + j
                x = tl.load(X + row * D + col).to(tl.float32) + tl.load(
                    R + row * D + col
                ).to(tl.float32)
                if ROUND:
                    xr = x.to(tl.bfloat16)
                    tl.store(ROUT + row * D + col, xr)
                    xf = xr.to(tl.float32)
                else:
                    tl.store(ROUT + row * D + col, x.to(tl.bfloat16))
                    xf = x
                acc += xf * xf
    elif FOLD == 1:  # sequential, j outer, b inner
        for j in tl.static_range(8):
            for b in tl.static_range(4):
                col = 8 * t + 256 * b + j
                x = tl.load(X + row * D + col).to(tl.float32) + tl.load(
                    R + row * D + col
                ).to(tl.float32)
                if ROUND:
                    xr = x.to(tl.bfloat16)
                    tl.store(ROUT + row * D + col, xr)
                    xf = xr.to(tl.float32)
                else:
                    tl.store(ROUT + row * D + col, x.to(tl.bfloat16))
                    xf = x
                acc += xf * xf
    else:  # pairwise tree over the 32 values in (b, j) order
        b = tl.arange(0, 4)
        j = tl.arange(0, 8)
        col = 8 * t[:, None, None] + 256 * b[None, :, None] + j[None, None, :]
        x = tl.load(X + row * D + col).to(tl.float32) + tl.load(R + row * D + col).to(
            tl.float32
        )
        if ROUND:
            xr = x.to(tl.bfloat16)
            tl.store(ROUT + row * D + col, xr)
            xf = xr.to(tl.float32)
        else:
            tl.store(ROUT + row * D + col, x.to(tl.bfloat16))
            xf = x
        sq = tl.reshape(xf * xf, [32, 32])
        v = tl.sum(tl.reshape(sq, [32, 16, 2]), 2)
        v = tl.sum(tl.reshape(v, [32, 8, 2]), 2)
        v = tl.sum(tl.reshape(v, [32, 4, 2]), 2)
        v = tl.sum(tl.reshape(v, [32, 2, 2]), 2)
        acc = tl.sum(v, 1)
    total = butterfly_up(acc)
    rstd = libdevice.rsqrt(total / D + eps)
    b = tl.arange(0, 4)
    j = tl.arange(0, 8)
    col = 8 * t[:, None, None] + 256 * b[None, :, None] + j[None, None, :]
    h = tl.load(ROUT + row * D + col).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    tl.store(OUT + row * D + col, (h * rstd * (w + 0.0)).to(tl.bfloat16))


cases = []
for seed in range(100):
    torch.manual_seed(seed)
    for rows in (1, 3, 12, 32):
        x = (torch.randn(rows, 1024, device=dev) * 3).to(torch.bfloat16)
        r = (torch.randn(rows, 1024, device=dev) * 3).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(1024, device=dev)).to(torch.bfloat16)
        cases.append((x, r, w))
for fold in (0, 1, 2):
    bad = tot = 0
    for x, r, w in cases:
        ref = rmsnorm(r + x, w, 1e-6)
        out = torch.empty_like(x)
        rout = torch.empty_like(r)
        kern[(x.shape[0],)](
            x,
            r,
            w,
            out,
            rout,
            1e-6,
            D=1024,
            FOLD=fold,
            ROUND=True,
            num_warps=1,
            enable_fp_fusion=False,
        )
        tot += x.shape[0]
        bad += int((out != ref).any(dim=1).sum())
    print(
        f"rounded add+norm, per-lane fold {fold}: rows differing from add-then-flashinfer {bad}/{tot}"
    )
