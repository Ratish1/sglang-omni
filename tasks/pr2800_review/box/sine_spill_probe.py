"""Where the bias SnakeBeta kernel's 8 spills come from: a kernel doing only libdevice's precise sine, against tl.sin.

Run on the box: CUDA_VISIBLE_DEVICES=0 python3 /workspace/box/tasks/pr2800_review/box/sine_spill_probe.py
Compiles two one-line kernels over a 1024-element tile, four warps, and prints each compiled kernel's registers and
spills. If the precise sine alone reports the spills, they are its large-argument reduction's local scratch, present
at every launch configuration, and not register pressure.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def precise_sine_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offsets, libdevice.sin(tl.load(x_ptr + offsets)))


@triton.jit
def fast_sine_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offsets, tl.sin(tl.load(x_ptr + offsets)))


@triton.jit
def copy_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offsets, tl.load(x_ptr + offsets) * 2.0)


@triton.jit
def approximate_sine_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    sine = tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;", "=r,r", [values], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(y_ptr + offsets, sine)


def main() -> None:
    x = torch.randn(4096, device="cuda")
    y = torch.empty_like(x)
    for name, kernel in (("libdevice.sin", precise_sine_kernel), ("tl.sin", fast_sine_kernel),
                         ("no sine", copy_kernel), ("sin.approx", approximate_sine_kernel)):
        compiled = kernel[(4,)](x, y, BLOCK=1024, num_warps=4)
        print(f"{name}: registers {compiled.n_regs}, spills {compiled.n_spills}")


if __name__ == "__main__":
    main()
