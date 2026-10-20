"""Registers, spills and the division and product instructions of the tree's HiFT kernels, launched as their
launchers launch them. usage: PYTHONPATH=<tree> python3 hift_kernel_registers_c20261018.py"""

import re

import torch
import triton

from sglang_omni.models.fun_cosyvoice3 import hift_kernels as kernels


def report(name: str, compiled) -> None:
    ptx = compiled.asm["ptx"]
    counts = {op: len(re.findall(re.escape(op), ptx)) for op in ("mul.rn.f32", "fma.rn.f32", "div.full.f32",
                                                                  "div.rn.f32")}
    print(f"{name}: regs {compiled.n_regs} spills {compiled.n_spills} {counts}")


x = torch.randn(1, 64, 2048, device="cuda")
alpha = torch.rand(64, device="cuda") + 0.5
out = torch.empty_like(x)
report("snake", kernels.snake_kernel[(triton.cdiv(2048, kernels.SAMPLES_PER_PROGRAM), 64, 1)](
    x, alpha, 1.0 / (alpha + 1e-9), out, 2048, 64, BLOCK_SAMPLES=kernels.SAMPLES_PER_PROGRAM))
padded = torch.randn(1, 1040, device="cuda")
basis = torch.randn(18, 16, device="cuda")
spectrum = torch.empty(1, 18, 257, device="cuda")
report("short time spectrum", kernels.short_time_spectrum_kernel[(triton.cdiv(257, kernels.FRAMES_PER_PROGRAM), 1)](
    padded, basis, spectrum, 257, 1040, N_FFT=16, HOP_LENGTH=4, BINS=9, BLOCK_FRAMES=kernels.FRAMES_PER_PROGRAM))
real = torch.randn(1, 9, 257, device="cuda")
signal = torch.empty(1, 1024, device="cuda")
report("overlap add inverse", kernels.overlap_add_inverse_kernel[
    (triton.cdiv(1024, kernels.OUTPUT_SAMPLES_PER_PROGRAM), 1)](
    real, real, basis, torch.rand(16, device="cuda") + 0.1, signal, 257, 1024, N_FFT=16, HOP_LENGTH=4, BINS=9,
    BLOCK_SAMPLES=kernels.OUTPUT_SAMPLES_PER_PROGRAM))
