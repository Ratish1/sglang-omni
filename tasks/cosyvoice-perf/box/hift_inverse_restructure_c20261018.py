"""The inverse STFT summed one bin at a time (the tree's overlap_add_inverse_kernel) against #2910's kernel (bins
padded to 16 and reduced across lanes, 256 samples on 4 warps, Triton's division) and that kernel on 64 samples and 2
warps: registers and spills, device time at the decode's shapes, the new kernel at other blocks, whether its output
is bit equal across blocks, and its difference from #2910's. usage: PYTHONPATH=<tree> python3 <this> <out_dir>"""

import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang_omni.models.fun_cosyvoice3 import hift_kernels

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda")
N_FFT, HOP, BINS = 16, 4, 9
SAMPLES_PER_FRAME = 120
CELLS = [(rows, frames) for frames in (64, 128, 256) for rows in (1, 4, 16)] + [(1, 400)]
NEW_BLOCKS = (128, 256, 512, 1024)


@triton.jit(do_not_specialize=["frames", "samples"])
def opened_kernel(
    real_ptr, imag_ptr, basis_ptr, squared_window_ptr, out_ptr, frames, samples,
    N_FFT: tl.constexpr, HOP: tl.constexpr, BINS: tl.constexpr, BINS_BLOCK: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    output = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    is_output = output < samples
    position = output + N_FFT // 2
    bins = tl.arange(0, BINS_BLOCK)
    is_bin = bins < BINS
    signal = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
    envelope = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
    for overlap in tl.static_range(N_FFT // HOP):
        frame = position // HOP - overlap
        tap = position - frame * HOP
        is_frame = is_output & (frame >= 0) & (frame < frames)
        mask = is_frame[:, None] & is_bin[None, :]
        offsets = (row * BINS + bins[None, :]) * frames + frame[:, None]
        real = tl.load(real_ptr + offsets, mask=mask, other=0.0)
        imag = tl.load(imag_ptr + offsets, mask=mask, other=0.0)
        cosine = tl.load(basis_ptr + bins[None, :] * N_FFT + tap[:, None], mask=mask, other=0.0)
        sine = tl.load(basis_ptr + (BINS + bins[None, :]) * N_FFT + tap[:, None], mask=mask, other=0.0)
        value = tl.sum(real * cosine + imag * sine, axis=1)
        signal += tl.where(is_frame, value, 0.0)
        envelope += tl.load(squared_window_ptr + tap, mask=is_frame, other=0.0)
    tl.store(out_ptr + row * samples + output, signal / envelope, mask=is_output)


def device_ms(launch, launches: int = 20, replays: int = 10, trials: int = 7) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            launch()
    per_launch = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        per_launch.append(start.elapsed_time(end) / (replays * launches))
    return statistics.median(per_launch)


def main() -> None:
    print(f"torch {torch.__version__} triton {triton.__version__} {torch.cuda.get_device_name()}", flush=True)
    print(f"hift_kernels {hift_kernels.__file__}", flush=True)
    generator = torch.Generator().manual_seed(0)
    basis = torch.randn(2 * BINS, N_FFT, generator=generator).to(DEVICE)
    squared_window = (torch.hann_window(N_FFT, periodic=False) ** 2 + 0.05).to(DEVICE)
    report = {"compiled": {}, "cells": {}}

    def launcher(arm, real, imaginary, out, frames, samples, rows):
        if arm.startswith("new"):
            block = int(arm.split()[1])
            grid = (triton.cdiv(samples, block), rows)
            return lambda: hift_kernels.overlap_add_inverse_kernel[grid](
                real, imaginary, basis, squared_window, out, frames, samples, N_FFT=N_FFT, HOP_LENGTH=HOP,
                BINS=BINS, BLOCK_SAMPLES=block)
        else:
            block, warps = (256, 4) if arm == "opened 256/4w" else (64, 2)
            grid = (triton.cdiv(samples, block), rows)
            return lambda: opened_kernel[grid](
                real, imaginary, basis, squared_window, out, frames, samples, N_FFT=N_FFT, HOP=HOP, BINS=BINS,
                BINS_BLOCK=16, BLOCK_SAMPLES=block, num_warps=warps)

    arms = ["opened 256/4w", "opened 64/2w", *(f"new {block}" for block in NEW_BLOCKS)]
    for rows, frames_of_mel in CELLS:
        frames = frames_of_mel * SAMPLES_PER_FRAME + 1
        samples = HOP * (frames - 1)
        real = torch.randn(rows, BINS, frames, generator=generator).to(DEVICE)
        imaginary = torch.randn(rows, BINS, frames, generator=generator).to(DEVICE)
        outputs = {}
        times = {}
        for arm in arms:
            out = torch.empty(rows, samples, device=DEVICE)
            launch = launcher(arm, real, imaginary, out, frames, samples, rows)
            compiled = launch()
            torch.cuda.synchronize()
            outputs[arm] = out.clone()
            report["compiled"][arm] = {"regs": compiled.n_regs, "spills": compiled.n_spills}
            times[arm] = device_ms(launch)
        reference = outputs["opened 256/4w"]
        new_reference = outputs["new 256"]
        cell = {}
        for arm in arms:
            difference = (outputs[arm] - reference).abs()
            cell[arm] = {
                "us": times[arm] * 1000,
                "vs_opened": times[arm] / times["opened 256/4w"] - 1,
                "max_abs_difference_vs_opened": difference.max().item(),
                "max_relative_difference_vs_opened": (difference / reference.abs().clamp_min(1e-3)).max().item(),
                "bit_equal_to_new_256": bool(torch.equal(outputs[arm], new_reference)),
            }
        report["cells"][f"rows={rows} F={frames_of_mel}"] = cell
        print(f"rows={rows} F={frames_of_mel}: opened 256/4w {times['opened 256/4w'] * 1000:.1f} us; " + ", ".join(
            f"{arm} {values['vs_opened']:+.1%} max|d| {values['max_abs_difference_vs_opened']:.1e} "
            f"equal-to-new-256 {values['bit_equal_to_new_256']}" for arm, values in cell.items()
            if arm != "opened 256/4w"), flush=True)
    print("compiled: " + "; ".join(f"{arm} {v['regs']} regs / {v['spills']} spills"
                                   for arm, v in report["compiled"].items()), flush=True)
    for arm in arms:
        ratios = [report["cells"][key][arm]["us"] / report["cells"][key]["opened 256/4w"]["us"]
                  for key in report["cells"]]
        geomean = statistics.geometric_mean(ratios)
        print(f"geomean {arm}: {geomean - 1:+.1%}", flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
