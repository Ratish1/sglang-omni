"""Known answer for the DCGM counters before any served number is read from them (run in the
container on one card while dcgm_sampler.py watches the same card from the host).

Phases of known shape, each bracketed by host epoch markers:
  - idle: every field 0;
  - a spin kernel of n CTAs (one warp group each, a dependent FMA chain, no memory traffic) for
    n = 1, SMs/4, SMs/2, SMs: GR active 1, SM active n / SMs if the CTA scheduler spreads CTAs
    over SMs, tensor and DRAM near 0;
  - a bf16 8192^3 matmul chain: GR active 1, SM active near 1, tensor high;
  - a square wave, 20 ms of matmuls then 20 ms idle: if DCGM really samples at the requested
    interval (10 ms), the per-sample GR values are bimodal; if it averages over a longer window
    they sit near 0.5.
The check also reports the lag between the markers and the first and last busy sample, the
offset any alignment with a trace must allow for.

usage (container): python3 dcgm_known_answer.py run --markers markers.txt
       (container): python3 dcgm_known_answer.py check --markers markers.txt --samples s.tsv --gpu 0
"""

from __future__ import annotations

import argparse
import collections
import time

import torch
import triton
import triton.language as tl

PHASE_SECONDS = 0.4
IDLE_SECONDS = 0.3
WAVE_PERIODS = 25
WAVE_HALF_SECONDS = 0.02
EDGE_GUARD_US = 50_000


@triton.jit
def spin_kernel(out_ptr, iterations, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = offsets.to(tl.float32)
    for _ in range(iterations):
        x = x * 0.999 + 1.0
    tl.store(out_ptr + offsets, x)


def busy_for(seconds: float, launch) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        launch()
        torch.cuda.synchronize()


def run(markers_path: str) -> None:
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    block = 128
    out = torch.empty(sms * block, device="cuda")
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)

    def spin(ctas: int, iterations: int) -> None:
        spin_kernel[(ctas,)](out, iterations, BLOCK=block, num_warps=4)

    # size the spin so one launch runs about 2 ms
    spin(1, 1 << 12)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    spin(1, 1 << 16)
    end.record()
    end.synchronize()
    iterations = int((1 << 16) * 2.0 / start.elapsed_time(end))
    torch.matmul(a, b)
    torch.cuda.synchronize()

    markers = []

    def phase(name: str, seconds: float, launch) -> None:
        begin = time.time_ns()
        if launch is None:
            time.sleep(seconds)
        else:
            busy_for(seconds, launch)
        markers.append((name, begin, time.time_ns()))

    phase("idle", IDLE_SECONDS, None)
    for ctas in (1, sms // 4, sms // 2, sms):
        phase(f"spin_{ctas}", PHASE_SECONDS, lambda ctas=ctas: spin(ctas, iterations))
        phase("idle", IDLE_SECONDS, None)
    phase("matmul", PHASE_SECONDS, lambda: torch.matmul(a, b))
    phase("idle", IDLE_SECONDS, None)
    begin = time.time_ns()
    for _ in range(WAVE_PERIODS):
        busy_for(WAVE_HALF_SECONDS, lambda: torch.matmul(a, b))
        time.sleep(WAVE_HALF_SECONDS)
    markers.append(("square_wave", begin, time.time_ns()))
    phase("idle", IDLE_SECONDS, None)
    with open(markers_path, "w") as f:
        f.write(f"sms {sms} spin_iterations {iterations}\n")
        for name, begin, finish in markers:
            f.write(f"{name} {begin} {finish}\n")
    print(f"{len(markers)} phases written to {markers_path}")


def check(markers_path: str, samples_path: str, gpu: int) -> None:
    with open(markers_path) as f:
        header = f.readline().split()
        sms = int(header[1])
        markers = [
            (name, int(begin) // 1000, int(finish) // 1000)
            for name, begin, finish in (line.split() for line in f)
        ]
    samples = collections.defaultdict(list)
    with open(samples_path) as f:
        f.readline()
        for line in f:
            ts, card, field, value = line.split("\t")
            if int(card) == gpu:
                samples[field].append((int(ts), float(value)))
    for values in samples.values():
        values.sort()
    fields = sorted(samples)
    gr = samples["gr_active"]
    intervals = [b - a for (a, _), (b, _) in zip(gr, gr[1:])]
    intervals.sort()
    print(
        f"{len(gr)} gr_active samples, interval p50 {intervals[len(intervals) // 2] / 1000:.1f} ms "
        f"p95 {intervals[int(len(intervals) * 0.95)] / 1000:.1f} ms"
    )
    print(
        f"{'phase':<14}{'expected sm':>12}"
        + "".join(f"{field:>15}" for field in fields)
    )
    for name, begin, finish in markers:
        expected = (
            f"{int(name.split('_')[1]) / sms:.3f}" if name.startswith("spin_") else ""
        )
        row = f"{name:<14}{expected:>12}"
        for field in fields:
            inside = [
                value
                for ts, value in samples[field]
                if begin + EDGE_GUARD_US <= ts <= finish - EDGE_GUARD_US // 2
            ]
            row += f"{sum(inside) / len(inside):15.3f}" if inside else f"{'none':>15}"
        print(row)
    for name, begin, finish in markers:
        if name != "matmul":
            continue
        busy = [
            ts
            for ts, value in gr
            if begin - 100_000 <= ts <= finish + 100_000 and value > 0.5
        ]
        print(
            f"matmul edges: first busy sample {(busy[0] - begin) / 1000:+.1f} ms after start, "
            f"last busy sample {(busy[-1] - finish) / 1000:+.1f} ms after end"
        )
    for name, begin, finish in markers:
        if name != "square_wave":
            continue
        inside = [value for ts, value in gr if begin <= ts <= finish]
        low = sum(value < 0.2 for value in inside)
        high = sum(value > 0.8 for value in inside)
        print(
            f"square wave (20 ms on, 20 ms off): {len(inside)} samples, {low} below 0.2, "
            f"{high} above 0.8, {len(inside) - low - high} between; mean "
            f"{sum(inside) / len(inside):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "check"))
    parser.add_argument("--markers", required=True)
    parser.add_argument("--samples")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.markers)
    else:
        check(args.markers, args.samples, args.gpu)


if __name__ == "__main__":
    main()
