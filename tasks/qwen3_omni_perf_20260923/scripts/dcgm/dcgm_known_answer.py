"""Known answer for the DCGM counters before any served number is read from them (run in the
container on one card while dcgm_sampler.py watches the same card from the host at 100 ms).

Phases of known shape, each bracketed by host epoch markers:
  - idle: every field 0;
  - a spin kernel of n CTAs (one warp group each, a dependent FMA chain, no memory traffic) for
    n = 1, SMs/4, SMs/2, SMs: GR active 1, SM active n / SMs if the CTA scheduler spreads CTAs
    over SMs, tensor and DRAM near 0;
  - a bf16 8192^3 matmul chain: GR active 1, SM active near 1, tensor high;
  - a square wave, 20 ms of matmuls then 20 ms idle: GR active near the duty, 0.5.

First reading (H100, 2026-09-30): DCGM reads its profiling counters every 100 ms whatever the
watch interval; at 10 ms nine samples in ten read 0, so the sampler runs at 100 ms. The check
fits the lag between a sample's timestamp and the window it covers (value at ts covers
[ts - lag - period, ts - lag]) against the markers, then reads each phase from the samples whose
window lies inside it. Any alignment with a trace uses the fitted lag.

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

PHASE_SECONDS = 1.0
IDLE_SECONDS = 0.5
WAVE_PERIODS = 50
WAVE_HALF_SECONDS = 0.02
MAX_LAG_US = 300_000


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


def overlap_us(low: int, high: int, spans) -> int:
    return sum(max(0, min(high, finish) - max(low, begin)) for begin, finish in spans)


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
    intervals = sorted(b - a for (a, _), (b, _) in zip(gr, gr[1:]))
    period = intervals[len(intervals) // 2]
    busy = [(b, f) for name, b, f in markers if name not in ("idle", "square_wave")]
    wave = [(b, f) for name, b, f in markers if name == "square_wave"]
    first, last = markers[0][1], markers[-1][2]
    fits = []
    for lag in range(0, MAX_LAG_US + 1, 2_000):
        errors = [
            (value - overlap_us(ts - lag - period, ts - lag, busy) / period) ** 2
            for ts, value in gr
            if ts - lag - period >= first
            and ts - lag <= last
            and overlap_us(ts - lag - period, ts - lag, wave) == 0
        ]
        fits.append(((sum(errors) / len(errors)) ** 0.5, lag))
    rms, lag = min(fits)
    print(
        f"{len(gr)} gr_active samples, period {period / 1000:.1f} ms; fitted lag "
        f"{lag / 1000:.0f} ms (the value at ts covers [ts - lag - period, ts - lag]), rms "
        f"against the markers {rms:.3f}"
    )
    print(
        f"{'phase':<14}{'n':>3}{'expected sm':>12}"
        + "".join(f"{field:>15}" for field in fields)
    )
    for name, begin, finish in markers:
        expected = {"square_wave": "gr 0.5"}.get(name, "")
        if name.startswith("spin_"):
            expected = f"{int(name.split('_')[1]) / sms:.3f}"
        row = ""
        count = 0
        for field in fields:
            inside = [
                value
                for ts, value in samples[field]
                if ts - lag - period >= begin and ts - lag <= finish
            ]
            count = len(inside)
            row += f"{sum(inside) / len(inside):15.3f}" if inside else f"{'none':>15}"
        print(f"{name:<14}{count:>3}{expected:>12}" + row)


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
