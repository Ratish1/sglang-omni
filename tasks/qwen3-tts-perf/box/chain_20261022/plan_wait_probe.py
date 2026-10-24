"""Does an op's first call at a new shape wait for other streams' work? Per op and shape: a side stream is given about
BUSY_MS of matmuls, then the op is called on the current stream at once; the host time until the call returns and
the device time until the op completes, against the side stream's work. A call that returns only after the side
stream drains (host or device time near the side stream's) waits device wide, as new cuFFT plans do; one that returns
at once does not. Ops: torch.stft at fresh frame counts (the speaker mel on the device), conv1d at fresh lengths with
the speaker encoder's first conv shape and with a strided conv like the reference encoder's, an elementwise add
(control), and each op again at a shape it has seen.

usage: python3 plan_wait_probe.py [--card 3]
"""

import argparse
import time

import torch
import torch.nn.functional as F

BUSY_MATMULS = 60
SIDE = 8192


def measure(op, device: torch.device, side: torch.cuda.Stream, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    torch.cuda.synchronize(device)
    side_start, side_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    op_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(side):
        side_start.record(side)
        for _ in range(BUSY_MATMULS):
            torch.matmul(a, b)
        side_end.record(side)
    host_start = time.perf_counter()
    op()
    host_ms = 1000 * (time.perf_counter() - host_start)
    op_end.record()
    torch.cuda.synchronize(device)
    return host_ms, side_start.elapsed_time(op_end), side_start.elapsed_time(side_end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    side = torch.cuda.Stream(device=device)
    a = torch.randn(SIDE, SIDE, device=device, dtype=torch.bfloat16)
    b = torch.randn(SIDE, SIDE, device=device, dtype=torch.bfloat16)
    window = torch.hann_window(1024, device=device)
    speaker_weight = torch.randn(512, 128, 5, device=device, dtype=torch.bfloat16)
    strided_weight = torch.randn(64, 32, 8, device=device, dtype=torch.bfloat16)
    torch.matmul(a, b)
    (a + b).sum()
    print("op | shape | host return ms | op done after side start ms | side work ms")
    for samples in (491_519, 983_039, 1_474_559, 983_039):
        clip = torch.randn(1, samples, device=device)
        host, done, busy = measure(
            lambda: torch.stft(clip, 1024, hop_length=256, window=window, center=False, return_complex=True),
            device, side, a, b,
        )
        print(f"stft | {samples} samples | {host:.2f} | {done:.2f} | {busy:.2f}")
    for frames in (1920, 3840, 5760, 3840):
        mels = torch.randn(1, 128, frames, device=device, dtype=torch.bfloat16)
        host, done, busy = measure(lambda: F.conv1d(mels, speaker_weight, padding=2), device, side, a, b)
        print(f"conv1d speaker | {frames} frames | {host:.2f} | {done:.2f} | {busy:.2f}")
    for length in (100_000, 200_003, 300_007, 200_003):
        values = torch.randn(1, 32, length, device=device, dtype=torch.bfloat16)
        host, done, busy = measure(lambda: F.conv1d(values, strided_weight, stride=4), device, side, a, b)
        print(f"conv1d strided | {length} samples | {host:.2f} | {done:.2f} | {busy:.2f}")
    control = torch.zeros(SIDE, SIDE, device=device)
    host, done, busy = measure(lambda: control.add_(1), device, side, a, b)
    print(f"add | {SIDE}x{SIDE} | {host:.2f} | {done:.2f} | {busy:.2f}")


if __name__ == "__main__":
    main()
