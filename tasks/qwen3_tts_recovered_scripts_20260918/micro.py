"""Isolate the two restage patterns and count the CUDA runtime calls each issues.

main:     buffer[:n] = torch.tensor(vals, device=cuda, dtype=D)
refactor: buffer[:n].copy_(torch.tensor(vals, dtype=D, pin_memory=True), non_blocking=True)
"""
import torch, collections
from torch.profiler import profile, ProfilerActivity

dev = torch.device("cuda")
MAXB, N, ITERS = 16, 6, 200
bufs = [torch.zeros(MAXB, dtype=d, device=dev) for d in
        (torch.long, torch.float32, torch.float32, torch.long, torch.long, torch.bool)]
vals = [[1]*8, [0.9]*8, [0.8]*8, [40]*8, [7]*8, [True]*8]

def main_pattern():
    for b, v in zip(bufs, vals):
        b[:8] = torch.tensor(v, device=dev, dtype=b.dtype)

def refactor_pattern():
    for b, v in zip(bufs, vals):
        b[:8].copy_(torch.tensor(v, dtype=b.dtype, pin_memory=True), non_blocking=True)

def count(fn, label):
    for _ in range(20): fn()            # warm the allocator and the caches
    torch.cuda.synchronize()
    # keep the device busy so a blocking copy has something real to wait on
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(ITERS):
            torch.cuda._sleep(2_000_000)
            fn()
    torch.cuda.synchronize()
    c = collections.Counter(); t = collections.Counter()
    for e in p.events():
        if str(getattr(e, "device_type", "")).endswith("CPU") or True:
            n = e.key
            if n.startswith("cuda"):
                c[n] += 1; t[n] += e.self_cpu_time_total
    print(f"--- {label}  ({ITERS} iterations x {N} buffers) ---")
    for n, k in c.most_common(8):
        print(f"    {n:26s} {k:6d} calls   {t[n]/1000:8.1f} ms   {t[n]/k:7.1f} us/call")

count(main_pattern, "main: torch.tensor(..., device=cuda)")
count(refactor_pattern, "refactor: pinned + non_blocking copy_")
