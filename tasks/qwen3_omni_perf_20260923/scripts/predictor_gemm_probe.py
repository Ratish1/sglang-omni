"""Kernel time (profiler) and bit equality of cuBLAS layouts for the predictor's GEMM shapes; batch invariance of main's own path."""

import torch
from torch.profiler import ProfilerActivity, profile

dev = "cuda"
shapes = {
    "qkv": (4096, 1024),
    "o_proj": (1024, 2048),
    "gate_up": (6144, 1024),
    "down": (1024, 3072),
    "lm_head": (2048, 1024),
}


def ktime(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = sum(
        e.self_device_time_total
        for e in p.key_averages()
        if e.device_type.name == "CUDA"
    )
    names = "+".join(
        sorted({e.key[:30] for e in p.key_averages() if e.device_type.name == "CUDA"})
    )
    return tot / iters, names


print(
    "kernel time per call (us), linear (weight as stored) against matmul with the pre-transposed weight"
)
print(f"{'proj':8s} {'M':>3s} {'linear':>7s} {'NN':>7s}  kernels(linear) | kernels(NN)")
for name, (N, K) in shapes.items():
    torch.manual_seed(0)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    w_t = w.t().contiguous()
    for M in (1, 4, 12, 32):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        t1, k1 = ktime(lambda: torch.nn.functional.linear(x, w))
        t2, k2 = ktime(lambda: torch.matmul(x, w_t))
        print(f"{name:8s} {M:3d} {t1:7.2f} {t2:7.2f}  {k1} | {k2}")
print(
    "\nbit equality over 20 seeds, linear against NN, and batch invariance of linear (row 0 alone against row 0 inside a batch of M)"
)
for name, (N, K) in shapes.items():
    eq_nn = {M: 0 for M in (1, 4, 12, 32)}
    inv = {M: 0 for M in (4, 12, 32)}
    for seed in range(20):
        torch.manual_seed(seed)
        w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        w_t = w.t().contiguous()
        xs = torch.randn(32, K, device=dev).to(torch.bfloat16)
        for M in (1, 4, 12, 32):
            x = xs[:M]
            eq_nn[M] += int(
                torch.equal(torch.nn.functional.linear(x, w), torch.matmul(x, w_t))
            )
        y1 = torch.nn.functional.linear(xs[:1], w)
        for M in (4, 12, 32):
            inv[M] += int(torch.equal(y1[0], torch.nn.functional.linear(xs[:M], w)[0]))
    print(f"{name:8s} linear==NN {eq_nn}  row0 invariant across M {inv}  (of 20)")
