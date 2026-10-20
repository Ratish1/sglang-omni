"""Known answers for nsys GPU metrics on this card: a saturated tensor-core matmul, an HBM-bound elementwise add, and a
64-row matmul of the talker's decode shape, each run alone for about 2 s between NVTX marks.

usage: nsys profile --gpu-metrics-devices=cuda-visible --gpu-metrics-set=gh100 --trace=cuda,nvtx python sm_issue_known_answers.py
"""

import time

import torch


def run(label: str, step, seconds: float = 2.0) -> None:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    began = time.monotonic()
    count = 0
    while time.monotonic() - began < seconds:
        for _ in range(20):
            step()
        torch.cuda.synchronize()
        count += 20
    torch.cuda.nvtx.range_pop()
    print(f"{label}: {count} calls in {time.monotonic() - began:.2f} s")


device = torch.device("cuda")
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
run("gemm 8192 cube bf16", lambda: a @ b)
x = torch.randn(2**28, dtype=torch.bfloat16, device=device)
y = torch.randn(2**28, dtype=torch.bfloat16, device=device)
out = torch.empty_like(x)
run("elementwise add 2^28 bf16", lambda: torch.add(x, y, out=out))
rows = torch.randn(64, 2048, dtype=torch.bfloat16, device=device)
weight = torch.randn(6144, 2048, dtype=torch.bfloat16, device=device)
run("matmul 64 x 2048 x 6144 bf16", lambda: torch.nn.functional.linear(rows, weight))
