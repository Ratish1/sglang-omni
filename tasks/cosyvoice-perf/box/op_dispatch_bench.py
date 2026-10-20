"""Host cost per call of one function registered as torch.library.custom_op against SGLang's
register_custom_op (Library.define plus impl on the CUDA key), called through torch.ops under
torch.inference_mode as the Flow runs, and the bare function for reference. The body only
allocates its output, so the difference is the dispatch path alone.
usage: python3 op_dispatch_bench.py
"""

import time

import torch
from sglang.srt.utils.custom_op import register_custom_op

CALLS = 20000


def body(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


torch_op = torch.library.custom_op(
    "op_dispatch_bench::torch_custom_op", mutates_args=(), device_types="cuda"
)(body)


@torch_op.register_fake
def fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


sglang_op = register_custom_op(op_name="op_dispatch_bench_sglang", mutates_args=[], out_shape=0)(body)


def per_call_us(fn, x, w):
    for _ in range(1000):
        fn(x, w)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(CALLS):
        fn(x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / CALLS * 1e6


def main():
    x = torch.randn(1, 600, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(16, 31, 64, 64, device="cuda", dtype=torch.bfloat16)
    print(f"torch {torch.__version__}")
    for mode in ("inference_mode", "no_grad"):
        context = torch.inference_mode if mode == "inference_mode" else torch.no_grad
        with context():
            rows = {
                "bare function": per_call_us(body, x, w),
                "torch.library.custom_op via torch.ops": per_call_us(
                    torch.ops.op_dispatch_bench.torch_custom_op, x, w
                ),
                "sglang register_custom_op via torch.ops": per_call_us(sglang_op, x, w),
            }
        for name, us in rows.items():
            print(f"{mode:15s} {name:42s} {us:6.2f} us")


if __name__ == "__main__":
    main()
