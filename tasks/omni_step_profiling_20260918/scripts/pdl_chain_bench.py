"""What a kernel boundary costs inside a CUDA graph on this card, and what PDL takes off.

Chains of dependent kernels at the predictor's bs 16 shapes, captured in one graph and
timed with events; the time per link is the kernel plus its boundary.

- floor: a one-element add, the cheapest dependent node;
- norm: FlashInfer rmsnorm (hidden 1024) after rmsnorm, PDL off and on;
- cuBLAS then norm: o_proj (2048 to 1024, weights from HBM) then rmsnorm, the norm's
  PDL off and on (cuBLAS never triggers its dependents);
- tiny then norm: SGLang's tiny GEMM (o_proj) then rmsnorm. The tiny GEMM's PDL is a
  compile-time flag of its JIT module, so --tiny-no-pdl runs this chain with it off
  in a separate process.

usage: python pdl_chain_bench.py [--tiny-no-pdl] [--rows 16]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

LINKS = 300
HIDDEN = 1024
O_PROJ_K = 2048


def graph_time_us(links) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for link in links[:4]:
            link()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for link in links:
            link()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(30):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / len(links))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny-no-pdl", action="store_true")
    parser.add_argument("--rows", type=int, default=16)
    args = parser.parse_args()
    import sglang.kernels.ops.gemm.tiny_gemm as tiny_module

    if args.tiny_no_pdl:
        tiny_module.is_arch_support_pdl = lambda: False
    else:
        pass
    import flashinfer

    device = torch.device("cuda")
    rows = args.rows
    torch.manual_seed(0)
    a = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    norm_weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    x = torch.randn(rows, O_PROJ_K, device=device, dtype=torch.bfloat16)
    weights = [
        torch.randn(HIDDEN, O_PROJ_K, device=device, dtype=torch.bfloat16) * 0.02
        for _ in range(64)
    ]
    one = torch.zeros(1, device=device)

    def norm(src, dst, pdl):
        return lambda: flashinfer.rmsnorm(
            src, norm_weight, 1e-6, out=dst, enable_pdl=pdl
        )

    print(f"rows {rows}; us per link, {LINKS} links in one graph")
    floor = graph_time_us([lambda: one.add_(1.0) for _ in range(LINKS)])
    print(f"  floor, one-element add                 {floor:6.2f}")
    for pdl in (False, True):
        links = [
            norm(a, b, pdl) if i % 2 == 0 else norm(b, a, pdl) for i in range(LINKS)
        ]
        print(
            f"  rmsnorm after rmsnorm, PDL {pdl!s:<5}        {graph_time_us(links):6.2f}"
        )
    alone = graph_time_us(
        [lambda w=weights[i % 64]: F.linear(x, w, out=None) for i in range(LINKS // 2)]
    )
    print(f"  cuBLAS o_proj alone                    {alone:6.2f}")
    for pdl in (False, True):
        links = []
        for i in range(LINKS // 2):
            w = weights[i % 64]
            links.append(lambda w=w: torch.matmul(x, w.t(), out=a))
            links.append(norm(a, b, pdl))
        per_pair = 2 * graph_time_us(links)
        print(
            f"  cuBLAS then rmsnorm, norm PDL {pdl!s:<5}   {per_pair:6.2f} per pair, "
            f"{per_pair - alone:5.2f} for the norm"
        )
    tiny_pdl = "off" if args.tiny_no_pdl else "on"
    tiny_alone = graph_time_us(
        [
            lambda w=weights[i % 64]: tiny_module.tiny_gemm_bf16(x, w, a, max_m=16)
            for i in range(LINKS // 2)
        ]
    )
    print(f"  tiny o_proj alone, PDL {tiny_pdl:<3}             {tiny_alone:6.2f}")
    for pdl in (False, True):
        links = []
        for i in range(LINKS // 2):
            w = weights[i % 64]
            links.append(lambda w=w: tiny_module.tiny_gemm_bf16(x, w, a, max_m=16))
            links.append(norm(a, b, pdl))
        per_pair = 2 * graph_time_us(links)
        print(
            f"  tiny (PDL {tiny_pdl}) then rmsnorm, norm PDL {pdl!s:<5} {per_pair:6.2f} per pair, "
            f"{per_pair - tiny_alone:5.2f} for the norm"
        )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        norm(a, b, True)()
        torch.matmul(x, weights[0].t(), out=a)
        tiny_module.tiny_gemm_bf16(x, weights[0], a, max_m=16)
        torch.cuda.synchronize()
    names = {
        event.name[:90] for event in prof.events() if event.device_type.name == "CUDA"
    }
    print("kernels:", *sorted(names), sep="\n  ")


if __name__ == "__main__":
    main()
