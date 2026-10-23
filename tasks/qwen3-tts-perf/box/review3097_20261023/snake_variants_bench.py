"""Why the fp32-once bias SnakeBeta runs slower than the eager-rounding one: the shipped kernel's arithmetic in four
variants on code2wav's 29 launches of a 10-frame decode (snake_kernel_bench.py's shapes), each timed in one CUDA graph
at rows 1 and 8, with the compiled kernel's registers and spills. Variants: eager rounding or fp32 rounded once, each
with FMA contraction off (as shipped) or on; and fp32 with the fast sine (tl.sin) for reference. Also the largest
error of each variant against an fp64 evaluation of the same bf16 inputs, in bf16 epsilons of the value.

usage: python3 snake_variants_bench.py [--card 2]
"""

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

LEVELS = ((768, 320), (384, 1600), (192, 6400), (96, 19200))
VECTORS_PER_PROGRAM = 256


@triton.jit(do_not_specialize=["row_vectors", "channel_vectors"])
def variant_kernel(
    conv_ptr, conv_batch_stride, bias_ptr, residual_ptr, sum_ptr, out_ptr, out_batch_stride, alpha_scale_ptr,
    inverse_beta_ptr, row_vectors, channel_vectors, HAS_RESIDUAL: tl.constexpr, WRITE_SUM: tl.constexpr,
    ROUND_LIKE_EAGER: tl.constexpr, FAST_SINE: tl.constexpr, VECTORS_PER_PROGRAM: tl.constexpr,
    VECTOR_ELEMENTS: tl.constexpr,
):
    out_dtype = out_ptr.dtype.element_ty
    batch = tl.program_id(1).to(tl.int64)
    vector = tl.program_id(0) * VECTORS_PER_PROGRAM + tl.arange(0, VECTORS_PER_PROGRAM)
    element = tl.arange(0, VECTOR_ELEMENTS)
    mask = (vector < row_vectors)[:, None]
    offsets = vector[:, None] * VECTOR_ELEMENTS + element[None, :]
    channel = (vector % channel_vectors)[:, None] * VECTOR_ELEMENTS + element[None, :]
    contiguous_base = batch * row_vectors * VECTOR_ELEMENTS
    alpha_scale = tl.load(alpha_scale_ptr + channel, mask=mask, other=0).to(tl.float32)
    inverse_beta = tl.load(inverse_beta_ptr + channel, mask=mask, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0).to(tl.float32)
    conv_values = tl.load(conv_ptr + batch * conv_batch_stride + offsets, mask=mask, other=0).to(tl.float32)
    if ROUND_LIKE_EAGER:
        total = (conv_values + bias).to(out_dtype).to(tl.float32)
    else:
        total = conv_values + bias
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + contiguous_base + offsets, mask=mask, other=0)
        total = total + residual.to(tl.float32)
        if ROUND_LIKE_EAGER:
            total = total.to(out_dtype).to(tl.float32)
        else:
            pass
    else:
        pass
    if WRITE_SUM:
        tl.store(sum_ptr + contiguous_base + offsets, total.to(out_dtype), mask=mask)
    else:
        pass
    if ROUND_LIKE_EAGER:
        phase = (total * alpha_scale).to(out_dtype).to(tl.float32)
        sine = libdevice.sin(phase).to(out_dtype).to(tl.float32)
        sine_squared = (sine * sine).to(out_dtype).to(tl.float32)
        scaled = (inverse_beta * sine_squared).to(out_dtype).to(tl.float32)
    elif FAST_SINE:
        sine = tl.sin(total * alpha_scale)
        scaled = inverse_beta * (sine * sine)
    else:
        sine = libdevice.sin(total * alpha_scale)
        scaled = inverse_beta * (sine * sine)
    tl.store(out_ptr + batch * out_batch_stride + offsets, (total + scaled).to(out_dtype), mask=mask)


VARIANTS = {
    "eager_nofma": (True, False, False),
    "eager_fma": (True, False, True),
    "fp32_nofma": (False, False, False),
    "fp32_fma": (False, False, True),
    "fp32_fastsin_nofma": (False, True, False),
}


def launch(conv, bias, alpha_scale, inverse_beta, out, residual, sum_out, variant):
    round_like_eager, fast_sine, fma = VARIANTS[variant]
    batch_size, length, channels = conv.shape
    vector_elements = 16 // out.element_size()
    row_vectors = length * channels // vector_elements
    grid = (triton.cdiv(row_vectors, VECTORS_PER_PROGRAM), batch_size)
    return variant_kernel[grid](
        conv, conv.stride(0), bias, conv if residual is None else residual, out if sum_out is None else sum_out, out,
        out.stride(0), alpha_scale, inverse_beta, row_vectors, channels // vector_elements,
        HAS_RESIDUAL=residual is not None, WRITE_SUM=sum_out is not None, ROUND_LIKE_EAGER=round_like_eager,
        FAST_SINE=fast_sine, VECTORS_PER_PROGRAM=VECTORS_PER_PROGRAM, VECTOR_ELEMENTS=vector_elements,
        num_warps=VECTORS_PER_PROGRAM // 32, enable_fp_fusion=fma, enable_reflect_ftz=False,
    )


def calls(rows: int, device: torch.device) -> list:
    def tensor(channels: int, length: int) -> torch.Tensor:
        return torch.randn(rows, length, channels, device=device, dtype=torch.bfloat16)

    def scales(channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.5
        beta = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.5
        return torch.exp(alpha), 1.0 / (torch.exp(beta) + 1e-9)

    first = tensor(1536, 40)
    result = [(first, torch.randn(1536, device=device, dtype=torch.bfloat16), *scales(1536), torch.empty_like(first),
               None, None)]
    for channels, length in LEVELS:
        bias = torch.randn(channels, device=device, dtype=torch.bfloat16)
        alpha_scale, inverse_beta = scales(channels)
        conv, out, running = tensor(channels, length), tensor(channels, length), tensor(channels, length)
        result.append((conv, bias, alpha_scale, inverse_beta, out, None, running))
        result.extend((conv, bias, alpha_scale, inverse_beta, out, None, None) for _ in range(3))
        result.extend((conv, bias, alpha_scale, inverse_beta, out, running, running) for _ in range(2))
        result.append((conv, bias, alpha_scale, inverse_beta, out, running, None))
    return result


def time_variant(variant: str, rows: int, device: torch.device) -> tuple[float, dict]:
    work = calls(rows, device)
    stream = torch.cuda.Stream(device)
    compiled = {}
    with torch.cuda.stream(stream):
        for _ in range(3):
            for item in work:
                kernel = launch(*item, variant)
                compiled[(item[5] is not None, item[6] is not None)] = kernel
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for item in work:
            launch(*item, variant)
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 200)
    resources = {f"residual{key[0]:d}_sum{key[1]:d}": (kernel.n_regs, kernel.n_spills) for key, kernel in compiled.items()}
    return statistics.median(times), resources


def max_ulps(variant: str, device: torch.device) -> float:
    torch.manual_seed(1)
    conv = torch.randn(1, 4096, 96, device=device, dtype=torch.bfloat16) * 4
    bias = torch.randn(96, device=device, dtype=torch.bfloat16)
    alpha_scale = torch.exp(torch.randn(96, device=device, dtype=torch.bfloat16))
    inverse_beta = 1.0 / (torch.exp(torch.randn(96, device=device, dtype=torch.bfloat16)) + 1e-9)
    out = torch.empty_like(conv)
    launch(conv, bias, alpha_scale, inverse_beta, out, None, None, variant)
    total = conv.double() + bias.double()
    exact = total + inverse_beta.double() * torch.sin(total * alpha_scale.double()) ** 2
    spacing = torch.finfo(torch.bfloat16).eps * exact.abs().clamp_min(1e-3)
    return float(((out.double() - exact).abs() / spacing).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    for variant in VARIANTS:
        row = {"variant": variant}
        for rows in (1, 8):
            microseconds, resources = time_variant(variant, rows, device)
            row[f"rows{rows}_us"] = round(microseconds, 2)
            row["regs_spills"] = resources
        row["max_error_bf16_eps"] = round(max_ulps(variant, device), 3)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
