"""V-e8: a flat-launch snake kernel against today's fused kernel and eager, per real shape.

Shapes: every [B, C, T] a SnakeBeta of the Qwen3-TTS decoder sees, collected by hooks over
one eager decode of every captured key (widths 1-8, 16, 32, 64 x batch 1, 2, 4, 8).
Candidate: y = x + r_c * sin(a_c * x)^2 with a and r built once by the eager expression,
one flat 1-D launch over numel, five bf16 rounding points. Switches: block size,
num_warps, 32 or 64 bit offsets, rounding by SGLang's round_bf16_to_fp32 or by a dtype
cast pair, pointers specialized on alignment or not.

Identity: every candidate against eager, torch.equal, on a random tensor and on a tensor
holding all 65,536 bf16 encodings (NaN, infinities, denormals included), per channel count.
Timing: a CUDA graph of CALLS calls per shape, median per call over REPS replays, so no
launch cost is in the number. Prints per shape eager, today's kernel (or "eager" where its
envelope rejects the shape) and each candidate, then per candidate the geometric mean and
the worst ratio against today's kernel.

usage: python vocoder_snake_v2_bench.py --model DIR
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
from triton.language.extra import libdevice
from vocoder_resident_bench import BATCHES, WIDTHS, load, random_codes

from sglang_omni.models.qwen3_tts import vocoder_kernels

CALLS = 20
REPS = 30


@triton.jit
def _round_keep_nan(value):
    # note(ratish): sin(inf) is the all-ones NaN; the bit rounding would carry it into
    # the sign bit, so a NaN passes through unrounded
    return tl.where(value != value, value, round_bf16_to_fp32(value))


def _snake(
    out_ptr,
    x_ptr,
    a_ptr,
    r_ptr,
    numel,
    channels,
    length,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    BITS: tl.constexpr,
):
    if WIDE:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    channel = (offs // length) % channels
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    if BITS:
        s = _round_keep_nan(x * a)
        sn = _round_keep_nan(libdevice.sin(s))
        p = _round_keep_nan(sn * sn)
        m = _round_keep_nan(r * p)
    else:
        s = (x * a).to(tl.bfloat16).to(tl.float32)
        sn = libdevice.sin(s).to(tl.bfloat16).to(tl.float32)
        p = (sn * sn).to(tl.bfloat16).to(tl.float32)
        m = (r * p).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offs, x + m, mask=mask)


SIZES = ["numel", "channels", "length"]
KERNELS = {
    "spec": triton.jit(do_not_specialize=SIZES)(_snake),
    "nospec": triton.jit(
        do_not_specialize=SIZES + ["out_ptr", "x_ptr", "a_ptr", "r_ptr"]
    )(_snake),
}


def candidate(block, warps, wide, bits, pointers):
    kernel = KERNELS[pointers]

    def run(x, a, r):
        out = torch.empty_like(x)
        numel = x.numel()
        kernel[(triton.cdiv(numel, block),)](
            out,
            x,
            a,
            r,
            numel,
            x.shape[1],
            x.shape[2],
            BLOCK=block,
            WIDE=wide,
            BITS=bits,
            num_warps=warps,
            enable_reflect_ftz=False,
            enable_fp_fusion=False,
        )
        return out

    return run


def per_call_us(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        for _ in range(CALLS):
            kept = fn()
    times = []
    for _ in range(REPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / CALLS)
    del graph, kept
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    tokenizer, incremental = load(args.model, device)
    print(
        f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}, triton {triton.__version__}"
    )
    snakes = [
        m for m in tokenizer.model.decoder.modules() if type(m).__name__ == "SnakeBeta"
    ]
    by_channels = {int(m.alpha.shape[0]): m for m in snakes}
    shapes: set[tuple[int, int, int]] = set()
    hooks = [
        m.register_forward_pre_hook(
            lambda _m, inputs: shapes.add(tuple(inputs[0].shape))
        )
        for m in snakes
    ]
    with torch.inference_mode():
        for width, batch in itertools.product(WIDTHS, BATCHES):
            state = incremental.init_state(batch, device=device, dtype=torch.bfloat16)
            incremental._decode_tensors(random_codes(batch, width, device), state)
    for hook in hooks:
        hook.remove()
    ordered = sorted(shapes, key=lambda s: (s[0] * s[1] * s[2], s))
    print(
        f"{len(snakes)} SnakeBeta modules, channel counts {sorted(by_channels)}, {len(ordered)} distinct shapes, numel {math.prod(ordered[0])} to {math.prod(ordered[-1])}"
    )

    constants = {}
    with torch.inference_mode():
        for channels, module in by_channels.items():
            alpha = torch.exp(module.alpha.unsqueeze(0).unsqueeze(-1))
            beta = torch.exp(module.beta.unsqueeze(0).unsqueeze(-1))
            r = 1.0 / (beta + module.no_div_by_zero)
            constants[channels] = (
                alpha.reshape(-1).contiguous(),
                r.reshape(-1).contiguous(),
            )

    configs = [
        (block, warps, False, bits, "nospec")
        for bits in (True, False)
        for block in (256, 512, 1024, 2048)
        for warps in (2, 4, 8)
    ]
    configs += [
        (512, 4, True, True, "nospec"),
        (512, 4, False, True, "spec"),
        (512, 4, False, False, "spec"),
    ]

    def label(c) -> str:
        return f"B{c[0]} w{c[1]} {'i64' if c[2] else 'i32'} {'bits' if c[3] else 'cast'} {c[4]}"

    all_bf16 = (
        torch.arange(-32768, 32768, dtype=torch.int32, device=device)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    print(
        "\nidentity against eager (random, all 65,536 bf16 encodings), per channel count"
    )
    torch.manual_seed(0)
    failed = set()
    with torch.inference_mode():
        for channels, module in by_channels.items():
            a, r = constants[channels]
            random = (
                torch.randn((2, channels, 257), dtype=torch.bfloat16, device=device) * 4
            )
            sweep = all_bf16.repeat(channels).reshape(1, channels, 65536).contiguous()
            for config in configs:
                fn = candidate(*config)
                for name, x in (("random", random), ("all", sweep)):
                    expected = module(x)
                    actual = fn(x, a, r)
                    same = torch.equal(
                        actual.view(torch.int16), expected.view(torch.int16)
                    )
                    if not same:
                        failed.add(config)
                        bad = (
                            (actual.view(torch.int16) != expected.view(torch.int16))
                            .sum()
                            .item()
                        )
                        print(
                            f"  MISMATCH C={channels} {label(config)} {name}: {bad} elements"
                        )
            old = vocoder_kernels.fused_snake_beta(sweep, module.alpha, module.beta)
            print(
                f"  C={channels}: candidates checked; today's kernel on all encodings: {'n/a' if old is None else torch.equal(old.view(torch.int16), module(sweep).view(torch.int16))}"
            )
    print(f"configs with any mismatch: {[label(c) for c in failed] or 'none'}")

    results: dict = {config: [] for config in configs}
    print(f"\nper call us in a CUDA graph ({CALLS} calls per graph, median of {REPS})")
    print(
        f"{'shape':>20} {'numel':>10} {'eager':>9} {'today':>9} "
        + " ".join(f"{label(c)[:22]:>22}" for c in configs[-3:])
        + f" {'best sweep':>26}"
    )
    for shape in ordered:
        channels = shape[1]
        module = by_channels[channels]
        a, r = constants[channels]
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
        eager_us = per_call_us(lambda: module(x))
        old_out = vocoder_kernels.fused_snake_beta(x, module.alpha, module.beta)
        today_us = (
            eager_us
            if old_out is None
            else per_call_us(
                lambda: vocoder_kernels.fused_snake_beta(x, module.alpha, module.beta)
            )
        )
        row = {}
        for config in configs:
            fn = candidate(*config)
            row[config] = per_call_us(lambda: fn(x, a, r))
            results[config].append(row[config] / today_us)
        best = min(configs[:-3], key=lambda c: row[c])
        print(
            f"{str(shape):>20} {math.prod(shape):>10} {eager_us:>9.1f} {(str(round(today_us, 1)) + ('*' if old_out is None else '')):>9} "
            + " ".join(f"{row[c]:>22.1f}" for c in configs[-3:])
            + f" {label(best)[:18]:>18} {row[best]:>7.1f}",
            flush=True,
        )
    print("\n* today's kernel rejects the shape and runs the eager chain")
    print(
        "\nper config against today's kernel: geometric mean ratio, worst ratio (above 1 is slower), shapes slower by more than 2 percent"
    )
    for config in configs:
        ratios = results[config]
        geo = math.exp(statistics.fmean(math.log(v) for v in ratios))
        print(
            f"  {label(config):32s} geo {geo:.3f}  worst {max(ratios):.3f}  slower>2% {sum(v > 1.02 for v in ratios)} of {len(ratios)}"
        )


if __name__ == "__main__":
    main()
