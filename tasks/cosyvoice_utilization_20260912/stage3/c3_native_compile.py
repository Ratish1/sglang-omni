# SPDX-License-Identifier: Apache-2.0
"""Compile variants of CosyVoice's native DiT forward, one per process.

    python c3_native_compile.py --variant main|nested|nested_disable|eager --out DIR

main: the tree's compile_dit_backbone (dynamic=True, chunk mask disabled, one
graph per streaming mode). nested: DiTBlock.forward as one nested region inlined
flat, automatic dynamic, the chunk mask compiled. nested_disable: nested with the
chunk mask disabled. Reports compile time, graphs, recompiles, AOT cache state,
the device time of one forward per shape, and saves the outputs for comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch._dynamo as dynamo
import torch._dynamo.utils as dynamo_utils

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.utils.checkpoint import resolve_checkpoint

SHAPES = ((2, 128), (4, 256), (32, 512), (8, 1024))


def inputs(batch: int, frames: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(batch * 10000 + frames)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    mask = torch.ones(batch, 1, frames, device="cuda", dtype=dtype)
    mask[batch // 2 :, :, frames * 3 // 4 :] = 0
    return (
        randn(batch, 80, frames),
        mask,
        randn(batch, 80, frames),
        torch.full((batch,), 0.37, device="cuda", dtype=dtype),
        randn(batch, 80),
        randn(batch, 80, frames),
    )


def compare_compiled_chunk_mask(chunk_size: int) -> dict[str, object]:
    """The patched chunk mask, eager against compiled with main's options, over
    the batch and frame sizes serving presents, both streaming modes."""
    from cosyvoice.flow.DiT import dit as dit_module

    chunk_mask = dit_module.add_optional_chunk_mask

    # DiT.forward passes its chunk size as a module attribute, which Dynamo
    # specializes even under dynamic=True; a closure or argument int becomes a
    # symbol, and Inductor's range analysis of the division then fails.
    class ChunkMask(torch.nn.Module):
        def __init__(self, static_chunk_size: int) -> None:
            super().__init__()
            self.static_chunk_size = static_chunk_size

        def forward(self, xs, masks):
            return chunk_mask(xs, masks, False, False, 0, self.static_chunk_size, -1)

    def masks_for(static: int):
        eager = ChunkMask(static)
        return eager, torch.compile(eager, dynamic=True)

    variants = {static: masks_for(static) for static in (chunk_size, 0)}
    generator = torch.Generator(device="cuda").manual_seed(0)
    checked, mismatches = 0, []
    for batch in (1, 2, 5, 16, 32):
        for frames in range(4, 1300, 37):
            lengths = torch.randint(
                1, frames + 1, (batch,), device="cuda", generator=generator
            )
            valid = torch.arange(frames, device="cuda")[None] < lengths[:, None]
            xs = torch.empty(batch, frames, 8, device="cuda")
            for static, (eager, compiled) in variants.items():
                expected = eager(xs, valid[:, None].clone())
                actual = compiled(xs, valid[:, None].clone())
                checked += 1
                if not torch.equal(expected, actual):
                    mismatches.append([batch, frames, static])
                else:
                    pass
    return {"chunk_size": chunk_size, "checked": checked, "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    flow, _ = stages.load_cosyvoice3_flow_hift(
        resolve_checkpoint(args.model), device="cuda:0"
    )
    stages.patch_chunk_mask()
    estimator = flow.decoder.estimator
    for module in estimator.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    dtype = torch.bfloat16

    start = time.perf_counter()
    if args.variant == "main":
        stages.compile_dit_backbone(flow, autocast_dtype=dtype)
    elif args.variant == "mask_compiled":
        mask_report = compare_compiled_chunk_mask(estimator.static_chunk_size)
        (args.out / "mask_compiled_mask.json").write_text(json.dumps(mask_report))
        print(json.dumps(mask_report))
        start = time.perf_counter()
        # Skips the torch.compiler.disable wrap; the rest is main's compile.
        stages.CHUNK_MASK_COMPILE_DISABLED = True
        stages.compile_dit_backbone(flow, autocast_dtype=dtype)
    elif args.variant in ("nested", "nested_disable"):
        from cosyvoice.flow.DiT import dit as dit_module
        from cosyvoice.flow.DiT.modules import DiTBlock

        if args.variant == "nested_disable":
            dit_module.add_optional_chunk_mask = torch.compiler.disable(
                dit_module.add_optional_chunk_mask
            )
        else:
            pass
        DiTBlock.forward = torch.compiler.nested_compile_region(DiTBlock.forward)
        dynamo.config.inline_invoke_subgraph = True
        estimator.forward = torch.compile(estimator.forward, fullgraph=False)
    else:
        pass

    warm_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for streaming in (False, True):
            for batch, frames in SHAPES[:2]:
                estimator(*inputs(batch, frames, dtype), streaming=streaming)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - start
    warm_s = time.perf_counter() - warm_start

    timings: dict[str, float] = {}
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for streaming in (False, True):
            for batch, frames in SHAPES:
                name = f"b{batch}_t{frames}_{'stream' if streaming else 'full'}"
                args_tuple = inputs(batch, frames, dtype)
                for _ in range(2):
                    estimator(*args_tuple, streaming=streaming)
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                begin.record()
                for _ in range(5):
                    out = estimator(*args_tuple, streaming=streaming)
                end.record()
                torch.cuda.synchronize()
                timings[name] = round(begin.elapsed_time(end) / 5, 3)
                outputs[name] = out.float().cpu()
    torch.save(outputs, args.out / f"{args.variant}.pt")
    counters = {
        group: dict(values)
        for group, values in dynamo_utils.counters.items()
        if group in ("aot_autograd", "stats", "recompiles", "graph_break")
    }
    report = {
        "variant": args.variant,
        "compile_and_warmup_s": round(compile_s, 2),
        "warmup_s": round(warm_s, 2),
        "forward_ms": timings,
        "nonfinite": {
            name: int((~torch.isfinite(out)).sum()) for name, out in outputs.items()
        },
        "counters": {
            k: {str(a)[:80]: b for a, b in v.items()} for k, v in counters.items()
        },
    }
    (args.out / f"{args.variant}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
