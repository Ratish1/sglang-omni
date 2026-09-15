#!/usr/bin/env python3
"""G0: what the shipped Flow CUDA graph table costs and buys.

Run on the H100 venv from the branch worktree, alone on the GPU, no server:

  python tasks/cosyvoice_utilization_20260912/stage0/g0_flow_graph_cost.py \
      --device cuda:0 --json g0.json

1. The shipped capture, FunCosyVoice3PipelineConfig's table through
   FlowCudaGraphRunner.capture as the factory calls it: wall time and allocator
   growth for the whole table.
2. The same table captured one shape at a time with the same body as
   FlowCudaGraphRunner.capture (one side stream, one shared pool): wall time,
   reserved growth and static buffer bytes per shape.
3. At every captured shape, the buffered Flow call three ways: graph replay,
   the padded eager solve with no runner, and the packed non streaming call
   the stream finals use; median wall of three synchronized calls after one warm
   call. Rows are equal length, so the padded frames are exactly the bucket.
4. CUDA runtime API counts for one call of each way at the smallest and the
   largest batch of the table.

The generated rows are random speech tokens behind one real reference prompt;
time and memory depend on shapes, not token values.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_ID,
    build_streams,
    flow_input,
    load_vocoder,
    provenance,
)

from sglang_omni.models.fun_cosyvoice3.config import (  # noqa: E402
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES as SHIPPED_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import VOCAB_SIZE  # noqa: E402
from sglang_omni.models.fun_cosyvoice3.stages import (  # noqa: E402
    CapturedFlowCudaGraph,
    FlowCudaGraphRunner,
    solve_flow_euler,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (  # noqa: E402
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
)

PROMPT_TOKENS = 2 * TOKEN_HOP_LEN
REPEATS = 3
MIB = 1024**2


def autocast(vocoder):
    return torch.autocast(
        device_type="cuda",
        dtype=vocoder.autocast_dtype,
        enabled=vocoder.autocast_dtype is not None,
    )


def capture_one_at_a_time(runner: FlowCudaGraphRunner, shapes) -> list[dict]:
    rows = []
    graphs = {}
    device = runner.device
    current_stream = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(current_stream)
    with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(stream):
        runner.pool = torch.cuda.graph_pool_handle()
        for batch, frames in shapes:
            torch.cuda.synchronize(device)
            reserved = torch.cuda.memory_reserved(device)
            started = time.perf_counter()
            static_inputs = runner.capture_inputs(batch, frames)
            with torch.autocast(
                device_type=device.type,
                dtype=runner.autocast_dtype,
                enabled=runner.autocast_dtype is not None,
            ):
                solve_flow_euler(runner.flow.decoder, *static_inputs)
            graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.graph(
                    cuda_graph=graph,
                    pool=runner.pool,
                    stream=stream,
                    capture_error_mode="thread_local",
                ),
                torch.autocast(
                    device_type=device.type,
                    dtype=runner.autocast_dtype,
                    enabled=runner.autocast_dtype is not None,
                ),
            ):
                static_output = solve_flow_euler(runner.flow.decoder, *static_inputs)
            torch.cuda.synchronize(device)
            graphs[(batch, frames)] = CapturedFlowCudaGraph(
                graph=graph, static_inputs=static_inputs, static_output=static_output
            )
            static_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (*static_inputs, static_output)
            )
            rows.append(
                {
                    "batch": batch,
                    "frames": frames,
                    "capture_s": time.perf_counter() - started,
                    "reserved_growth_mib": (
                        torch.cuda.memory_reserved(device) - reserved
                    )
                    / MIB,
                    "static_mib": static_bytes / MIB,
                }
            )
    current_stream.wait_stream(stream)
    runner.graphs = graphs
    return rows


def timed_ms(call) -> float:
    call()
    samples = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def runtime_counts(call) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    for event in profiler.events():
        if event.name.startswith(("cuda", "cu")) or "Memcpy" in event.name:
            counts[event.name] += 1
    return dict(counts.most_common())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json")
    args = parser.parse_args()

    info = provenance(args.device)
    device = torch.device(args.device)
    checkpoint, vocoder = load_vocoder(args.model, args.device, torch.bfloat16)
    stream = build_streams(
        checkpoint, args.device, count=1, prompt_tokens=PROMPT_TOKENS, min_generated=1
    )[0]
    print(
        f"prompt {stream.sample_id}, {PROMPT_TOKENS} tokens; {len(SHIPPED_SHAPES)} shapes"
    )

    torch.cuda.synchronize(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    shipped = FlowCudaGraphRunner(
        vocoder.flow, device=device, autocast_dtype=vocoder.autocast_dtype
    )
    shipped.capture(SHIPPED_SHAPES)
    torch.cuda.synchronize(device)
    shipped_capture = {
        "shapes": len(SHIPPED_SHAPES),
        "capture_s": time.perf_counter() - started,
        "reserved_growth_mib": (torch.cuda.memory_reserved(device) - reserved) / MIB,
        "allocated_growth_mib": (torch.cuda.memory_allocated(device) - allocated) / MIB,
    }
    print(f"\n1. shipped capture: {shipped_capture}")
    del shipped
    torch.cuda.empty_cache()

    runner = FlowCudaGraphRunner(
        vocoder.flow, device=device, autocast_dtype=vocoder.autocast_dtype
    )
    per_shape = capture_one_at_a_time(runner, SHIPPED_SHAPES)
    vocoder.flow.attach_cuda_graph_runner(runner)
    hits: list[bool] = []
    original_run = runner.run

    def counted_run(*inputs):
        result = original_run(*inputs)
        hits.append(result is not None)
        return result

    runner.run = counted_run

    generator = torch.Generator().manual_seed(0)
    by_shape = {(row["batch"], row["frames"]): row for row in per_shape}
    print("\n2 and 3. per shape")
    print(
        f"{'batch':>6}{'frames':>8}{'capture s':>11}{'reserved MiB':>14}{'replay ms':>11}"
        f"{'eager ms':>10}{'packed ms':>11}{'eager/replay':>14}"
    )
    runtime = {}
    batches = sorted({batch for batch, _ in SHIPPED_SHAPES})
    with torch.inference_mode():
        for batch, frames in sorted(SHIPPED_SHAPES):
            generated = frames // TOKEN_MEL_RATIO - PROMPT_TOKENS
            items = [
                flow_input(
                    replace(
                        stream,
                        tokens=torch.randint(
                            0,
                            VOCAB_SIZE,
                            (1, generated),
                            dtype=torch.int32,
                            generator=generator,
                        ),
                    ),
                    generated,
                )
                for _ in range(batch)
            ]

            def replay():
                with autocast(vocoder):
                    return vocoder.flow.inference(items)

            def eager():
                vocoder.flow.cuda_graph_runner = None
                try:
                    with autocast(vocoder):
                        return vocoder.flow.inference(items)
                finally:
                    vocoder.flow.cuda_graph_runner = runner

            def packed():
                with autocast(vocoder):
                    return vocoder.flow.inference_leftover(items)

            hits.clear()
            row = by_shape[(batch, frames)]
            row["replay_ms"] = timed_ms(replay)
            row["replay_hits"] = hits.count(True)
            row["replay_misses"] = hits.count(False)
            row["eager_ms"] = timed_ms(eager)
            row["packed_ms"] = timed_ms(packed)
            print(
                f"{batch:>6}{frames:>8}{row['capture_s']:>11.2f}"
                f"{row['reserved_growth_mib']:>14.1f}{row['replay_ms']:>11.1f}"
                f"{row['eager_ms']:>10.1f}{row['packed_ms']:>11.1f}"
                f"{row['eager_ms'] / row['replay_ms']:>14.2f}"
                + ("" if row["replay_misses"] == 0 else "  replay missed")
            )
            if batch in (batches[0], batches[-1]) and batch not in runtime:
                runtime[batch] = {
                    "frames": frames,
                    "replay": runtime_counts(replay),
                    "eager": runtime_counts(eager),
                    "packed": runtime_counts(packed),
                }

    print("\n4. runtime API counts per call")
    for batch, counts in runtime.items():
        for way in ("replay", "eager", "packed"):
            print(
                f"batch {batch} frames {counts['frames']} {way}: "
                + ", ".join(f"{k} {v}" for k, v in counts[way].items())
            )
    total_capture = sum(row["capture_s"] for row in per_shape)
    print(f"\none shape at a time: {total_capture:.1f} s capture in total")

    if args.json:
        with open(args.json, "w") as out:
            json.dump(
                {
                    "provenance": info,
                    "shipped_capture": shipped_capture,
                    "per_shape": per_shape,
                    "runtime": runtime,
                },
                out,
                indent=1,
            )


if __name__ == "__main__":
    main()
