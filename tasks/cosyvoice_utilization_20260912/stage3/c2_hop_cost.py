# SPDX-License-Identifier: Apache-2.0
"""Cost of one streaming Flow call on the tree it runs from.

Builds the vocoder as serving does, then times hop_batch and leftover_batch at
1, 4 and 16 rows: median wall per call, and from one profiled call the CUDA
kernel count, the device time and the host time spent launching.

    python c2_hop_cost.py --out /data/c3/main.json [--no-compile]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput

PROMPT_TOKENS = 75
HOP_TOKENS = (28, 53, 103)
LOOKAHEAD = 3
REPEATS = 20


def make_items(
    rows: int, target_tokens: int, flow, prompt_tokens: int = PROMPT_TOKENS
) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(rows * 1000 + target_tokens)
    return [
        FlowBatchInput(
            token=torch.randint(0, 6561, (1, target_tokens), generator=generator),
            prompt_token=torch.randint(
                0, 6561, (1, prompt_tokens), generator=generator
            ),
            prompt_feat=torch.randn(
                1, prompt_tokens * 2, flow.output_size, generator=generator
            ),
            embedding=torch.randn(
                1, flow.spk_embed_affine_layer.in_features, generator=generator
            ),
        )
        for _ in range(rows)
    ]


def measure(call, items) -> dict[str, float]:
    for _ in range(3):
        call(items)
    torch.cuda.synchronize()
    walls, hosts = [], []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call(items)
        hosts.append((time.perf_counter() - start) * 1e3)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - start) * 1e3)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call(items)
        torch.cuda.synchronize()
    events = [event for event in prof.events() if event.device_type.name == "CUDA"]
    kernels = [
        event for event in events if event.name != "Memcpy HtoD (Pageable -> Device)"
    ]
    by_name: dict[str, list[float]] = {}
    for event in kernels:
        count_and_ms = by_name.setdefault(event.name[:90], [0, 0.0])
        count_and_ms[0] += 1
        count_and_ms[1] += event.device_time / 1e3
    top = sorted(by_name.items(), key=lambda pair: -pair[1][1])[:25]
    return {
        "wall_ms_median": round(statistics.median(walls), 2),
        "host_ms_median": round(statistics.median(hosts), 2),
        "kernels": len(kernels),
        "device_ms": round(sum(event.device_time for event in kernels) / 1e3, 2),
        "top_kernels": [[name, count, round(ms, 2)] for name, (count, ms) in top],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--no-compile", action="store_true")
    # One row, the shapes streaming c1 presents: every hop size and finals over
    # the whole utterance range, at three prompt lengths.
    parser.add_argument("--c1-sweep", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scheduler = stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        enable_dit_torch_compile=not args.no_compile,
        flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
    )
    vocoder = scheduler.vocoder
    report: dict[str, dict[str, float]] = {}
    with vocoder.stream_context:
        for prompt in (50, 100, 150) if args.c1_sweep else ():
            for target in HOP_TOKENS:
                items = make_items(1, target + LOOKAHEAD, vocoder.flow, prompt)
                report[f"hop prompt={prompt} tokens={target}"] = measure(
                    vocoder.hop_batch, items
                )
            for target in (50, 100, 150, 200, 250, 300, 350):
                items = make_items(1, target, vocoder.flow, prompt)
                report[f"final prompt={prompt} tokens={target}"] = measure(
                    vocoder.leftover_batch, items
                )
        for rows in () if args.c1_sweep else (1, 4, 16):
            for target in HOP_TOKENS:
                items = make_items(rows, target + LOOKAHEAD, vocoder.flow)
                report[f"hop rows={rows} tokens={target}"] = measure(
                    vocoder.hop_batch, items
                )
            items = make_items(rows, HOP_TOKENS[-1], vocoder.flow)
            report[f"final rows={rows} tokens={HOP_TOKENS[-1]}"] = measure(
                vocoder.leftover_batch, items
            )

            def buffered(batch: list[FlowBatchInput]) -> list[torch.Tensor]:
                with torch.autocast("cuda", dtype=vocoder.autocast_dtype):
                    return vocoder.flow.inference(batch)

            report[f"buffered rows={rows} tokens={HOP_TOKENS[-1]}"] = measure(
                buffered, items
            )
        # Captured (requests, frames) shapes: these replay the buffered CUDA graph.
        graphed = () if args.c1_sweep else ((16, 576), (9, 560), (5, 544), (3, 528))
        for rows, frames in graphed:
            items = make_items(rows, frames // 2 - PROMPT_TOKENS, vocoder.flow)
            report[f"graphed rows={rows} frames={frames}"] = measure(buffered, items)
    for name, values in report.items():
        print(
            name, {key: value for key, value in values.items() if key != "top_kernels"}
        )
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
