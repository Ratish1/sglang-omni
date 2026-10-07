# SPDX-License-Identifier: Apache-2.0
"""Sweep the vocoder's eager compiled Flow paths over served shapes in the serving
configuration, one synchronized call per shape, to find a shape that faults: buffered
calls (graph hits and eager misses), finals (leftover, packed) and whole history hops.
Run under CUDA_LAUNCH_BLOCKING=1 so a fault names its launch.

    cd <tree> && python shape_sweep.py --paths buffered final hop --rows 1 2 --out <log>
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput


def make_items(
    flow, rows: int, prompt_tokens: int, tokens: int, seed: int
) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(seed)
    return [
        FlowBatchInput(
            token=torch.randint(0, 6561, (1, tokens), generator=generator),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--paths", nargs="+", default=["buffered", "final", "hop"])
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--prompts", nargs="+", type=int, default=[30, 75, 151])
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=560)
    args = parser.parse_args()
    scheduler = stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=True,
        enable_flow_prefix_cuda_graph=True,
        flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
        flow_prefix_cache_gb=24.0,
    )
    vocoder = scheduler.vocoder
    calls = 0
    started = time.perf_counter()
    for path in args.paths:
        for rows in args.rows:
            for prompt_tokens in args.prompts:
                for tokens in range(args.min_tokens, args.max_tokens + 1):
                    items = make_items(
                        vocoder.flow, rows, prompt_tokens, tokens, seed=calls
                    )
                    frames = 2 * (prompt_tokens + tokens)
                    try:
                        with vocoder.stream_context:
                            if path == "buffered":
                                with torch.autocast(
                                    "cuda", dtype=vocoder.autocast_dtype
                                ):
                                    vocoder.flow.inference(items)
                            elif path == "final":
                                vocoder.leftover_batch(items)
                            else:
                                vocoder.hop_batch(items)
                        torch.cuda.synchronize()
                    except Exception as exc:
                        print(
                            f"FAULT path={path} rows={rows} prompt={prompt_tokens} tokens={tokens} "
                            f"frames={frames} after {calls} calls: {exc!r}",
                            flush=True,
                        )
                        raise
                    calls += 1
                    if calls % 100 == 0:
                        print(
                            f"ok {calls} calls, at path={path} rows={rows} prompt={prompt_tokens} "
                            f"tokens={tokens} frames={frames}, {time.perf_counter() - started:.0f} s",
                            flush=True,
                        )
    print(
        f"done {calls} calls without a fault, {time.perf_counter() - started:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
