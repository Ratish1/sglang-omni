# SPDX-License-Identifier: Apache-2.0
"""One side of the omni-gpu-deep-dive trace pair for #2406's cached Flow hop and the
whole-history hop it replaces, at the same served shape.

mapping: DiT compile off (the prefix forward runs eager), stacks on; it names the
sglang_omni line of every kernel. formal: the serving configuration, stacks off; it gives
every number. Each side runs in its own process. A cached case fills its rows' prefix
once, then every call restores the rows' frame counts and conv contexts, so each call
recomputes the same hop and rewrites the same pages with the same values.

    cd <tree> && python prefix_trace_pair.py --side mapping --out /data/t1
    cd <tree> && python prefix_trace_pair.py --side formal  --out /data/t1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput

sys.path.insert(0, str(Path.cwd() / ".claude/skills/omni-gpu-deep-dive/scripts"))
PROMPT_TOKENS = 75
LOOKAHEAD = 3
# (path, cached tokens before the hop, hop tokens, rows): the first three hops of the
# default plan (25, 50, 100), each on the cache and as the whole-history hop, at the
# largest step (16 rows), and hop 2 on the cache at 1 and 4 rows for the per step floor
CASES = {
    "prefix_first": ("prefix", 0, 25, 16),
    "whole_first": ("whole", 0, 25, 16),
    "prefix_hop2": ("prefix", 25, 50, 16),
    "whole_hop2": ("whole", 25, 50, 16),
    "prefix_hop3": ("prefix", 75, 100, 16),
    "whole_hop3": ("whole", 75, 100, 16),
    "prefix_hop2_r1": ("prefix", 25, 50, 1),
    "prefix_hop2_r4": ("prefix", 25, 50, 4),
}


def make_streams(flow, total_tokens: int, rows: int) -> list[dict]:
    generator = torch.Generator().manual_seed(rows * 1000 + total_tokens)
    return [
        dict(
            token=torch.randint(0, 6561, (1, total_tokens), generator=generator),
            prompt_token=torch.randint(
                0, 6561, (1, PROMPT_TOKENS), generator=generator
            ),
            prompt_feat=torch.randn(
                1, PROMPT_TOKENS * 2, flow.output_size, generator=generator
            ),
            embedding=torch.randn(
                1, flow.spk_embed_affine_layer.in_features, generator=generator
            ),
        )
        for _ in range(rows)
    ]


def items_until(streams: list[dict], tokens: int) -> list[FlowBatchInput]:
    return [
        FlowBatchInput(
            token=stream["token"][:, : tokens + LOOKAHEAD],
            prompt_token=stream["prompt_token"],
            prompt_feat=stream["prompt_feat"],
            embedding=stream["embedding"],
        )
        for stream in streams
    ]


def frames_of(tokens: int) -> int:
    return (PROMPT_TOKENS + tokens) * 2


def case_body(vocoder, case: str):
    path, offset, hop, rows = CASES[case]
    streams = make_streams(vocoder.flow, offset + hop + LOOKAHEAD, rows)
    items = items_until(streams, offset + hop)
    caches: list = []
    if path == "whole":

        def body():
            return vocoder.hop_batch(items)

    else:
        caches = [vocoder.prefix_cache_rows(frames_of(offset + hop)) for _ in streams]
        assert all(cache is not None for cache in caches)
        with torch.inference_mode(), vocoder.stream_context:
            done, length = 0, 25
            while done < offset:
                vocoder.hop_batch_prefix(items_until(streams, done + length), caches)
                done += length
                length = min(100, length * 2)
            assert done == offset, (done, offset)
        saved = [
            [(row.committed_frames, row.conv_context) for row in pair]
            for pair in caches
        ]

        def body():
            for pair, rows in zip(caches, saved, strict=True):
                for row, (frames, context) in zip(pair, rows, strict=True):
                    row.committed_frames, row.conv_context = frames, context
            return vocoder.hop_batch_prefix(items, caches)

    def in_stream():
        with torch.inference_mode(), vocoder.stream_context:
            return body()

    return in_stream, caches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--side", choices=("mapping", "formal"), required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=sorted(CASES), default=list(CASES)
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from omni_trace_pair import capture

    serving = args.side == "formal"
    scheduler = stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        enable_dit_torch_compile=serving,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=serving,
        flow_prefix_cache_gb=24.0,
    )
    vocoder = scheduler.vocoder
    assert vocoder.flow.prefix_pool is not None
    for case in args.cases:
        body, caches = case_body(vocoder, case)
        capture(
            output_dir=Path(args.out) / case,
            tag=args.side,
            body=body,
            iters=args.iters,
            warmup=args.warmup,
            with_stack=not serving,
        )
        for cache in caches:
            vocoder.release_prefix_cache(cache)


if __name__ == "__main__":
    main()
