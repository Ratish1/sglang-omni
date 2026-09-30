# SPDX-License-Identifier: Apache-2.0
"""One side of the omni-gpu-deep-dive trace pair for one vocoder call at a served extreme.

mapping: the vocoder built with DiT compile and the Flow CUDA graph off, stacks on; it
names the sglang_omni line of every kernel. formal: the serving configuration, stacks off;
it gives every number. Each side runs in its own process so the eager side never shares
a process with Inductor's state. Both write into --out (mapping/, formal/), which
analyze_llm_torch_profile.py --mapping-input --formal-input reads.

    cd <tree> && python vocoder_trace_pair.py --side mapping --cases hop1 hop16 --out /data/p1
    cd <tree> && python vocoder_trace_pair.py --side formal  --cases hop1 hop16 --out /data/p1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput

# run from the root of the tree under test, as the skill documents
sys.path.insert(0, str(Path.cwd() / ".claude/skills/omni-gpu-deep-dive/scripts"))
PROMPT_TOKENS = 75
LOOKAHEAD = 3
# hop and final at the smallest and largest row count a step takes (max_batch_size 16);
# buffered at the smallest and largest capture shape; HiFT over a first hop's mel and
# over a 16 s history
CASES = {
    "hop1": ("hop", 1, 25 + LOOKAHEAD),
    "hop16": ("hop", 16, 25 + LOOKAHEAD),
    "final1": ("final", 1, 100),
    "final16": ("final", 16, 100),
    "buffered_small": ("buffered", 1, 416 // 2 - PROMPT_TOKENS),
    "buffered_large": ("buffered", 16, 576 // 2 - PROMPT_TOKENS),
    "hift_short": ("hift", 1, 56),
    "hift_long": ("hift", 1, 800),
}


def make_items(rows: int, tokens: int, flow) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(rows * 1000 + tokens)
    return [
        FlowBatchInput(
            token=torch.randint(0, 6561, (1, tokens), generator=generator),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--side", choices=("mapping", "formal"), required=True)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), required=True)
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
        enable_flow_cuda_graph=serving,
        flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
    )
    vocoder = scheduler.vocoder
    for case in args.cases:
        capture(
            output_dir=Path(args.out) / case,
            tag=args.side,
            body=case_body(vocoder, case),
            iters=args.iters,
            warmup=args.warmup,
            with_stack=not serving,
        )


def case_body(vocoder, case: str):
    kind, rows, size = CASES[case]
    if kind == "hift":
        mel = torch.randn(1, vocoder.flow.output_size, size, device="cuda")

        def body():
            return vocoder.hift_delta(
                mel, hift_mel=None, speech_offset=0, finalize=False
            )

    else:
        items = make_items(rows, size, vocoder.flow)

        def buffered():
            with torch.autocast("cuda", dtype=vocoder.autocast_dtype):
                return vocoder.flow.inference(items)

        body = {
            "hop": lambda: vocoder.hop_batch(items),
            "final": lambda: vocoder.leftover_batch(items),
            "buffered": buffered,
        }[kind]

    def in_stream():
        with vocoder.stream_context:
            return body()

    return in_stream


if __name__ == "__main__":
    main()
