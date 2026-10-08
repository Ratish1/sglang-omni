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
import inspect
import sys
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput, HiftStepRow

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
    # finals inside the final graph tiers (4,096 frames per CFG half at most): 8 rows of
    # 350 frames, 16 rows of 250
    "final8": ("final", 8, 100),
    "final16_fit": ("final", 16, 50),
    "buffered_small": ("buffered", 1, 416 // 2 - PROMPT_TOKENS),
    "buffered_large": ("buffered", 16, 576 // 2 - PROMPT_TOKENS),
    "buffered_miss8": ("buffered", 8, 80),
    "buffered_miss2": ("buffered", 2, 400),
    "hift_short": ("hift", 1, 56),
    "hift_long": ("hift", 1, 800),
    # the batched streaming HiFT step (#2392) at 1 and 16 rows: a hop whose history
    # is 150 frames with 92 already emitted, a final at 250 with 192 emitted
    "hiftstep_hop1": ("hiftstep", 1, (150, 92, False)),
    "hiftstep_hop16": ("hiftstep", 16, (150, 92, False)),
    "hiftstep_final1": ("hiftstep", 1, (250, 192, True)),
    "hiftstep_final16": ("hiftstep", 16, (250, 192, True)),
    # finals over long histories, where F0 and the sine source still run over the whole
    # history while the decode runs over the window: 16 rows of 15 s, one row of 60 s
    "hiftstep_final16_long": ("hiftstep", 16, (750, 692, True)),
    "hiftstep_final1_long": ("hiftstep", 1, (3000, 2900, True)),
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
    # trees with the final graphs take their switch as a required argument
    final_graph_switch = {
        name: serving
        for name in ("enable_flow_final_cuda_graph",)
        if name in inspect.signature(stages.create_vocoder_executor).parameters
    }
    scheduler = stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        enable_dit_torch_compile=serving,
        enable_flow_cuda_graph=serving,
        enable_flow_prefix_cuda_graph=serving,
        flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
        flow_prefix_cache_gb=24.0,
        **final_graph_switch,
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
    if kind == "hiftstep":
        history_frames, emitted_frames, is_final = size
        step_rows = [
            HiftStepRow(
                history=torch.randn(
                    1, vocoder.flow.output_size, history_frames, device="cuda"
                ),
                emitted_samples=emitted_frames * vocoder.hift_samples_per_mel_frame,
                is_final=is_final,
            )
            for _ in range(rows)
        ]

        def body():
            return vocoder.hift_step(step_rows)

    elif kind == "hift":
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
