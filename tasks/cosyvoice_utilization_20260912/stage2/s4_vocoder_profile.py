#!/usr/bin/env python3
"""Every vocoder call kind under Nsight, each kernel mapped to the module that
launched it.

One process, the branch's own factory with the hop cache and its captured
steps. Forward hooks push an NVTX range named by module path around every
module of the DiT, the Flow conditioning and HiFT, so a kernel's innermost
range is its module. Over the S1 gate's staggered schedule, at the chosen
steps, one call of each kind runs inside the capture range, after one run
outside it:

  plain_hop      vocoder.hop_batch, the whole prefix, eager
  cached_eager   vocoder.hop_batch_cached with no step sizes
  cached_replay  the same call through the captured steps
  final          vocoder.leftover_batch over the rows' tokens so far
  hift           vocoder.hift_delta per row, with the row's mel history

Run it under Nsight, then nsys_module_ledger.py on the export:

  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --trace=cuda,nvtx,osrt --cuda-graph-trace=node -o $OUT/vocoder \
    python s4_vocoder_profile.py --out $OUT --steps-profiled 0 3 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage0"))
from common import MODEL_ID, build_streams, extend_tokens, provenance  # noqa: E402

from sglang_omni.models.fun_cosyvoice3.flow_hop_cache import CachedDiT  # noqa: E402
from sglang_omni.models.fun_cosyvoice3.stages import (  # noqa: E402
    FlowBatchInput,
    create_vocoder_executor,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (  # noqa: E402
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint  # noqa: E402

PROMPT_TOKENS = (40, 55, 63, 78, 91, 110, 127, 144)
BYTES_PER_FRAME = 1_802_240
nvtx = torch.cuda.nvtx


def name_modules(root: torch.nn.Module, prefix: str) -> None:
    """An NVTX range named by module path around every forward under root."""
    for path, module in root.named_modules():
        name = f"{prefix}.{path}" if path else prefix
        module.register_forward_pre_hook(
            lambda module, args, name=name: nvtx.range_push(name)
        )
        module.register_forward_hook(
            lambda module, args, output: nvtx.range_pop(), always_call=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--stagger", type=int, default=4)
    parser.add_argument("--steps-profiled", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--samples", type=int, default=1088)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    info = provenance(args.device)
    checkpoint = resolve_checkpoint(args.model)

    prompts = tuple(
        PROMPT_TOKENS[index % len(PROMPT_TOKENS)] for index in range(args.streams)
    )
    hop, tokens = TOKEN_HOP_LEN, 0
    for _ in range(args.steps):
        tokens += hop
        hop = next_stream_hop_len(hop)
    references = build_streams(
        checkpoint,
        args.device,
        count=4 * args.streams,
        prompt_tokens=prompts * 4,
        min_generated=TOKEN_HOP_LEN + PRE_LOOKAHEAD_LEN,
        samples=args.samples,
    )
    streams = extend_tokens(references, args.streams, tokens + PRE_LOOKAHEAD_LEN)
    prompts_padded = [
        pad_flow_prompt_to_hop(
            stream.prompt_token, stream.prompt_feat, hop_len=TOKEN_HOP_LEN
        )
        for stream in streams
    ]

    plan: list[list[tuple[int, int, int]]] = []
    state = [(0, TOKEN_HOP_LEN) for _ in streams]
    final_frames = [0 for _ in streams]
    for step in range(args.steps):
        participants = []
        for index, stream in enumerate(streams):
            offset, hop = state[index]
            if step < index % args.stagger:
                continue
            if offset + hop + PRE_LOOKAHEAD_LEN > int(stream.tokens.shape[1]):
                continue
            participants.append((index, offset, hop))
            prompt_len = int(prompts_padded[index][0].shape[1])
            final_frames[index] = (prompt_len + offset + hop) * TOKEN_MEL_RATIO
            state[index] = (offset + hop, next_stream_hop_len(hop))
        plan.append(participants)

    # The buffered solver graphs stay off: they serve buffered traffic only and
    # would take the card's memory from the pool this schedule needs.
    budget = int(sum(final_frames) * BYTES_PER_FRAME * 1.1)
    device_type, _, card = args.device.partition(":")
    scheduler = create_vocoder_executor(
        args.model,
        device=device_type,
        gpu_id=int(card or 0),
        enable_flow_cuda_graph=False,
        flow_kv_cache_bytes=budget,
    )
    vocoder = scheduler.vocoder
    cache = vocoder.flow_hop_cache
    flow = vocoder.flow
    flow.cached_estimator = CachedDiT(
        flow.decoder.estimator, cache, device=args.device, capture_steps=True
    )
    scheduler.warmup_now()
    estimator = flow.cached_estimator
    sizes = estimator.sizes
    # After the capture, so the ranges annotate eager code only; a replayed
    # segment's kernels carry the range of the call that replays them.
    # The DiT is flow.decoder.estimator, so one walk of the Flow names it too.
    name_modules(flow.flow, "flow")
    name_modules(vocoder.hift, "hift")
    info["sizes"] = list(sizes)

    handles = {}
    history: dict[int, tuple[torch.Tensor | None, int]] = {}
    scenarios: list[dict] = []
    for step, participants in enumerate(plan):
        if not participants:
            continue
        items = [
            FlowBatchInput(
                token=streams[index].tokens[:, : offset + hop + PRE_LOOKAHEAD_LEN],
                prompt_token=prompts_padded[index][0],
                prompt_feat=prompts_padded[index][1],
                embedding=streams[index].embedding,
            )
            for index, offset, hop in participants
        ]
        for index, _, _ in participants:
            if index not in handles:
                handles[index] = cache.open_stream()
        row_handles = [handles[index] for index, _, _ in participants]
        starts = [handle.frames for handle in row_handles]
        for handle, (index, offset, hop) in zip(row_handles, participants, strict=True):
            prompt_len = int(prompts_padded[index][0].shape[1])
            if not cache.reserve(handle, (prompt_len + offset + hop) * TOKEN_MEL_RATIO):
                raise RuntimeError("the probe's pool is too small for its schedule")
        lanes = [lane for handle in row_handles for lane in handle.lanes]
        tails = cache.conv_tails[:, :, lanes]

        def rewind():
            for handle, start in zip(row_handles, starts, strict=True):
                handle.frames = start
            cache.conv_tails[:, :, lanes] = tails

        def cached():
            rewind()
            return vocoder.hop_batch_cached(items, row_handles)

        def cached_eager():
            estimator.sizes = ()
            try:
                return cached()
            finally:
                estimator.sizes = sizes

        emitted = cached()

        def hift():
            for (index, _, _), mel in zip(participants, emitted, strict=True):
                hift_mel, speech_offset = history.get(index, (None, 0))
                vocoder.hift_delta(
                    mel, hift_mel=hift_mel, speech_offset=speech_offset, finalize=False
                )

        calls = {
            "plain_hop": lambda: vocoder.hop_batch(items),
            "cached_eager": cached_eager,
            "cached_replay": cached,
            "final": lambda: vocoder.leftover_batch(items),
            "hift": hift,
        }
        if step in args.steps_profiled:
            for name, call in calls.items():
                call()
                torch.cuda.synchronize()
            torch.cuda.profiler.start()
            for name, call in calls.items():
                nvtx.range_push(f"scenario:{name}:step{step}")
                call()
                torch.cuda.synchronize()
                nvtx.range_pop()
            torch.cuda.profiler.stop()
            new_frames = sum(
                handle.reserved - start for handle, start in zip(row_handles, starts)
            )
            scenarios.append(
                {
                    "step": step,
                    "rows": len(items),
                    "new_frames": new_frames,
                    "window_frames": sum(handle.reserved for handle in row_handles),
                    "history_frames": [
                        (
                            0
                            if history.get(index, (None, 0))[0] is None
                            else int(history[index][0].shape[2])
                        )
                        for index, _, _ in participants
                    ],
                }
            )
            print(json.dumps(scenarios[-1]))
        # The rows' HiFT history moves on as the scheduler moves it.
        for (index, _, _), mel in zip(participants, emitted, strict=True):
            hift_mel, speech_offset = history.get(index, (None, 0))
            _, hift_mel, speech_offset = vocoder.hift_delta(
                mel, hift_mel=hift_mel, speech_offset=speech_offset, finalize=False
            )
            history[index] = (hift_mel, speech_offset)

    with open(os.path.join(args.out, "s4_scenarios.json"), "w") as out:
        json.dump({"info": info, "scenarios": scenarios}, out, indent=1)


if __name__ == "__main__":
    main()
