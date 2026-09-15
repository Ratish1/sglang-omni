#!/usr/bin/env python3
"""V0: every synchronizing call and every runtime copy inside one call of each
vocoder kind, at the branch head, attributed to the Python line that issued it.

Run on the H100 venv from the branch worktree, alone on the GPU, no server:

  python tasks/cosyvoice_utilization_20260912/stage0/v0_flow_call_syncs.py \
      --device cuda:0 --json v0.json

Each call runs twice to warm, then once under torch.cuda.set_sync_debug_mode
("warn") with every warning's stack recorded, then once under torch.profiler
with the CUDA runtime API events counted. The profiler count includes the one
device synchronize the measurement adds after the call.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import traceback
import warnings
from collections.abc import Callable

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_ID,
    build_streams,
    flow_input,
    hop_window,
    load_vocoder,
    provenance,
)

from sglang_omni.models.fun_cosyvoice3.stages import (  # noqa: E402
    FLOW_CUDA_GRAPH_FRAME_BUCKET,
    FlowCudaGraphRunner,
    pack_flow_inputs,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (  # noqa: E402
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
)

PROMPT_TOKENS = 2 * TOKEN_HOP_LEN
HOPS = ((0, TOKEN_HOP_LEN), (TOKEN_HOP_LEN, 2 * TOKEN_HOP_LEN))
ROWS = 8
MARKERS = ("sglang_omni/", "cosyvoice/", "x_transformers/", "torch/")


def short(filename: str) -> str:
    for marker in MARKERS:
        index = filename.rfind(marker)
        if index >= 0:
            return filename[index:]
    return os.path.basename(filename)


def record_syncs(call: Callable[[], object]) -> dict[str, int]:
    sites: collections.Counter[str] = collections.Counter()

    def hook(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return
        frames = [
            frame
            for frame in traceback.extract_stack()[:-1]
            if not frame.filename.endswith("warnings.py")
        ]
        sites[
            " <- ".join(
                f"{short(frame.filename)}:{frame.lineno} {frame.name}"
                for frame in reversed(frames[-4:])
            )
            + f" [{str(message)[:60]}]"
        ] += 1

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = hook
        torch.cuda.set_sync_debug_mode("warn")
        try:
            call()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    return dict(sites.most_common())


def record_runtime(call: Callable[[], object]) -> dict[str, int]:
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
        name = event.name
        if name.startswith(("cuda", "cu")) or "Memcpy" in name or "Memset" in name:
            counts[name] += 1
    return dict(counts.most_common())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1088)
    parser.add_argument("--json")
    args = parser.parse_args()

    info = provenance(args.device)
    checkpoint, vocoder = load_vocoder(args.model, args.device, torch.bfloat16)
    streams = build_streams(
        checkpoint,
        args.device,
        count=ROWS,
        prompt_tokens=PROMPT_TOKENS,
        min_generated=hop_window(*HOPS[-1]),
        samples=args.samples,
    )
    print(f"streams {[stream.sample_id for stream in streams]}")
    hop_rows = [
        flow_input(stream, hop_window(*HOPS[index % len(HOPS)]))
        for index, stream in enumerate(streams)
    ]
    final_rows = [flow_input(stream, stream.tokens.shape[1]) for stream in streams]

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        first = vocoder.hop_batch([flow_input(streams[0], hop_window(*HOPS[0]))])[0]
        _, history, emitted = vocoder.hift_delta(
            first, hift_mel=None, speech_offset=0, finalize=False
        )
        second = vocoder.hop_batch([flow_input(streams[0], hop_window(*HOPS[1]))])[0]
        second = second[:, :, HOPS[1][0] * TOKEN_MEL_RATIO :]
        leftover = vocoder.leftover_batch([final_rows[0]])[0]
        leftover = leftover[:, :, (HOPS[1][0] + HOPS[1][1]) * TOKEN_MEL_RATIO :]
        with autocast():
            buffered_mels = vocoder.flow.inference(final_rows)

        pair = final_rows[:2]
        pair_frames = max(pack_flow_inputs(vocoder.flow.flow, pair).total_mel_lengths)
        bucket = (
            -(-pair_frames // FLOW_CUDA_GRAPH_FRAME_BUCKET)
            * FLOW_CUDA_GRAPH_FRAME_BUCKET
        )
        runner = FlowCudaGraphRunner(
            vocoder.flow,
            device=torch.device(args.device),
            autocast_dtype=vocoder.autocast_dtype,
        )
        runner.capture(((len(pair), bucket),))
        vocoder.flow.attach_cuda_graph_runner(runner)
        hits: list[bool] = []
        original_run = runner.run

        def counted_run(*inputs):
            result = original_run(*inputs)
            hits.append(result is not None)
            return result

        runner.run = counted_run

        def buffered(rows):
            def call():
                with autocast():
                    return vocoder.flow.inference(rows)

            return call

        calls: dict[str, Callable[[], object]] = {
            "hop_batch rows=1": lambda: vocoder.hop_batch(hop_rows[:1]),
            f"hop_batch rows={ROWS} mixed offsets": lambda: vocoder.hop_batch(hop_rows),
            "leftover_batch rows=1": lambda: vocoder.leftover_batch(final_rows[:1]),
            f"leftover_batch rows={ROWS}": lambda: vocoder.leftover_batch(final_rows),
            "hift_delta hop with history": lambda: vocoder.hift_delta(
                second, hift_mel=history, speech_offset=emitted, finalize=False
            ),
            "hift_delta final with history": lambda: vocoder.hift_delta(
                leftover, hift_mel=history, speech_offset=emitted, finalize=True
            ),
            f"buffered flow.inference rows=2 graph key ({len(pair)}, {bucket})": buffered(
                pair
            ),
            f"buffered flow.inference rows={ROWS} no captured key": buffered(
                final_rows
            ),
            f"mel2wav_batch rows={ROWS}": lambda: vocoder.mel2wav_batch(buffered_mels),
        }

        results = {}
        for name, call in calls.items():
            hits.clear()
            call()
            call()
            torch.cuda.synchronize()
            syncs = record_syncs(call)
            runtime = record_runtime(call)
            results[name] = {
                "sync_warnings": sum(syncs.values()),
                "sync_sites": syncs,
                "runtime_events": runtime,
                "graph_hits": hits.count(True),
                "graph_misses": hits.count(False),
            }
            print(f"\n== {name}")
            print(
                f"sync warnings {sum(syncs.values())}, graph hits "
                f"{hits.count(True)}, misses {hits.count(False)}"
            )
            for site, count in syncs.items():
                print(f"  {count:3d}  {site}")
            print(
                "  runtime events: " + ", ".join(f"{k} {v}" for k, v in runtime.items())
            )

    if args.json:
        with open(args.json, "w") as out:
            json.dump({"provenance": info, "calls": results}, out, indent=1)


if __name__ == "__main__":
    main()
