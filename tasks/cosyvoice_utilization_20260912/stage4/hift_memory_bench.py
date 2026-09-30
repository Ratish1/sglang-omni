# SPDX-License-Identifier: Apache-2.0
"""Streaming HiFT step memory on the tree it runs from, on the real checkpoint's HiFT.

Replays one seeded sequence of streaming steps: 16 streams whose mel lengths follow the
SeedTTS range, each growing through the hop schedule, every step carrying the streams that
have a hop ready. Reports the peak allocated memory (live tensors) and the peak reserved
memory (the caching allocator's footprint) of the whole sequence, and the largest step.

    python hift_memory_bench.py --out /data/hm/ours.json
"""

from __future__ import annotations

import argparse
import json
import random

import torch

from sglang_omni.models.fun_cosyvoice3 import stages

STREAMS = 16
STEPS = 400
# mel frames per hop as a stream grows (25, 50, 100 tokens at two frames per token)
HOP_FRAMES = (50, 100, 200)
SAMPLES_PER_FRAME = 480
# the non final hold back both trees apply (F0 look right 3, conv_pre look right 4, one ISTFT frame)
HOLD_FRAMES = 8


def run_step(vocoder, rows: list[tuple[torch.Tensor, int, bool]]) -> None:
    if hasattr(stages, "HiftStepRow"):
        vocoder.hift_step(
            [
                stages.HiftStepRow(history=h, emitted_samples=e, is_final=f)
                for h, e, f in rows
            ]
        )
    else:
        vocoder.hift_step(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint_dir = stages.resolve_checkpoint(args.model)
    flow, hift = stages.load_cosyvoice3_flow_hift(checkpoint_dir, device="cuda")
    vocoder = stages.CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    rng = random.Random(0)
    generator = torch.Generator(device="cuda").manual_seed(0)

    def new_stream() -> dict:
        return {"length": rng.randint(100, 500), "frames": 0, "hops": 0, "emitted": 0}

    streams = [new_stream() for _ in range(STREAMS)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    largest = 0
    with torch.inference_mode(), vocoder.stream_context:
        for _ in range(STEPS):
            rows, members = [], []
            for index, stream in enumerate(streams):
                hop = HOP_FRAMES[min(stream["hops"], len(HOP_FRAMES) - 1)]
                is_final = stream["frames"] + hop >= stream["length"]
                total = stream["length"] if is_final else stream["frames"] + hop
                history = torch.randn(1, 80, total, device="cuda", generator=generator)
                rows.append((history, stream["emitted"], is_final))
                members.append((index, total, is_final))
            largest = max(largest, sum(total for _, total, _ in members))
            run_step(vocoder, rows)
            for index, total, is_final in members:
                stream = streams[index]
                if is_final:
                    streams[index] = new_stream()
                else:
                    stream["frames"] = total
                    stream["hops"] += 1
                    stream["emitted"] = max(total - HOLD_FRAMES, 0) * SAMPLES_PER_FRAME
    torch.cuda.synchronize()
    report = {
        "peak_allocated_mib": round(
            (torch.cuda.max_memory_allocated() - base_allocated) / 2**20
        ),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
        "reserved_at_end_mib": round(torch.cuda.memory_reserved() / 2**20),
        "largest_step_history_frames": largest,
    }
    print(report, flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
