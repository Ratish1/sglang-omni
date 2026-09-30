# SPDX-License-Identifier: Apache-2.0
"""Streaming HiFT step cost on the tree it runs from, on the real checkpoint's HiFT.

One step of R rows, each row a mel history of H frames of which the last N are new, as a
causal step presents them: main's per request hift_delta loop, or the tree's hift_step
(the #2392 tuple rows or HiftStepRow). Median wall per step with the device synchronized,
and the CUDA kernel count of one step.

    python hift_step_bench.py --out /data/hb/main.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3 import stages

# rows and new frames at the hop sizes a stream grows through (25, 50, 100 tokens), each
# over a history of the frames already emitted
CASES = (
    (1, 50, 50),
    (1, 200, 550),
    (4, 100, 250),
    (16, 50, 50),
    (16, 100, 150),
    (16, 200, 350),
    (16, 200, 950),
)
REPEATS = 20


def run_step(vocoder, histories: list[torch.Tensor], new_frames: int) -> None:
    samples_per_frame = 480
    if hasattr(stages, "HiftStepRow"):
        vocoder.hift_step(
            [
                stages.HiftStepRow(
                    history=history,
                    emitted_samples=(history.shape[2] - new_frames) * samples_per_frame,
                    is_final=False,
                )
                for history in histories
            ]
        )
    elif hasattr(vocoder, "hift_step"):
        vocoder.hift_step(
            [
                (history, (history.shape[2] - new_frames) * samples_per_frame, False)
                for history in histories
            ]
        )
    else:
        for history in histories:
            vocoder.hift_delta(
                history[:, :, history.shape[2] - new_frames :],
                hift_mel=history[:, :, : history.shape[2] - new_frames],
                speech_offset=(history.shape[2] - new_frames) * samples_per_frame,
                finalize=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint_dir = stages.resolve_checkpoint(args.model)
    flow, hift = stages.load_cosyvoice3_flow_hift(checkpoint_dir, device="cuda")
    vocoder = stages.CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(0)
    report: dict[str, dict[str, float]] = {}
    with torch.inference_mode(), vocoder.stream_context:
        for rows, new_frames, history_frames in CASES:
            histories = [
                torch.randn(1, 80, history_frames, device="cuda", generator=generator)
                for _ in range(rows)
            ]
            for _ in range(3):
                run_step(vocoder, histories, new_frames)
            torch.cuda.synchronize()
            walls = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                run_step(vocoder, histories, new_frames)
                torch.cuda.synchronize()
                walls.append((time.perf_counter() - start) * 1e3)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run_step(vocoder, histories, new_frames)
                torch.cuda.synchronize()
            kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
            name = f"rows={rows} new={new_frames} history={history_frames}"
            report[name] = {
                "wall_ms_median": round(statistics.median(walls), 2),
                "kernels": len(kernels),
                "device_ms": round(sum(e.device_time for e in kernels) / 1e3, 2),
            }
            print(name, report[name], flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
